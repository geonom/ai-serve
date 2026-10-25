"""vision-serve as an ai-serve `vision` plugin (ai-serve-ui docs/plugins.md): faces in photos -- boxes, 5 landmarks,
quality, and embeddings for recognising people (AuraFace + YuNet, both free for any use). Objects are a task of the
contract too; no model for them yet.

The models live in a worker process (worker.py), started on the first request after asking ai-serve's GPU manager
where there is room (POST /api/coordination/gpu-preflight without a GPU); a no means the CPU (slower, never a
failure). Stopped after VISION_KEEP_ALIVE_S idle seconds or on `release`.

Privacy (owner + Tapestry, 2026-10-25): images and face embeddings are biometric data. This plugin keeps nothing --
no files, no caches -- and logs only counts and times."""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile

from vision_plugin import install

HERE = Path(__file__).resolve().parent.parent
logger = logging.getLogger("vision-serve")


def _load_env() -> None:
    f = HERE / ".env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
GATEWAY = os.environ.get("AI_SERVE_URL", "http://127.0.0.1:8030").rstrip("/")
NEED_MB = int(os.environ.get("VISION_NEED_MB", "1000"))
KEEP_ALIVE_S = int(os.environ.get("VISION_KEEP_ALIVE_S", "300"))
MAX_IMAGES = 16
MAX_IMAGE_BYTES = 40 * 1024 * 1024
LOG = HERE / "logs" / "vision-serve.log"
FACE_MODEL = "auraface"


def installed() -> bool:
    return install.present("yunet") and install.present("auraface")


def catalog() -> Dict[str, Any]:
    return {"object": "list", "data": [{
        "id": FACE_MODEL, "label": "AuraFace (faces)", "tasks": ["faces"], "default": True,
        "available": installed(), "max_images": MAX_IMAGES, "aliases": ["faces", "auraface-v1"],
        "description": "Finds faces (YuNet: box, 5 landmarks, quality) and describes each as a 512-number vector "
                       "for recognising people (AuraFace, ResNet100 with ArcFace loss).",
        "license": "Apache-2.0 + MIT", "commercial_use": True,
        # vectors of another model or version are not comparable; thresholds: cosine of two faces' vectors
        "embedding": {"model": "auraface-v1", "dim": 512, "metric": "cosine",
                      "thresholds": {"maybe": 0.30, "likely": 0.45}}}]}


class _Retry(Exception):
    pass


class Worker:
    def __init__(self):
        self.proc: Optional[mp.process.BaseProcess] = None
        self.conn = None
        self.gpu: Optional[int] = None
        self.device = ""
        self.busy = 0
        self.last_used = 0.0
        self.lock = asyncio.Lock()
        self.idle: Optional[asyncio.Task] = None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc and self.proc.is_alive() else None

    async def _gateway(self, method: str, path: str, **kw) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60.0) as c:
            return await c.request(method, f"{GATEWAY}{path}", **kw)

    async def _start(self, gpu: Optional[int]) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        from vision_plugin import worker
        self.proc = ctx.Process(target=worker.main, daemon=True,
                                args=(child, gpu, str(install.path_of("yunet")), str(install.path_of("auraface"))))
        self.proc.start()
        child.close()
        self.conn, self.gpu = parent, gpu
        msg = await asyncio.to_thread(self._recv, 180)
        if not msg.get("ready"):
            await self.stop()
            if msg.get("oom"):
                raise _Retry("out of memory while loading")
            raise HTTPException(status_code=500, detail=f"vision worker didn't start: {msg.get('error')}")
        self.device = msg.get("device", "")
        logger.info(f"worker {self.proc.pid} ready on {'GPU ' + str(gpu) if gpu is not None else 'CPU'} ({self.device})")

    def _recv(self, timeout: float) -> Dict[str, Any]:
        if not self.conn.poll(timeout):
            return {"ready": False, "error": "no answer from the worker"}
        try:
            return self.conn.recv()
        except (EOFError, OSError):
            return {"ready": False, "error": "the worker ended"}

    async def stop(self) -> bool:
        if not self.proc:
            return False
        was = self.proc.is_alive()
        self.proc.terminate()
        await asyncio.to_thread(self.proc.join, 10)
        if self.proc.is_alive():
            self.proc.kill()
            await asyncio.to_thread(self.proc.join, 5)
        self.proc, self.conn, self.device = None, None, ""
        if was:
            logger.info("worker stopped")
        return was

    async def _ensure(self, cpu: bool) -> Optional[str]:
        """A running worker; returns a reservation id to release after the first batch."""
        if self.pid and not (cpu and self.gpu is not None):
            return None
        if self.pid:
            await self.stop()
        if cpu:
            await self._start(None)
            return None
        try:
            r = await self._gateway("POST", "/api/coordination/gpu-preflight", json={
                "purpose": "vision:faces", "need_mb": NEED_MB, "current_gpu": None, "movable": True})
            d = r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            d = {}
        if not d.get("granted") or d.get("gpu") is None:
            logger.info(f"no GPU room ({d.get('reason') or 'ai-serve unreachable or refused'}) -- using the CPU")
            await self._start(None)
            return None
        await self._start(int(d["gpu"]))
        rid = d.get("reservation_id")
        if rid and self.pid:
            try:
                await self._gateway("POST", f"/api/coordination/reservations/{rid}/confirm", json={"pid": self.pid})
            except httpx.HTTPError:
                pass
        return rid

    async def analyze(self, images: List[bytes], embed: bool) -> Dict[str, Any]:
        async with self.lock:   # one batch at a time
            self.busy += 1
            try:
                for attempt in (1, 2, 3):        # GPU, GPU again after the manager made room, then the CPU
                    rid = None
                    try:
                        rid = await self._ensure(cpu=attempt == 3)
                        self.conn.send({"images": images, "embed": embed})
                        msg = await asyncio.to_thread(self._recv, 600)
                        if "ok" in msg:
                            logger.info(f"{len(images)} image(s), {sum(len(i.get('faces', [])) for i in msg['ok'])} "
                                        f"face(s) in {msg.get('seconds')} s on {'GPU ' + str(self.gpu) if self.gpu is not None else 'CPU'}")
                            return {"images": msg["ok"]}
                        if msg.get("oom") or msg.get("error") in ("the worker ended", "no answer from the worker"):
                            raise _Retry(msg.get("error", ""))
                        logger.error(f"worker error: {msg.get('error')}")
                        raise HTTPException(status_code=500, detail="face analysis failed")
                    except _Retry as e:
                        logger.warning(f"{e} (attempt {attempt}) -- starting the worker again")
                        await self.stop()
                    finally:
                        if rid:
                            try:
                                await self._gateway("DELETE", f"/api/coordination/reservations/{rid}")
                            except httpx.HTTPError:
                                pass
                raise HTTPException(status_code=507, detail="face analysis ran out of memory -- try again in a moment")
            finally:
                self.busy -= 1
                self.last_used = time.time()
                self._schedule_idle_stop()

    def _schedule_idle_stop(self) -> None:
        if self.idle and not self.idle.done():
            self.idle.cancel()
        if KEEP_ALIVE_S <= 0:
            return

        async def wait():
            try:
                await asyncio.sleep(KEEP_ALIVE_S)
                if self.busy == 0:
                    await self.stop()
            except asyncio.CancelledError:
                pass
        self.idle = asyncio.create_task(wait())


WORKER = Worker()
app = FastAPI(title="vision-serve plugin for ai-serve")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/vision/models")
async def models():
    return catalog()


@app.post("/v1/vision/analyze")
async def analyze(image: List[UploadFile] = File(...), model: str = Form(FACE_MODEL), tasks: str = Form("faces"),
                  embed: bool = Form(True), labels: Optional[str] = Form(None)):
    if model.strip().lower() not in (FACE_MODEL, "faces", "auraface-v1"):
        raise HTTPException(status_code=400, detail=f"unknown model {model!r} (this plugin: {FACE_MODEL})")
    wanted = [t.strip() for t in tasks.split(",") if t.strip()]
    if wanted != ["faces"]:
        raise HTTPException(status_code=400, detail="AuraFace does faces only")
    if not 1 <= len(image) <= MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"1 to {MAX_IMAGES} images per call")
    if not installed():
        raise HTTPException(status_code=400, detail="AuraFace is not installed (POST /plugin/install)")
    data = []
    for u in image:
        b = await u.read(MAX_IMAGE_BYTES + 1)
        if len(b) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"an image is larger than {MAX_IMAGE_BYTES >> 20} MB")
        data.append(b)
    return await WORKER.analyze(data, embed)


# ------------------------------------------------------------------ plugin contract: status, release, logs, install

@app.get("/plugin/status")
async def status():
    pid = WORKER.pid
    where = ("GPU " + str(WORKER.gpu)) if WORKER.gpu is not None else "CPU"
    workers = [{"id": "faces", "label": "AuraFace", "pid": pid, "loaded": True, "busy": WORKER.busy > 0,
                "last_used": WORKER.last_used or None, "gpu": WORKER.gpu}] if pid and WORKER.gpu is not None else []
    facts = [{"label": "Faces", "value": f"loaded on {where}" if pid else
              ("off (starts with the next image)" if installed() else "not installed")}]
    return {"busy": WORKER.busy > 0, "ready": True, "workers": workers, "facts": facts}


@app.post("/plugin/release")
async def release(body: Any = Body(default={})):
    body = body if isinstance(body, dict) else {}
    if body.get("worker") not in (None, "faces"):
        raise HTTPException(status_code=404, detail="no such worker")
    if WORKER.busy and body.get("force") is not True:
        raise HTTPException(status_code=409, detail="busy; retry later or force")
    return {"released": ["faces"] if await WORKER.stop() else []}


@app.get("/plugin/logs")
async def logs(tail: int = 200):
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(tail, 5000)):]
    except OSError:
        lines = []
    return {"logs": "\n".join(lines)}


_install: Dict[str, Any] = {"state": None, "done": 0, "error": ""}


def install_item() -> Dict[str, Any]:
    state = "installed" if installed() else (_install["state"] or "missing")
    return {"id": FACE_MODEL, "label": "AuraFace + YuNet", "description": "Faces: find them, recognise people",
            "size_mb": 249, "state": state, "license": "Apache-2.0 + MIT",
            "done_mb": _install["done"] >> 20 if state == "downloading" else None,
            "error": _install["error"] if state == "failed" else ""}


def _run_install() -> None:
    def progress(n: int) -> None:
        _install["done"] += n
    try:
        for name in ("yunet", "auraface"):
            install.fetch(name, progress)
        _install.update(state=None, error="")
    except BaseException as e:
        _install.update(state="failed", error=f"{type(e).__name__}: {e}"[:300])


@app.get("/plugin/install")
async def install_list():
    return {"items": [install_item()]}


@app.post("/plugin/install", status_code=202)
async def install_start(body: Any = Body(...)):
    if not isinstance(body, dict) or body.get("id") != FACE_MODEL:
        raise HTTPException(status_code=404, detail=f"nothing to install under that id (only {FACE_MODEL})")
    if not installed() and _install["state"] != "downloading":
        _install.update(state="downloading", done=0, error="")
        asyncio.get_running_loop().run_in_executor(None, _run_install)
    return install_item()


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8068")))
    a = ap.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOG, encoding="utf-8")])
    # uvicorn's access log names only method, path and status -- no bodies; kept off anyway (warning level)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
