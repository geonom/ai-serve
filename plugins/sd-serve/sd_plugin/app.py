"""sd-serve as an ai-serve `image` plugin (ai-serve-ui docs/plugins.md): this small server is what ai-serve starts and
talks to. It starts sd-server (serve.sh, Iris-3B Turbo) only when an image is asked for -- after asking ai-serve's GPU
manager where there is room (POST /api/coordination/gpu-preflight without a GPU: it picks one, and says "move" when
an idle sd-server should restart on a roomier GPU) -- because sd-server places its weights once, at process start,
from the VRAM free then. It stops sd-server after SD_KEEP_ALIVE_S idle seconds or on `release` (sd-server has no
unload call). Rules without I/O (incl. the prompt-injection guard): rules.py.
(Moved here from ai-serve-ui's backend/images/iris.py, 2026-10-21.)"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import psutil
from fastapi import Body, FastAPI, HTTPException

from sd_plugin import rules

HERE = Path(__file__).resolve().parent.parent
logger = logging.getLogger("sd-plugin")


def _load_env() -> None:
    f = HERE / ".env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
SD_PORT = int(os.environ.get("SD_SERVER_PORT", "8064"))       # sd-server itself, loopback only
MODELS = os.environ.get("SD_MODELS_DIR") or str(HERE / "models")
NEED_MB = int(os.environ.get("SD_NEED_MB", "7000"))            # measured 5.1 GB resident, 6.6 GB peak
KEEP_ALIVE_S = int(os.environ.get("SD_KEEP_ALIVE_S", "300"))
GATEWAY = os.environ.get("AI_SERVE_URL", "http://127.0.0.1:8030").rstrip("/")
TIMEOUT_S = 300
LOG = HERE / "logs" / "sd-server.log"
SD = f"http://127.0.0.1:{SD_PORT}"
MODEL_FILES = ("diffusion_models/iris/iris-3b-turbo-Q4_K.gguf", "text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf")


def _gpu_name(gpu: int) -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _device_for(gpu: int) -> str:
    """sd-server's device for ai-serve's GPU index (Vulkan numbers GPUs differently from nvidia-smi; a Mac: Metal)."""
    env = dict(os.environ, LD_LIBRARY_PATH=str(HERE / "bin"), DYLD_LIBRARY_PATH=str(HERE / "bin"))
    try:
        out = subprocess.run([str(HERE / "bin" / "sd-server"), "--list-devices"], capture_output=True, text=True,
                             timeout=30, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return rules.pick_device(out, _gpu_name(gpu))


class Server:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.gpu: Optional[int] = None
        self.loaded = False
        self.busy = 0
        self.last_used = 0.0
        self.lock = asyncio.Lock()
        self.idle: Optional[asyncio.Task] = None

    @staticmethod
    def find() -> Optional[psutil.Process]:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = p.info["cmdline"] or []
                if any(c.endswith("sd-server") for c in cmd) and str(SD_PORT) in cmd:
                    return p
            except (psutil.Error, TypeError):
                continue
        return None

    def alive(self) -> bool:
        if self.pid and psutil.pid_exists(self.pid):
            try:
                if psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:
                    return True
            except psutil.Error:
                pass
        p = self.find()
        if p is None:
            self.pid, self.loaded = None, False
            return False
        self.pid = p.pid
        try:
            self.gpu = int(p.environ().get("CUDA_VISIBLE_DEVICES", ""))
        except (psutil.Error, ValueError):
            pass
        return True

    @staticmethod
    async def health() -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (await c.get(f"{SD}/v1/models")).status_code == 200
        except httpx.HTTPError:
            return False

    async def _gateway(self, method: str, path: str, **kw) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120.0) as c:
            return await c.request(method, f"{GATEWAY}{path}", **kw)

    async def _start(self, gpu: int) -> None:
        LOG.parent.mkdir(exist_ok=True)
        log = open(LOG, "w", encoding="utf-8")
        env = dict(os.environ, PORT=str(SD_PORT), SD_GPU_ID=str(gpu), SD_MODELS_DIR=MODELS)
        if not (HERE / "sdcpp" / "build" / "bin" / "sd-server").exists():   # the prebuilt one: device by GPU name
            env["SD_DEVICE"] = await asyncio.to_thread(_device_for, gpu)
        try:
            self.proc = subprocess.Popen([str(HERE / "serve.sh")], cwd=HERE, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        self.pid, self.gpu, self.loaded = self.proc.pid, gpu, False
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if await self.health():
                return
            if self.proc.poll() is not None:
                raise HTTPException(status_code=500, detail="Iris: sd-server exited at start -- see /plugin/logs")
            await asyncio.sleep(0.5)
        raise HTTPException(status_code=503, detail="Iris: sd-server didn't come up")

    async def stop(self) -> bool:
        if not self.alive():
            return False
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.to_thread(psutil.Process(self.pid).wait, 15)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.proc, self.pid, self.loaded = None, None, False
        logger.info("sd-server stopped")
        return True

    async def generate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        async with self.lock:   # one request at a time: one GPU, one model
            self.busy += 1
            reservation = None
            try:
                running = self.alive()
                if not running or not self.loaded:
                    # where to run: the GPU manager picks (start / stay / move) and reserves before the load
                    r = await self._gateway("POST", "/api/coordination/gpu-preflight", json={
                        "purpose": "Iris-3B", "need_mb": NEED_MB, "current_gpu": self.gpu if running else None,
                        "movable": True})
                    d = r.json() if r.status_code == 200 else {}
                    if r.status_code == 507:
                        raise HTTPException(status_code=507, detail=r.json().get("detail", "GPUs held exclusively"))
                    if not d.get("granted"):
                        raise HTTPException(status_code=507, detail=f"Iris: not enough GPU memory -- {d.get('reason') or r.text[:200]}")
                    reservation = d.get("reservation_id")
                    if running and d.get("mode") == "move":
                        await self.stop()
                        running = False
                    if not running:
                        await self._start(int(d["gpu"]))
                    if reservation and self.pid:
                        await self._gateway("POST", f"/api/coordination/reservations/{reservation}/confirm",
                                            json={"pid": self.pid})
                async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
                    r = await c.post(f"{SD}/v1/images/generations", json=body)
                if r.status_code != 200:
                    logger.error(f"sd-server {r.status_code}: {r.text[:300]}")
                    raise HTTPException(status_code=500, detail="Iris: image generation failed")
                data = [{"b64_json": d["b64_json"]} for d in r.json().get("data", []) if d.get("b64_json")]
                if not data:
                    raise HTTPException(status_code=500, detail="Iris: no image returned")
                self.loaded = True
                return {"created": int(time.time()), "data": data}
            except httpx.HTTPError as e:
                raise HTTPException(status_code=503, detail=f"Iris: {type(e).__name__} (ai-serve or sd-server unreachable)")
            finally:
                self.busy -= 1
                self.last_used = time.time()
                if reservation:
                    try:
                        await self._gateway("DELETE", f"/api/coordination/reservations/{reservation}")
                    except httpx.HTTPError:
                        pass
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
                    logger.info("Iris idle -- stopping sd-server")
                    await self.stop()
            except asyncio.CancelledError:
                pass
        self.idle = asyncio.create_task(wait())


SERVER = Server()
app = FastAPI(title="sd-serve plugin for ai-serve")


def installed() -> bool:
    binary = (HERE / "sdcpp/build/bin/sd-server").is_file() or (HERE / "bin/sd-server").is_file()
    return binary and all(os.path.isfile(os.path.join(MODELS, f)) for f in MODEL_FILES)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/images/models")
async def models():
    return rules.catalog(installed())


@app.post("/v1/images/generations")
async def generations(body: Any = Body(...)):
    try:
        req = rules.from_openai(body)
    except rules.InputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not installed():
        raise HTTPException(status_code=400, detail="Iris-3B Turbo is not installed (see sd-serve's CLAUDE.md)")
    return await SERVER.generate(req)


@app.get("/plugin/status")
async def status():
    running = SERVER.alive()
    workers = [{"id": "iris", "label": "Iris-3B Turbo", "pid": SERVER.pid, "loaded": SERVER.loaded,
                "busy": SERVER.busy > 0, "last_used": SERVER.last_used or None, "gpu": SERVER.gpu}] if running else []
    facts = [{"label": "Iris", "value": ("loaded" if SERVER.loaded else "started") + (f" on GPU {SERVER.gpu}" if SERVER.gpu is not None else "")
              if running else "off (starts with the next image)"}]
    return {"busy": SERVER.busy > 0, "ready": True, "workers": workers, "facts": facts}


@app.post("/plugin/release")
async def release(body: Any = Body(default={})):
    body = body if isinstance(body, dict) else {}
    if body.get("worker") not in (None, "iris"):
        raise HTTPException(status_code=404, detail="no such worker")
    if SERVER.busy and body.get("force") is not True:
        raise HTTPException(status_code=409, detail="Iris is busy; retry later or force")
    return {"released": ["iris"] if await SERVER.stop() else []}


@app.get("/plugin/logs")
async def logs(tail: int = 200):
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(tail, 5000)):]
    except OSError:
        lines = []
    return {"logs": "\n".join(lines)}


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8065")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
