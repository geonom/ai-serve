"""Strata as an ai-serve `llm` plugin (ai-serve-ui docs/plugins.md): this small server is what ai-serve starts and
talks to; it starts Strata itself only when a model request (or the gateway's `load`) needs it, takes every GPU for
it through the gateway's GPU manager first (`gpu: exclusive`), forwards the OpenAI API, and stops Strata on
`release` (or after STRATA_KEEP_ALIVE idle seconds, if set).

Strata lives elsewhere (STRATA_DIR, its own install); nothing of it is copied here. Settings: .env (see
.env.example). Rules without I/O: rules.py."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from strata_plugin import rules

HERE = Path(__file__).resolve().parent.parent
logger = logging.getLogger("strata-plugin")


def _load_env() -> None:
    f = HERE / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
STRATA_DIR = os.environ.get("STRATA_DIR", "")
STRATA_CONFIG = os.environ.get("STRATA_CONFIG") or (os.path.join(STRATA_DIR, "strata-iq3_s.json") if STRATA_DIR else "")
STRATA_PORT = int(os.environ.get("STRATA_PORT", "8080"))
START_TIMEOUT_S = int(os.environ.get("STRATA_START_TIMEOUT", "1800"))
KEEP_ALIVE_S = int(os.environ.get("STRATA_KEEP_ALIVE", "0"))
ALIASES = [a.strip() for a in os.environ.get("STRATA_ALIASES", "").split(",") if a.strip()]
GATEWAY = os.environ.get("AI_SERVE_URL", "http://127.0.0.1:8030").rstrip("/")
LOG = HERE / "logs" / "strata.log"
STRATA = f"http://127.0.0.1:{STRATA_PORT}"


class Server:
    """Strata's server process: started here, or adopted when it already runs (started elsewhere / before)."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.inflight = 0
        self.last_used = 0.0
        self.lock = asyncio.Lock()
        self.keep_alive: Optional[asyncio.Task] = None
        self.error: Optional[str] = None

    # ---- looking
    @staticmethod
    def find_pid() -> Optional[int]:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = p.info["cmdline"] or []
                if any(x.endswith(os.path.join("serve", "server.py")) for x in cmd) and "strata" in cmd \
                        and str(STRATA_PORT) in cmd:
                    return p.info["pid"]
            except (psutil.Error, TypeError):
                continue
        return None

    def alive(self) -> bool:
        if self.pid and psutil.pid_exists(self.pid):
            try:
                return psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE
            except psutil.Error:
                return False
        self.pid = self.find_pid()
        return self.pid is not None

    @staticmethod
    async def health() -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{STRATA}/health")
            data = r.json() if r.status_code == 200 else None
            return data if isinstance(data, dict) and data.get("service") == "strata" else None
        except (httpx.HTTPError, ValueError):
            return None

    @staticmethod
    async def strata_busy() -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{STRATA}/status")
            d = r.json() if r.status_code == 200 else {}
            return bool(d.get("busy")) or int(d.get("queued") or 0) > 0
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    async def state(self) -> str:
        return rules.state_of(self.alive(), await self.health())

    # ---- starting
    async def _reserve_gpus(self) -> List[str]:
        """Every GPU for Strata, from the gateway's GPU manager (it unloads idle models; a fast no otherwise)."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as c:
                r = await c.post(f"{GATEWAY}/api/coordination/gpu-exclusive", json={"purpose": "Strata"})
            d = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise HTTPException(status_code=503, detail=f"ai-serve (the GPU owner) not reachable at {GATEWAY}: {e}")
        if r.status_code != 200 or not d.get("granted"):
            raise HTTPException(status_code=409, detail=f"Can't start Strata now: {d.get('reason') or d}")
        return list(d.get("reservation_ids") or [])

    @staticmethod
    async def _reservations(ids: List[str], pid: Optional[int]) -> None:
        async with httpx.AsyncClient(timeout=10.0) as c:
            for rid in ids:
                try:
                    if pid:
                        await c.post(f"{GATEWAY}/api/coordination/reservations/{rid}/confirm", json={"pid": pid})
                    else:
                        await c.delete(f"{GATEWAY}/api/coordination/reservations/{rid}")
                except httpx.HTTPError:
                    pass

    async def start(self) -> None:
        python = os.path.join(STRATA_DIR, ".venv", "bin", "python")
        if not STRATA_DIR or not os.path.exists(python) or not os.path.isfile(STRATA_CONFIG):
            raise HTTPException(status_code=500, detail=(
                f"Strata not found: need {python or 'STRATA_DIR'} and {STRATA_CONFIG or 'STRATA_CONFIG'} (.env)"))
        ids = await self._reserve_gpus()
        LOG.parent.mkdir(exist_ok=True)
        log = open(LOG, "w", encoding="utf-8")
        cmd = [python, os.path.join("serve", "server.py"), "--engine", "strata", "--config", STRATA_CONFIG,
               "--host", "127.0.0.1", "--port", str(STRATA_PORT)]
        try:
            self.proc = subprocess.Popen(cmd, cwd=STRATA_DIR, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, STRATA_ENGINE_READY_S=str(START_TIMEOUT_S)),
                                         start_new_session=True)
        except OSError:
            await self._reservations(ids, None)
            raise
        finally:
            log.close()
        self.pid, self.error = self.proc.pid, None
        await self._reservations(ids, self.proc.pid)   # the GPUs stay Strata's for as long as it lives
        logger.info(f"Started Strata (pid {self.pid}) on 127.0.0.1:{STRATA_PORT}")

    async def ready(self, timeout_s: float = START_TIMEOUT_S) -> None:
        """Start Strata if needed and wait until it serves."""
        async with self.lock:
            st = await self.state()
            if st == "ready":
                return
            if st == "stopped":
                await self.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            h = await self.health()
            if h and h.get("loaded"):
                return
            if self.proc is not None and self.proc.poll() is not None:
                code, self.proc, self.pid = self.proc.returncode, None, None
                self.error = f"Strata exited while loading (code {code}); see /plugin/logs"
                raise HTTPException(status_code=500, detail=self.error)
            if self.proc is None and not self.alive():
                raise HTTPException(status_code=500, detail="Strata stopped while loading")
            await asyncio.sleep(3)
        raise HTTPException(status_code=504, detail=f"Strata didn't finish loading in {int(timeout_s)} s; "
                                                    f"it keeps loading -- retry later")

    # ---- stopping
    async def stop(self) -> List[str]:
        pid = self.pid if self.alive() else None
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.to_thread(psutil.Process(pid).wait, 60)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                logger.warning(f"Strata (pid {pid}) ignored SIGTERM for 60 s; killing it")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        # leftover engine processes of *this* install -- by exe path, never by name alone
        root = os.path.realpath(STRATA_DIR) if STRATA_DIR else None
        left = []
        for p in psutil.process_iter(["name", "exe"]):
            try:
                if root and p.info["name"] == "strata" and (p.info["exe"] or "").startswith(root):
                    left.append(p)
            except psutil.Error:
                continue
        for p in left:
            try:
                p.terminate()
            except psutil.Error:
                pass
        _, alive = await asyncio.to_thread(psutil.wait_procs, left, 10)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass
        self.proc, self.pid = None, None
        logger.info("Strata stopped" + (f" ({len(left)} leftover engine process(es) ended)" if left else ""))
        return ["strata"] if pid else []

    async def busy(self) -> bool:
        return self.inflight > 0 or (self.alive() and await self.strata_busy())

    # ---- requests
    def begin(self) -> None:
        self.inflight += 1
        self.last_used = time.time()
        if self.keep_alive and not self.keep_alive.done():
            self.keep_alive.cancel()

    def end(self) -> None:
        self.inflight = max(0, self.inflight - 1)
        self.last_used = time.time()
        if KEEP_ALIVE_S > 0 and self.inflight == 0:
            self.keep_alive = asyncio.create_task(self._idle_stop())

    async def _idle_stop(self) -> None:
        try:
            await asyncio.sleep(KEEP_ALIVE_S)
            if not await self.busy():
                logger.info(f"Strata idle for {KEEP_ALIVE_S} s; stopping it")
                await self.stop()
        except asyncio.CancelledError:
            pass


SERVER = Server()
app = FastAPI(title="Strata plugin for ai-serve")


def _config() -> Dict[str, Any]:
    try:
        return json.loads(Path(STRATA_CONFIG).read_text(encoding="utf-8")) if STRATA_CONFIG else {}
    except (OSError, ValueError):
        return {}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def models():
    """The catalog -- from Strata's config; Strata itself isn't started for it."""
    if not STRATA_DIR or not os.path.isdir(STRATA_DIR):
        return {"object": "list", "data": []}
    return rules.catalog(_config(), ALIASES)


@app.get("/plugin/status")
async def status():
    st = await SERVER.state()
    busy = await SERVER.busy()
    workers = [{"id": "strata", "label": "Strata", "pid": SERVER.pid, "loaded": True, "busy": busy}] \
        if st != "stopped" and SERVER.pid else []
    facts = [{"label": "Strata", "value": {"stopped": "off", "loading": "loading (4–10 min)", "ready": "ready"}[st]}]
    if SERVER.error:
        facts.append({"label": "Error", "value": SERVER.error})
    if not STRATA_DIR:
        facts.append({"label": "Einrichtung", "value": "STRATA_DIR fehlt (.env)"})
    return {"busy": busy, "ready": st == "ready", "state": st, "workers": workers, "facts": facts}


@app.post("/plugin/load")
async def load(body: Any = Body(default={})):
    model = body.get("model") if isinstance(body, dict) else None
    if model not in (None, rules.MODEL_ID):
        raise HTTPException(status_code=404, detail=f"this plugin serves {rules.MODEL_ID!r} only")
    await SERVER.ready()
    return {"status": "loaded"}


@app.post("/plugin/release")
async def release(body: Any = Body(default={})):
    body = body if isinstance(body, dict) else {}
    if body.get("worker") not in (None, "strata"):
        raise HTTPException(status_code=404, detail="no such worker")
    if await SERVER.busy() and body.get("force") is not True:
        raise HTTPException(status_code=409, detail="Strata is busy; retry later or force")
    return {"released": await SERVER.stop()}


@app.get("/plugin/logs")
async def logs(tail: int = 200):
    try:
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(tail, 5000)):]
    except OSError:
        lines = []
    return {"logs": "\n".join(lines)}


_HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive"}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def forward(path: str, request: Request):
    """Everything else goes to Strata: POSTs under /v1/ start it first (the first one waits for the load); other
    calls (/props, /slots ...) only reach a running Strata."""
    body = await request.body()
    if request.method == "POST" and path.startswith("v1/"):
        await SERVER.ready()
        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    body = json.dumps(rules.upstream_payload(data)).encode()
            except ValueError:
                pass
    elif await SERVER.state() == "stopped":
        return JSONResponse({"detail": "Strata is not running"}, status_code=503)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    SERVER.begin()
    client = httpx.AsyncClient(timeout=httpx.Timeout(START_TIMEOUT_S, connect=10.0))
    try:
        req = client.build_request(request.method, f"{STRATA}/{path}", params=request.query_params,
                                   headers=headers, content=body)
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        SERVER.end()
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Strata: {type(e).__name__}")
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP}

    async def chunks():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()
            SERVER.end()

    if "text/event-stream" in resp.headers.get("content-type", ""):
        return StreamingResponse(chunks(), status_code=resp.status_code, headers=out_headers)
    out_headers.pop("content-encoding", None)        # aread() hands over the decoded body
    try:
        content = await resp.aread()
    finally:
        await resp.aclose()
        await client.aclose()
        SERVER.end()
    return Response(content=content, status_code=resp.status_code, headers=out_headers)


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8075)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
