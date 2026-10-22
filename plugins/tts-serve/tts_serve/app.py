"""tts-serve HTTP API (binds 127.0.0.1 -- reachability beyond this machine is chosen
in ai-serve-ui's Services tab, never here).

GET  /health                      liveness
GET  /v1/engines                  providers, their capabilities and runtime state
GET  /v1/status                   overall status (most severe engine state)
POST /v1/generate                 TtsRequest -> {engine, audio_base64, device} or audio/wav
POST /v1/engines/{id}/unload      ?force=true to unload while busy
POST /v1/unload_idle              ?gpu=N -- free idle engines (GPU owner asks for room)
GET  /v1/logs                     ?engine=&tail=
"""
import time
import base64
import logging
import os
import platform
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tts_serve.domain import RequestError, TtsRequest, overall_status, status_facts
from tts_serve.providers import PROVIDERS_DIR
from tts_serve.install import Installer
from tts_serve.workers import CoordinationRefused, HttpCoordinator, WorkerError, WorkerPool, nvidia_smi_probe

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def create_app(providers_dir: Optional[Path] = None, logs_dir: Optional[Path] = None, gpu_probe=None,
               coordinator=None, default_engine: Optional[str] = None) -> FastAPI:
    pool_args = dict(
        providers_dir=providers_dir or PROVIDERS_DIR,
        logs_dir=logs_dir or Path(os.getenv("TTS_SERVE_LOGS_DIR", ROOT / "logs")),
        gpu_probe=gpu_probe or nvidia_smi_probe,
        coordinator=coordinator or HttpCoordinator(os.getenv("TTS_SERVE_GATEWAY_URL", os.environ.get("AI_SERVE_URL") or "http://127.0.0.1:8030")),
        default_engine=default_engine or os.getenv("TTS_SERVE_DEFAULT_ENGINE", "cosyvoice"),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pool = WorkerPool(**pool_args)
        app.state.installer = Installer(app.state.pool.registry, app.state.pool.logs_dir,
                                        lambda: bool(app.state.pool.gpu_probe()), platform.system(), platform.machine())
        yield
        await app.state.pool.shutdown()  # workers have no adoption -- don't orphan their VRAM

    app = FastAPI(title="tts-serve", lifespan=lifespan)

    @app.exception_handler(RequestError)
    async def _req_error(request: Request, e: RequestError):
        return JSONResponse({"detail": e.message}, status_code=e.status)

    @app.exception_handler(WorkerError)
    async def _worker_error(request: Request, e: WorkerError):
        return JSONResponse({"detail": e.message}, status_code=e.status)

    @app.exception_handler(CoordinationRefused)
    async def _refused(request: Request, e: CoordinationRefused):
        return JSONResponse({"detail": e.message}, status_code=507)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/engines")
    async def engines(request: Request):
        return request.app.state.pool.engines()

    @app.get("/v1/status")
    async def status(request: Request):
        view = request.app.state.pool.engines()
        states = [e["status"] for e in view["engines"]]
        overall = overall_status(states)
        return {"status": overall, "ready": overall not in ("starting", "loading", "error"),
                "busy": overall in ("loading", "generating"), "default_engine": view["default_engine"],
                "engines": {e["id"]: e for e in view["engines"]},
                # plugin contract (ai-serve-ui docs/plugins.md): the processes holding GPU memory
                "workers": [{"id": e["id"], "label": e["name"], "pid": e["pid"], "loaded": True,
                             "busy": bool(e["inflight"]) or e["status"] in ("loading", "generating", "starting"),
                             "last_used": e["last_used"], "gpu": e.get("gpu")} for e in view["engines"] if e["loaded"]],
                "facts": status_facts(view["engines"], view["default_engine"], time.time())}

    @app.post("/v1/release")
    async def release(request: Request, body: Any = Body(default={})):
        """Plugin contract: free GPU memory. {"worker": engine} unloads that engine (409 while busy unless
        force); without a worker every idle engine (on {"gpu": n} only) goes."""
        pool = request.app.state.pool
        body = body if isinstance(body, dict) else {}
        worker, gpu, force = body.get("worker"), body.get("gpu"), body.get("force") is True
        if worker is not None:
            if not isinstance(worker, str):
                raise RequestError("worker must be an engine id")
            m = pool.resolve(worker)
            result = await pool.unload(m.id, force=force)
            return {"released": [m.id] if result == "unloaded" else []}
        if gpu is not None and (type(gpu) is not int or gpu < 0):
            raise RequestError("gpu must be a GPU index")
        return {"released": await pool.unload_idle(gpu)}

    @app.post("/v1/generate")
    async def generate(request: Request, body: Any = Body(...)):
        req = TtsRequest.from_dict(body)
        result = await request.app.state.pool.generate(req)
        if req.response_format == "wav":
            return Response(content=base64.b64decode(result["audio_base64"]), media_type="audio/wav",
                            headers={"X-TTS-Engine": result["engine"]})
        return result

    @app.post("/v1/engines/{engine_id}/unload")
    async def unload(engine_id: str, request: Request, force: bool = False):
        pool = request.app.state.pool
        m = pool.resolve(engine_id)
        return {"id": m.id, "status": await pool.unload(m.id, force=force)}

    @app.post("/v1/unload_idle")
    async def unload_idle(request: Request, gpu: Optional[int] = None):
        return {"unloaded": await request.app.state.pool.unload_idle(gpu)}

    # plugin contract: install -- set up an engine that isn't yet (its own setup.sh), owner-only at the gateway
    @app.get("/v1/install")
    async def install_list(request: Request):
        return {"items": request.app.state.installer.items()}

    @app.post("/v1/install", status_code=202)
    async def install_start(request: Request, body: Any = Body(...)):
        try:
            return request.app.state.installer.start(body.get("id") if isinstance(body, dict) else None)
        except LookupError as e:
            return JSONResponse({"detail": str(e)}, status_code=404)

    @app.get("/v1/logs")
    async def logs(request: Request, engine: Optional[str] = None, tail: int = 200):
        pool = request.app.state.pool
        pid = pool.resolve(engine).id if engine else None
        return {"logs": pool.tail(pid, max(1, min(tail, 5000)))}

    return app


app = create_app()
