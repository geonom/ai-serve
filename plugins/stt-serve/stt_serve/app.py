"""stt-serve HTTP API (binds 127.0.0.1 -- reachability beyond this machine is chosen
in ai-serve-ui's Services tab, never here).

GET  /health
GET  /v1/status                          models loaded / busy, live sessions
POST /v1/transcribe                      STTTranscribeRequest (audio as base64 only)
GET  /v1/diarize/backends
POST /v1/unload | /v1/diarize/unload | /v1/diarize/nemotron/unload
POST /v1/unload_all?force=               everything (GPU owner asks, e.g. for Strata); 409 while busy
POST /v1/live/start | chunk | stop, GET /v1/live/poll | sessions, PATCH /v1/live/settings
GET  /v1/logs?tail=                      Nemotron worker log
"""
import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException

from stt_serve.facts import status_facts
from stt_serve.install import Installer, has_cuda as install_has_cuda
from stt_serve import engine
from stt_serve.state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _nemotron_running() -> bool:
    p = state.nemotron_diarize_process
    return p is not None and p.poll() is None


def _moss_running() -> bool:
    p = state.moss_transcribe_process
    return p is not None and p.poll() is None


def _busy() -> bool:
    return (state.whisperx_status in ("transcribing", "loading") or state.diarize_status in ("diarizing", "loading")
            or state.nemotron_diarize_status in ("diarizing", "loading", "starting")
            or state.moss_transcribe_status in ("transcribing", "loading", "starting"))


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(engine.LOGS_DIR, exist_ok=True)
        yield
        for sid in list(state.live_stt_sessions):
            task = state.live_stt_sessions[sid].get("keep_alive_task")
            if task:
                task.cancel()
        state.live_stt_sessions.clear()
        engine._unload_nemotron_diarize()  # its own process -- don't orphan its VRAM
        engine._unload_moss_transcribe()
        engine._unload_whisperx()
        engine._unload_diarize()

    app = FastAPI(title="stt-serve", lifespan=lifespan)
    app.include_router(engine.router)

    # plugin contract: install -- set up MOSS-Transcribe or Nemotron (their own setup.sh), owner-only at the gateway
    bins = {"moss": lambda: engine.MOSS_TRANSCRIBE_PYTHON_BIN, "nemotron": lambda: engine.NEMOTRON_DIARIZE_PYTHON_BIN}
    installer = Installer(engine.ROOT, lambda k: os.path.exists(bins[k]()), install_has_cuda, engine.LOGS_DIR)
    app.state.installer = installer

    @app.get("/v1/install")
    async def install_list():
        return {"items": app.state.installer.items()}

    @app.post("/v1/install", status_code=202)
    async def install_start(body: Any = Body(...)):
        try:
            return app.state.installer.start(body.get("id") if isinstance(body, dict) else None)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))

    def _workers():
        """The processes holding models (plugin contract): this process for WhisperX/pyannote, the Nemotron and
        MOSS children for theirs."""
        out = []
        if state.whisperx_model is not None or state.diarize_pipeline is not None or state.align_model is not None:
            out.append({"id": "stt-serve", "label": "WhisperX / diarization", "pid": os.getpid(), "loaded": True,
                        "busy": state.whisperx_status == "transcribing" or state.diarize_status == "diarizing"
                                or bool(state.live_stt_sessions)})
        if _nemotron_running():
            out.append({"id": "nemotron", "label": "Nemotron diarization", "pid": state.nemotron_diarize_pid,
                        "loaded": True, "busy": state.nemotron_diarize_status == "diarizing"})
        if _moss_running():
            out.append({"id": "moss", "label": "MOSS-Transcribe", "pid": state.moss_transcribe_pid, "loaded": True,
                        "busy": state.moss_transcribe_status == "transcribing"})
        return out

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status():
        view = {
            "whisperx": {"loaded": state.whisperx_model is not None, "status": state.whisperx_status},
            "diarize": {"loaded": state.diarize_pipeline is not None, "status": state.diarize_status,
                        "device": state.diarize_device, "align_language": state.align_language},
            "nemotron": {"loaded": _nemotron_running(), "status": state.nemotron_diarize_status,
                         "pid": state.nemotron_diarize_pid if _nemotron_running() else None,
                         "device": state.nemotron_diarize_device},
            "moss": {"loaded": _moss_running(), "status": state.moss_transcribe_status,
                     "pid": state.moss_transcribe_pid if _moss_running() else None,
                     "device": state.moss_transcribe_device},
            "live_sessions": len(state.live_stt_sessions),
            "busy": _busy(),
        }
        view["facts"] = status_facts(view)
        view["workers"] = _workers()
        return view

    @app.post("/v1/release")
    async def release(body: Any = Body(default={})):
        """Plugin contract (ai-serve-ui docs/plugins.md): free GPU memory -- one worker (stt-serve = WhisperX +
        pyannote in this process, nemotron, moss) or, without one, all of them. 409 while busy unless force."""
        body = body if isinstance(body, dict) else {}
        worker, force = body.get("worker"), body.get("force") is True
        if worker is not None and not isinstance(worker, str):
            raise HTTPException(status_code=400, detail="worker must be text")
        loaded = {w["id"]: w for w in _workers()}
        if worker is not None and worker not in ("stt-serve", "nemotron", "moss"):
            raise HTTPException(status_code=404, detail=f"no worker {worker!r}")
        targets = [worker] if worker else list(loaded)
        busy = [t for t in targets if t in loaded and loaded[t]["busy"]]
        if busy and not force:
            raise HTTPException(status_code=409, detail=f"busy: {', '.join(busy)}; retry later or force")
        done = []
        for t in targets:
            if t not in loaded:
                continue
            if t == "stt-serve":
                engine._unload_whisperx()
                engine._unload_diarize()
            elif t == "nemotron":
                engine._unload_nemotron_diarize()
            elif t == "moss":
                engine._unload_moss_transcribe()
            done.append(t)
        return {"released": done}

    @app.post("/v1/unload_all")
    async def unload_all(force: bool = False):
        if _busy() and not force:
            raise HTTPException(status_code=409, detail="STT is busy (transcribing/diarizing); retry later or force.")
        done = []
        if state.whisperx_model is not None:
            engine._unload_whisperx()
            done.append("whisperx")
        if state.diarize_pipeline is not None or state.align_model is not None:
            engine._unload_diarize()
            done.append("diarization")
        if _nemotron_running():
            engine._unload_nemotron_diarize()
            done.append("nemotron")
        if _moss_running():
            engine._unload_moss_transcribe()
            done.append("moss")
        return {"unloaded": done}

    @app.get("/v1/logs")
    async def logs(tail: int = 200):
        out = []
        for name in ("nemotron_diarize.log", "moss_transcribe.log"):
            path = os.path.join(engine.LOGS_DIR, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    out.append(f"--- {name}\n" + "".join(f.readlines()[-max(1, min(tail, 5000)):]))
        return {"logs": "\n".join(out)}

    return app


app = create_app()
