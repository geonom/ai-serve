"""Slim orchestrator — logging, app, middleware, routers, static mount."""
import asyncio
import logging
import os
import logging.handlers
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import API_KEY, FRONTEND_DIR, LOGS_DIR
from backend.public_api.middleware import PublicExposureMiddleware
from backend.model_manager import scan_existing_llama_servers
from backend.state import state

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Endpoints polled every few seconds by the dashboard UI (and internally by
# /api/status, which itself fires an httpx /health check per instance) —
# logging every hit at INFO drowns out anything worth seeing within hours.
_NOISY_POLL_PATHS = (
    "/api/gpu/status", "/api/status", "/health",
    "/api/system/status", "/api/tts/status",
)


class _QuietPollingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            method, path, status = args[1], args[2], args[4]
            if method == "GET" and status == 200 and any(
                str(path).startswith(p) for p in _NOISY_POLL_PATHS
            ):
                return False
        return True


_backend_log_path = f"{LOGS_DIR}/backend.log"
_file_handler = logging.handlers.RotatingFileHandler(
    _backend_log_path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_file_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setFormatter(_fmt)
root_logger = logging.getLogger()
# LLAMA_SERVE_LOG_TO_FILE=0 keeps logs/backend.log clean -- set by the unit tests,
# whose deliberate bad requests ("secret /some/private path", testclient 401s) used
# to land in the live log and read like real incidents.
root_logger.handlers = ([_file_handler] if os.getenv("LLAMA_SERVE_LOG_TO_FILE", "1") != "0" else []) + [_console_handler]
root_logger.setLevel(logging.INFO)
logger = logging.getLogger("ai-serve-ui")

logging.getLogger("uvicorn.access").addFilter(_QuietPollingFilter())
# httpx logs an INFO line for every internal /health probe (issued once per
# instance on every /api/status poll) — not useful, just noise.
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        lg = logging.getLogger(logger_name)
        if _file_handler not in lg.handlers:
            lg.addHandler(_file_handler)
    scan_existing_llama_servers()
    try:
        # Owner gates live in this process; Serve routes persist in tailscaled.
        from backend.routers import services as services_router
        errors = await services_router.apply_exposure()
        if errors:
            logger.warning(f"Service exposure not fully applied: {errors}")
    except Exception as e:
        logger.warning(f"Service exposure restore failed: {e}")
    # Stops device instances the gateway started once they've been idle (never deletes files)
    from backend.cluster.service import SERVICE as _cluster
    reaper_task = asyncio.create_task(_cluster.run_reaper())

    async def _warm_model_traits():
        # GET /v1/llm/models reads every GGUF header once (~15 s for ~100 files); do it now, not on first use
        try:
            from backend.model_manager import get_model_traits
            from backend.routers.dashboard import list_models
            paths = [m["path"] for m in (await list_models())["models"]]
            await asyncio.get_running_loop().run_in_executor(None, lambda: [get_model_traits(p) for p in paths])
        except Exception as e:
            logger.warning(f"Warming model traits failed: {e}")
    warm_task = asyncio.create_task(_warm_model_traits())
    try:
        # Services switched to autostart in the dashboard (e.g. nextcloud-bridge)
        from backend.routers import services as services_router
        await services_router.run_autostart()
    except Exception as e:
        logger.warning(f"Service autostart failed: {e}")
    yield
    reaper_task.cancel()
    warm_task.cancel()
    try:
        from backend.routers import services as services_router
        await services_router.GATES.stop_all()
    except Exception as e:
        logger.warning(f"Stopping service gates failed: {e}")
    # Deliberately NOT terminating managed llama-server processes here.
    # They're started with start_new_session=True specifically so they survive
    # this process exiting -- scan_existing_llama_servers()
    # re-adopt them on the next startup (see their own docstrings), and
    # get_status()'s self-heal prunes anything that actually died. This shutdown
    # hook used to call proc.terminate() on everything unconditionally, which
    # defeated that adoption path for every *graceful* shutdown (including a
    # routine `touch restart.flag` reload) -- confirmed live: a restart.flag
    # cycle killed every loaded model, forcing a full cold reload on the next
    # request, even though the whole adoption mechanism exists to avoid exactly
    # that. Only closing our own log file handles here, which is harmless --
    # the child processes hold their own fd to the same file regardless.
    for info in state.managed_processes.values():
        log_file = info.get("log_file")
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
    # TTS workers belong to tts-serve now (it terminates them on its own shutdown).


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI-Serve Dashboard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Who may use the dashboard/APIs: this machine, the Tailscale owner's devices, others
# only with LLAMA_SERVE_API_KEY; /public/ and /api/training/ have their own keys.
# Rules in backend/access/policy.py.
from backend.access.middleware import AccessMiddleware
from backend.access.policy import AccessPolicy, ApiKey
from backend.network import NoneProvider, ProviderError, active, load_provider, set_active
from backend.config import NETWORK_CONFIG_FILE, NETWORK_PROVIDER

try:
    NETWORK = load_provider(NETWORK_PROVIDER, NETWORK_CONFIG_FILE)
except ProviderError as e:  # fail closed: no private network, loopback + API key only
    logger.error(f"Network provider unusable ({e}) -- only this machine and the API key get in")
    NETWORK = NoneProvider()
set_active(NETWORK)
# asks the active provider per request (not one bound at import), so it can be replaced without a restart
from backend.access.hosts import known_hosts  # noqa: E402  (Host/Origin check: DNS rebinding, cross-site requests)
app.add_middleware(AccessMiddleware, policy=AccessPolicy(ApiKey(API_KEY), is_private=lambda ip: active().is_private(ip),
                                                         known_hosts=known_hosts),
                   owner=lambda: active().owner(), whois=lambda ip: active().identify(ip))
# Added last = outermost: a tunnel-marked request (Tailscale Funnel / Cloudflare) is
# turned away before any other layer unless its path is under /public/ -- see
# backend/public_api/domain.py's ExposureGuard.
app.add_middleware(PublicExposureMiddleware)

# Routers — order matters only for duplicate paths (first registered wins)
from backend.routers import cluster, comfyui_compat, dashboard, door, gpu, gateway, hf_download, media, coordination, ocr, overview, plugins, public_gateway, services, smoketest, training, vision  # noqa: E402

app.include_router(dashboard.router)
app.include_router(cluster.router)
app.include_router(gpu.router)
app.include_router(gateway.router)
app.include_router(media.router)
app.include_router(ocr.router)
app.include_router(public_gateway.router)
app.include_router(services.router)
app.include_router(door.router)
app.include_router(overview.router)
app.include_router(coordination.router)
app.include_router(smoketest.router)
app.include_router(plugins.router)
app.include_router(hf_download.router)
app.include_router(comfyui_compat.router)
app.include_router(training.router)
app.include_router(vision.router)

# Static UI — must be last so /api and /v1 routes take priority
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


if __name__ == "__main__":
    # X-Forwarded-For is honoured only from loopback (Tailscale Serve/Funnel proxy from
    # there) -- the access policy relies on that: a rewritten peer was loopback.
    from backend.config import GATEWAY_HOST, GATEWAY_PORT
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT, proxy_headers=True, forwarded_allow_ips="127.0.0.1")
