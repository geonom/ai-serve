"""train-serve (a separate app, port 8038): fine-tuning jobs moved out of
this backend on 2026-10-17. Started on demand through the Services registry (like
tts-serve); the GPU manager reads its running jobs to count their VRAM."""
import asyncio
import logging
from typing import Any, Dict, List

import httpx
from fastapi import HTTPException

from backend.config import TRAIN_SERVE_SERVICE_ID, TRAIN_SERVE_URL

logger = logging.getLogger("ai-serve-ui.train-client")
_start_lock = asyncio.Lock()


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            return (await c.get(f"{TRAIN_SERVE_URL}/health")).status_code == 200
    except httpx.HTTPError:
        return False


async def ensure_running(timeout_s: float = 30.0) -> bool:
    if await health():
        return True
    async with _start_lock:
        if await health():
            return True
        from backend.routers import services as svc
        from backend.services import runtime
        from backend.services.domain import ServiceError, plan_start
        try:
            m = svc._registry().get(TRAIN_SERVE_SERVICE_ID)
            procs = await asyncio.to_thread(runtime.snapshot)
            plan = plan_start(m, procs)
        except (ServiceError, HTTPException) as e:
            logger.warning(f"train-serve not reachable and not startable: {getattr(e, 'message', e)}")
            return False
        logger.info("Starting train-serve on demand")
        svc._launched[m.id] = runtime.launch(plan, svc._log_path(m.id))
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if await health():
                return True
            await asyncio.sleep(0.5)
        return False


async def running_jobs() -> List[Dict[str, Any]]:
    """Jobs train-serve reports as running (with their pid); [] when it isn't up."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{TRAIN_SERVE_URL}/api/training-monitor/jobs")
        jobs = r.json() if r.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return []
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs", [])
    return [j for j in jobs if isinstance(j, dict) and j.get("status") == "running" and j.get("pid")]
