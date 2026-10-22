"""Running image jobs that a caller can cancel (POST /v1/images/cancel).

A caller (the Spielwiese's comparison runner, say) tags a generation with an `X-Request-Id` it made up --
random, unguessable, 16-64 of [A-Za-z0-9_-]. While the job runs, the provider path registers how to stop
exactly that job: for ComfyUI and MOSS's ComfyUI, delete its prompt from the queue if it hasn't started and
interrupt it if it's the one running (ComfyUI's /interrupt with `prompt_id` only stops that prompt), so one
person's cancel never stops another's image. Image plugins cancel their own jobs (POST {cancel}).
A cancel that arrives before the job registered is remembered, so the job stops as soon as it starts.
Tests: tests/unit/test_image_jobs.py.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Awaitable, Callable, Dict, Optional

import httpx

logger = logging.getLogger("ai-serve-ui.images.jobs")
_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_REMEMBER_S = 600


def valid_request_id(value) -> Optional[str]:
    return value if isinstance(value, str) and _ID.match(value) else None


class Jobs:
    def __init__(self):
        self._stop: Dict[str, Callable[[], Awaitable[None]]] = {}
        self._cancelled: Dict[str, float] = {}

    def register(self, request_id: str, stop: Callable[[], Awaitable[None]]) -> None:
        self._stop[request_id] = stop

    def unregister(self, request_id: str) -> None:
        self._stop.pop(request_id, None)

    def cancelled(self, request_id: Optional[str]) -> bool:
        return bool(request_id) and request_id in self._cancelled

    async def cancel(self, request_id: str) -> bool:
        """True if a running job was stopped. Remembered either way for a job that hasn't registered yet."""
        now = time.time()
        self._cancelled = {k: t for k, t in self._cancelled.items() if now - t < _REMEMBER_S}
        self._cancelled[request_id] = now
        stop = self._stop.pop(request_id, None)
        if stop is None:
            return False
        try:
            await stop()
        except Exception as e:   # the job's own path reports the outcome
            logger.warning(f"stopping image job failed: {e}")
        return True


JOBS = Jobs()


async def interrupt_comfy(base_url: str, prompt_id: str) -> None:
    """Remove `prompt_id` from a ComfyUI queue if it's still waiting, interrupt it if it's running."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as c:
        await c.post("/queue", json={"delete": [prompt_id]})
        await c.post("/interrupt", json={"prompt_id": prompt_id})
