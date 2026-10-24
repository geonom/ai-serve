"""Asks the GPU owner (ai-serve-ui, STT_SERVE_GATEWAY_URL) before a model loads:
507 there (e.g. Strata holds the GPUs) becomes a 507 here. Unreachable -> go ahead
(standalone use).

A granted preflight is a reservation (ai-serve-ui's GPU manager): it is confirmed
right away with this process's PID -- models load in this process or in a child
(Nemotron worker), which the gateway attributes to it -- and released once an
in-process load is done, so the gateway counts the real usage from then on."""
import os
import re
from typing import NamedTuple, Optional

import httpx
from fastapi import HTTPException

GATEWAY_URL = os.getenv("STT_SERVE_GATEWAY_URL", os.environ.get("AI_SERVE_URL") or "http://127.0.0.1:8030").rstrip("/")
# Rough needs (MiB) the gateway may compare against free VRAM; it only refuses for Strata today.
NEEDS_MB = {"whisperx": 1500, "diarize": 1200, "nemotron_diarize": 4000, "moss_transcribe": 6000}
# What WhisperX needs per model size, loaded AND after its first transcription (float16 + pyannote VAD + CUDA context;
# measured 2026-10-24: medium 2690 MB -- 1500 was announced and it ran out of memory). The GPU manager also learns
# the real peak per purpose, so these only have to be in the right range.
_WHISPER_NEEDS = (("tiny", 900), ("base", 1100), ("small", 1800), ("medium", 3200), ("turbo", 3000), ("large", 5000))


def whisper_need_mb(model: str) -> int:
    """MiB for a WhisperX model name (a size, a Hugging Face repo or a folder); unknown -> the largest."""
    words = set(re.split(r"[^a-z0-9]+", os.path.basename(model.rstrip("/")).lower()))
    for word, mb in _WHISPER_NEEDS:
        if word in words:
            return mb
    return 5000


class Grant(NamedTuple):
    rid: Optional[str]     # reservation to release after the first use (None: nothing to release)
    gpu: Optional[int]     # where to load (None: the gateway is unreachable -- the default GPU)


def _result(r: httpx.Response) -> Optional[str]:
    """Reservation id (None: nothing to confirm -- refused-but-200, old gateway,
    or no reservation)."""
    if r.status_code in (409, 507):
        try:
            detail = r.json().get("detail") or r.text
        except ValueError:
            detail = r.text
        raise HTTPException(status_code=507, detail=str(detail))
    if r.status_code == 200:
        return r.json().get("reservation_id")
    return None


def _confirm_sync(rid: Optional[str]) -> None:
    if rid:
        try:
            httpx.post(f"{GATEWAY_URL}/api/coordination/reservations/{rid}/confirm",
                       json={"pid": os.getpid()}, timeout=10)
        except httpx.HTTPError:
            pass


def release_sync(rid: Optional[str]) -> None:
    """The load finished (or failed): from now on the gateway counts real usage."""
    if rid and GATEWAY_URL:
        try:
            httpx.delete(f"{GATEWAY_URL}/api/coordination/reservations/{rid}", timeout=10)
        except httpx.HTTPError:
            pass


def reserve_sync(purpose: str, need_mb: int) -> Grant:
    """Ask the GPU manager for room without naming a GPU: it picks one with room (and may unload idle models) and
    says which. No -> HTTPException 507 (the caller decides: CPU, or retry later). Unreachable -> Grant(None, None)."""
    if not GATEWAY_URL:
        return Grant(None, None)
    try:
        r = httpx.post(f"{GATEWAY_URL}/api/coordination/gpu-preflight", timeout=30,
                       json={"purpose": purpose, "need_mb": int(need_mb)})
    except httpx.HTTPError:
        return Grant(None, None)
    rid = _result(r)
    if r.status_code != 200:
        return Grant(None, None)
    body = r.json()
    if not body.get("granted", True):   # an older gateway said no with a 200
        raise HTTPException(status_code=507, detail=str(body.get("reason") or "no room on the GPUs"))
    _confirm_sync(rid)
    gpu = body.get("gpu")
    return Grant(rid, gpu if isinstance(gpu, int) else None)


async def reserve(purpose: str, need_mb: int) -> Grant:
    """reserve_sync for async callers (the workers). The caller confirms with the started worker's PID (confirm)."""
    if not GATEWAY_URL:
        return Grant(None, None)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{GATEWAY_URL}/api/coordination/gpu-preflight",
                             json={"purpose": purpose, "need_mb": int(need_mb)})
    except httpx.HTTPError:
        return Grant(None, None)
    rid = _result(r)
    if r.status_code != 200:
        return Grant(None, None)
    body = r.json()
    if not body.get("granted", True):
        raise HTTPException(status_code=507, detail=str(body.get("reason") or "no room on the GPUs"))
    gpu = body.get("gpu")
    return Grant(rid, gpu if isinstance(gpu, int) else None)


async def confirm(rid: Optional[str], pid: int) -> None:
    if rid and GATEWAY_URL:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{GATEWAY_URL}/api/coordination/reservations/{rid}/confirm", json={"pid": pid})
        except httpx.HTTPError:
            pass


def preflight_sync(purpose: str, gpu: int = 0) -> Optional[str]:
    if not GATEWAY_URL:
        return None
    try:
        r = httpx.post(f"{GATEWAY_URL}/api/coordination/gpu-preflight", timeout=30,
                       json={"purpose": f"stt:{purpose}", "gpu": gpu, "need_mb": NEEDS_MB.get(purpose, 2000)})
    except httpx.HTTPError:
        return None
    rid = _result(r)
    _confirm_sync(rid)
    return rid


async def preflight(purpose: str, gpu: int = 0) -> Optional[str]:
    if not GATEWAY_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{GATEWAY_URL}/api/coordination/gpu-preflight",
                             json={"purpose": f"stt:{purpose}", "gpu": gpu, "need_mb": NEEDS_MB.get(purpose, 2000)})
    except httpx.HTTPError:
        return None
    rid = _result(r)
    if rid:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{GATEWAY_URL}/api/coordination/reservations/{rid}/confirm",
                             json={"pid": os.getpid()})
        except httpx.HTTPError:
            pass
    return rid
