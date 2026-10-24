"""GPU coordination for out-of-process backends (plugins: tts-serve, stt-serve, the Strata adapter ...). This
gateway stays the GPU owner: before a cold start, such a backend asks here -- an exclusive holder's guard applies
to it like to everything else; a `gpu: exclusive` plugin asks for every GPU at once (gpu-exclusive). When it's short on room, the central
GPU manager (backend/cluster/local.py) unloads idle, unpinned consumers of any kind
-- LLMs, ComfyUI, TTS engines, STT -- least recently used first; then it learns the
free VRAM. Blocked through tunnels like every non-/public path."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.cluster.local import MANAGER
from backend.model_manager import fetch_gpu_status

logger = logging.getLogger("ai-serve-ui.coordination")
router = APIRouter(tags=["coordination"])


class PreflightRequest(BaseModel):
    purpose: str = Field(..., min_length=1, max_length=80, description="e.g. 'tts:cosyvoice'")
    gpu: Optional[int] = Field(None, ge=0, description="omitted: the manager picks the GPU with room (answer: gpu, mode)")
    need_mb: int = Field(..., ge=0)
    current_gpu: Optional[int] = Field(None, ge=0, description="the GPU the service runs on now (None: not running)")
    movable: bool = Field(True, description="the service may be restarted on another GPU (it picks its GPU at start)")


@router.post("/api/coordination/gpu-preflight", summary="May an out-of-process backend cold-start on this GPU?")
async def gpu_preflight(req: PreflightRequest):
    """Fast yes/no (no waiting list -- the caller retries). Yes = a reservation of
    `need_mb` (idle consumers unloaded if needed): confirm it with the started
    worker's PID within `hold_s` (POST .../reservations/{id}/confirm), then it's
    bound to that process until it has allocated the memory or exits."""
    # 507 (raised by the manager) while a gpu-exclusive plugin holds the GPU
    gpu, mode = req.gpu, None
    if gpu is None:   # the manager picks (cluster/consumers.py choose_gpu): start / stay / move / evict
        from backend.cluster.consumers import choose_gpu
        # for what this purpose really needed before, not only what it announces (NeedBook)
        gpu, mode = choose_gpu(await MANAGER.room(), MANAGER.need(req.purpose, req.need_mb), req.current_gpu,
                               req.movable)
    res, unloaded, reason = await MANAGER.reserve(gpu, req.need_mb, req.purpose)
    if unloaded:
        logger.info(f"{req.purpose} needs ~{req.need_mb} MiB on GPU {gpu}: unloaded {', '.join(unloaded)}")
    elif res is None and reason:
        logger.info(f"{req.purpose}: {reason}")
    gpus = {g["index"]: g for g in (await fetch_gpu_status()).get("gpus", [])}
    body = {"gpu": gpu, "mode": mode, "free_mb": gpus.get(gpu, {}).get("free"), "granted": res is not None,
            "reservation_id": res.id if res else None, "hold_s": int(MANAGER.reservations.hold_s),
            "reason": reason or None}
    if res is None:
        # a no is 507 (2026-10-24): as 200 + granted false, stt-serve and tts-serve loaded anyway and ran out of memory
        return JSONResponse(status_code=507, content={**body, "detail": reason or f"no room on GPU {gpu}"})
    return body


@router.get("/api/coordination/room", summary="Memory each GPU could give right now without unloading anyone")
async def gpu_room():
    """Hardware-free minus what is promised to loads in progress; 0 under an exclusive holder. For plugins that
    decide themselves where to run (e.g. image-serve moving an idle ComfyUI to a roomier GPU)."""
    return {"room": {str(g): mb for g, mb in (await MANAGER.room()).items()}}


class ExclusiveRequest(BaseModel):
    purpose: str = Field(..., min_length=1, max_length=80, description="e.g. 'Strata'")


@router.post("/api/coordination/gpu-exclusive", summary="May a gpu-exclusive plugin take every GPU now?")
async def gpu_exclusive(req: ExclusiveRequest):
    """Fast yes/no. Yes = idle, unpinned consumers on every GPU unloaded and one reservation per GPU: confirm each
    with the started process' PID (POST .../reservations/{id}/confirm); they last as long as it lives. No = busy or
    pinned consumers (or other programs) are in the way -- nothing was unloaded."""
    gpus = [g["index"] for g in (await fetch_gpu_status()).get("gpus", [])]
    held, unloaded, reason = await MANAGER.reserve_exclusive(gpus, req.purpose)
    if held is None:
        logger.info(f"{req.purpose} (exclusive): {reason}")
    elif unloaded:
        logger.info(f"Unloaded for {req.purpose}: {', '.join(unloaded)}")
    return {"granted": held is not None, "reservation_ids": [r.id for r in held or []],
            "unloaded": list(unloaded or []), "reason": reason or None}


class ConfirmRequest(BaseModel):
    pid: int = Field(..., gt=1, strict=True)


@router.post("/api/coordination/reservations/{reservation_id}/confirm",
             summary="Bind a GPU reservation to the process that was started for it")
async def confirm_reservation(reservation_id: str, req: ConfirmRequest):
    if not MANAGER.confirm(reservation_id, req.pid):
        raise HTTPException(status_code=410, detail="reservation unknown or expired -- ask again")
    return {"reservation_id": reservation_id, "pid": req.pid}


@router.delete("/api/coordination/reservations/{reservation_id}",
               summary="Release a GPU reservation (the worker is loaded, or its start was given up)")
async def release_reservation(reservation_id: str):
    await MANAGER.release_measured(reservation_id)
    return {"released": reservation_id}
