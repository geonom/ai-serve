"""Central GPU manager endpoints: who holds VRAM on this machine's GPUs (measured
per process), pin a consumer so it's never unloaded automatically, unload one by
hand. Rules: backend/cluster/consumers.py; I/O: backend/cluster/local.py."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StrictBool

from backend.cluster.local import MANAGER

router = APIRouter(tags=["gpu"])


class PinRequest(BaseModel):
    pinned: StrictBool


@router.get("/api/gpu/consumers", summary="Who holds VRAM on which GPU (measured), pinned / busy / evictable")
async def consumers():
    return {"gpus": await MANAGER.view()}


@router.post("/api/gpu/consumers/{consumer_id}/pin", summary="Pin (never unload automatically) or unpin")
async def pin(consumer_id: str, req: PinRequest):
    return {"id": consumer_id, "pinned": sorted(MANAGER.set_pin(consumer_id, req.pinned))}


@router.post("/api/gpu/consumers/{consumer_id}/unload",
             summary="Unload a consumer from VRAM now (refused while busy; never deletes files)")
async def unload(consumer_id: str):
    try:
        await MANAGER.evict(consumer_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"unloaded": consumer_id}
