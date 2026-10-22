"""ai-serve's GPU manager, reached over its coordination API (it stays the GPU owner) -- the same calls ComfyUI's code
made in-process before it moved here: room per GPU, reserve, confirm with a pid, release, what ComfyUI holds.
`choose_gpu` is the manager's own rule (copied, pure)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import httpx
from fastapi import HTTPException

from image_serve.config import GATEWAY

logger = logging.getLogger("image-serve.gpu")
MOVE_MARGIN_MB = 2048
PLUGIN_ID = "comfyui"      # how ai-serve lists this plugin's workers: comfyui:comfyui


def choose_gpu(free_mb: Mapping[int, int], need_mb: int, current: Optional[int], movable: bool) -> Tuple[int, str]:
    """ai-serve-ui backend/cluster/consumers.py choose_gpu: start / stay / move / evict."""
    if current is None:
        return max(free_mb, key=lambda g: (free_mb[g], -g)), "start"
    best = max(free_mb, key=lambda g: (free_mb[g], -g))
    if (movable and best != current and free_mb[best] >= need_mb
            and free_mb[best] - free_mb.get(current, 0) >= MOVE_MARGIN_MB):
        return best, "move"
    if free_mb.get(current, 0) >= need_mb:
        return current, "stay"
    if movable:
        roomy = [g for g, f in free_mb.items() if g != current and f >= need_mb]
        if roomy:
            return max(roomy, key=lambda g: (free_mb[g], -g)), "move"
    return current, "evict"


@dataclass
class Reservation:
    id: str
    gpu: Optional[int] = None
    mb: int = 0
    purpose: str = ""
    created: float = 0.0


async def _call(method: str, path: str, **kw) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120.0) as c:
        return await c.request(method, f"{GATEWAY}{path}", **kw)


class Manager:
    async def room(self) -> Dict[int, int]:
        try:
            r = await _call("GET", "/api/coordination/room")
            return {int(k): int(v) for k, v in r.json().get("room", {}).items()} if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}

    async def held_mb(self, gpu: int) -> int:
        """What ComfyUI measurably holds on `gpu` (ai-serve attributes it to this plugin's worker)."""
        try:
            r = await _call("GET", "/api/gpu/consumers")
            for g in r.json().get("gpus", []):
                if g.get("index") == gpu:
                    return sum(int(c.get("mb") or 0) for c in g.get("consumers", [])
                               if str(c.get("id", "")).startswith(f"{PLUGIN_ID}:"))
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return 0

    async def reserve(self, gpu: int, need_mb: int, purpose: str):
        """(Reservation | None, [], reason). 507 while an exclusive plugin holds the GPUs (raised)."""
        try:
            r = await _call("POST", "/api/coordination/gpu-preflight",
                            json={"purpose": purpose[:80], "gpu": gpu, "need_mb": max(0, int(need_mb))})
        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"ai-serve (the GPU owner) not reachable at {GATEWAY}: {e}")
        if r.status_code == 507:
            raise HTTPException(status_code=507, detail=r.json().get("detail", "GPUs held exclusively"))
        d = r.json() if r.status_code == 200 else {}
        if d.get("granted") and d.get("reservation_id"):
            return Reservation(d["reservation_id"]), [], ""
        return None, [], d.get("reason") or r.text[:200]

    async def raise_if_exclusive(self, gpu: Optional[int], purpose: str) -> None:
        res, _, _ = await self.reserve(gpu if gpu is not None else 0, 0, purpose)   # 507 raised when held
        if res is not None:
            self.release(res.id)

    def confirm(self, reservation_id: str, pid: int) -> None:
        self._later(_call("POST", f"/api/coordination/reservations/{reservation_id}/confirm", json={"pid": pid}))

    def release(self, reservation_id: str) -> None:
        self._later(_call("DELETE", f"/api/coordination/reservations/{reservation_id}"))

    @staticmethod
    def _later(coro) -> None:
        async def run():
            try:
                await coro
            except httpx.HTTPError as e:
                logger.warning(f"GPU manager call failed: {e}")
        try:
            asyncio.get_running_loop().create_task(run())
        except RuntimeError:
            coro.close()


MANAGER = Manager()
