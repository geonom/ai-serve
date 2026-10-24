"""/api/comfyui/* -- ComfyUI's job API (generate / edit / style-transfer / batch / task / loras / workflows / images /
start / stop / unload / status ...), from before image plugins existed and still used by several apps (Munin,
dynasty, language_learning, train-serve, Liruna, the Playground). Since 2026-10-21 ComfyUI is the image-serve plugin;
this forwards the whole prefix to the image plugin that declares `comfyui_api` (docs/plugins.md), unchanged.
Tests: tests/unit/test_comfyui_compat.py."""
from __future__ import annotations

import re
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.plugins import registry

router = APIRouter(tags=["comfyui (forwarded)"])
_HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "upgrade", "accept-encoding"}
_SEGMENT = re.compile(r"^[^/\x00-\x1f\\]{1,200}$")


def _plugin():
    return next((p for p in registry.providers("image") if "comfyui_api" in p.provision.endpoints), None)


@router.api_route("/api/comfyui/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def forward(path: str, request: Request):
    p = _plugin()
    if p is None:
        raise HTTPException(status_code=404, detail="No image plugin offers ComfyUI's API (see Apps & Services).")
    segments = path.split("/")
    if any(s in ("", ".", "..") or not _SEGMENT.match(s) for s in segments):
        raise HTTPException(status_code=400, detail="invalid path")
    if not await registry.ensure_running(p):
        raise HTTPException(status_code=503, detail=f"{p.name} is not reachable -- see Apps & Services")
    # the plugin's own loopback port and declared prefix; each segment quoted (file names may hold spaces)
    target = registry.url(p, "comfyui_api") + "/" + "/".join(quote(s, safe="") for s in segments)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    try:
        async with registry._client(registry.TIMEOUT_S) as c:
            r = await c.request(request.method, target, params=request.query_params, headers=headers,
                                content=await request.body())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"{p.name}: {type(e).__name__}")
    out = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "content-disposition", "cache-control")}
    return Response(content=r.content, status_code=r.status_code, headers=out)
