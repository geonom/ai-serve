"""Dashboard endpoints for Hugging Face GGUF downloads (backend/hf_download.py, rules in backend/hf_models.py).
Behind the dashboard access policy like every /api path: this machine, the owner's devices, or the API key."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend import hf_download
from backend.hf_models import HfLinkError

router = APIRouter()


def _body(body: Any, allowed: set) -> dict:
    if not isinstance(body, dict) or not set(body) <= allowed or not isinstance(body.get("url"), str):
        raise HTTPException(status_code=400, detail=f"send a JSON object with url (and only {', '.join(sorted(allowed))})")
    for k in allowed - {"url"}:
        if body.get(k) is not None and not isinstance(body[k], str):
            raise HTTPException(status_code=400, detail=f"{k} must be a file path from the listing, or null")
    return body


@router.post("/api/models/hf/inspect", summary="List the GGUF models of a Hugging Face repo (from a pasted address)")
async def inspect(body: Any = Body(...)):
    b = _body(body, {"url"})
    try:
        return await asyncio.to_thread(hf_download.inspect, b["url"])
    except HfLinkError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/models/hf/download", status_code=202, summary="Download one GGUF model (+ vision projector)")
async def download(body: Any = Body(...)):
    b = _body(body, {"url", "model", "mmproj"})
    try:
        plan = await asyncio.to_thread(hf_download.make_plan, b["url"], b.get("model"), b.get("mmproj"))
    except HfLinkError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return hf_download.start(plan)


@router.get("/api/models/hf/downloads", summary="Hugging Face downloads of this run and how far they got")
async def downloads():
    return {"downloads": hf_download.jobs()}
