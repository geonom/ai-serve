"""/api/training/* -> train-serve. Fine-tuning moved into its own service on 2026-10-17
(the train-serve app, 127.0.0.1:8038); remote clients (finetuning
client on the Mac) keep using http://<this machine>:8030/api/training.

train-serve checks the bearer token itself (fail closed); the access policy lets this
prefix through for that reason. Only this prefix is forwarded -- train-serve's
token-free monitor API (/api/training-monitor/*, for its own page) is never reachable
through the gateway. Request and response bodies stream (dataset uploads, adapter/GGUF
downloads)."""
import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from backend import train_client
from backend.config import TRAIN_SERVE_URL

logger = logging.getLogger("ai-serve-ui.training-proxy")
router = APIRouter(tags=["training"])

_PATH = re.compile(r"\A[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\Z")
_PASS = ("authorization", "content-type", "accept", "range")
_BACK = ("content-type", "content-disposition", "content-length", "cache-control", "content-range", "accept-ranges")
_transport: Optional[httpx.AsyncBaseTransport] = None  # tests put a MockTransport here


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=TRAIN_SERVE_URL, transport=_transport,
                             timeout=httpx.Timeout(30.0, read=None, write=None))


@router.api_route("/api/training/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                  summary="Fine-tuning API (forwarded to train-serve; bearer token)")
async def forward(path: str, request: Request):
    raw = (request.scope.get("raw_path") or b"").lower()
    if (not _PATH.match(path) or any(seg in (".", "..") for seg in path.split("/"))
            or b"%2f" in raw or b"%2e" in raw or b"%5c" in raw):
        raise HTTPException(status_code=400, detail="invalid path")
    headers = {k: v for k, v in request.headers.items() if k.lower() in _PASS}
    if request.client:
        headers["X-Forwarded-For"] = request.client.host
    url = f"/api/training/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    body = await request.body()
    for attempt in (1, 2):
        client = _client()
        try:
            upstream = await client.send(client.build_request(request.method, url, headers=headers, content=body),
                                         stream=True)
            break
        except httpx.ConnectError:
            await client.aclose()
            if attempt == 2 or not await train_client.ensure_running():
                raise HTTPException(status_code=503, detail="train-serve is not running and could not be "
                                                            "started (Apps & Services → Training).")
    out = {k: v for k, v in upstream.headers.items() if k.lower() in _BACK}

    async def close():
        await upstream.aclose()
        await client.aclose()
    return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code, headers=out,
                             background=BackgroundTask(close))
