"""/v1/llm/advice/*: which language model suits a task, from `advice` plugins (backend/plugins/advice.py) -- for apps
(domains, recommend), people (ask) and agents (mcp). Tests: tests/unit/test_advice_plugins.py."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from backend.config import DEFAULT_CHAT_MODEL, GATEWAY_PORT
from backend.plugins import advice
from backend.version import BACKEND_VERSION

router = APIRouter()
ASK_TIMEOUT_S = 600.0                 # a cold model load, then two short completions


LLM_CATALOG_TTL_S = 300.0           # /v1/llm/models reads every GGUF header and asks devices: 20 s to 2 min here
_llm_cache: Dict[str, Any] = {"at": 0.0, "data": None}


async def _llm_catalog() -> Dict[str, Any]:
    """What this machine can serve, for `installed_only` -- cached: which files exist changes rarely, and a ranking
    must not wait minutes for it. (`loaded` may lag by up to the TTL.)"""
    from backend.routers.gateway import list_llm_models
    if _llm_cache["data"] is None or time.monotonic() - _llm_cache["at"] > LLM_CATALOG_TTL_S:
        _llm_cache["data"] = await list_llm_models()
        _llm_cache["at"] = time.monotonic()
    return _llm_cache["data"]


@router.get("/v1/llm/advice/domains", summary="What model advice can rank for: use cases and benchmark dimensions")
async def advice_domains():
    entries, errors = await advice.catalog()
    rows = [{**{k: v for k, v in e.items() if k != "plugin"}, "provider": e["plugin"]} for e in entries]
    return {"object": "list", "data": rows, **({"errors": errors} if errors else {})}


@router.post("/v1/llm/advice/recommend", summary="Rank the language models for a domain under speed / memory limits")
async def advice_recommend(body: Any = Body(...)):
    try:
        return await advice.recommend(body, _llm_catalog)
    except advice.AdviceError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    model: Optional[str] = Field(None, max_length=200, description="chat model that reads the question and words "
                                                                    "the answer (default: DEFAULT_CHAT_MODEL)")
    limit: int = Field(5, ge=1, le=20)


async def _chat(model: str, messages, response_format=None, max_tokens: int = 600) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "thinking": False}
    if response_format:
        payload["response_format"] = response_format
    async with httpx.AsyncClient(timeout=ASK_TIMEOUT_S) as c:
        r = await c.post(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions", json=payload)
    if r.status_code != 200:
        raise advice.AdviceError(f"{model}: chat answered {r.status_code}: {r.text[:200]}", 502)
    return r.json()["choices"][0]["message"].get("content") or ""


@router.post("/v1/llm/advice/ask", summary="A question in words; the answer from the ranking, with the ranking")
async def advice_ask(req: AskRequest):
    """A chat model turns the question into a recommend request (JSON schema over the domain list), the advice
    plugin ranks, and the model words the answer from those rows only. `request` and `recommendation` come back
    with it, so an app (or person) can check every number against the ranking."""
    model = req.model or DEFAULT_CHAT_MODEL
    try:
        entries, errors = await advice.catalog()
        if not entries:
            raise advice.AdviceError("no advice plugin answered" + (f": {'; '.join(errors)}" if errors else ""), 503)
        ids = [e["id"] for e in entries]
        raw = await _chat(model, advice.plan_messages(req.question, entries),
                          {"type": "json_schema", "json_schema": {"name": "advice_request",
                                                                  "schema": advice.plan_schema(ids)}}, 300)
        try:
            planned = json.loads(raw)
        except ValueError:
            raise advice.AdviceError(f"{model} answered no JSON request", 502)
        request = advice.plan_request(planned, ids, req.limit)
        rec = await advice.recommend(request, _llm_catalog)
        text = await _chat(model, advice.answer_messages(req.question, advice.facts(rec, top=req.limit)))
    except advice.AdviceError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"chat model: {type(e).__name__}")
    return {"object": "advice.answer", "answer": text.strip(), "model": model, "request": request,
            "recommendation": rec}


@router.post("/v1/llm/advice/mcp", summary="MCP server (streamable HTTP) for agents: list_model_domains, recommend_models")
async def advice_mcp(request: Request):
    try:
        msg = await request.json()
    except ValueError:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}},
                            status_code=400)
    if isinstance(msg, list):
        out = [r for r in [await advice.mcp_handle(m, _llm_catalog, str(BACKEND_VERSION)) for m in msg] if r]
        return JSONResponse(out) if out else Response(status_code=202)
    out = await advice.mcp_handle(msg, _llm_catalog, str(BACKEND_VERSION))
    return JSONResponse(out) if out is not None else Response(status_code=202)


@router.get("/v1/llm/advice/mcp", include_in_schema=False)
async def advice_mcp_stream():
    # no server-initiated messages: answers come back on the POST (streamable HTTP allows 405 here)
    return Response(status_code=405, headers={"Allow": "POST"})
