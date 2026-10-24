"""OpenAI-compatible gateway: /api/chat/completions, /v1/chat/completions, /v1/models, /v1/embeddings, /v1/rerank."""
import asyncio
import glob
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.config import MODEL_DIRS
from backend.model_manager import (
    check_model_file_exists,
    get_or_load_model,
    get_upstream_error_detail,
    reset_keep_alive,
    resolve_model_path,
    find_active_model_port,
    find_draft_model_path,
)
from backend.response_processing import gguf_has_bundled_mtp, is_loadable_gguf, is_mtp_draft, process_completion_response
from backend.plugins import llm as llm_plugins
from backend.cluster.service import SERVICE as CLUSTER, Upstream
from backend.state import state

logger = logging.getLogger("ai-serve-ui.gateway")
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# Shared across every request model below that has a gpu_id field -- confirmed live
# (2026-08-27): get_or_load_model()/_load_locked() has no load balancing across GPUs.
# When gpu_id is left unset and this model is already running on more than one GPU,
# the existing-instance lookup returns the FIRST matching entry in
# state.managed_processes (insertion order, i.e. whichever instance was started
# first) -- never round-robined, and unrelated to that GPU's current load. A second
# GPU's instance of the same model sits fully idle unless a caller sets gpu_id
# explicitly to target it. When no instance is running yet, leaving this unset
# auto-picks the GPU with the most free VRAM at load time.
_GPU_ID_DESC = (
    "Which GPU to route this request to. Left unset, an already-running instance of "
    "this model is matched by model path alone -- if the same model is loaded on "
    "multiple GPUs, the FIRST one registered is always picked (no load balancing), "
    "regardless of its current load; the other GPU's instance stays idle. Set this "
    "explicitly to target a specific GPU's instance. If no instance is running yet, "
    "leaving this unset auto-picks the GPU with the most free VRAM."
)


class UnifiedChatRequest(BaseModel):
    model_name: Optional[str] = None
    messages: List[Dict[str, Any]]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    thinking: Optional[bool] = None
    gpu_id: Optional[int] = Field(None, description=_GPU_ID_DESC)
    keep_alive: int = 300
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None  # llama-server -np at auto-load; default 1, 0 = llama.cpp auto (4)
    response_format: Optional[Dict[str, Any]] = None
    retries: int = 3
    mtp: Optional[bool] = None


class UnifiedChatResponse(BaseModel):
    status: str
    content: str
    data: Optional[Any] = None
    thinking: str
    model_used: str
    runtime_seconds: float
    tokens: int
    tps: float


class ChatCompletionChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    tps: Optional[float] = None
    runtime_seconds: Optional[float] = None


class ChatCompletionRequest(BaseModel):
    # llama-server's OpenAI-compat layer accepts every standard OpenAI sampling field
    # (stop, seed, top_p, presence_penalty, frequency_penalty, logit_bias, logprobs,
    # top_logprobs, ...) PLUS its own native extensions (top_k, min_p, repeat_penalty,
    # mirostat*, dry_*, xtc_*, grammar, samplers, cache_prompt, reasoning_format, ...) --
    # confirmed via llama.cpp's tools/server/server-schema.cpp field_* registrations.
    # Explicitly enumerating all of those here would be fragile (llama.cpp adds new
    # sampler params over time) and this endpoint's docstring already claims "OpenAI-
    # compatible" -- so instead of whitelisting each one, allow arbitrary extra fields
    # and forward them verbatim (see the payload-building code below, which merges
    # `model_extra`). Without this, FastAPI/Pydantic silently drops any field not
    # explicitly declared below, before the handler ever sees it -- confirmed live as
    # the root cause of chat_template_kwargs.enable_thinking being ignored by this
    # gateway (see CLAUDE.md) despite working correctly against llama-server directly;
    # the same silent-drop applies to every other undeclared field.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    voice: Optional[str] = None
    thinking: Optional[bool] = None
    gpu_id: Optional[int] = Field(None, description=_GPU_ID_DESC)
    keep_alive: Optional[int] = 300
    # Only takes effect on an auto-load. If a matching model_path is already
    # running, model_manager._load_locked() compares ctx_size against that
    # instance's own: bigger request -> the instance is restarted with the
    # larger context; equal/smaller -> the running instance is reused as-is.
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None  # llama-server -np at auto-load; default 1, 0 = llama.cpp auto (4)
    retries: Optional[int] = 3
    tools: Optional[List[Dict[str,Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    mtp: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[Union[str, List[int]]]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    gpu_id: Optional[int] = Field(None, description=_GPU_ID_DESC)
    keep_alive: Optional[int] = 300
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None  # llama-server -np at auto-load; default 1, 0 = llama.cpp auto (4)


class EmbeddingResponseDataItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponseUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingResponseDataItem]
    model: str
    usage: EmbeddingResponseUsage


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    gpu_id: Optional[int] = Field(None, description=_GPU_ID_DESC)
    keep_alive: Optional[int] = 300
    ctx_size: Optional[int] = None
    parallel: Optional[int] = None  # llama-server -np at auto-load; default 1, 0 = llama.cpp auto (4)
    # Rarely needed in practice -- llama-server's own --reranking flag already forces
    # pooling type to "rank" server-side (common/arg.cpp), regardless of the model's
    # GGUF metadata. Exposed only as an escape hatch, same spirit as StartRequest.pooling.
    pooling: Optional[str] = None


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    object: str = "list"
    model: str
    results: List[RerankResultItem]
    usage: EmbeddingResponseUsage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_chat_template_kwargs(
    thinking: Optional[bool], explicit: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Translate this gateway's own `thinking` flag into llama-server's native
    chat_template_kwargs.enable_thinking -- confirmed live (backend/routers/gateway.py's
    only reliable lever for a model whose chat template bakes thinking in at the Jinja
    level, e.g. gemma-4-12b-it, which starts every turn with an unconditional <|think|>
    token). server-common.cpp reads body.chat_template_kwargs.enable_thinking and passes
    it straight into chat-template rendering, actually skipping the reasoning block
    entirely -- a materially different (and more reliable) mechanism than
    _inject_thinking_instruction()'s plain system-prompt nudge, which a template-level
    default can simply override regardless of what the prompt asks for.

    `explicit` is a caller-supplied chat_template_kwargs dict (the OpenAI/vLLM-style
    passthrough convention some external tools already use directly) -- if it already
    sets enable_thinking, that wins over the derived value from `thinking`, since it's
    the more specific, deliberately-typed signal (must be a real JSON boolean; llama-server
    rejects a string here).
    """
    kwargs = dict(explicit) if explicit else {}
    if thinking is not None and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = thinking
    return kwargs or None


def _is_tool_format_error(err: str) -> bool:
    """True when an upstream error is llama.cpp's own tool-call-grammar rejection
    (chat.cpp's common_chat_parse: 'The model produced output that does not match
    the expected {format} format') rather than some other upstream failure (a JSON-
    schema validation failure, a timeout, a connection error, ...). Substring match,
    not a full-string comparison, since the format name varies by architecture
    (peg-gemma4, peg-llama3, ...) -- shared by _build_retry_nudge() and the
    parallel_tool_calls-forcing retry override right below it, so both react to the
    same failure class consistently instead of drifting apart if one changes."""
    return "does not match the expected" in err and "format" in err


def _build_retry_nudge(last_error: str) -> str:
    """Build the corrective message appended before a retry -- differentiated by what
    actually failed, since one generic nudge doesn't fit every failure class.

    Confirmed live (2026-08-27, an external caller's tool-calling workflow against
    gemma-4-E4B-it-qat): llama.cpp's own chat.cpp raises exactly
    "The model produced output that does not match the expected {format} format"
    (see chat.cpp's common_chat_parse) when the model's raw text fails its
    architecture-specific tool-call grammar (a PEG parser, hence formats named
    "peg-<arch>") -- confirmed here by reading the model's actual rejected output
    straight out of llama-server's own log: it had chained two tool calls into one
    turn (calling the second with a placeholder before the first's result even came
    back) and appended trailing narrative prose after the closing tag, both of which
    violate the strict single-call grammar.

    Before this, every retry -- including this one -- reused the JSON-validation
    nudge ("Please output valid JSON") regardless of failure kind. That's actively
    wrong advice for a tool-call-format rejection (the output wasn't JSON at all, and
    telling the model to fix its JSON does nothing to stop it chaining calls or
    adding prose), which is almost certainly why this failure mode ran out every one
    of the default 4 attempts (req.retries or 3) as often as it did rather than
    self-correcting on retry the way the generic JSON nudge does for its own failure
    class. Only a substring match (the exact upstream format name varies by
    architecture, e.g. "peg-gemma4"), not a full-string comparison, so this stays
    correct if llama.cpp adds another architecture's peg-format down the line.
    """
    if _is_tool_format_error(last_error):
        return (
            f"Your previous response was rejected by the server: {last_error}. Call "
            "exactly ONE tool per turn, wait for its result before calling another "
            "tool, and output nothing else -- no other tool calls, no explanatory "
            "text -- before or after the tool call."
        )
    return f"Your previous response failed validation: {last_error}. Please output valid JSON."


def _inject_thinking_instruction(messages: List[Dict[str, Any]], thinking: Optional[bool]) -> List[Dict[str, Any]]:
    msgs = list(messages)
    if thinking is False:
        instruction = "\nDo NOT use think tags or output reasoning. Output the answer directly."
    elif thinking is True:
        instruction = "\nPlease write your thinking process enclosed in <think>...</think> tags before your final response."
    else:
        return msgs
    if msgs and msgs[0]["role"] == "system":
        msgs[0] = dict(msgs[0])
        msgs[0]["content"] += instruction
    else:
        msgs.insert(0, {"role": "system", "content": instruction.lstrip("\n")})
    return msgs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# A model an llm plugin lists (backend/plugins/llm.py; e.g. Strata through its adapter) goes to that plugin's
# OpenAI server; everything else is a local llama-server or a device. These keep the handlers backend-agnostic.
PLUGIN_UPSTREAM_TIMEOUT_S = 1800.0   # a plugin may load for minutes on the first request and serve slowly


async def _resolve_upstream(model_name: Optional[str], **load_kwargs) -> Upstream:
    """Where a request goes: an llm plugin that lists the model, a local llama-server, or -- when this machine
    can't serve the model (507: no VRAM / an exclusive plugin owns the GPUs, or the file isn't here) -- an enrolled
    device (backend/cluster). An explicit gpu_id that doesn't fit stays a plain 400, as before."""
    found = await llm_plugins.find(model_name)
    if found is not None:
        p, entry = found
        if not await llm_plugins.registry.ensure_running(p):
            raise HTTPException(status_code=503, detail=f"{p.name} (serves {entry['id']}) is not reachable -- "
                                                        f"see Apps & Services, logs/service-{p.id}.log")
        return Upstream(kind="plugin", base=llm_plugins.base_url(p), label=f"{entry['id']} @ {p.id}", port=p.port,
                        model=entry["id"])
    path = resolve_model_path(model_name)
    local = os.path.isfile(path)
    devices = CLUSTER.has_nodes()
    local_error = None
    if local or not devices:
        try:
            port = await get_or_load_model(model_path=path, **load_kwargs)
            return Upstream(kind="local", base=f"http://127.0.0.1:{port}", port=port,
                            label=os.path.basename(state.managed_processes[port]["model"]))
        except HTTPException as e:
            if e.status_code != 507 or not devices:
                raise
            local_error = e
    mode = "reranking" if load_kwargs.get("reranking") else "embedding" if load_kwargs.get("embedding") else "chat"
    try:
        return await CLUSTER.upstream_for(
            os.path.basename(path), path if local else None,
            ctx_size=load_kwargs.get("ctx_size") or 32000, keep_alive=load_kwargs.get("keep_alive") or 300,
            local_size_mb=(os.path.getsize(path) >> 20) if local else None,
            mode=mode, pooling=load_kwargs.get("pooling"))
    except HTTPException as e:
        if local_error is not None and e.status_code == 507:
            raise HTTPException(status_code=507, detail=f"{local_error.detail}\nOther devices: {e.detail}")
        if not local and os.path.islink(path) and e.status_code == 507:
            # the file is here, but on a drive that isn't mounted: say that, not just "on no device" (2026-10-24)
            try:
                check_model_file_exists(path, "Model")
            except HTTPException as missing:
                raise HTTPException(status_code=400, detail=f"{missing.detail}\nOther devices: {e.detail}")
        raise


def _begin_request(up: Upstream) -> None:
    if up.kind == "plugin":
        return            # the plugin keeps its own books (busy, keep-alive)
    if up.kind == "node":
        CLUSTER.begin(up)
    else:
        state.managed_processes[up.port]["active_requests"] += 1
        state.managed_processes[up.port]["last_used"] = time.time()


def _end_request(up: Upstream, keep_alive: Optional[int]) -> None:
    if up.kind == "plugin":
        return
    if up.kind == "node":
        CLUSTER.end(up, keep_alive)
    elif up.port in state.managed_processes:
        state.managed_processes[up.port]["active_requests"] = max(
            0, state.managed_processes[up.port].get("active_requests", 1) - 1
        )
        state.managed_processes[up.port]["last_used"] = time.time()
        reset_keep_alive(up.port, keep_alive)


def _model_label(up: Upstream) -> str:
    return up.label

def _upstream_client(up: Upstream, default_timeout: float) -> httpx.AsyncClient:
    """A plugin's server is reached like every plugin endpoint (registry's client: loopback, patchable in tests);
    it may load for minutes on the first request and serve slowly."""
    if up.kind == "plugin":
        return llm_plugins.registry._client(PLUGIN_UPSTREAM_TIMEOUT_S)
    return httpx.AsyncClient(timeout=default_timeout)


def _for_upstream(payload: Dict[str, Any], up: Upstream) -> Dict[str, Any]:
    """A plugin may serve several models: tell it which one (llama-server ignores the field)."""
    if up.model:
        payload["model"] = up.model
    return payload

@router.get("/api/props", summary="Proxy llama-server /props for tool-template introspection")
async def get_props(model: Optional[str] = None):
    port = find_active_model_port(model)
    if port is None:
        raise HTTPException(status_code=404, detail="No active model server found.")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"http://127.0.0.1:{port}/props")
        resp.raise_for_status()
        return resp.json()

@router.post("/api/chat/completions", summary="Unified chat completion", response_model=UnifiedChatResponse)
async def unified_chat_completion(req: UnifiedChatRequest):
    up = await _resolve_upstream(
        req.model_name, gpu_id=req.gpu_id, keep_alive=req.keep_alive,
        ctx_size=req.ctx_size or 32000, parallel=req.parallel,
        use_mtp=True if req.mtp is None else req.mtp,
    )
    _begin_request(up)

    try:
        processed_messages: List[Dict[str, Any]] = []
        has_system = False
        for msg in req.messages:
            if msg.get("role") == "system":
                has_system = True
            processed_messages.append(msg)
        if not has_system and req.system_prompt:
            processed_messages.insert(0, {"role": "system", "content": req.system_prompt})

        final_messages = _inject_thinking_instruction(processed_messages, req.thinking)
        retries_left = req.retries
        last_error = None

        for attempt in range(retries_left + 1):
            try:
                current_messages = list(final_messages)
                if attempt > 0 and last_error:
                    current_messages.append({
                        "role": "user",
                        "content": _build_retry_nudge(last_error),
                    })
                payload: Dict[str, Any] = {
                    "messages": current_messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "stream": False,
                }
                ctk = _build_chat_template_kwargs(req.thinking)
                if ctk:
                    payload["chat_template_kwargs"] = ctk
                if req.response_format:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "structured_schema", "strict": True, "schema": req.response_format},
                    }

                async with _upstream_client(up, 180.0) as client:
                    t0 = time.time()
                    resp = await client.post(up.url("/v1/chat/completions"), json=_for_upstream(payload, up),
                                             headers=up.headers)
                    duration = time.time() - t0
                    resp.raise_for_status()
                    data = resp.json()

                message = data["choices"][0]["message"]
                upstream_finish_reason = data["choices"][0].get("finish_reason", "stop")
                try:
                    raw_content, reasoning, parsed_data = process_completion_response(
                        message, req.thinking, req.response_format, finish_reason=upstream_finish_reason
                    )
                except Exception as val_err:
                    last_error = str(val_err)
                    logger.warning(f"Validation attempt {attempt + 1} failed: {val_err}")
                    if attempt < retries_left:
                        continue
                    raise HTTPException(
                        status_code=422,
                        detail=f"JSON validation failed after {req.retries} retries: {val_err}. "
                               f"Last candidate's finish_reason was '{upstream_finish_reason}' "
                               f"(if 'length', the schema-constrained response was truncated by "
                               f"max_tokens before it could complete -- raise max_tokens rather than "
                               f"assuming the model won't follow the schema). "
                               f"Content was: {message.get('content', '')}",
                    )

                tokens = data.get("usage", {}).get("completion_tokens", 0)
                tps = round(tokens / duration, 2) if duration > 0 else 0
                return {
                    "status": "success",
                    "content": raw_content,
                    "data": parsed_data,
                    "thinking": reasoning,
                    "model_used": _model_label(up),
                    "runtime_seconds": round(duration, 2),
                    "tokens": tokens,
                    "tps": tps,
                }
            except httpx.HTTPError as http_err:
                err_detail = get_upstream_error_detail(http_err)
                logger.error(f"HTTP error: {err_detail}")
                if attempt < retries_left:
                    last_error = err_detail
                    continue
                sc = http_err.response.status_code if isinstance(http_err, httpx.HTTPStatusError) else 502
                raise HTTPException(status_code=sc, detail=f"Upstream error: {err_detail}")
    finally:
        _end_request(up, req.keep_alive)


@router.post("/v1/chat/completions", summary="OpenAI-compatible chat completions", response_model=ChatCompletionResponse)
async def openai_chat_completion(req: ChatCompletionRequest):
    # Handle audio input (STT preprocessing)
    processed_messages: List[Dict[str, Any]] = []
    for msg in req.messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    from backend.routers.media import STTTranscribeRequest, transcribe_stt
                    stt_req = STTTranscribeRequest(audio_base64=part["input_audio"]["data"])
                    stt_res = await transcribe_stt(stt_req)
                    new_content.append({"type": "text", "text": stt_res["transcript"]})
                else:
                    new_content.append(part)
            processed_messages.append({"role": role, "content": new_content})
        else:
            processed_messages.append(msg)

    up = await _resolve_upstream(
        req.model, gpu_id=req.gpu_id, keep_alive=req.keep_alive or 300,
        ctx_size=req.ctx_size or 32000, parallel=req.parallel,
        use_mtp=True if req.mtp is None else req.mtp,
    )
    _begin_request(up)

    # Streaming fast-path
    if req.stream and not req.response_format and not req.modalities:
        stream_messages = _inject_thinking_instruction(processed_messages, req.thinking)
        payload = {
            "messages": stream_messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }
        ctk = _build_chat_template_kwargs(req.thinking, req.chat_template_kwargs)
        if ctk:
            payload["chat_template_kwargs"] = ctk
        # Forward any llama-server-native/OpenAI-standard field this schema doesn't
        # explicitly declare (stop, seed, top_p, presence_penalty, repeat_penalty,
        # mirostat*, dry_*, grammar, ...) straight through -- see ChatCompletionRequest's
        # extra="allow" docstring. None of these keys can collide with the ones set
        # above, since model_extra only ever contains fields NOT declared on the model.
        if req.model_extra:
            payload.update(req.model_extra)

        if req.tools:
            payload["tools"] = req.tools
        if req.tool_choice:
            payload["tool_choice"] = req.tool_choice
        if req.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = req.parallel_tool_calls

        # Open the upstream first: if it refuses (wrong kind of model, bad params ...) the caller gets that status
        # and message, not a 200 event stream with an error JSON inside (2026-10-20: an empty answer bubble)
        client = _upstream_client(up, 300.0)
        try:
            resp = await client.send(client.build_request("POST", up.url("/v1/chat/completions"), json=_for_upstream(payload, up),
                                                          headers=up.headers), stream=True)
        except httpx.HTTPError as e:
            await client.aclose()
            _end_request(up, req.keep_alive or 300)
            raise HTTPException(status_code=502, detail=f"Upstream unreachable: {type(e).__name__}")
        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            await client.aclose()
            _end_request(up, req.keep_alive or 300)
            try:
                err = json.loads(body)
                detail = (err.get("error") or {}).get("message") if isinstance(err.get("error"), dict) else err.get("detail")
            except ValueError:
                detail = None
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream error: {detail or body[:500].decode(errors='replace')}")

        async def _stream_sse():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
                _end_request(up, req.keep_alive or 300)

        return StreamingResponse(
            _stream_sse(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        final_messages = _inject_thinking_instruction(processed_messages, req.thinking)
        retries_left = req.retries or 3
        last_error = None

        for attempt in range(retries_left + 1):
            try:
                current_messages = list(final_messages)
                is_tool_format_retry = attempt > 0 and last_error is not None and _is_tool_format_error(last_error)
                if attempt > 0 and last_error:
                    current_messages.append({
                        "role": "user",
                        "content": _build_retry_nudge(last_error),
                    })
                payload = {
                    "messages": current_messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "stream": False,
                }
                ctk = _build_chat_template_kwargs(req.thinking, req.chat_template_kwargs)
                if ctk:
                    payload["chat_template_kwargs"] = ctk
                # See the streaming fast-path above / ChatCompletionRequest's extra="allow"
                # docstring: forward any undeclared llama-server-native/OpenAI-standard
                # field (stop, seed, top_p, presence_penalty, repeat_penalty, mirostat*,
                # dry_*, grammar, ...) straight through instead of silently dropping it.
                if req.model_extra:
                    payload.update(req.model_extra)
                if req.tools:
                    payload["tools"] = req.tools
                if req.tool_choice:
                    payload["tool_choice"] = req.tool_choice
                if req.parallel_tool_calls is not None:
                    payload["parallel_tool_calls"] = req.parallel_tool_calls
                if is_tool_format_retry:
                    # Confirmed against llama.cpp's own grammar builder (common/chat.cpp):
                    # it's parallel_tool_calls, not tool_choice, that controls how many
                    # <|tool_call>...<tool_call|> repeats the peg-gemma4 grammar accepts in
                    # one turn (max = parallel_tool_calls ? -1 : 1) -- tool_choice only
                    # changes the *minimum* (0 vs 1) and whether the grammar is applied
                    # lazily or eagerly, it doesn't add a rule permitting a second call or
                    # trailing content either way. The observed failure is the model
                    # chaining a second tool call (sometimes with freely-invented argument
                    # names, sometimes with narrative text around it) before the first
                    # call's result was even available -- forcing max=1 here directly
                    # targets that, unlike toggling tool_choice which wouldn't change the
                    # parse-time grammar structure at all. Overrides the caller's own
                    # parallel_tool_calls for this and every further attempt in this
                    # request, for as long as the same failure class keeps recurring --
                    # not a one-shot override, since there's no reason to expect the model
                    # would stop chaining calls on its own.
                    payload["parallel_tool_calls"] = False
                if req.response_format:
                    schema_data = req.response_format.get("json_schema", {}).get("schema") or req.response_format
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "structured_schema", "strict": True, "schema": schema_data},
                    }

                async with _upstream_client(up, 180.0) as client:
                    t0 = time.time()
                    resp = await client.post(up.url("/v1/chat/completions"), json=_for_upstream(payload, up),
                                             headers=up.headers)
                    duration = time.time() - t0
                    resp.raise_for_status()
                    data = resp.json()

                message = data["choices"][0]["message"]
                upstream_finish_reason = data["choices"][0].get("finish_reason", "stop")
                try:
                    raw_content, reasoning, parsed_data = process_completion_response(
                        message, req.thinking, req.response_format, finish_reason=upstream_finish_reason
                    )
                except Exception as val_err:
                    last_error = str(val_err)
                    logger.warning(f"Validation attempt {attempt + 1} failed: {val_err}")
                    if attempt < retries_left:
                        continue
                    raise HTTPException(
                        status_code=422,
                        detail=f"JSON validation failed after {req.retries} retries: {val_err}. "
                               f"Last candidate's finish_reason was '{upstream_finish_reason}' "
                               f"(if 'length', the schema-constrained response was truncated by "
                               f"max_tokens before it could complete -- raise max_tokens rather than "
                               f"assuming the model won't follow the schema). "
                               f"Content was: {message.get('content', '')}",
                    )

                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
                total_tokens = data.get("usage", {}).get("total_tokens", 0)
                tps = round(completion_tokens / duration, 2) if duration > 0 else 0

                # Audio modalities
                audio_res = None
                if req.modalities and "audio" in req.modalities:
                    from backend.routers.media import TTSGenerateRequest, generate_tts
                    voice_name = req.voice or "alloy"
                    ref_voice_path = None
                    instruct = None
                    voices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "voices")
                    if os.path.exists(voices_dir):
                        for f in os.listdir(voices_dir):
                            if f.lower().startswith(voice_name.lower()):
                                ref_voice_path = os.path.join(voices_dir, f)
                                break
                    if not ref_voice_path:
                        instruct = voice_name
                    tts_req = TTSGenerateRequest(
                        text=raw_content, language="en",
                        ref_voice_path=ref_voice_path, instruct=instruct, response_format="base64"
                    )
                    tts_res = await generate_tts(tts_req)
                    audio_res = {"id": "audio_" + str(uuid.uuid4()), "data": tts_res["audio_base64"], "transcript": raw_content}

                return ChatCompletionResponse(
                    id="chatcmpl-" + str(uuid.uuid4()),
                    created=int(time.time()),
                    model=_model_label(up),
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionChoiceMessage(
                            role="assistant", content=raw_content,
                            reasoning_content=reasoning if reasoning else None,
                            audio=audio_res,
                            tool_calls=message.get("tool_calls"),
                        ),
                        finish_reason=upstream_finish_reason,
                    )],
                    usage=ChatCompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                    tps=tps,
                    runtime_seconds=round(duration, 2),
                )
            except httpx.HTTPError as http_err:
                err_detail = get_upstream_error_detail(http_err)
                logger.error(f"HTTP error: {err_detail}")
                # llama.cpp throws this while re-parsing a *previous* tool call's arguments
                # out of the conversation history during chat-template rendering, before any
                # generation happens. Every retry resends the same poisoned history and fails
                # identically, so retrying is pure wasted latency — fail fast instead.
                if "Failed to parse tool call arguments as JSON" in err_detail:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "A previous assistant message in this conversation contains a tool_call "
                            "with malformed JSON in its arguments. llama.cpp fails to re-parse it while "
                            "rendering the chat template, before generation starts, so this request cannot "
                            f"succeed on retry. Remove or fix that tool_call from the history. Upstream: {err_detail}"
                        ),
                    )
                if attempt < retries_left:
                    last_error = err_detail
                    continue
                sc = http_err.response.status_code if isinstance(http_err, httpx.HTTPStatusError) else 502
                raise HTTPException(status_code=sc, detail=f"Upstream error: {err_detail}")
    finally:
        _end_request(up, req.keep_alive or 300)


@router.get("/v1/llm/models", summary="Language models on this machine, on devices and from llm plugins -- kind and abilities")
async def list_llm_models():
    """Everything chat/embeddings/rerank can reach, with what each model is (backend/llm_catalog.py)."""
    from backend.llm_catalog import build_catalog
    from backend.model_manager import get_model_traits
    from backend.routers.dashboard import list_models

    found = (await list_models())["models"]

    def _local() -> List[Dict[str, Any]]:
        out = []
        for m in found:
            if os.path.islink(m["path"]) and not os.path.exists(m["path"]):
                continue      # on a drive that isn't mounted -- can't be served until it is back (2026-10-24)
            try:
                size_mb = round(os.path.getsize(m["path"]) / 1048576)
            except OSError:
                size_mb = None
            out.append({"name": m["name"], "size_mb": size_mb, "context_length": m.get("context_length"),
                        "vision": bool(m.get("mmproj_path") or m.get("vision_bundled")),
                        "mtp": bool(m.get("mtp_draft_path") or m.get("mtp_bundled")),
                        "traits": get_model_traits(m["path"])})
        return out

    local = await asyncio.get_running_loop().run_in_executor(None, _local)
    loaded = {}
    for info in list(state.managed_processes.values()):
        proc = info.get("process")
        if proc is None or proc.poll() is None:
            loaded[os.path.basename(info.get("model") or "")] = {"gpu_id": info.get("gpu_id"), "ctx_size": info.get("ctx_size")}
    devices = []
    if CLUSTER.has_nodes():
        for snap in await CLUSTER.snapshots():
            devices.append({"id": snap.id, "online": snap.online,
                            "models": {os.path.basename(k): v for k, v in snap.models.items()},
                            "instances": [dict(i, model=os.path.basename(i.get("model") or "")) for i in snap.instances]})
    plugin_models, errors = await llm_plugins.catalog()
    states = {}
    for p in llm_plugins.registry.providers("llm"):
        if "status" in p.provision.endpoints:
            st = await llm_plugins.registry.status(p)
            state_ = "stopped" if st is None else "ready" if st["ready"] else "stopped" if not st["workers"] else "loading"
        else:                       # a standard OpenAI server (Ollama, LM Studio ...): it serves when it answers
            state_ = "ready" if await llm_plugins.registry.running(p) else "stopped"
        states[p.id] = {"state": state_, "exclusive": p.provision.gpu == "exclusive", "name": p.name}
    out = build_catalog(local, loaded_local=loaded, devices=devices, plugins=plugin_models, plugin_states=states)
    if errors:
        out["errors"] = errors
    return out


@router.get("/v1/models", summary="OpenAI-compatible model list")
async def list_openai_models():
    models = []
    seen_paths: set = set()
    for directory in MODEL_DIRS:
        if not os.path.isdir(directory):
            continue
        for f in glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True):
            abs_path = os.path.abspath(f)
            name = os.path.basename(f)
            if (
                abs_path not in seen_paths
                and "mmproj" not in name.lower()
                and not is_mtp_draft(name)
                and is_loadable_gguf(name)
            ):
                draft_path = find_draft_model_path(abs_path)
                models.append({
                    "id": name,
                    "object": "model",
                    "created": int(os.path.getmtime(f)),
                    "owned_by": "local",
                    "mtp_available": draft_path is not None or gguf_has_bundled_mtp(abs_path),
                })
                seen_paths.add(abs_path)
    return {"object": "list", "data": models}


@router.post("/v1/embeddings", summary="OpenAI-compatible embeddings", response_model=EmbeddingResponse)
@router.post("/api/embeddings", summary="Unified embeddings endpoint", response_model=EmbeddingResponse)
async def openai_embeddings(req: EmbeddingRequest):
    up = await _resolve_upstream(
        req.model, gpu_id=req.gpu_id, keep_alive=req.keep_alive or 300,
        ctx_size=req.ctx_size or 32000, parallel=req.parallel, embedding=True, use_mtp=False,
    )
    _begin_request(up)
    try:
        payload: Dict[str, Any] = {"input": req.input, "model": up.model or req.model}
        if req.encoding_format:
            payload["encoding_format"] = req.encoding_format
        if req.dimensions:
            payload["dimensions"] = req.dimensions
        if req.user:
            payload["user"] = req.user
        async with _upstream_client(up, 60.0) as client:
            resp = await client.post(up.url("/v1/embeddings"), json=payload, headers=up.headers)
            resp.raise_for_status()
            return EmbeddingResponse(**resp.json())
    except httpx.HTTPError as http_err:
        err_detail = get_upstream_error_detail(http_err)
        sc = http_err.response.status_code if isinstance(http_err, httpx.HTTPStatusError) else 502
        raise HTTPException(status_code=sc, detail=f"Upstream error: {err_detail}")
    finally:
        _end_request(up, req.keep_alive or 300)


@router.post("/v1/rerank", summary="OpenAI/Jina-style reranking", response_model=RerankResponse)
@router.post("/api/rerank", summary="Unified reranking endpoint", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """Score/rank `documents` against `query` using a dedicated cross-encoder reranker
    model (e.g. bge-reranker-v2-m3) -- distinct from an embedding model like bge-m3,
    which only produces vectors, not relevance scores. Auto-loads with --reranking
    (which implies --embedding and pooling type "rank" on llama-server's own side).

    bge-m3's own dense embeddings (via /v1/embeddings) only cover half of what bge-m3
    can do -- it also supports sparse (lexical) embeddings, which llama-server's
    /v1/embeddings does not expose at all. This endpoint doesn't fill that gap either;
    a hybrid dense+lexical retrieval setup still needs a separate lexical stage (e.g.
    SQLite FTS5) on the caller's side, with this rerank endpoint applied as a final
    cross-encoder pass over the combined candidate set.
    """
    up = await _resolve_upstream(
        req.model, gpu_id=req.gpu_id, keep_alive=req.keep_alive or 300,
        ctx_size=req.ctx_size or 32000, parallel=req.parallel, embedding=True, reranking=True, pooling=req.pooling,
        use_mtp=False,
    )
    _begin_request(up)
    try:
        payload: Dict[str, Any] = {"model": up.model or req.model, "query": req.query, "documents": req.documents}
        if req.top_n is not None:
            payload["top_n"] = req.top_n
        async with _upstream_client(up, 60.0) as client:
            resp = await client.post(up.url("/v1/rerank"), json=payload, headers=up.headers)
            resp.raise_for_status()
            data = resp.json()
            results = [
                RerankResultItem(
                    index=item["index"],
                    relevance_score=item["relevance_score"],
                    document=req.documents[item["index"]] if 0 <= item["index"] < len(req.documents) else None,
                )
                for item in data.get("results", [])
            ]
            return RerankResponse(
                model=data.get("model", req.model),
                results=results,
                usage=EmbeddingResponseUsage(**data.get("usage", {})),
            )
    except httpx.HTTPError as http_err:
        err_detail = get_upstream_error_detail(http_err)
        sc = http_err.response.status_code if isinstance(http_err, httpx.HTTPStatusError) else 502
        raise HTTPException(status_code=sc, detail=f"Upstream error: {err_detail}")
    finally:
        _end_request(up, req.keep_alive or 300)
