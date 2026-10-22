"""Curated OpenAI-compatible surface at /public/v1 -- the ONLY part of this backend
meant to be reachable from outside the machine (via the tunnel), e.g. for a Nextcloud
on shared hosting using its "OpenAI and LocalAI integration" app.

This module is the application layer: every security rule lives in
backend/public_api/domain.py (key, catalog, per-kind request policies, admission,
error disclosure); tunnel exposure and body limits are enforced one layer further
out by backend/public_api/middleware.py. Here we only wire those rules to the
existing /v1 handlers.

Request lifecycle (PublicRoute): authenticate -> admit (POST only) -> endpoint
(policy.sanitize -> delegate) -> map any failure through ErrorPolicy. Auth runs
before the body is even parsed.
"""
import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import PUBLIC_API_KEY, PUBLIC_MAX_CONCURRENT, PUBLIC_MODELS_FILE
from backend.public_api.domain import (
    AdmissionControl,
    BodyLimits,
    ChatPolicy,
    ErrorPolicy,
    ImagePolicy,
    ModelCatalog,
    ModelKind,
    PolicyViolation,
    PublicApiKey,
    SpeechPolicy,
    TranscriptionPolicy,
    authenticate,
)
from backend.routers import media
from backend.routers.gateway import ChatCompletionRequest, openai_chat_completion

logger = logging.getLogger("ai-serve-ui.public")

API_KEY: Optional[PublicApiKey] = PublicApiKey.from_secret(PUBLIC_API_KEY)
if PUBLIC_API_KEY and API_KEY is None:
    logger.warning(f"LLAMA_SERVE_PUBLIC_API_KEY is shorter than {PublicApiKey.MIN_LENGTH} characters "
                   "-- the public API stays disabled.")
ADMISSION = AdmissionControl(PUBLIC_MAX_CONCURRENT)
BODY_LIMITS = BodyLimits()
ERRORS = ErrorPolicy()
CHAT = ChatPolicy()
SPEECH = SpeechPolicy()
IMAGE = ImagePolicy()
TRANSCRIPTION = TranscriptionPolicy()

_AUDIO_FORMATS = {
    # response_format -> (ffmpeg output args, media type); keys == SpeechPolicy.FORMATS
    "mp3": (["-f", "mp3", "-c:a", "libmp3lame", "-q:a", "2"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-c:a", "libopus"], "audio/ogg"),
    "aac": (["-f", "adts", "-c:a", "aac"], "audio/aac"),
    "flac": (["-f", "flac"], "audio/flac"),
    "wav": (["-f", "wav"], "audio/wav"),
    "pcm": (["-f", "s16le", "-ac", "1", "-ar", "24000"], "audio/pcm"),
}


def _auth_shape(request: Request) -> str:
    """What the Authorization header looks like, without revealing the secret --
    enough to tell "missing", "Basic instead of Bearer" or "pasted with extra
    characters" (wrong length) apart when a client is misconfigured."""
    header = request.headers.get("authorization")
    if header is None:
        return "no Authorization header"
    scheme, _, value = header.partition(" ")
    expected = len(PUBLIC_API_KEY) if PUBLIC_API_KEY else None
    hint = []
    if value != value.strip():
        hint.append("surrounding whitespace")
    if value[:1] in "\"'" and value:
        hint.append("starts with a quote")
    if "=" in value[:30] and value.upper().startswith("LLAMA_SERVE"):
        hint.append("includes the variable name")
    return f"auth scheme {scheme!r}, value length {len(value)} (expected {expected})" + (
        f", {', '.join(hint)}" if hint else "")


def _client(request: Request) -> str:
    return request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")


class PublicRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        inner = super().get_route_handler()
        admits = "POST" in (self.methods or set())  # GPU work; listing models is free

        async def handler(request: Request) -> Response:
            ref = uuid.uuid4().hex[:12]
            try:
                authenticate(API_KEY, request.headers.get("authorization"))
                if admits:
                    ADMISSION.acquire()
            except PolicyViolation as e:
                logger.warning(f"[public {ref}] refused {request.method} {request.url.path} "
                               f"from {_client(request)}: {e.status} {e.message} ({_auth_shape(request)})")
                status, body = ERRORS.to_public(e, ref)
                return JSONResponse(body, status_code=status)

            release = admits
            try:
                response = await inner(request)
                if admits and isinstance(response, StreamingResponse):
                    response.body_iterator = _release_when_done(response.body_iterator)
                    release = False
                return response
            except PolicyViolation as e:
                logger.info(f"[public {ref}] {request.url.path}: {e.status} {e.message}")
                status, body = ERRORS.to_public(e, ref)
            except StarletteHTTPException as e:  # fastapi's HTTPException is a subclass
                cause = _domain_cause(e)
                if cause is not None:  # e.g. FastAPI wrapping the body limit's refusal
                    logger.info(f"[public {ref}] {request.url.path}: {cause.status} {cause.message}")
                    status, body = ERRORS.to_public(cause, ref)
                else:
                    logger.warning(f"[public {ref}] {request.url.path}: upstream {e.status_code} {e.detail}")
                    status, body = ERRORS.to_public_http(e.status_code, e.detail, ref)
            except RequestValidationError as e:
                logger.info(f"[public {ref}] {request.url.path}: validation {e.errors()}")
                status, body = ERRORS.to_public_http(422, None, ref)
            except Exception as e:
                logger.exception(f"[public {ref}] {request.url.path}: unexpected {e!r}")
                status, body = ERRORS.to_public(e, ref)
            finally:
                if release:
                    ADMISSION.release()
            return JSONResponse(body, status_code=status)

        return handler


def _domain_cause(exc: BaseException) -> Optional[PolicyViolation]:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, PolicyViolation):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


async def _release_when_done(chunks):
    try:
        async for chunk in chunks:
            yield chunk
    finally:
        ADMISSION.release()


router = APIRouter(prefix="/public/v1", tags=["public"], route_class=PublicRoute)


def load_catalog() -> ModelCatalog:
    """Re-read per request: edits to the file need no restart."""
    try:
        with open(PUBLIC_MODELS_FILE, encoding="utf-8") as f:
            return ModelCatalog.from_dict(json.load(f))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Public model catalog {PUBLIC_MODELS_FILE}: {e}")


@router.get("/models", summary="Public model list (aliases only)")
async def public_models():
    return {
        "object": "list",
        "data": [
            {"id": a.id, "object": "model", "created": 0, "owned_by": "ai-serve-ui", "kind": a.kind.value}
            for a in load_catalog().aliases
        ],
    }


@router.post("/chat/completions", summary="Public chat completions")
async def public_chat_completions(body: Dict[str, Any] = Body(...)):
    alias = load_catalog().resolve(body.get("model"), ModelKind.CHAT)
    payload = CHAT.sanitize(body, alias)
    try:
        req = ChatCompletionRequest(**payload)
    except ValidationError:
        raise PolicyViolation("Invalid chat request.")
    result = await openai_chat_completion(req)
    # Report the alias, never the GGUF filename/path behind it.
    if isinstance(result, BaseModel) and hasattr(result, "model"):
        return result.model_copy(update={"model": alias.id})
    if isinstance(result, dict) and "model" in result:
        return {**result, "model": alias.id}
    if isinstance(result, StreamingResponse):
        return StreamingResponse(_rewrite_sse_model(result.body_iterator, alias.id),
                                 media_type=result.media_type, status_code=result.status_code)
    return result


async def _rewrite_sse_model(chunks, alias: str):
    """Streamed chunks carry llama-server's `model` (the absolute GGUF path) --
    replace it with the alias, line by line, so no local path leaves the machine."""
    buf = ""
    async for chunk in chunks:
        buf += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        *lines, buf = buf.split("\n")
        for line in lines:
            yield _rewrite_sse_line(line, alias) + "\n"
    if buf:
        yield _rewrite_sse_line(buf, alias)


def _rewrite_sse_line(line: str, alias: str) -> str:
    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
        return line
    try:
        obj = json.loads(line[5:])
    except ValueError:
        return line
    if isinstance(obj, dict) and "model" in obj:
        obj["model"] = alias
        return "data: " + json.dumps(obj, ensure_ascii=False)
    return line


@router.post("/audio/transcriptions", summary="Public transcription")
async def public_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),        # accepted for OpenAI compatibility, unused
    temperature: Optional[float] = Form(None),  # accepted for OpenAI compatibility, unused
):
    alias = load_catalog().resolve(model, ModelKind.STT)
    content = await file.read()
    cmd = TRANSCRIPTION.sanitize(language=language or alias.defaults.get("language"),
                                 response_format=response_format, size_bytes=len(content))
    await file.seek(0)
    return await media.openai_audio_transcriptions(
        file=file, model=alias.target, language=cmd.language, response_format=cmd.response_format)


async def _transcode_audio(wav: bytes, response_format: str, speed: float) -> Response:
    out_args, media_type = _AUDIO_FORMATS[response_format]
    if response_format == "wav" and speed == 1.0:
        return Response(content=wav, media_type=media_type)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
    if speed != 1.0:
        cmd += ["-filter:a", f"atempo={speed}"]  # one stage covers the policy's 0.5-2.0
    cmd += out_args + ["pipe:1"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate(wav)
    if proc.returncode != 0 or not out:
        raise HTTPException(status_code=500, detail=f"ffmpeg -> {response_format} failed: "
                            f"{err.decode(errors='replace')[-300:]}")
    return Response(content=out, media_type=media_type)


@router.post("/audio/speech", summary="Public speech generation")
async def public_speech(body: Dict[str, Any] = Body(...)):
    alias = load_catalog().resolve(body.get("model"), ModelKind.TTS)
    cmd = SPEECH.sanitize(body, alias)
    speech_req = media.OpenAISpeechRequest(model=cmd.engine, input=cmd.text, voice=cmd.voice,
                                           language=cmd.language)
    wav_resp = await media.openai_audio_speech(speech_req)
    if not isinstance(wav_resp, Response):
        raise HTTPException(status_code=500, detail="TTS engine returned no audio.")
    return await _transcode_audio(wav_resp.body, cmd.response_format, cmd.speed)


@router.post("/images/generations", summary="Public image generation", response_model=media.ImageGenerationResponse)
async def public_image_generation(body: Dict[str, Any] = Body(...)):
    alias = load_catalog().resolve(body.get("model"), ModelKind.IMAGE)
    cmd = IMAGE.sanitize(body, alias)
    items = []
    for _ in range(cmd.n):  # sequential: one GPU; the image plugin of the alias's model renders it
        req = media.ImageGenerationRequest(prompt=cmd.prompt, model=cmd.model_name, aspect_ratio=cmd.aspect_ratio,
                                           steps=cmd.steps, seed=random.randint(1, 10_000_000), n=1)
        res = await media.openai_image_generation(req, None)
        items.append(media.ImageGenerationResponseDataItem(b64_json=res["data"][0]["b64_json"]))
    return media.ImageGenerationResponse(created=int(time.time()), data=items)
