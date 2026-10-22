"""/public/v1 over HTTP: the domain rules (test_public_domain.py) wired into FastAPI --
exposure guard middleware, key gate, error mapping, admission control, delegation."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient

from backend.public_api.domain import AdmissionControl, PublicApiKey
from backend.public_api.middleware import PublicExposureMiddleware
from backend.routers import media, public_gateway as pg

KEY = "k" * 43
FUNNEL = {"Tailscale-Funnel-Request": "?1"}
MODELS = {
    "models": [
        {"id": "chat", "kind": "chat", "target": "small.gguf", "defaults": {"thinking": False, "ctx_size": 8192}},
        {"id": "chat-large", "kind": "chat", "target": "large.gguf"},
        {"id": "whisper-1", "kind": "stt"},
        {"id": "tts", "kind": "tts", "target": "cosyvoice", "voices": ["narrator"]},
    ]
}


def _make_app():
    app = FastAPI()
    app.add_middleware(PublicExposureMiddleware)

    @app.get("/api/status")
    async def internal_status():
        return {"internal": True}

    app.include_router(pg.router)
    return app


@pytest.fixture
def app(tmp_path, monkeypatch):
    cfg = tmp_path / "public_models.json"
    cfg.write_text(json.dumps(MODELS))
    monkeypatch.setattr(pg, "PUBLIC_MODELS_FILE", str(cfg))
    monkeypatch.setattr(pg, "API_KEY", PublicApiKey.from_secret(KEY))
    monkeypatch.setattr(pg, "ADMISSION", AdmissionControl(max_concurrent=4))
    return _make_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def _auth(extra=None):
    return {"Authorization": f"Bearer {KEY}", **(extra or {})}


def _fake_chat(monkeypatch, result=None, seen=None):
    async def fake(req):
        if seen is not None:
            seen["req"] = req
        return result if result is not None else {"model": req.model, "choices": []}
    monkeypatch.setattr(pg, "openai_chat_completion", fake)


# --------------------------------------------------------------------------- exposure

def test_tunnel_request_to_internal_route_is_404(client):
    r = client.get("/api/status", headers=FUNNEL)
    assert r.status_code == 404 and "internal" not in r.text


def test_tunnel_request_with_traversal_is_404(client):
    assert client.get("/public/../api/status", headers=FUNNEL).status_code == 404


def test_local_request_to_internal_route_still_works(client):
    assert client.get("/api/status").json() == {"internal": True}


def test_tunnel_request_to_public_route_works(client):
    assert client.get("/public/v1/models", headers=_auth(FUNNEL)).status_code == 200


def test_oversized_body_is_413_before_handler(client, monkeypatch):
    called = {}
    _fake_chat(monkeypatch, seen=called)
    big = "x" * (pg.BODY_LIMITS.for_path("/public/v1/chat/completions") + 10)
    r = client.post("/public/v1/chat/completions", headers=_auth({"Content-Type": "application/json"}),
                    content=json.dumps({"messages": [{"role": "user", "content": big}]}))
    assert r.status_code == 413 and "req" not in called


# --------------------------------------------------------------------------- key

def test_unset_key_disables_everything(client, monkeypatch):
    monkeypatch.setattr(pg, "API_KEY", None)
    r = client.get("/public/v1/models", headers=_auth())
    assert r.status_code == 503 and r.json()["error"]["message"]


def test_wrong_or_missing_key_401(client):
    assert client.get("/public/v1/models").status_code == 401
    assert client.get("/public/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_key_checked_before_body_validation(client):
    r = client.post("/public/v1/chat/completions", json={"nonsense": True})
    assert r.status_code == 401


# --------------------------------------------------------------------------- behavior

def test_models_lists_aliases_only(client):
    data = client.get("/public/v1/models", headers=_auth()).json()["data"]
    assert [m["id"] for m in data] == ["chat", "chat-large", "whisper-1", "tts"]


def test_unknown_model_400_openai_error_shape(client):
    r = client.post("/public/v1/chat/completions", headers=_auth(),
                    json={"model": "bge-m3-Q8_0.gguf", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400 and r.json()["error"]["type"] == "model_not_found"
    assert "chat, chat-large" in r.json()["error"]["message"]


def test_chat_policy_applied(client, monkeypatch):
    seen = {}
    _fake_chat(monkeypatch, seen=seen)
    r = client.post("/public/v1/chat/completions", headers=_auth(), json={
        "messages": [{"role": "user", "content": "x"}],
        "gpu_id": 0, "ctx_size": 131072, "keep_alive": 99999, "modalities": ["audio"], "voice": "x",
    })
    assert r.status_code == 200 and r.json()["model"] == "chat"
    req = seen["req"]
    assert req.model == "small.gguf" and req.gpu_id is None and req.keep_alive == 300
    assert req.modalities is None and req.voice is None and req.ctx_size == 8192 and req.thinking is False


def test_chat_ssrf_url_rejected_400(client, monkeypatch):
    seen = {}
    _fake_chat(monkeypatch, seen=seen)
    msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:8030/api/x"}}]}]
    r = client.post("/public/v1/chat/completions", headers=_auth(), json={"messages": msg})
    assert r.status_code == 400 and "req" not in seen


def test_inner_error_details_hidden(client, monkeypatch):
    async def boom(req):
        raise HTTPException(status_code=400, detail="Model file not found: /media/owner/Extreme SSD/x.gguf")
    monkeypatch.setattr(pg, "openai_chat_completion", boom)
    r = client.post("/public/v1/chat/completions", headers=_auth(), json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert "/media" not in r.text and "alex" not in r.text and r.json()["error"]["reference"]


def test_unexpected_exception_hidden(client, monkeypatch):
    async def boom(req):
        raise RuntimeError("secret /data path")
    monkeypatch.setattr(pg, "openai_chat_completion", boom)
    r = TestClient(client.app, raise_server_exceptions=False).post(
        "/public/v1/chat/completions", headers=_auth(), json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 500 and "/mnt" not in r.text


def test_stream_model_field_rewritten(client, monkeypatch):
    async def gen():
        yield 'data: {"model": "/mnt/x/small.gguf", "choices": []}\n\n'
        yield "data: [DONE]\n\n"
    _fake_chat(monkeypatch, result=StreamingResponse(gen(), media_type="text/event-stream"))
    r = client.post("/public/v1/chat/completions", headers=_auth(),
                    json={"model": "chat", "stream": True, "messages": [{"role": "user", "content": "x"}]})
    assert "gguf" not in r.text and '"model": "chat"' in r.text and "data: [DONE]" in r.text


def test_tts_voice_outside_allowlist_falls_back(client, monkeypatch):
    voices = []

    async def fake_speech(req):
        voices.append(req.voice)
        return Response(content=b"RIFF", media_type="audio/wav")

    async def fake_transcode(wav, fmt, speed):
        return Response(content=wav, media_type="audio/wav")

    monkeypatch.setattr(media, "openai_audio_speech", fake_speech)
    monkeypatch.setattr(pg, "_transcode_audio", fake_transcode)
    for voice in ("some-actor-clip", "narrator"):
        assert client.post("/public/v1/audio/speech", headers=_auth(),
                           json={"input": "Hallo", "voice": voice}).status_code == 200
    assert voices == ["alloy", "narrator"]


def test_transcription_bad_language_rejected(client):
    r = client.post("/public/v1/audio/transcriptions", headers=_auth(),
                    files={"file": ("a.wav", b"RIFF0000")}, data={"language": "de; rm -rf /"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- admission

def test_concurrency_limit_429_and_release_after_stream(app, monkeypatch):
    monkeypatch.setattr(pg, "ADMISSION", AdmissionControl(max_concurrent=1))
    gate = asyncio.Event()

    async def slow_gen():
        await gate.wait()
        yield "data: [DONE]\n\n"

    async def fake(req):
        return StreamingResponse(slow_gen(), media_type="text/event-stream")

    monkeypatch.setattr(pg, "openai_chat_completion", fake)
    body = {"stream": True, "messages": [{"role": "user", "content": "x"}]}

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            first = asyncio.create_task(c.post("/public/v1/chat/completions", headers=_auth(), json=body))
            await asyncio.sleep(0.05)
            second = await c.post("/public/v1/chat/completions", headers=_auth(), json=body)
            gate.set()
            first_resp = await first
            third = await c.get("/public/v1/models", headers=_auth())
            return first_resp, second, third

    first, second, third = asyncio.run(scenario())
    assert second.status_code == 429
    assert first.status_code == 200 and third.status_code == 200
    assert pg.ADMISSION.in_flight == 0


# --------------------------------------------------------------------------- transcription formats (media helper)

def test_transcription_formats():
    result = {"transcript": "Hallo Welt.", "segments": [{"start": 0.0, "end": 1.5, "text": " Hallo Welt."}],
              "duration_s": 1.5}
    assert media.openai_transcription_response(result, "json", None) == {"text": "Hallo Welt."}
    vj = media.openai_transcription_response(result, "verbose_json", "de")
    assert vj["language"] == "de" and vj["duration"] == 1.5 and vj["segments"][0]["end"] == 1.5
    srt = media.openai_transcription_response(result, "srt", None).body.decode()
    assert srt == "1\n00:00:00,000 --> 00:00:01,500\nHallo Welt.\n"
    vtt = media.openai_transcription_response(result, "vtt", None).body.decode()
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500")


def test_chunked_oversized_body_is_413(client, monkeypatch):
    seen = {}
    _fake_chat(monkeypatch, seen=seen)
    limit = pg.BODY_LIMITS.for_path("/public/v1/chat/completions")

    def chunks():  # generator body -> no Content-Length, Transfer-Encoding: chunked
        yield b'{"messages": [{"role": "user", "content": "'
        for _ in range(limit // (1024 * 1024) + 2):
            yield b"x" * (1024 * 1024)
        yield b'"}]}'

    r = client.post("/public/v1/chat/completions", headers=_auth({"Content-Type": "application/json"}),
                    content=chunks())
    assert r.status_code == 413 and "req" not in seen


def test_real_app_has_exposure_guard():
    """The production app (backend.main), not just this file's test app."""
    from backend.main import app as real_app
    c = TestClient(real_app)
    for path in ("/api/status", "/v1/models", "/", "/docs", "/openapi.json", "/api/training/jobs", "/api/services"):
        assert c.get(path, headers=FUNNEL).status_code == 404, path
    assert c.get("/public/v1/models", headers=FUNNEL).status_code in (401, 503)


@pytest.mark.parametrize("path, body", [
    ("/public/v1/images/generations", {"prompt": "a", "model": "invalid-model"}),
    ("/public/v1/audio/speech", {"input": "a", "voice": "invalid-voice", "model": "invalid-model",
                                 "response_format": "mp3"}),
])
def test_nextcloud_capability_probes_see_an_existing_endpoint(client, path, body):
    """integration_openai's isT2IAvailable()/isTTSAvailable() send exactly these and
    count anything but 404/401 as "available"."""
    r = client.post(path, headers=_auth(), json=body)
    assert r.status_code not in (401, 404)
