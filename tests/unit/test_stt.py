"""The gateway's STT surface after WhisperX/diarization moved to stt-serve
(~/projects/stt-serve): /api/stt/* and /v1/audio/transcriptions forward
there. The gateway keeps voices/ -- it resolves audio_path (inside voices/ only),
sends audio as base64 (stt-serve accepts no paths) and writes the .txt sidecar.
Transcription logic itself is tested in stt-serve; stt-serve is faked here."""
import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import media
from backend.routers.media import STTTranscribeRequest

client = TestClient(app, client=("127.0.0.1", 50000))  # this machine (access policy)


STT = {"catalog": "/v1/engines", "transcribe": "/v1/transcribe", "live": "/v1/live", "status": "/v1/status",
       "release": "/v1/release", "logs": "/v1/logs", "gpu": "shared"}


class FakeSttServe:
    """The stt plugin at the HTTP level: the gateway reaches it only through the plugin registry."""
    def __init__(self):
        self.up = True
        self.calls = []
        self.error = None
        self.response = {"status": "success", "transcript": "SPEAKER_00: hallo welt",
                         "plain_transcript": "hallo welt", "segments": [{"start": 0, "end": 1, "text": "hallo welt"}]}

    async def ensure_running(self, p, timeout_s=40.0):
        return self.up

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("down")
        assert (request.url.host, request.url.port) == ("127.0.0.1", 8060)
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if path == "/v1/status":
            return httpx.Response(200, json={
                "whisperx": {"loaded": True, "status": "idle"}, "diarize": {"loaded": False, "status": "idle"},
                "busy": False, "workers": [
                    {"id": "stt-serve", "label": "WhisperX", "pid": 70, "loaded": True, "busy": False},
                    {"id": "nemotron", "label": "Nemotron", "pid": 77, "loaded": True, "busy": False, "gpu": 0}]})
        if path == "/v1/logs":
            return httpx.Response(200, json={"logs": "nemotron log line"})
        if path == "/v1/engines":
            return httpx.Response(200, json={"engines": [{"id": "whisperx", "fields": [{"name": "diarize_backend",
                "type": "choice", "choices": [{"value": "whisperx", "label": "pyannote", "available": True},
                                              {"value": "nemotron", "label": "Nemotron", "available": False}]}]},
                {"id": "moss", "fields": []}], "default_engine": "whisperx"})   # as stt-serve answers
        self.calls.append((request.method, path, body, dict(request.url.params) or None))
        if self.error:
            return httpx.Response(self.error[0], json={"detail": self.error[1]})
        if path == "/v1/transcribe":
            return httpx.Response(200, json=dict(self.response))
        if path == "/v1/release":
            return httpx.Response(200, json={"released": [body.get("worker")]})
        return httpx.Response(200, json={"ok": True, "path": path})


@pytest.fixture(autouse=True)
def stt(monkeypatch):
    from backend.plugins import registry
    from backend.plugins.contracts import parse_provides
    from backend.routers import services as svc
    from backend.services.domain import ServiceManifest, ServiceRegistry
    fake = FakeSttServe()
    m = ServiceManifest(id="stt-serve", name="stt-serve", description="d", project_dir=Path("/x"),
                        command=("./run.sh",), env={}, port=8060, bind="127.0.0.1", match=("stt_serve.app",),
                        identity="none", ui=False, provides=parse_provides({"stt": STT}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(registry, "ensure_running", fake.ensure_running)
    return fake


def _b64(data=b"RIFF0000WAVEfmt "):
    return base64.b64encode(data).decode()


def test_request_model_keeps_audio_path_for_callers():
    r = STTTranscribeRequest(audio_path="voices/x.wav", diarize=True)
    assert r.audio_path == "voices/x.wav" and r.diarize_backend == "whisperx"


def test_transcribe_forwards_without_paths(stt):
    r = client.post("/api/stt/transcribe", json={"audio_base64": _b64(), "language": "de", "diarize": True})
    assert r.status_code == 200 and r.json()["transcript"] == "SPEAKER_00: hallo welt"
    method, path, body, _ = stt.calls[0]
    assert (method, path) == ("POST", "/v1/transcribe")
    assert body["audio_base64"] == _b64() and body["language"] == "de" and body["diarize"] is True
    assert "audio_path" not in body


def test_audio_path_is_read_here_and_sidecar_written(stt, tmp_path, monkeypatch):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "clip.wav").write_bytes(b"RIFFclip")
    monkeypatch.setattr(media, "_VOICES_DIR", str(voices))
    r = client.post("/api/stt/transcribe", json={"audio_path": str(voices / "clip.wav")})
    assert r.status_code == 200
    body = stt.calls[0][2]
    assert base64.b64decode(body["audio_base64"]) == b"RIFFclip" and "audio_path" not in body
    assert (voices / "clip.txt").read_text() == "hallo welt"  # plain -- never speaker-labelled


@pytest.mark.parametrize("path", ["/etc/passwd", "voices/../backend/config.py"])
def test_audio_path_outside_voices_rejected(stt, path):
    r = client.post("/api/stt/transcribe", json={"audio_path": path})
    assert r.status_code == 400 and stt.calls == []


def test_missing_audio_is_400(stt):
    assert client.post("/api/stt/transcribe", json={}).status_code == 400


def test_stt_serve_down_is_503(stt):
    stt.up = False
    r = client.post("/api/stt/transcribe", json={"audio_base64": _b64()})
    assert r.status_code == 503


def test_errors_pass_through(stt):
    stt.error = (507, "Strata belegt die GPUs")
    assert client.post("/api/stt/transcribe", json={"audio_base64": _b64()}).status_code == 507


def test_openai_transcriptions_forward_language(stt):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF")},
                    data={"language": "de", "response_format": "text"})
    assert r.status_code == 200 and r.text == "SPEAKER_00: hallo welt"
    assert stt.calls[0][2]["language"] == "de"


@pytest.mark.parametrize("method, url, body, target", [
    ("post", "/api/stt/live/start", {"language": "de"}, "/v1/live/start"),
    ("post", "/api/stt/live/chunk", {"session_id": "s", "audio_base64": "eA=="}, "/v1/live/chunk"),
    ("post", "/api/stt/live/stop", {"session_id": "s"}, "/v1/live/stop"),
    ("patch", "/api/stt/live/settings", {"session_id": "s", "silence_ms": 300}, "/v1/live/settings"),
    ("post", "/api/stt/unload", {"worker": "stt-serve"}, "/v1/release"),
    ("post", "/api/stt/diarize/unload", {"worker": "stt-serve"}, "/v1/release"),
    ("post", "/api/stt/diarize/nemotron/unload", {"worker": "nemotron"}, "/v1/release"),
])
def test_endpoints_proxy(stt, method, url, body, target):
    sent = None if target == "/v1/release" else body
    r = getattr(client, method)(url, json=sent) if sent is not None else getattr(client, method)(url)
    assert r.status_code == 200, r.text
    forwarded = stt.calls[-1][2]
    assert stt.calls[-1][1] == target and body.items() <= forwarded.items()


def test_live_poll_and_sessions_proxy(stt):
    assert client.get("/api/stt/live/poll", params={"session_id": "s", "since": 2}).status_code == 200
    assert stt.calls[-1][1:] == ("/v1/live/poll", None, {"session_id": "s", "since": "2"})
    assert client.get("/api/stt/live/sessions").status_code == 200
    r = client.get("/api/stt/diarize/backends")       # derived from the catalog + status
    assert r.json()["backends"] == {"whisperx": {"available": True, "loaded": True, "description": "pyannote"},
                                    "nemotron": {"available": False, "loaded": True, "description": "Nemotron"}}


def test_unknown_fields_and_live_validation_stay_strict(stt):
    assert client.post("/api/stt/live/chunk", json={"audio_base64": "eA=="}).status_code == 422


def test_dashboard_lists_loaded_stt_models(stt):
    inst = client.get("/api/status").json()["instances"]
    eps = {i.get("unload_endpoint"): i for i in inst}
    assert "/api/plugins/stt-serve/release?worker=stt-serve" in eps
    assert eps["/api/plugins/stt-serve/release?worker=nemotron"]["gpu_id"] == 0
    sysstat = client.get("/api/system/status").json()
    assert sysstat["whisperx"]["status"] == "idle" and sysstat["diarize"]["loaded"] is False


def test_logs_include_stt_serve(stt):
    assert "nemotron log line" in client.get("/api/logs/whisperx").json()["logs"]


def test_moss_engine_and_hotwords_are_forwarded(stt):
    r = client.post("/api/stt/transcribe", json={"audio_base64": _b64(), "engine": "moss", "hotwords": ["Liruna"],
                                                  "diarize": True})
    assert r.status_code == 200
    body = stt.calls[0][2]
    assert body["engine"] == "moss" and body["hotwords"] == ["Liruna"]


def test_engine_names_are_stt_serves_business(stt):
    # no fixed list here any more (2026-10-20): a new engine in stt-serve works without a gateway change;
    # stt-serve refuses names it doesn't know. Only the shape is checked here.
    r = client.post("/api/stt/transcribe", json={"audio_base64": _b64(), "engine": "whisper-large"})
    assert r.status_code == 200 and stt.calls[0][2]["engine"] == "whisper-large"
    r = client.post("/api/stt/transcribe", json={"audio_base64": _b64(), "engine": "a b; rm"})
    assert r.status_code == 422 and len(stt.calls) == 1


@pytest.mark.parametrize("model, engine", [("whisper-1", "whisperx"), ("moss-transcribe", "moss"),
                                           ("MOSS-Transcribe-Diarize", "moss")])
def test_openai_transcriptions_pick_engine_by_model(stt, model, engine):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"RIFF")}, data={"model": model})
    assert r.status_code == 200
    transcribed = [c for c in stt.calls if c[1] == "/v1/transcribe"]
    assert transcribed[0][2]["engine"] == engine
