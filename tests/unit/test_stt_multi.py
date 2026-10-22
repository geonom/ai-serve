"""More than one transcription plugin (backend/stt_client.py), like speech output (test_tts_multi.py): engines merged
into one catalog, a transcription goes to the plugin that lists its engine, a live session stays with the plugin that
started it, release/logs reach every plugin. Plugins faked at the HTTP level."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import registry
from backend.plugins.contracts import parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
B64 = "UklGRgAAAABXQVZF"

PLUGINS = {   # id -> (port, engines, default, has live)
    "parakeet-stt": (8062, [{"id": "parakeet", "label": "Parakeet", "available": True, "fields": []},
                            {"id": "whisperx", "label": "duplicate, must lose", "available": True}], "parakeet", False),
    "stt-serve": (8060, [{"id": "whisperx", "label": "WhisperX", "available": True, "fields": [
                      {"name": "diarize_backend", "choices": [{"value": "whisperx", "available": True, "label": "pyannote"}]}]},
                         {"id": "moss", "label": "MOSS", "available": False, "fields": []}], "whisperx", True),
}


def _manifest(pid, port, live):
    eps = {"catalog": "/v1/engines", "transcribe": "/v1/transcribe", "status": "/v1/status",
           "release": "/v1/release", "logs": "/v1/logs", "gpu": "shared"}
    if live:
        eps["live"] = "/v1/live"
    return ServiceManifest(id=pid, name=pid, description="d", project_dir=Path("/x"), command=("./run.sh",), env={},
                           port=port, bind="127.0.0.1", match=(pid,), identity="none", ui=False,
                           provides=parse_provides({"stt": eps}))


class Fake:
    def __init__(self):
        self.calls = []
        self.down = set()

    async def ensure_running(self, p, timeout_s=40.0):
        return p.id not in self.down

    def __call__(self, request: httpx.Request) -> httpx.Response:
        pid = next(k for k, v in PLUGINS.items() if v[0] == request.url.port)
        if pid in self.down:
            raise httpx.ConnectError("down")
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((pid, request.method, path, body, dict(request.url.params)))
        port, engines, default, live = PLUGINS[pid]
        if path == "/v1/engines":
            return httpx.Response(200, json={"engines": engines, "default_engine": default})
        if path == "/v1/transcribe":
            return httpx.Response(200, json={"text": f"from {pid}", "engine": body.get("engine")})
        if path == "/v1/live/start":
            return httpx.Response(200, json={"session_id": f"{pid}-s1"})
        if path.startswith("/v1/live/"):
            return httpx.Response(200, json={"ok": True, "plugin": pid})
        if path == "/v1/status":
            return httpx.Response(200, json={"busy": False, "workers": [{"id": f"{pid}-worker"}]})
        if path == "/v1/release":
            return httpx.Response(200, json={"released": [f"{pid}-worker"]})
        if path == "/v1/logs":
            return httpx.Response(200, json={"logs": f"{pid} log"})
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    from backend.routers import services as svc
    f = Fake()
    reg = ServiceRegistry([_manifest(pid, v[0], v[3]) for pid, v in PLUGINS.items()], [], {})
    monkeypatch.setattr(svc, "_registry", lambda: reg)
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(f))
    monkeypatch.setattr(registry, "ensure_running", f.ensure_running)
    from backend import stt_client
    monkeypatch.setattr(stt_client, "DEFAULT_STT_ENGINE", "")
    monkeypatch.setattr(stt_client, "_owner", {})
    monkeypatch.setattr(stt_client, "_sessions", {})
    return f


def _transcribe(**body):
    body.setdefault("audio_base64", B64)
    return client.post("/api/stt/transcribe", json=body)


def test_engines_are_merged_and_a_duplicate_stays_with_the_first():
    import asyncio
    from backend import stt_client
    view = asyncio.run(stt_client.engines())
    by_id = {e["id"]: e for e in view["engines"]}
    assert set(by_id) == {"parakeet", "whisperx", "moss"}
    assert by_id["whisperx"]["plugin"] == "parakeet-stt" and by_id["moss"]["plugin"] == "stt-serve"
    assert any("whisperx" in e for e in view["errors"])
    assert view["default_engine"] == "parakeet"


def test_a_transcription_goes_to_the_engines_plugin(fake):
    assert _transcribe(engine="moss").json()["text"] == "from stt-serve"
    assert _transcribe(engine="parakeet").json()["text"] == "from parakeet-stt"
    r = _transcribe()                                      # no engine: the default engine, named explicitly
    assert r.json() == {"text": "from parakeet-stt", "engine": "parakeet"}


def test_the_default_can_be_chosen(fake, monkeypatch):
    from backend import stt_client
    monkeypatch.setattr(stt_client, "DEFAULT_STT_ENGINE", "moss")
    assert _transcribe().json() == {"text": "from stt-serve", "engine": "moss"}


def test_a_live_session_stays_with_the_plugin_that_started_it(fake):
    sid = client.post("/api/stt/live/start", json={}).json()["session_id"]
    assert sid == "stt-serve-s1"                           # the only plugin with live sessions
    client.post("/api/stt/live/chunk", json={"session_id": sid, "audio_base64": B64})
    client.get("/api/stt/live/poll", params={"session_id": sid})
    client.post("/api/stt/live/stop", json={"session_id": sid})
    live = [(pid, path) for pid, m, path, b, q in fake.calls if path.startswith("/v1/live/")]
    assert {pid for pid, _ in live} == {"stt-serve"} and len(live) == 4


def test_release_and_logs_reach_every_plugin(fake):
    import asyncio
    from backend import stt_client
    out = asyncio.run(stt_client.unload_all())
    assert set(out["released"]) == {"parakeet-stt-worker", "stt-serve-worker"}
    text = asyncio.run(stt_client.logs())
    assert "parakeet-stt log" in text and "stt-serve log" in text


def test_one_plugin_down_leaves_the_other(fake):
    fake.down.add("parakeet-stt")
    r = _transcribe(engine="moss")
    assert r.status_code == 200 and r.json()["text"] == "from stt-serve"


def test_audio_models_lists_both():
    text = json.dumps(client.get("/v1/audio/models").json()["stt"])
    assert '"parakeet"' in text and '"moss"' in text
