"""More than one speech plugin (backend/tts_client.py): their engines are merged into one catalog (each tagged with
its plugin; a duplicate engine id stays with the plugin that sorts first), and every call about an engine goes to
the plugin that lists it. Found 2026-10-22 while building a Piper example plugin: the gateway only ever talked to the
first `tts` plugin, so a second one was ignored -- or, sorting first, hid tts-serve. Plugins faked at the HTTP level."""
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
WAV = b"RIFF\x00\x00\x00\x00WAVEfake"

PLUGINS = {   # id -> (port, engines, default)
    "ai-serve-piper": (8091, [{"id": "piper", "name": "Piper", "aliases": ["piper-tts"], "installed": True,
                               "status": "idle", "loaded": False, "default_reference": False,
                               "modes": [{"variant": "voice", "inputs": {"speaker": "optional"},
                                          "speakers": ["de_DE-thorsten-medium", "en_US-ljspeech-medium"]}]},
                              {"id": "chatterbox", "name": "duplicate, must lose", "installed": True}], "piper"),
    "tts-serve": (8050, [{"id": "chatterbox", "name": "Chatterbox", "aliases": ["chatter"], "installed": True,
                          "status": "unloaded", "loaded": False, "default_reference": False},
                         {"id": "cosyvoice", "name": "CosyVoice 3", "installed": True, "status": "idle",
                          "loaded": True, "pid": 11, "default_reference": False}], "chatterbox"),
}


def _manifest(pid, port):
    return ServiceManifest(id=pid, name=pid, description="d", project_dir=Path("/x"), command=("./run.sh",), env={},
                           port=port, bind="127.0.0.1", match=(pid,), identity="none", ui=False,
                           provides=parse_provides({"tts": {"catalog": "/v1/engines", "generate": "/v1/generate",
                                                            "release": "/plugin/release", "logs": "/plugin/logs",
                                                            "gpu": "none"}}))


class Fake:
    def __init__(self):
        self.calls = []
        self.down = set()

    async def ensure_running(self, p, timeout_s=30.0):
        return p.id not in self.down

    def __call__(self, request: httpx.Request) -> httpx.Response:
        pid = next(k for k, v in PLUGINS.items() if v[0] == request.url.port)
        if pid in self.down:
            raise httpx.ConnectError("down")
        body = json.loads(request.content) if request.content else None
        self.calls.append((pid, request.method, request.url.path, body))
        port, engines, default = PLUGINS[pid]
        if request.url.path == "/v1/engines":
            return httpx.Response(200, json={"engines": engines, "default_engine": default, "errors": []})
        if request.url.path == "/v1/generate":
            return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})
        if request.url.path == "/plugin/release":
            w = (body or {}).get("worker")
            return httpx.Response(200, json={"released": [w] if w else [e["id"] for e in engines]})
        if request.url.path == "/plugin/logs":
            return httpx.Response(200, json={"logs": f"{pid} log"})
        return httpx.Response(404, json={"detail": request.url.path})


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    from backend.routers import services as svc
    f = Fake()
    reg = ServiceRegistry([_manifest(pid, v[0]) for pid, v in PLUGINS.items()], [], {})
    monkeypatch.setattr(svc, "_registry", lambda: reg)
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(f))
    monkeypatch.setattr(registry, "ensure_running", f.ensure_running)
    monkeypatch.setattr("backend.routers.media.DEFAULT_TTS_ENGINE", "")
    return f


def _generate(**body):
    body.setdefault("text", "Hallo")
    return client.post("/api/tts/generate", json=body)


def test_engines_of_both_plugins_are_listed_and_a_duplicate_stays_with_the_first():
    import asyncio
    from backend import tts_client
    view = asyncio.run(tts_client.engines())
    by_id = {e["id"]: e for e in view["engines"]}
    assert set(by_id) == {"piper", "chatterbox", "cosyvoice"}
    assert by_id["piper"]["plugin"] == "ai-serve-piper" and by_id["cosyvoice"]["plugin"] == "tts-serve"
    assert by_id["chatterbox"]["plugin"] == "ai-serve-piper" and by_id["chatterbox"]["name"] == "duplicate, must lose"
    assert any("chatterbox" in e for e in view["errors"])          # the ignored duplicate is reported
    assert view["default_engine"] == "piper"                        # the first plugin's default


def test_generate_goes_to_the_plugin_that_lists_the_engine(fake):
    assert _generate(engine="cosyvoice").status_code == 200
    assert _generate(engine="piper-tts").status_code == 200        # by alias
    gens = [(pid, body["engine"]) for pid, m, path, body in fake.calls if path == "/v1/generate"]
    assert gens == [("tts-serve", "cosyvoice"), ("ai-serve-piper", "piper")]


def test_one_plugin_down_leaves_the_other_usable(fake):
    fake.down.add("tts-serve")
    r = _generate(engine="piper")
    assert r.status_code == 200, r.text
    r = _generate(engine="cosyvoice")
    assert r.status_code == 400 and "piper" in r.json()["detail"]   # unknown here, says what exists


def test_unload_reaches_the_owner_and_unload_idle_reaches_all(fake):
    import asyncio
    from backend import tts_client
    assert asyncio.run(tts_client.unload("cosyvoice"))["status"] == "unloaded"
    rel = [(pid, body) for pid, m, path, body in fake.calls if path == "/plugin/release"]
    assert rel == [("tts-serve", {"worker": "cosyvoice", "force": False})]
    fake.calls.clear()
    released = asyncio.run(tts_client.unload_idle())
    assert {pid for pid, m, path, body in fake.calls if path == "/plugin/release"} == {"ai-serve-piper", "tts-serve"}
    assert "cosyvoice" in released and "piper" in released


def test_audio_models_lists_every_plugins_engines():
    r = client.get("/v1/audio/models")
    assert r.status_code == 200
    text = json.dumps(r.json())
    assert '"piper"' in text and '"cosyvoice"' in text


def test_openai_voice_naming_a_speaker_of_the_engine_becomes_that_speaker(fake):
    """/v1/audio/speech: `voice` = one of the engine's own speakers (Piper's voices, Qwen3's presets) -> speaker,
    not a style instruction (Piper would refuse one)."""
    r = client.post("/v1/audio/speech", json={"model": "piper", "input": "Hallo", "voice": "de_DE-thorsten-medium",
                                              "language": "de"})
    assert r.status_code == 200, r.text
    body = [b for pid, m, path, b in fake.calls if path == "/v1/generate"][-1]
    assert body["speaker"] == "de_DE-thorsten-medium" and body.get("variant") == "voice" and "instruct" not in body


def test_the_default_prefers_an_installed_engine(monkeypatch):
    import asyncio
    from backend import tts_client
    piper = PLUGINS["ai-serve-piper"]
    monkeypatch.setitem(PLUGINS, "ai-serve-piper", (piper[0], [dict(piper[1][0], installed=False)], "piper"))
    assert asyncio.run(tts_client.engines())["default_engine"] == "chatterbox"
