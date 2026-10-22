"""The plugin registry (backend/plugins/registry.py): which apps provide what, calling them only at their declared
endpoints, starting them on demand, and what the GPU manager learns from them (workers, release)."""
import json
from pathlib import Path

import httpx
import pytest

from backend.plugins import registry
from backend.plugins.contracts import parse_provides

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeManifest:
    def __init__(self, sid, port, provides, match=("x",)):
        self.id, self.port, self.name, self.match = sid, port, sid.title(), match
        self.provides = parse_provides(provides) if provides else {}
        self.health_path = None


TTS = {"catalog": "/v1/engines", "generate": "/v1/generate", "status": "/v1/status", "release": "/v1/release",
       "logs": "/v1/logs", "gpu": "shared"}


@pytest.fixture
def plugins(monkeypatch):
    seen = []
    up = {"tts-b": True, "tts-a": True, "door": True}

    def handler(request: httpx.Request):
        seen.append((request.method, str(request.url)))
        port = request.url.port
        name = {9001: "tts-a", 9002: "tts-b", 9003: "door"}[port]
        if not up[name]:
            raise httpx.ConnectError("down")
        if request.url.path == "/v1/status":
            return httpx.Response(200, json={"busy": False, "workers": [
                {"id": "cosyvoice", "label": "CosyVoice 3", "pid": 4242, "busy": False, "loaded": True, "last_used": 100.0},
                {"id": "omnivoice", "label": "OmniVoice", "pid": 4343, "busy": True, "loaded": True}]})
        if request.url.path == "/v1/release":
            body = json.loads(request.content or b"{}")
            if body.get("worker") == "omnivoice":
                return httpx.Response(409, json={"detail": "busy"})
            return httpx.Response(200, json={"released": [body.get("worker")]})
        if request.url.path == "/status":
            return httpx.Response(200, json={"facts": ["Anmeldung: läuft"], "domain": "apps.example.org"})
        return httpx.Response(200, json={"ok": True})

    manifests = [FakeManifest("tts-b", 9002, {"tts": TTS}), FakeManifest("tts-a", 9001, {"tts": TTS}),
                 FakeManifest("door", 9003, {"access": {"shares": "/api/routes", "status": "/status"}}),
                 FakeManifest("plain-app", 9004, {})]
    monkeypatch.setattr(registry, "_manifests", lambda: [m for m in manifests])
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(registry, "_root_pid", lambda m: 777)
    return seen, up


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
def test_providers_in_stable_order(plugins, anyio_backend):
    assert [p.id for p in registry.providers("tts")] == ["tts-a", "tts-b"]
    assert registry.provider("tts").id == "tts-a"
    assert registry.provider("image") is None
    assert registry.provider("access").id == "door"


async def test_calls_only_declared_endpoints_on_loopback(plugins):
    seen, _ = plugins
    p = registry.provider("tts")
    r = await registry.call(p, "generate", "POST", json={"text": "x"})
    assert r.status_code == 200 and seen[-1] == ("POST", "http://127.0.0.1:9001/v1/generate")
    with pytest.raises(registry.ContractError):
        await registry.call(p, "transcribe", "POST")


async def test_status_and_access_domain(plugins):
    st = await registry.status(registry.provider("access"))
    assert st["domain"] == "apps.example.org" and st["facts"][0]["value"] == "Anmeldung: läuft"


async def test_a_plugin_that_is_down_has_no_status(plugins):
    _, up = plugins
    up["tts-a"] = False
    assert await registry.status(registry.provider("tts")) is None
    assert await registry.running(registry.provider("tts")) is False


async def test_gpu_consumers_come_from_the_plugins_workers(plugins):
    consumers = await registry.gpu_consumers()
    by = {c.id: c for c in consumers}
    assert set(by) == {"tts-a:cosyvoice", "tts-a:omnivoice", "tts-b:cosyvoice", "tts-b:omnivoice"}   # door: no GPU
    c = by["tts-a:cosyvoice"]
    assert c.kind == "tts" and c.plugin == "tts-a" and c.root_pids == {4242} and c.label == "CosyVoice 3" and c.last_used == 100.0
    assert by["tts-a:omnivoice"].busy is True


async def test_release_a_worker(plugins):
    seen, _ = plugins
    await registry.release("tts-b:cosyvoice")
    assert seen[-1] == ("POST", "http://127.0.0.1:9002/v1/release")
    with pytest.raises(PermissionError):
        await registry.release("tts-a:omnivoice")          # 409 from the plugin: busy
    with pytest.raises(KeyError):
        await registry.release("nobody:x")
