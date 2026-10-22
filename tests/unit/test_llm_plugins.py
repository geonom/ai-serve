"""llm plugins (backend/plugins/llm.py): another OpenAI-compatible server -- e.g. Strata through its adapter -- whose
models the gateway lists and routes to, without knowing it by name. The plugin is faked at the HTTP level and reached
only through the plugin registry."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import llm as llm_plugins
from backend.plugins import registry
from backend.plugins.contracts import parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8075


class FakeModelServer:
    def __init__(self):
        self.calls, self.ready, self.loaded_by = [], False, None
        self.busy = False

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.url.host, request.url.port) == ("127.0.0.1", PORT)
        path, body = request.url.path, json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "big-moe", "label": "Big MoE", "kind": "chat", "context_length": 65536, "thinking": True,
                 "aliases": ["Big-MoE-IQ3_S-strata.gguf"]}]})
        if path == "/plugin/status":
            workers = [{"id": "server", "label": "Big MoE", "pid": 4321, "loaded": True, "busy": self.busy}] \
                if self.ready else []
            return httpx.Response(200, json={"busy": self.busy, "ready": self.ready, "workers": workers})
        if path == "/plugin/load":
            self.ready, self.loaded_by = True, body
            return httpx.Response(200, json={"status": "loaded"})
        if path == "/plugin/release":
            if self.busy and not body.get("force"):
                return httpx.Response(409, json={"detail": "busy"})
            self.ready = False
            return httpx.Response(200, json={"released": ["server"]})
        if path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Paris"},
                                                          "finish_reason": "stop"}], "usage": {}})
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def server(monkeypatch):
    from backend.routers import services as svc
    fake = FakeModelServer()
    m = ServiceManifest(id="bigmoe", name="Big MoE server", description="d", project_dir=Path("/x"),
                        command=("./run.sh",), env={}, port=PORT, bind="127.0.0.1", match=("bigmoe_adapter",),
                        identity="none", ui=False, provides=parse_provides({"llm": {
                            "openai": "/v1", "load": "/plugin/load", "status": "/plugin/status",
                            "release": "/plugin/release", "gpu": "exclusive"}}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(registry, "ensure_running", fake.ensure_running)
    monkeypatch.setattr(llm_plugins, "_cache", (0.0, [], []))
    return fake


def test_listed_with_the_plugin_and_its_state(server, monkeypatch, tmp_path):
    from backend import model_manager
    from backend.routers import dashboard, gateway
    for mod in (model_manager, gateway, dashboard):      # never this machine's model folders
        monkeypatch.setattr(mod, "MODEL_DIRS", [str(tmp_path)])
    rows = {r["id"]: r for r in client.get("/v1/llm/models").json()["models"]}
    r = rows["big-moe"]
    assert r["exclusive"] and r["context_length"] == 65536 and r["thinking"]
    assert r["where"] == [{"place": "plugin", "plugin": "bigmoe", "name": "Big MoE server", "state": "stopped",
                           "loaded": False}]


@pytest.mark.parametrize("name", ["big-moe", "BIG-MOE", "Big-MoE-IQ3_S-strata.gguf", "/m/Big-MoE-IQ3_S-strata.gguf"])
def test_chat_goes_to_the_plugin_by_id_or_alias(server, name):
    r = client.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "Hauptstadt?"}]})
    assert r.status_code == 200, r.text
    method, path, body = server.calls[-1]
    assert (method, path, body["model"]) == ("POST", "/v1/chat/completions", "big-moe")


def test_api_start_loads_and_answers_with_the_plugins_port(server):
    r = client.post("/api/start", json={"model_path": "Big-MoE-IQ3_S-strata.gguf", "port": 8123, "ctx_size": 32000})
    assert r.status_code == 200, r.text
    assert r.json()["port"] == PORT and r.json()["requested_port"] == 8123 and r.json()["status"] == "started"
    assert server.loaded_by == {"model": "big-moe"}
    assert client.post("/api/start", json={"model_path": "big-moe", "port": 8123, "ctx_size": 131072}).status_code == 400


def test_stop_by_port_releases_unless_busy(server):
    server.ready, server.busy = True, True
    assert client.post(f"/api/stop?port={PORT}").status_code == 409
    assert client.post(f"/api/stop?port={PORT}&force=true").status_code == 200
    assert server.ready is False


def test_status_lists_the_running_server_with_its_port(server):
    server.ready = True
    inst = [i for i in client.get("/api/status").json()["instances"] if i.get("plugin") == "bigmoe"]
    assert inst and inst[0]["port"] == PORT and inst[0]["unload_endpoint"] == "/api/plugins/bigmoe/release?worker=server"
    assert client.post("/api/plugins/bigmoe/release?worker=server").status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_exclusive_plugin_holds_the_gpus_only_while_its_server_runs(server, anyio_backend):
    assert [c for c in await registry.gpu_consumers()] == []            # adapter up, model server stopped
    server.ready = True
    [c] = await registry.gpu_consumers()
    assert (c.id, c.kind, c.busy, c.evictable_kind, c.root_pids) == ("bigmoe:server", "exclusive", True, False, {4321})


def test_unknown_names_stay_local(server):
    assert client.post("/v1/chat/completions", json={"model": "no-such-model.gguf",
                                                    "messages": [{"role": "user", "content": "x"}]}).status_code in (400, 404)
