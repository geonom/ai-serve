"""External model servers (backend/plugins/external.py): a standard OpenAI-compatible server the owner already runs
-- Ollama, LM Studio, llama-server, vLLM -- added by port only, without an adapter app (owner, 2026-10-21: the
Spielwiese and every extension should work with standard Ollama or LM Studio). Rules: loopback only (the gateway
never calls elsewhere), plain ids, no ports of this gateway, the OpenAI path ends in /v1; never started or stopped
from here (the owner runs them), no GPU of their own in the GPU manager's books."""
import json

import pytest

from backend.plugins import external
from backend.plugins.contracts import ContractError, normalize_catalog


def test_a_server_by_port():
    s = external.parse_server({"id": "ollama", "name": "Ollama", "port": 11434})
    assert (s.id, s.name, s.port, s.openai) == ("ollama", "Ollama", 11434, "/v1")
    assert s.provision.capability == "llm" and s.provision.gpu == "none"
    assert s.provision.endpoints == {"openai": "/v1"}


@pytest.mark.parametrize("raw", [
    {"id": "ollama", "port": 8030},                         # this gateway
    {"id": "ollama", "port": 80},                           # privileged
    {"id": "Ollama!", "port": 11434},                       # plain ids only
    {"id": "x", "port": 11434, "openai": "/api"},           # OpenAI path ends in /v1
    {"id": "x", "port": 11434, "openai": "http://evil/v1"}, # a path, never a URL
    {"id": "x", "port": 11434, "host": "192.168.1.5"},      # loopback only: no host field at all
    {"id": "x", "port": "11434"},                           # strict types
    {"id": "x"},
])
def test_refused(raw):
    with pytest.raises(ContractError):
        external.parse_server(raw)


def test_file_keeps_good_entries_and_reports_bad_ones():
    servers, errors = external.parse_file({"servers": [{"id": "ollama", "port": 11434},
                                                       {"id": "lmstudio", "port": 1234, "name": "LM Studio"},
                                                       {"id": "ollama", "port": 1235}, {"id": "x", "port": 8030}]})
    assert [s.id for s in servers] == ["ollama", "lmstudio"]
    assert len(errors) == 2                                  # duplicate id, gateway port


def test_known_servers_to_look_for():
    assert {(k.id, k.port) for k in external.KNOWN} >= {("ollama", 11434), ("lmstudio", 1234)}


@pytest.mark.parametrize("model_id", ["llama3.2:3b", "qwen/qwen3-8b", "nomic-embed-text:latest",
                                      "Qwen3-8B-GGUF", "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"])
def test_real_world_model_ids_are_accepted(model_id):
    """Ollama names models `name:tag`, LM Studio `publisher/model` -- they must be usable as ids."""
    entries, errors = normalize_catalog("llm", "ollama", {"data": [{"id": model_id, "object": "model"}]})
    assert not errors and entries[0]["id"] == model_id


@pytest.mark.parametrize("model_id", ["../x", "a b", "a\nb", "", "x" * 200, "/abs"])
def test_odd_model_ids_are_refused(model_id):
    entries, errors = normalize_catalog("llm", "ollama", {"data": [{"id": model_id}]})
    assert entries == [] and errors


def test_kind_from_the_name_when_the_server_says_nothing():
    entries, _ = normalize_catalog("llm", "ollama", {"data": [{"id": "nomic-embed-text:latest"}, {"id": "bge-reranker"},
                                                               {"id": "llama3.2:3b"}]})
    assert [e["kind"] for e in entries] == ["embedding", "rerank", "chat"]


# ------------------------------------------------------------------ wiring: registry + dashboard endpoints

import httpx
from fastapi.testclient import TestClient

from backend.plugins import llm as llm_plugins
from backend.plugins import registry

_REAL_ENSURE_RUNNING = registry.ensure_running     # captured at import, before conftest's "never start" patch


@pytest.fixture
def ollama(monkeypatch, tmp_path):
    """An Ollama on 11434 with two models; LM Studio's port answers too (to be offered), nothing else does."""
    from backend.routers import plugins as plugins_router
    from backend.routers import services as svc
    from backend.services.domain import ServiceRegistry
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, str(request.url)))
        assert request.url.host == "127.0.0.1"
        if request.url.port == 11434 and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "llama3.2:3b", "object": "model", "owned_by": "library"},
                {"id": "nomic-embed-text:latest", "object": "model"}]})
        if request.url.port == 11434 and request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Sofia"},
                                                          "finish_reason": "stop"}], "usage": {}})
        if request.url.port == 1234 and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen/qwen3-8b"}]})
        raise httpx.ConnectError("nothing there")

    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(registry, "ensure_running", _REAL_ENSURE_RUNNING)
    monkeypatch.setattr(external, "FILE", str(tmp_path / "model_servers.json"))
    monkeypatch.setattr(llm_plugins, "_cache", (0.0, [], []))
    from backend import model_manager
    from backend.routers import dashboard, gateway
    for mod in (model_manager, gateway, dashboard):      # never this machine's model folders
        monkeypatch.setattr(mod, "MODEL_DIRS", [str(tmp_path)])
    from backend.main import app
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.seen = seen
    c.file = tmp_path / "model_servers.json"
    return c


def test_nothing_added_but_running_servers_are_offered(ollama):
    r = ollama.get("/api/model-servers").json()
    assert r["servers"] == []
    assert {d["id"]: d["models"] for d in r["detected"]} == {"ollama": 2, "lmstudio": 1}


def test_added_server_serves_chat_through_the_gateway(ollama):
    r = ollama.put("/api/model-servers/ollama", json={"name": "Ollama", "port": 11434})
    assert r.status_code == 200, r.text
    assert json.loads(ollama.file.read_text())["servers"] == [{"id": "ollama", "name": "Ollama", "port": 11434, "openai": "/v1"}]
    listing = ollama.get("/api/model-servers").json()
    assert listing["servers"][0]["running"] and listing["servers"][0]["models"] == 2
    assert [d["id"] for d in listing["detected"]] == ["lmstudio"]
    rows = {m["id"]: m for m in ollama.get("/v1/llm/models").json()["models"]}
    assert rows["llama3.2:3b"]["where"][0]["plugin"] == "ollama" and rows["nomic-embed-text:latest"]["kind"] == "embedding"
    assert rows["llama3.2:3b"]["exclusive"] is False and rows["llama3.2:3b"]["where"][0]["state"] == "ready"
    c = ollama.post("/v1/chat/completions", json={"model": "llama3.2:3b", "messages": [{"role": "user", "content": "?"}]})
    assert c.status_code == 200 and c.json()["choices"][0]["message"]["content"] == "Sofia"
    assert ("POST", "http://127.0.0.1:11434/v1/chat/completions") in ollama.seen
    assert ollama.delete("/api/model-servers/ollama").status_code == 200
    assert json.loads(ollama.file.read_text())["servers"] == []


@pytest.mark.parametrize("body", [{"port": 8030}, {"port": 11434, "host": "10.0.0.1"}, {"port": "11434"},
                                  {"port": 11434, "openai": "http://x/v1"}])
def test_refused_entries(ollama, body):
    assert ollama.put("/api/model-servers/ollama", json=body).status_code in (400, 422)
    assert not ollama.file.exists()


def test_no_gpu_books_and_never_started(ollama):
    ollama.put("/api/model-servers/ollama", json={"port": 11434})
    import asyncio
    assert asyncio.run(registry.gpu_consumers()) == []


def test_a_llama_server_of_an_added_model_server_is_never_adopted(monkeypatch, tmp_path):
    """Found live: the owner's own llama-server on 1234, added as a model server, was adopted at a gateway restart and
    unloaded by the keep-alive timer 5 minutes later."""
    from backend import model_manager
    from backend.state import state

    class P:
        pid = 4242
    cmd = ["/opt/llama-server", "--port", "1234", "-m", "/m/qwen3-4b.gguf"]
    monkeypatch.setattr(model_manager, "processes", lambda attrs: [(P(), {"pid": 4242, "name": "llama-server", "cmdline": cmd})])
    f = tmp_path / "model_servers.json"
    f.write_text(json.dumps({"servers": [{"id": "lmstudio", "name": "LM Studio", "port": 1234}]}))
    monkeypatch.setattr(external, "FILE", str(f))
    monkeypatch.setattr(registry, "_manifests", lambda: [])
    saved = dict(state.managed_processes)
    state.managed_processes.clear()
    try:
        model_manager.scan_existing_llama_servers()
        assert 1234 not in state.managed_processes
        assert 1234 in model_manager._foreign_ports()
    finally:
        state.managed_processes.clear()
        state.managed_processes.update(saved)
