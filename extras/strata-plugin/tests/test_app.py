"""The adapter's HTTP surface without a real Strata: catalog from the config, status/release while stopped, the GPU
owner's refusal passed on, and nothing forwarded to a Strata that isn't running."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from strata_plugin import app as plugin


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = tmp_path / "strata.json"
    cfg.write_text(json.dumps({"model_name": "qwen-x", "args": ["--max-context", "65536"]}))
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setattr(plugin, "STRATA_DIR", str(tmp_path))
    monkeypatch.setattr(plugin, "STRATA_CONFIG", str(cfg))
    monkeypatch.setattr(plugin, "STRATA", "http://127.0.0.1:9")            # nothing listens there
    monkeypatch.setattr(plugin, "ALIASES", [])                             # not this machine's .env
    monkeypatch.setattr(plugin.Server, "find_pid", staticmethod(lambda: None))
    monkeypatch.setattr(plugin, "SERVER", plugin.Server())
    return TestClient(plugin.app)


def test_catalog_without_starting_strata(client):
    [m] = client.get("/v1/models").json()["data"]
    assert m["id"] == "strata" and m["context_length"] == 65536 and m["aliases"] == ["qwen-x"]


def test_stopped(client):
    st = client.get("/plugin/status").json()
    assert (st["ready"], st["busy"], st["workers"], st["state"]) == (False, False, [], "stopped")
    assert client.post("/plugin/release", json={}).json() == {"released": []}
    assert client.get("/props").status_code == 503


def test_gpu_owner_says_no(client, monkeypatch):
    real = httpx.AsyncClient

    def refusing(*a, **kw):
        return real(transport=httpx.MockTransport(lambda r: httpx.Response(
            200, json={"granted": False, "reason": "GPU 0: CosyVoice is busy"})), timeout=5)
    monkeypatch.setattr(plugin.httpx, "AsyncClient", refusing)
    r = client.post("/plugin/load", json={"model": "strata"})
    assert r.status_code == 409 and "CosyVoice is busy" in r.json()["detail"]
    assert client.post("/plugin/load", json={"model": "other"}).status_code == 404
