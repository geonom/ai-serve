"""Cluster endpoints (backend/routers/cluster.py): create a one-time code, serve the
agent file, enroll, list and remove nodes. Rules: test_cluster_enrollment.py."""
import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import network
from backend.cluster.enrollment import EnrollmentBook, NodeRegistry
from backend.network import StaticProvider
from backend.routers import cluster

TOKEN = "n" * 40


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster, "REGISTRY", NodeRegistry(str(tmp_path / "nodes.json")))
    monkeypatch.setattr(cluster, "BOOK", EnrollmentBook())
    network.set_active(StaticProvider.from_dict({"networks": ["100.64.0.0/10"], "owner": "owner@github",
                                                 "peers": {}}))
    a = FastAPI()
    a.include_router(cluster.router)
    yield a
    network.set_active(None)


def c(app, ip):
    return TestClient(app, client=(ip, 40000))


def test_full_enrollment(app):
    code = c(app, "127.0.0.1").post("/api/cluster/enrollments").json()
    assert code["code"] and code["agent_sha256"] and "--enroll" in code["command"]
    body = {"code": code["code"], "node_name": "macbook", "agent_url": "http://100.64.7.11:8040", "token": TOKEN}
    r = c(app, "100.64.7.11").post("/api/cluster/enroll", json=body)
    assert r.status_code == 201 and r.json() == {"node_id": "macbook", "coordinator_login": "owner@github"}
    nodes = c(app, "127.0.0.1").get("/api/cluster/nodes").json()["nodes"]
    assert [n["id"] for n in nodes] == ["macbook"] and TOKEN not in json.dumps(nodes)
    # code is single-use
    assert c(app, "100.64.7.11").post("/api/cluster/enroll", json=body).status_code == 403
    assert c(app, "127.0.0.1").delete("/api/cluster/nodes/macbook").status_code == 200
    assert c(app, "127.0.0.1").delete("/api/cluster/nodes/macbook").status_code == 404


def test_bad_code_and_mismatched_address(app):
    code = c(app, "127.0.0.1").post("/api/cluster/enrollments").json()["code"]
    body = {"code": "wrong", "node_name": "m", "agent_url": "http://100.64.7.11:8040", "token": TOKEN}
    assert c(app, "100.64.7.11").post("/api/cluster/enroll", json=body).status_code == 403
    body["code"] = code
    body["agent_url"] = "http://100.90.1.2:8040"
    r = c(app, "100.64.7.11").post("/api/cluster/enroll", json=body)
    assert r.status_code == 400 and "address" in r.json()["detail"]


def test_agent_file_with_hash(app):
    r = c(app, "100.64.7.11").get("/api/cluster/agent.py")
    assert r.status_code == 200 and r.headers["x-agent-sha256"] == hashlib.sha256(r.content).hexdigest()
    assert b"llama-serve node agent" in r.content


def release():
    tag = "b11681"
    names = ["llama-b11681-bin-macos-arm64.tar.gz", "llama-b11681-bin-ubuntu-x64.tar.gz"]
    return {"tag_name": tag, "assets": [
        {"name": n, "size": 1000, "digest": "sha256:" + "a" * 64,
         "browser_download_url": f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{n}"}
        for n in names]}


def _enrolled(app):
    code = c(app, "127.0.0.1").post("/api/cluster/enrollments").json()["code"]
    c(app, "100.64.7.11").post("/api/cluster/enroll", json={
        "code": code, "node_name": "macbook", "agent_url": "http://100.64.7.11:8040", "token": TOKEN})


def test_install_runtime_picks_and_sends(app, monkeypatch):
    _enrolled(app)
    calls = []

    async def agent_call(node, method, path, body=None, timeout=10.0):
        calls.append((node.id, method, path, body))
        if path == "/v1/node":
            return {"facts": {"platform": "darwin", "machine": "arm64", "gpu_vendors": [],
                              "cuda_driver": None, "rocm": False, "vulkan": False}}
        return {"state": "installing"}

    async def fetch_release(tag=None):
        return release()

    monkeypatch.setattr(cluster, "agent_call", agent_call)
    monkeypatch.setattr(cluster, "fetch_release", fetch_release)
    r = c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/runtime", json={})
    assert r.status_code == 202, r.text
    assert r.json()["variant"] == "metal"
    sent = calls[-1]
    assert sent[1:3] == ("POST", "/v1/runtimes")
    assert sent[3]["files"][0]["url"].startswith("https://github.com/ggml-org/llama.cpp/releases/download/b11681/")
    assert len(sent[3]["files"][0]["sha256"]) == 64


def test_install_runtime_unknown_node(app):
    assert c(app, "127.0.0.1").post("/api/cluster/nodes/nope/runtime", json={}).status_code == 404


def test_install_runtime_bad_variant(app, monkeypatch):
    _enrolled(app)

    async def agent_call(node, method, path, body=None, timeout=10.0):
        return {"facts": {"platform": "linux", "machine": "x86_64", "gpu_vendors": [], "cuda_driver": None,
                          "rocm": False, "vulkan": False}}

    async def fetch_release(tag=None):
        return release()
    monkeypatch.setattr(cluster, "agent_call", agent_call)
    monkeypatch.setattr(cluster, "fetch_release", fetch_release)
    r = c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/runtime", json={"variant": "cuda-99"})
    assert r.status_code == 400


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "gguf"
    d.mkdir()
    (d / "m.gguf").write_bytes(b"GGUF" + b"x" * 1000)
    (d / "notes.txt").write_text("x")
    monkeypatch.setattr(cluster, "MODEL_DIRS", [str(d)])
    monkeypatch.setattr(cluster, "HASHES", cluster.HashCache(str(tmp_path / "hashes.json")))
    return d


def test_serve_model_file(app, models_dir):
    r = c(app, "100.64.7.11").get("/api/cluster/models/m.gguf", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.content == b"GGUF"
    for bad in ("notes.txt", "..%2Fm.gguf", "missing.gguf"):
        assert c(app, "100.64.7.11").get(f"/api/cluster/models/{bad}").status_code == 404


def test_send_local_model(app, models_dir, monkeypatch):
    _enrolled(app)
    sent = []

    async def agent_call(node, method, path, body=None, timeout=10.0):
        sent.append((path, body))
        return {"state": "downloading"}
    monkeypatch.setattr(cluster, "agent_call", agent_call)
    r = c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/models", json={"model": "m.gguf"})
    assert r.status_code == 202, r.text
    path, body = sent[-1]
    assert path == "/v1/models/download"
    assert body["sha256"] == hashlib.sha256((models_dir / "m.gguf").read_bytes()).hexdigest()
    assert body["sources"] == ["http://testserver/api/cluster/models/m.gguf"]


def test_send_hf_model(app, models_dir, monkeypatch):
    _enrolled(app)
    sent = []

    async def agent_call(node, method, path, body=None, timeout=10.0):
        sent.append(body)
        return {}

    async def hf_meta(ref):
        return (12345, "b" * 64)
    monkeypatch.setattr(cluster, "agent_call", agent_call)
    monkeypatch.setattr(cluster, "hf_meta", hf_meta)
    r = c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/models",
                                 json={"hf_repo": "unsloth/Qwen3-0.6B-GGUF", "hf_file": "Qwen3-0.6B-Q4_K_M.gguf"})
    assert r.status_code == 202, r.text
    assert sent[-1]["sha256"] == "b" * 64 and sent[-1]["sources"][-1].startswith("https://huggingface.co/")


def test_send_model_errors(app, models_dir):
    _enrolled(app)
    assert c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/models", json={"model": "../m.gguf"}).status_code == 400
    assert c(app, "127.0.0.1").post("/api/cluster/nodes/macbook/models", json={}).status_code == 400


def test_delete_download_is_explicit_passthrough(app, monkeypatch):
    _enrolled(app)
    calls = []

    async def agent_call(node, method, path, body=None, timeout=10.0):
        calls.append((method, path))
        return {"deleted": "x.gguf"}
    monkeypatch.setattr(cluster, "agent_call", agent_call)
    r = c(app, "127.0.0.1").delete("/api/cluster/nodes/macbook/downloads/x.gguf")
    assert r.status_code == 200 and calls == [("DELETE", "/v1/downloads/x.gguf")]
    assert c(app, "127.0.0.1").delete("/api/cluster/nodes/macbook/downloads/..%2Fx.gguf").status_code in (400, 404)


def test_set_budgets_passthrough(app, monkeypatch):
    _enrolled(app)
    calls = []

    async def agent_call(node, method, path, body=None, timeout=10.0):
        calls.append((method, path, body))
        return {"budget_mb": body}
    monkeypatch.setattr(cluster, "agent_call", agent_call)
    r = c(app, "127.0.0.1").put("/api/cluster/nodes/macbook/budgets", json={"0": 9000})
    assert r.status_code == 200 and calls == [("PUT", "/v1/budgets", {"0": 9000})]
    assert c(app, "127.0.0.1").put("/api/cluster/nodes/macbook/budgets", json=["x"]).status_code == 422
