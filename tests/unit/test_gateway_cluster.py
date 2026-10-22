"""Gateway -> device fallback (backend/routers/gateway.py::_resolve_upstream).

Local first, as before. A device is used only when this machine can't serve the
model: 507 (no VRAM anywhere / Strata owns the GPUs) or the file isn't here --
and only when devices are enrolled. An explicit gpu_id that doesn't fit (400)
stays a 400. If no device helps either, the caller sees both reasons."""
import asyncio

import pytest
from fastapi import HTTPException

from backend.cluster.service import Upstream
from backend.routers import gateway


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * (3 << 20))
    monkeypatch.setattr(gateway, "resolve_model_path", lambda name: str(f))
    return f


def devices(monkeypatch, on=True, result=None, error=None):
    calls = []

    async def upstream_for(name, local_path, ctx_size, keep_alive, local_size_mb=None, mode="chat", pooling=None):
        calls.append((name, local_path, local_size_mb))
        if error:
            raise error
        return result or Upstream(kind="node", base="http://100.1.1.1:8040/v1/instances/i/proxy", label="m @ mac")
    monkeypatch.setattr(gateway.CLUSTER, "has_nodes", lambda: on)
    monkeypatch.setattr(gateway.CLUSTER, "upstream_for", upstream_for)
    return calls


def failing_local(monkeypatch, status, detail="Insufficient VRAM on all GPUs"):
    async def load(**kw):
        raise HTTPException(status_code=status, detail=detail)
    monkeypatch.setattr(gateway, "get_or_load_model", load)


def resolve():
    return asyncio.run(gateway._resolve_upstream("m.gguf", ctx_size=8192, keep_alive=300))


def test_507_goes_to_device(local_model, monkeypatch):
    failing_local(monkeypatch, 507)
    calls = devices(monkeypatch)
    up = resolve()
    assert up.kind == "node" and calls == [("m.gguf", str(local_model), 3)]


def test_400_explicit_gpu_stays(local_model, monkeypatch):
    failing_local(monkeypatch, 400, "Not enough VRAM on GPU 1")
    calls = devices(monkeypatch)
    with pytest.raises(HTTPException) as e:
        resolve()
    assert e.value.status_code == 400 and calls == []


def test_no_devices_keeps_507(local_model, monkeypatch):
    failing_local(monkeypatch, 507)
    devices(monkeypatch, on=False)
    with pytest.raises(HTTPException) as e:
        resolve()
    assert e.value.status_code == 507


def test_missing_file_goes_to_device_without_local_load(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "resolve_model_path", lambda name: str(tmp_path / "nowhere.gguf"))

    async def load(**kw):
        raise AssertionError("must not try a local load for a file that isn't here")
    monkeypatch.setattr(gateway, "get_or_load_model", load)
    calls = devices(monkeypatch)
    assert resolve().kind == "node" and calls == [("nowhere.gguf", None, None)]


def test_both_reasons_when_nothing_helps(local_model, monkeypatch):
    failing_local(monkeypatch, 507, "Insufficient VRAM on all GPUs")
    devices(monkeypatch, error=HTTPException(status_code=507, detail="no device has room"))
    with pytest.raises(HTTPException) as e:
        resolve()
    assert "Insufficient VRAM" in e.value.detail and "no device has room" in e.value.detail


def test_copying_503_passes_through(local_model, monkeypatch):
    failing_local(monkeypatch, 507)
    devices(monkeypatch, error=HTTPException(status_code=503, detail="being copied", headers={"Retry-After": "120"}))
    with pytest.raises(HTTPException) as e:
        resolve()
    assert e.value.status_code == 503 and e.value.headers["Retry-After"] == "120"


def test_mode_derived_from_load_flags(local_model, monkeypatch):
    failing_local(monkeypatch, 507)
    seen = {}

    async def upstream_for(name, local_path, ctx_size, keep_alive, local_size_mb=None, mode="chat", pooling=None):
        seen.update(mode=mode, pooling=pooling)
        return Upstream(kind="node", base="http://x/v1/instances/i/proxy", label="m @ mac")
    monkeypatch.setattr(gateway.CLUSTER, "has_nodes", lambda: True)
    monkeypatch.setattr(gateway.CLUSTER, "upstream_for", upstream_for)
    asyncio.run(gateway._resolve_upstream("m.gguf", ctx_size=8192, keep_alive=300, embedding=True))
    assert seen == {"mode": "embedding", "pooling": None}
    asyncio.run(gateway._resolve_upstream("m.gguf", ctx_size=8192, keep_alive=300,
                                          embedding=True, reranking=True, pooling="rank"))
    assert seen == {"mode": "reranking", "pooling": "rank"}


def test_embeddings_endpoint_uses_device(local_model, monkeypatch):
    """/v1/embeddings goes through the same fallback and the device's proxy."""
    import httpx
    from fastapi.testclient import TestClient
    from backend.main import app
    failing_local(monkeypatch, 507)
    devices(monkeypatch, result=Upstream(kind="node", base="http://node.test/v1/instances/i/proxy",
                                         headers={"Authorization": "Bearer tok"}, label="m @ mac"))
    calls = []

    def handler(request):
        calls.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"object": "list", "model": "m", "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 1, "total_tokens": 1}})
    real = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler), **kw))
    r = TestClient(app, client=("127.0.0.1", 50000)).post("/v1/embeddings", json={"model": "m.gguf", "input": "hi"})
    assert r.status_code == 200, r.text
    assert calls == [("http://node.test/v1/instances/i/proxy/v1/embeddings", "Bearer tok")]
