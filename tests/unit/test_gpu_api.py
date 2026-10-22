"""GPU manager endpoints (backend/routers/gpu.py): the owner sees who holds VRAM,
pins a consumer (never unloaded automatically) and unloads one by hand. Unloading
refuses busy consumers (409) and unknown ones (404); it never deletes files."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.cluster import local
from backend.cluster.consumers import Consumer
from backend.routers import gpu


@pytest.fixture
def client(tmp_path, monkeypatch):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    cons = [Consumer(id="comfyui", kind="image", label="ComfyUI", root_pids={1}),
            Consumer(id="tts:cosy", kind="tts", label="CosyVoice", root_pids={2}, busy=True)]

    async def view():
        return [{"index": 0, "consumers": [{"id": c.id} for c in cons]}]

    async def consumers():
        return cons
    unloaded = []

    async def evict(cid, consumers=None):
        unloaded.append(cid)
    monkeypatch.setattr(m, "view", view)
    monkeypatch.setattr(m, "consumers", consumers)
    monkeypatch.setattr(m, "_do_evict", evict, raising=False)
    monkeypatch.setattr(gpu, "MANAGER", m)
    app = FastAPI()
    app.include_router(gpu.router)
    c = TestClient(app)
    c.unloaded = unloaded
    c.manager = m
    return c


def test_view(client):
    assert client.get("/api/gpu/consumers").json()["gpus"][0]["consumers"][0]["id"] == "comfyui"


def test_pin_roundtrip(client):
    assert client.post("/api/gpu/consumers/comfyui/pin", json={"pinned": True}).json()["pinned"] == ["comfyui"]
    assert client.manager.pins() == frozenset({"comfyui"})
    assert client.post("/api/gpu/consumers/comfyui/pin", json={"pinned": "yes"}).status_code == 422
    assert client.post("/api/gpu/consumers/comfyui/pin", json={"pinned": False}).json()["pinned"] == []


def test_unload(client, monkeypatch):
    async def evict(cid, consumers=None):
        if cid == "tts:cosy":
            raise PermissionError("CosyVoice is busy -- not unloading it")
        if cid == "nope":
            raise LookupError("nope is not using a GPU")
        client.unloaded.append(cid)
    monkeypatch.setattr(client.manager, "evict", evict)
    assert client.post("/api/gpu/consumers/comfyui/unload").status_code == 200 and client.unloaded == ["comfyui"]
    assert client.post("/api/gpu/consumers/tts:cosy/unload").status_code == 409
    assert client.post("/api/gpu/consumers/nope/unload").status_code == 404
