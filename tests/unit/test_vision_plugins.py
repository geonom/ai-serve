"""vision plugins (backend/plugins/vision.py, /v1/vision/*): an app says which tasks it wants -- faces, objects or
both -- and each goes to the model for it (one call when one model does both). The plugins are faked at the HTTP
level and reached only through the plugin registry."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import registry
from backend.plugins import vision
from backend.plugins.contracts import normalize_catalog, parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
FACES_PORT, OBJECTS_PORT = 8090, 8091
FACE = {"box": [10, 20, 30, 40], "landmarks": [[1, 2]] * 5, "score": 0.9, "embedding": [0.1] * 4}


class FakeVision:
    def __init__(self):
        self.calls = []

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        port, path = request.url.port, request.url.path
        if path == "/v1/vision/models":
            if port == FACES_PORT:
                return httpx.Response(200, json={"data": [
                    {"id": "auraface", "label": "AuraFace", "available": True, "tasks": ["faces"], "default": True,
                     "max_images": 16, "license": "Apache-2.0", "commercial_use": True,
                     "embedding": {"model": "auraface-v1", "dim": 512, "metric": "cosine",
                                   "thresholds": {"maybe": 0.3, "likely": 0.45, "bogus": 3}}},
                    {"id": "other-face", "label": "Other", "available": False, "tasks": ["faces"]}]})
            return httpx.Response(200, json={"data": [
                {"id": "owl", "label": "OWL", "available": True, "tasks": ["objects"], "max_images": 4,
                 "open_vocabulary": True},
                {"id": "both", "label": "Both", "available": True, "tasks": ["faces", "objects"]}]})
        if path == "/v1/vision/analyze":
            body = request.content
            self.calls.append((port, body))
            n = body.count(b'name="image"')
            tasks = body.split(b'name="tasks"\r\n\r\n')[1].split(b"\r\n")[0].decode().split(",")
            if b"boom" in body:
                return httpx.Response(507, json={"detail": "no room on any GPU"})
            row = {"width": 100, "height": 80}
            if "faces" in tasks:
                row["faces"] = [FACE]
            if "objects" in tasks:
                row["objects"] = [{"label": "ship", "box": [0, 0, 5, 5], "score": 0.5}]
            return httpx.Response(200, json={"images": [row] * n})
        if path == "/status":
            return httpx.Response(200, json={"busy": False, "workers": []})
        return httpx.Response(404, json={"detail": path})


def _manifest(pid, port):
    return ServiceManifest(id=pid, name=pid, description="d", project_dir=Path("/x"), command=("./run.sh",), env={},
                           port=port, bind="127.0.0.1", match=("x",), identity="none", ui=False,
                           provides=parse_provides({"vision": {"catalog": "/v1/vision/models",
                                                               "analyze": "/v1/vision/analyze", "status": "/status",
                                                               "release": "/release", "gpu": "shared"}}))


@pytest.fixture
def fake(monkeypatch):
    from backend.routers import services as svc
    f = FakeVision()
    ms = [_manifest("vision-serve", FACES_PORT), _manifest("zz-objects", OBJECTS_PORT)]
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry(ms, [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(f))
    monkeypatch.setattr(registry, "ensure_running", f.ensure_running)
    monkeypatch.setattr(vision, "_cache", (0.0, [], []))
    return f


def post(tasks=None, n=1, **data):
    files = [("image", (f"p{i}.jpg", b"\xff\xd8 jpeg", "image/jpeg")) for i in range(n)]
    return client.post("/v1/vision/analyze", files=files, data={**({"tasks": tasks} if tasks else {}), **data})


def test_catalog_lists_every_plugins_models_with_their_embedding(fake):
    rows = {r["id"]: r for r in client.get("/v1/vision/models").json()["data"]}
    assert set(rows) == {"auraface", "other-face", "owl", "both"}
    emb = rows["auraface"]["embedding"]
    assert emb == {"model": "auraface-v1", "dim": 512, "metric": "cosine", "thresholds": {"maybe": 0.3, "likely": 0.45}}
    assert rows["auraface"]["provider"] == "vision-serve" and rows["owl"]["open_vocabulary"] is True
    assert [r["id"] for r in client.get("/v1/vision/models?task=objects").json()["data"]] == ["owl", "both"]


def test_faces_by_default_go_to_the_default_face_model(fake):
    r = post()
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["images"][0]["faces"] == [FACE] and "objects" not in d["images"][0]
    assert d["models"]["faces"]["id"] == "auraface" and d["models"]["faces"]["embedding"]["dim"] == 512
    assert [c[0] for c in fake.calls] == [FACES_PORT]


def test_both_tasks_on_different_models_are_merged_per_image(fake):
    r = post("faces,objects", n=2)
    assert r.status_code == 200, r.text
    imgs = r.json()["images"]
    assert len(imgs) == 2 and all(i["faces"] and i["objects"] for i in imgs)
    assert sorted(c[0] for c in fake.calls) == [FACES_PORT, OBJECTS_PORT]
    assert r.json()["models"]["objects"]["id"] == "owl"


def test_one_model_doing_both_gets_one_call(fake):
    r = post("both", face_model="both", object_model="both")
    assert r.status_code == 200 and len(fake.calls) == 1
    assert b"faces,objects" in fake.calls[0][1]


@pytest.mark.parametrize("data,why", [
    ({"tasks": "nonsense"}, "tasks"),
    ({"face_model": "nope"}, "unknown"),
    ({"face_model": "owl"}, "doesn't do faces"),
    ({"face_model": "other-face"}, "not installed"),
])
def test_bad_requests_are_400(fake, data, why):
    r = post(**data)
    assert r.status_code == 400 and why in r.json()["detail"]
    assert fake.calls == []


def test_image_count_is_capped_by_the_model(fake):
    assert post("objects", n=5).status_code == 400          # owl takes 4
    assert post("objects", n=4).status_code == 200


def test_plugin_refusals_keep_their_status(fake):
    r = client.post("/v1/vision/analyze", files=[("image", ("x.jpg", b"boom", "image/jpeg"))])
    assert r.status_code == 507 and "room" in r.json()["detail"]


def test_without_a_vision_plugin_it_says_so(monkeypatch):
    from backend.routers import services as svc
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([], [], {}))
    monkeypatch.setattr(vision, "_cache", (0.0, [], []))
    r = post()
    assert r.status_code == 400 and "no installed model for faces" in r.json()["detail"]


def test_entry_shape():
    entries, errors = normalize_catalog("vision", "p", {"data": [
        {"id": "a", "tasks": ["faces"], "embedding": {"model": "m", "dim": 512}},
        {"id": "b", "tasks": ["poetry"]}]})
    assert [e["id"] for e in entries] == ["a"] and errors and entries[0]["max_images"] == 1
    assert entries[0]["embedding"]["metric"] == "cosine"
