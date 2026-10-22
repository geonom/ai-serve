"""/api/comfyui/* goes unchanged to the image plugin that declares `comfyui_api` (image-serve) -- apps from before
image plugins (Munin, dynasty, language_learning, train-serve, Liruna, the Spielwiese) keep working."""
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


@pytest.fixture
def plugin(monkeypatch):
    from backend.routers import services as svc
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, request.url.port, request.url.raw_path.decode(), request.content))
        if request.url.path.endswith("/images/my picture.png"):
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})
        return httpx.Response(202, json={"status": "queued", "task_id": "t1"})

    async def up(p, timeout_s=40.0):
        return True
    m = ServiceManifest(id="comfyui", name="ComfyUI", description="d", project_dir=Path("/x"), command=("./run.sh",),
                        env={}, port=8189, bind="127.0.0.1", match=("image_serve.app",), identity="single", ui=True,
                        provides=parse_provides({"image": {"catalog": "/v1/images/models", "generate": "/v1/images/generations",
                                                           "comfyui_api": "/api/comfyui", "status": "/plugin/status",
                                                           "release": "/plugin/release", "gpu": "shared"}}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(handler))
    monkeypatch.setattr(registry, "ensure_running", up)
    return seen


def test_forwarded_unchanged(plugin):
    r = client.post("/api/comfyui/generate?x=1", json={"prompt": "a fox", "model_name": "z_image_turbo-Q6_K.gguf"})
    assert r.status_code == 202 and r.json()["task_id"] == "t1"
    method, port, path, body = plugin[-1]
    assert (method, port, path) == ("POST", 8189, "/api/comfyui/generate?x=1")
    assert json.loads(body)["prompt"] == "a fox"


def test_file_names_with_spaces(plugin):
    r = client.get("/api/comfyui/images/my picture.png")
    assert r.status_code == 200 and r.content == b"\x89PNG" and r.headers["content-type"] == "image/png"
    assert plugin[-1][2] == "/api/comfyui/images/my%20picture.png"


@pytest.mark.parametrize("path", ["/api/comfyui/images/..%2F..%2Fetc%2Fpasswd", "/api/comfyui/a/%2e%2e/b"])
def test_traversal_refused(plugin, path):
    assert client.get(path).status_code in (400, 404)
    assert not any(".." in p[2] for p in plugin)


def test_without_a_plugin_404(monkeypatch):
    from backend.routers import services as svc
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([], [], {}))
    assert client.get("/api/comfyui/status").status_code == 404
