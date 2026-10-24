"""image plugins (backend/plugins/images.py): image backends outside the gateway -- Iris in sd-serve, MOSS in its own
ComfyUI, later the main ComfyUI -- list their models, and /v1/images/* goes to the plugin that lists the model.
Since 2026-10-21 every image model comes from a plugin (ComfyUI is image-serve). The plugin is faked at the
HTTP level and reached only through the plugin registry."""
import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import images as image_plugins
from backend.plugins import registry
from backend.plugins.contracts import parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8066
PNG = base64.b64encode(b"\x89PNG fake").decode()


class FakeImageServer:
    def __init__(self):
        self.calls = []

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.url.host, request.url.port) == ("127.0.0.1", PORT)
        path = request.url.path
        ctype = request.headers.get("content-type", "")
        body = json.loads(request.content) if request.content and ctype.startswith("application/json") else request.content
        self.calls.append((request.method, path, body, request.headers.get("x-request-id")))
        if path == "/v1/images/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "iris-turbo", "label": "Iris-3B Turbo (Plugin)", "available": True, "tasks": ["generate"],
                 "aspect_ratios": ["1:1", "16:9"], "steps": {"default": 4, "adjustable": False, "max": 4},
                 "license": "Apache-2.0", "commercial_use": True, "aliases": ["iris"]},
                {"id": "pix-edit", "label": "Pix Edit", "available": True, "tasks": ["edit"]},
                {"id": "group-edit", "label": "Group Edit", "available": True, "tasks": ["edit"], "max_images": 5,
                 "aspect_ratios": ["1:1", "16:9"]},
                {"id": "cutter", "label": "Cutter", "available": True, "tasks": ["remove_background"],
                 "default": True}]})
        if path == "/v1/images/generations":
            if body.get("prompt") == "boom":
                return httpx.Response(400, json={"detail": "steps can't be set for this model"})
            return httpx.Response(200, json={"created": 1, "data": [{"b64_json": PNG}]})
        if path == "/v1/images/edits":
            return httpx.Response(200, json={"created": 1, "data": [{"b64_json": PNG}]})
        if path == "/v1/images/cancel":
            return httpx.Response(200, json={"cancelled": True})
        if path == "/status":
            return httpx.Response(200, json={"busy": False, "workers": []})
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def server(monkeypatch):
    from backend.routers import services as svc
    fake = FakeImageServer()
    m = ServiceManifest(id="sd-serve", name="Iris (sd-serve)", description="d", project_dir=Path("/x"),
                        command=("./run.sh",), env={}, port=PORT, bind="127.0.0.1", match=("sd_plugin",),
                        identity="none", ui=False, provides=parse_provides({"image": {
                            "catalog": "/v1/images/models", "generate": "/v1/images/generations",
                            "edit": "/v1/images/edits", "cancel": "/v1/images/cancel", "status": "/status",
                            "release": "/release", "gpu": "shared"}}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(registry, "ensure_running", fake.ensure_running)
    monkeypatch.setattr(image_plugins, "_cache", (0.0, [], []))
    return fake


def test_models_come_from_the_plugins():
    rows = {r["id"]: r for r in client.get("/v1/images/models").json()["data"]}
    assert rows["iris-turbo"]["label"] == "Iris-3B Turbo (Plugin)" and rows["iris-turbo"]["provider"] == "sd-serve"
    assert rows["iris-turbo"]["license"] == "Apache-2.0"
    assert "pix-edit" not in rows                                           # edit-only: not in the generate list
    assert "pix-edit" in {r["id"] for r in client.get("/v1/images/models?task=edit").json()["data"]}
    assert "z-image-turbo" not in rows                                      # nothing built in any more


@pytest.mark.parametrize("name", ["iris-turbo", "IRIS"])
def test_generation_goes_to_the_plugin_with_the_request_id(server, name):
    r = client.post("/v1/images/generations", json={"model": name, "prompt": "a fox", "aspect_ratio": "16:9"},
                    headers={"X-Request-Id": "req-0123456789abcdef"})
    assert r.status_code == 200, r.text
    assert r.json()["data"][0]["b64_json"] == PNG
    method, path, body, rid = server.calls[-1]
    assert (method, path, body["model"], body["prompt"], body["aspect_ratio"], rid) == \
           ("POST", "/v1/images/generations", "iris-turbo", "a fox", "16:9", "req-0123456789abcdef")
    assert "size" not in body and "response_format" not in body     # only what the caller sent (found live)


def test_plugin_refusals_keep_their_status_and_message():
    r = client.post("/v1/images/generations", json={"model": "iris-turbo", "prompt": "boom"})
    assert r.status_code == 400 and "steps" in r.json()["detail"]


def test_an_edit_model_can_not_generate():
    r = client.post("/v1/images/generations", json={"model": "pix-edit", "prompt": "x"})
    assert r.status_code == 400


def test_edit_goes_to_the_plugin_as_multipart(server):
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG", "image/png")},
                    data={"prompt": "make it blue", "model": "pix-edit", "seed": "7"})
    assert r.status_code == 200, r.text
    method, path, body, _ = server.calls[-1]
    assert (method, path) == ("POST", "/v1/images/edits")
    assert b"make it blue" in body and b"pix-edit" in body and b"\x89PNG" in body


def test_edits_take_several_images_in_order_and_an_aspect_ratio(server):
    """A background + up to four figures (owner 2026-10-24): `image` repeated or OpenAI's `image[]`, forwarded as
    repeated `image` parts in the caller's order; `aspect_ratio` = the output's shape."""
    row = next(r for r in client.get("/v1/images/models?task=edit").json()["data"] if r["id"] == "group-edit")
    assert row["max_images"] == 5
    for field in ("image", "image[]"):
        files = [(field, (f"{i}.png", f"\x89PNG-{i}".encode(), "image/png")) for i in range(5)]
        r = client.post("/v1/images/edits", files=files,
                        data={"prompt": "the four from picture 2 in picture 1", "model": "group-edit",
                              "aspect_ratio": "16:9"})
        assert r.status_code == 200, r.text
        _, _, body, _ = server.calls[-1]
        assert body.count(b'name="image"') == 5 and b"image[]" not in body
        assert [body.index(f"\x89PNG-{i}".encode()) for i in range(5)] == sorted(body.index(f"\x89PNG-{i}".encode()) for i in range(5))
        assert b'name="aspect_ratio"' in body and b"16:9" in body


def test_too_many_images_or_a_wrong_shape_are_refused_before_the_plugin(server):
    def edits():
        return [c for c in server.calls if c[1] == "/v1/images/edits"]
    before = len(edits())
    two = [("image", ("1.png", b"\x89PNG")), ("image", ("2.png", b"\x89PNG"))]
    r = client.post("/v1/images/edits", files=two, data={"prompt": "p", "model": "pix-edit"})
    assert r.status_code == 400 and "1" in r.json()["detail"]      # no max_images = one image
    six = [("image", (f"{i}.png", b"\x89PNG")) for i in range(6)]
    r = client.post("/v1/images/edits", files=six, data={"prompt": "p", "model": "group-edit"})
    assert r.status_code == 400 and "5" in r.json()["detail"]
    r = client.post("/v1/images/edits", files=two, data={"prompt": "p", "model": "group-edit", "aspect_ratio": "4:3"})
    assert r.status_code == 400 and "16:9" in r.json()["detail"]
    both = [("image", ("1.png", b"\x89PNG")), ("image[]", ("2.png", b"\x89PNG"))]
    assert client.post("/v1/images/edits", files=both, data={"prompt": "p", "model": "group-edit"}).status_code == 400
    assert client.post("/v1/images/edits", data={"prompt": "p", "model": "group-edit"}).status_code == 400
    assert len(edits()) == before


def test_remove_background_needs_no_prompt_and_goes_to_its_model(server):
    rows = client.get("/v1/images/models?task=remove_background").json()
    assert [r["id"] for r in rows["data"]] == ["cutter"] and rows["default"] == "cutter"
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG", "image/png")},
                    data={"task": "remove_background"})
    assert r.status_code == 200, r.text
    method, path, body, _ = server.calls[-1]
    assert (method, path) == ("POST", "/v1/images/edits")
    assert b"remove_background" in body and b"cutter" in body and b'name="prompt"' not in body


def test_remove_background_refuses_a_model_that_cannot():
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG")},
                    data={"task": "remove_background", "model": "pix-edit"})
    assert r.status_code == 400 and "remove_background" in r.json()["detail"]


def test_edit_and_style_still_need_a_prompt():
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG")}, data={"model": "pix-edit"})
    assert r.status_code == 400 and "prompt" in r.json()["detail"]


def test_cancel_reaches_the_plugins(server):
    r = client.post("/v1/images/cancel", json={"request_id": "req-0123456789abcdef"})
    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert ("POST", "/v1/images/cancel", {"request_id": "req-0123456789abcdef"}, None) in server.calls
