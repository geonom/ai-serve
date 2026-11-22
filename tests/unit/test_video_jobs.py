"""Video through `video` plugins (backend/plugins/video_gen.py, backend/routers/video_jobs.py, 2026-11-13). The plugin
is faked at the HTTP level and reached only through the plugin registry. What matters: jobs go to the plugin that
lists the model (a start picture picks image_to_video), the catalog keeps LoRAs/sound/lengths, the gateway's job id
says which plugin to ask, and a result file is only ever fetched from below that plugin's own `jobs` path."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import registry, video_gen
from backend.plugins.contracts import parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8189


class FakeVideoPlugin:
    def __init__(self):
        self.calls = []
        self.file_url = "/v1/video/jobs/vid-1/files/vid-1.mp4"

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.url.host, request.url.port) == ("127.0.0.1", PORT)
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if path == "/v1/video/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "ltx-2.5", "label": "LTX-2.5", "available": True, "tasks": ["text_to_video"], "default": True,
                 "max_seconds": 8, "min_seconds": 2, "default_seconds": 4, "fps": 24, "audio": True,
                 "resolutions": ["960x544", "1280x704"], "loras": ["style A", "../evil"], "sage_attention": False,
                 "steps": {"default": 8, "min": 4, "max": 20, "adjustable": True}},
                {"id": "wan-2.2", "label": "Wan 2.2", "available": True, "tasks": ["image_to_video"], "default": True,
                 "max_seconds": 6, "fps": 16, "resolutions": ["832x480"], "loras": [], "sage_attention": True},
                {"id": "wan-2.2-fp8", "label": "Wan 2.2 fp8", "available": False, "tasks": ["image_to_video"]}]})
        if path == "/v1/video/jobs" and request.method == "POST":
            if body.get("prompt") == "bad":
                return httpx.Response(400, json={"detail": "lora for LTX-2.5: style A"})
            if body.get("prompt") == "full":
                return httpx.Response(507, json={"detail": "No room for the image model on GPU 0: busy"})
            return httpx.Response(202, json={"id": "vid-1", "status": "queued"})
        if path == "/v1/video/jobs/vid-1":
            return httpx.Response(200, json={"id": "vid-1", "status": "done", "progress": 1.0, "result": {
                "files": [{"name": "vid-1.mp4", "type": "video/mp4", "url": self.file_url}],
                "data": {"frames": 97, "runtime_seconds": 48.2}}})
        if path == "/v1/video/jobs/vid-1/cancel":
            return httpx.Response(200, json={"id": "vid-1", "status": "cancelled", "progress": None})
        if path == "/v1/video/jobs/vid-1/files/vid-1.mp4":
            return httpx.Response(200, content=b"MP4fake", headers={"content-type": "video/mp4"})
        if path == "/elsewhere/secret.mp4":
            return httpx.Response(200, content=b"SECRET")
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def plugin(monkeypatch):
    from backend.routers import services as svc
    fake = FakeVideoPlugin()
    m = ServiceManifest(id="comfyui", name="ComfyUI", description="d", project_dir=Path("/x"),
                        command=("./run.sh",), env={}, port=PORT, bind="127.0.0.1", match=("image_serve.app",),
                        identity="single", ui=True, provides=parse_provides({"video": {
                            "catalog": "/v1/video/models", "jobs": "/v1/video/jobs", "status": "/plugin/status",
                            "release": "/plugin/release", "gpu": "shared"}}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(registry, "ensure_running", fake.ensure_running)
    monkeypatch.setattr(video_gen, "_cache", (0.0, [], []))
    return fake


def test_catalog_keeps_sound_lengths_and_plain_lora_names():
    rows = {m["id"]: m for m in client.get("/v1/video/models").json()["data"]}
    ltx = rows["ltx-2.5"]
    assert ltx["provider"] == "comfyui" and ltx["audio"] is True and ltx["fps"] == 24
    assert ltx["min_seconds"] == 2 and ltx["default_seconds"] == 4 and ltx["resolutions"] == ["960x544", "1280x704"]
    assert ltx["loras"] == ["style A"]                         # a path-like name is dropped
    assert rows["wan-2.2"]["audio"] is False
    assert rows["ltx-2.5"]["sage_attention"] is False and rows["wan-2.2"]["sage_attention"] is True
    assert rows["wan-2.2-fp8"]["sage_attention"] is None
    assert ltx["steps"] == {"default": 8, "min": 4, "max": 20, "adjustable": True}
    assert rows["wan-2.2"]["steps"] is None                     # not offered: the plugin's fixed count
    i2v = client.get("/v1/video/models?task=image_to_video").json()["data"]
    assert {m["id"] for m in i2v} == {"wan-2.2", "wan-2.2-fp8"}
    assert client.get("/v1/video/models?task=music").status_code == 400


def test_a_job_goes_to_the_plugin_listing_its_model(plugin):
    r = client.post("/v1/video/jobs", json={"prompt": "Ein Fuchs im Schnee", "seconds": 4, "lora": "style A"})
    assert r.status_code == 202 and r.json() == {"id": "comfyui.vid-1", "status": "queued", "model": "ltx-2.5"}
    sent = [c for c in plugin.calls if c[:2] == ("POST", "/v1/video/jobs")][-1][2]
    assert sent == {"prompt": "Ein Fuchs im Schnee", "seconds": 4, "lora": "style A", "model": "ltx-2.5"}
    client.post("/v1/video/jobs", json={"prompt": "er springt", "image_base64": "iVBOR"})   # picture: image_to_video
    assert [c for c in plugin.calls if c[:2] == ("POST", "/v1/video/jobs")][-1][2]["model"] == "wan-2.2"


def test_wrong_models_and_tasks_are_400_and_plugin_refusals_keep_their_reason():
    assert client.post("/v1/video/jobs", json={"model": "sora", "prompt": "x"}).status_code == 400
    assert client.post("/v1/video/jobs", json={"model": "ltx-2.5", "task": "image_to_video",
                                               "prompt": "x"}).status_code == 400
    assert client.post("/v1/video/jobs", json={"task": "music", "prompt": "x"}).status_code == 400
    r = client.post("/v1/video/jobs", json={"prompt": "bad"})
    assert r.status_code == 400 and "lora for" in r.text


def test_no_gpu_room_is_the_plugins_507_with_its_reason():
    """Fast yes/no: the plugin asks for GPU room before it accepts a job; its no reaches the caller unchanged."""
    r = client.post("/v1/video/jobs", json={"prompt": "full"})
    assert r.status_code == 507 and "No room" in r.json()["detail"]


def test_status_and_file_come_through_the_gateways_own_paths():
    st = client.get("/v1/video/jobs/comfyui.vid-1").json()
    assert st["id"] == "comfyui.vid-1" and st["status"] == "done" and st["result"]["data"]["frames"] == 97
    assert st["result"]["files"] == [{"name": "vid-1.mp4", "type": "video/mp4",
                                      "url": "/v1/video/jobs/comfyui.vid-1/files/vid-1.mp4"}]
    f = client.get("/v1/video/jobs/comfyui.vid-1/files/vid-1.mp4")
    assert f.status_code == 200 and f.content == b"MP4fake" and f.headers["content-type"] == "video/mp4"
    assert client.get("/v1/video/jobs/comfyui.vid-1/files/other.mp4").status_code == 404
    assert client.post("/v1/video/jobs/comfyui.vid-1/cancel").json()["status"] == "cancelled"


def test_a_file_url_outside_the_jobs_path_is_never_fetched(plugin):
    plugin.file_url = "/v1/video/jobs/vid-2/files/vid-1.mp4"
    assert client.get("/v1/video/jobs/comfyui.vid-1/files/vid-1.mp4").status_code == 404
    plugin.file_url = "/elsewhere/secret.mp4"
    r = client.get("/v1/video/jobs/comfyui.vid-1/files/vid-1.mp4")
    assert r.status_code in (404, 502) and b"SECRET" not in r.content
    assert not any(c[1] == "/elsewhere/secret.mp4" for c in plugin.calls)


def test_job_ids_name_a_known_plugin_and_nothing_else():
    for ref in ("vid-1", "nope.vid-1", "comfyui.", "comfyui.a/b", "comfyui.a.b", "..%2f.x"):
        assert client.get(f"/v1/video/jobs/{ref}").status_code == 404, ref
