"""Music and sound effects through `audio` plugins (backend/plugins/audio_gen.py, backend/routers/audio_jobs.py,
2026-10-30). The plugin is faked at the HTTP level and reached only through the plugin registry. What matters:
jobs go to the plugin that lists the model, the gateway's job id says which plugin to ask, and a result file is only
ever fetched from below that plugin's own `jobs` path."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import audio_gen, registry
from backend.plugins.contracts import parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8189


class FakeAudioPlugin:
    def __init__(self):
        self.calls = []
        self.file_url = "/v1/audio/jobs/aud-1/files/aud-1.mp3"
        self.status = "done"

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.url.host, request.url.port) == ("127.0.0.1", PORT)
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if path == "/v1/audio/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "sa3-sfx", "label": "Stable Audio 3 Small SFX", "available": True, "tasks": ["sound"],
                 "max_seconds": 120, "min_seconds": 1, "default_seconds": 8, "default": True,
                 "license": "Stability AI Community License", "bundle": "sa3-sfx"},
                {"id": "ace-step", "label": "ACE-Step 1.5 Turbo", "available": True, "tasks": ["music"],
                 "max_seconds": 300, "lyrics": True, "default": True}]})
        if path == "/v1/audio/jobs" and request.method == "POST":
            if body.get("prompt") == "boom":
                return httpx.Response(400, json={"detail": "prompt: describe the sound"})
            return httpx.Response(202, json={"id": "aud-1", "status": "queued"})
        if path == "/v1/audio/jobs/aud-1":
            st = {"id": "aud-1", "status": self.status, "progress": 1.0 if self.status == "done" else 0.3}
            if self.status == "done":
                st["result"] = {"files": [{"name": "aud-1.mp3", "type": "audio/mpeg", "url": self.file_url}],
                                "data": {"seconds": 8}}
            return httpx.Response(200, json=st)
        if path == "/v1/audio/jobs/aud-1/cancel":
            return httpx.Response(200, json={"id": "aud-1", "status": "cancelled", "progress": None})
        if path == "/v1/audio/jobs/aud-1/files/aud-1.mp3":
            return httpx.Response(200, content=b"ID3fake", headers={"content-type": "audio/mpeg"})
        if path == "/elsewhere/secret.mp3":
            return httpx.Response(200, content=b"SECRET")
        return httpx.Response(404, json={"detail": path})


@pytest.fixture(autouse=True)
def plugin(monkeypatch):
    from backend.routers import services as svc
    fake = FakeAudioPlugin()
    m = ServiceManifest(id="comfyui", name="ComfyUI", description="d", project_dir=Path("/x"),
                        command=("./run.sh",), env={}, port=PORT, bind="127.0.0.1", match=("image_serve.app",),
                        identity="single", ui=True, provides=parse_provides({"audio": {
                            "catalog": "/v1/audio/models", "jobs": "/v1/audio/jobs", "status": "/plugin/status",
                            "release": "/plugin/release", "gpu": "shared"}}))
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([m], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setattr(registry, "ensure_running", fake.ensure_running)
    monkeypatch.setattr(audio_gen, "_cache", (0.0, [], []))
    return fake


def test_models_per_task_and_the_speech_catalog_stays_where_it_was():
    r = client.get("/v1/audio/models?task=sound").json()
    assert [m["id"] for m in r["data"]] == ["sa3-sfx"] and r["default"] == "sa3-sfx"
    assert r["data"][0]["provider"] == "comfyui" and r["data"][0]["max_seconds"] == 120
    music = client.get("/v1/audio/models?task=music").json()["data"]
    assert music[0]["id"] == "ace-step" and music[0]["lyrics"] is True
    assert len(client.get("/v1/audio/models?task=all").json()["data"]) == 2
    assert client.get("/v1/audio/models?task=speech").status_code == 400


def test_a_job_goes_to_the_plugin_listing_its_model(plugin):
    r = client.post("/v1/audio/jobs", json={"task": "sound", "prompt": "Regen", "seconds": 10})
    assert r.status_code == 202 and r.json() == {"id": "comfyui.aud-1", "status": "queued", "model": "sa3-sfx"}
    sent = [c for c in plugin.calls if c[:2] == ("POST", "/v1/audio/jobs")][-1][2]
    assert sent == {"task": "sound", "prompt": "Regen", "seconds": 10, "model": "sa3-sfx"}
    client.post("/v1/audio/jobs", json={"prompt": "Tavernenmusik"})                 # no task, no model: music
    assert [c for c in plugin.calls if c[:2] == ("POST", "/v1/audio/jobs")][-1][2]["model"] == "ace-step"


def test_wrong_models_and_tasks_are_400_and_plugin_refusals_keep_their_reason():
    assert client.post("/v1/audio/jobs", json={"model": "dall-e", "prompt": "x"}).status_code == 400
    assert client.post("/v1/audio/jobs", json={"model": "sa3-sfx", "task": "music", "prompt": "x"}).status_code == 400
    assert client.post("/v1/audio/jobs", json={"task": "speech", "prompt": "x"}).status_code == 400
    r = client.post("/v1/audio/jobs", json={"task": "sound", "prompt": "boom"})
    assert r.status_code == 400 and "describe the sound" in r.text


def test_status_and_file_come_through_the_gateways_own_paths():
    st = client.get("/v1/audio/jobs/comfyui.aud-1").json()
    assert st["id"] == "comfyui.aud-1" and st["status"] == "done"
    assert st["result"]["files"] == [{"name": "aud-1.mp3", "type": "audio/mpeg",
                                      "url": "/v1/audio/jobs/comfyui.aud-1/files/aud-1.mp3"}]
    f = client.get("/v1/audio/jobs/comfyui.aud-1/files/aud-1.mp3")
    assert f.status_code == 200 and f.content == b"ID3fake" and f.headers["content-type"] == "audio/mpeg"
    assert client.get("/v1/audio/jobs/comfyui.aud-1/files/other.mp3").status_code == 404
    assert client.post("/v1/audio/jobs/comfyui.aud-1/cancel").json()["status"] == "cancelled"


def test_a_file_url_outside_the_jobs_path_is_never_fetched(plugin):
    plugin.file_url = "/v1/audio/jobs/aud-2/files/aud-1.mp3"        # another job's directory
    assert client.get("/v1/audio/jobs/comfyui.aud-1/files/aud-1.mp3").status_code == 404
    plugin.file_url = "/elsewhere/secret.mp3"                        # not even under jobs
    r = client.get("/v1/audio/jobs/comfyui.aud-1/files/aud-1.mp3")
    assert r.status_code in (404, 502) and b"SECRET" not in r.content
    assert not any(c[1] == "/elsewhere/secret.mp3" for c in plugin.calls)


def test_job_ids_name_a_known_plugin_and_nothing_else():
    for ref in ("aud-1", "nope.aud-1", "comfyui.", "comfyui.a/b", "comfyui.a.b", "..%2f.x"):
        assert client.get(f"/v1/audio/jobs/{ref}").status_code == 404, ref
