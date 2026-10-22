"""image-serve's plugin surface: catalog with licenses, edit/style checks, status/release without a running ComfyUI."""
import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import comfy
from image_serve.state import state

client = TestClient(plugin.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def nothing_running(monkeypatch):
    monkeypatch.setattr(state, "comfyui_process", None)
    monkeypatch.setattr(state, "comfyui_loaded_models", None)
    monkeypatch.setattr(comfy, "find_comfyui_process", lambda: None)


def test_catalog_is_the_contracts_shape():
    rows = {r["id"]: r for r in client.get("/v1/images/models").json()["data"]}
    assert {"z-image-turbo", "qwen-edit-2511"} <= set(rows)
    assert rows["z-image-turbo"]["license"] == "Apache-2.0" and rows["z-image-turbo"]["commercial_use"] is True
    assert rows["qwen-edit-2511"]["tasks"] == ["edit", "style"] and "file" not in rows["z-image-turbo"]
    assert {r["id"] for r in client.get("/v1/images/models?task=style").json()["data"]} == {"qwen-edit-2511", "qwen-edit-2511-q3"}


def test_style_needs_a_reference_and_tasks_are_checked():
    files = {"image": ("a.png", b"\x89PNG")}
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "task": "style"}).status_code == 400
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "task": "upscale"}).status_code == 400
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "model": "z-image-turbo"}).status_code == 400


def test_style_goes_to_style_transfer_with_both_images(monkeypatch):
    seen = []

    async def fake_style(req):
        seen.append((req.model_name, len(req.content_image_base64) > 0, len(req.style_image_base64) > 0))
        return {"task_id": "t1"}

    async def fake_await(pid):
        return "UE5H"
    monkeypatch.setattr(comfy, "comfyui_style_transfer", fake_style)
    monkeypatch.setattr(comfy, "_await_comfyui_task", fake_await)
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG"), "reference": ("b.png", b"\x89PNG2")},
                    data={"prompt": "as watercolor", "task": "style", "model": "qwen-edit-2511"})
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "UE5H"
    assert seen == [("qwen-image-edit-2511-Q2_K.gguf", True, True)]


def test_status_and_release_when_nothing_runs():
    st = client.get("/plugin/status").json()
    assert st["workers"] == [] and st["busy"] is False and "off" in st["facts"][0]["value"]
    assert client.post("/plugin/release", json={"worker": "other"}).status_code == 404
    assert client.post("/plugin/release", json={}).json() == {"released": []}


def test_layout_is_refused_here():
    r = client.post("/v1/images/generations", json={"prompt": "x", "model": "z-image-turbo", "layout": [{"box": [0, 0, 1, 1]}]})
    assert r.status_code == 400


def test_a_callers_port_never_moves_comfyui_onto_this_apps_port(monkeypatch):
    """The dashboard's old start button (and the API docs' example) send port 8189 -- image-serve's own port now."""
    import asyncio
    started = []

    class P:
        pid = 4242

        def poll(self):
            return None

    def popen(cmd, **kw):
        started.append(cmd)
        return P()
    monkeypatch.setattr(comfy.subprocess, "Popen", popen)
    monkeypatch.setattr(comfy.MANAGER if hasattr(comfy, "MANAGER") else __import__("image_serve.gpu").gpu.MANAGER,
                        "raise_if_exclusive", lambda *a: asyncio.sleep(0))
    monkeypatch.setattr(state, "comfyui_process", None)
    r = asyncio.run(comfy.start_comfyui(comfy.ComfyUIStartRequest(port=8189, gpu_id=0)))
    assert started[0][started[0].index("--port") + 1] == str(state.comfyui_port)
    assert r["port"] == state.comfyui_port != 8189          # never this app's own port
