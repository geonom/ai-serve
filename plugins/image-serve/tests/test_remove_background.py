"""Background removal (task remove_background): ComfyUI's own BiRefNet node behind /v1/images/edits -- an image in, the
same image with an alpha channel out. For character art in visual novels (V-Noveller) and anything else that needs
cut-out figures."""
import json

import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import bundles, catalog, comfy
from image_serve.state import state

client = TestClient(plugin.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def nothing_running(monkeypatch):
    monkeypatch.setattr(state, "comfyui_process", None)
    monkeypatch.setattr(comfy, "find_comfyui_process", lambda: None)


def test_birefnet_is_listed_for_its_task_only():
    rows = {r["id"]: r for r in client.get("/v1/images/models").json()["data"]}
    b = rows["birefnet"]
    assert b["tasks"] == ["remove_background"] and b["license"] == "MIT" and b["commercial_use"] is True
    only = client.get("/v1/images/models?task=remove_background").json()["data"]
    assert [r["id"] for r in only] == ["birefnet"] and only[0]["default"] is True
    assert "birefnet" not in {r["id"] for r in client.get("/v1/images/models?task=generate").json()["data"]}
    assert "birefnet" not in {r["id"] for r in client.get("/v1/images/models?task=edit").json()["data"]}


def test_birefnet_comes_as_a_pinned_bundle():
    (f,) = bundles.files_of("birefnet")
    assert f.repo == "Comfy-Org/BiRefNet" and f.dir == "background_removal" and f.name == "birefnet.safetensors"
    assert f.sha256 == "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154" and f.size == 444473596
    assert bundles.BUNDLES["birefnet"]["license"] == "MIT" and catalog.BUNDLE_OF["birefnet"] == "birefnet"


def test_the_workflow_is_comfys_own_chain():
    wf = comfy.remove_background_workflow("upload-1.png", "birefnet.safetensors")
    by_type = {n["class_type"]: (nid, n["inputs"]) for nid, n in wf.items()}
    assert set(by_type) == {"LoadImage", "LoadBackgroundRemovalModel", "RemoveBackground", "InvertMask",
                            "JoinImageWithAlpha", "SaveImage"}
    load, model = by_type["LoadImage"], by_type["LoadBackgroundRemovalModel"]
    assert load[1]["image"] == "upload-1.png" and model[1]["bg_removal_name"] == "birefnet.safetensors"
    # RemoveBackground gives the foreground; JoinImageWithAlpha takes 1 - alpha, hence InvertMask in between
    assert by_type["RemoveBackground"][1] == {"bg_removal_model": [model[0], 0], "image": [load[0], 0]}
    assert by_type["InvertMask"][1] == {"mask": [by_type["RemoveBackground"][0], 0]}
    assert by_type["JoinImageWithAlpha"][1] == {"image": [load[0], 0], "alpha": [by_type["InvertMask"][0], 0]}
    assert by_type["SaveImage"][1]["images"] == [by_type["JoinImageWithAlpha"][0], 0]
    # the GPU manager is asked for the model's room like for any other workflow
    assert comfy.comfy_model_files(wf) == frozenset({"birefnet.safetensors"})
    json.dumps(wf)


def test_comfy_is_told_where_downloaded_background_models_are(tmp_path):
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "models").mkdir(parents=True)
    args = comfy.extra_model_paths_args(str(comfy_dir), str(tmp_path / "models"), str(tmp_path))
    assert "background_removal: |\n        background_removal/" in open(args[1]).read()


def _fake_queue(monkeypatch, seen):
    async def fake_upload(data):
        seen.append(("upload", data))
        return "upload-1.png"

    async def fake_queue(wf, label):
        seen.append(("queue", wf, label))
        return {"status": "queued", "task_id": "t1"}

    async def fake_await(pid):
        return "UE5HQUxQSEE="
    monkeypatch.setattr(comfy, "_upload_image_to_comfyui", fake_upload)
    monkeypatch.setattr(comfy, "queue_workflow", fake_queue)
    monkeypatch.setattr(comfy, "_await_comfyui_task", fake_await)
    monkeypatch.setattr(comfy, "_ensure_comfyui_running", _noop)


async def _noop(*a, **kw):
    return None


def test_remove_background_needs_no_prompt(monkeypatch):
    seen = []
    _fake_queue(monkeypatch, seen)
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG-figure")}, data={"task": "remove_background"})
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "UE5HQUxQSEE="
    assert seen[0] == ("upload", b"\x89PNG-figure")
    _, wf, label = seen[1]
    assert label == "birefnet" and comfy.comfy_model_files(wf) == frozenset({"birefnet.safetensors"})


def test_remove_background_takes_its_model_and_refuses_others(monkeypatch):
    seen = []
    _fake_queue(monkeypatch, seen)
    files = {"image": ("a.png", b"\x89PNG")}
    assert client.post("/v1/images/edits", files=files, data={"task": "remove_background", "model": "birefnet"}).status_code == 200
    r = client.post("/v1/images/edits", files=files, data={"task": "remove_background", "model": "qwen-edit-2511"})
    assert r.status_code == 400 and "remove_background" in r.json()["detail"]
    # and BiRefNet can't edit by instruction
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "model": "birefnet"}).status_code == 400


def test_remove_background_when_not_installed(monkeypatch):
    monkeypatch.setattr(bundles, "present", lambda bundle, root: bundle != "birefnet")
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG")}, data={"task": "remove_background"})
    assert r.status_code == 400 and "not installed" in r.json()["detail"]


def test_edit_and_style_still_need_a_prompt():
    files = {"image": ("a.png", b"\x89PNG")}
    r = client.post("/v1/images/edits", files=files, data={"task": "edit"})
    assert r.status_code == 400 and "prompt" in r.json()["detail"]
