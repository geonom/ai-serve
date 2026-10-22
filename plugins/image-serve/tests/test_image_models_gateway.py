"""GET /v1/images/models and /v1/images/generations by catalog id + aspect_ratio (image_serve/app.py; moved from ai-serve-ui)."""
import pytest
from fastapi.testclient import TestClient

from image_serve import catalog
from image_serve import comfy as media


@pytest.fixture
def client(monkeypatch):
    from image_serve.app import app
    seen = {"comfy": []}

    async def fake_comfy(req):
        seen["comfy"].append(req)
        return {"task_id": "t1"}

    async def fake_await(task_id):
        return "Q09NRlk="


    monkeypatch.setattr(media, "comfyui_generate", fake_comfy)
    monkeypatch.setattr(media, "_await_comfyui_task", fake_await)
    monkeypatch.setattr(catalog, "_on_disk", lambda root, rel: "ideogram" not in rel)
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.seen = seen
    return c


def test_models_endpoint(client):
    r = client.get("/v1/images/models")
    assert r.status_code == 200
    rows = {m["id"]: m for m in r.json()["data"]}
    assert {"z-image-turbo", "ernie-turbo"} <= set(rows)
    assert rows["z-image-turbo"]["available"] and "ideogram-4" not in rows   # missing and not downloadable: hidden


@pytest.mark.parametrize("model, file", [("z-image-turbo", "z_image_turbo-Q6_K.gguf"),
                                         ("z_image_turbo-Q6_K.gguf", "z_image_turbo-Q6_K.gguf"),
                                         ("ernie-turbo", "ernie-image-turbo-Q6_K.gguf")])
def test_comfy_models_with_aspect(client, model, file):
    r = client.post("/v1/images/generations", json={"model": model, "prompt": "a fox", "aspect_ratio": "16:9"})
    assert r.status_code == 200, r.text
    req = client.seen["comfy"][0]
    assert (req.model_name, req.aspect_ratio, req.base_resolution) == (file, "16:9", 1024)


@pytest.mark.parametrize("body", [
    {"model": "z-image-turbo", "layout": [{"box": [0, 0, 1, 1]}]},          # no built-in model places things
    {"model": "qwen-image-2.1", "aspect_ratio": "21:9"},
    {"model": "z-image-turbo", "aspect_ratio": "16:9", "size": "1024x1024"},   # one or the other
    {"model": "ideogram-4"},                                                  # not on disk
])
def test_refusals(client, body):
    r = client.post("/v1/images/generations", json={"prompt": "x", **body})
    assert r.status_code == 400, r.text
    assert not any(client.seen.values())


def test_unknown_model_keeps_the_default(client):
    """OpenAI clients send e.g. dall-e-3; that kept working before the catalog and still does."""
    assert client.post("/v1/images/generations", json={"model": "dall-e-3", "prompt": "a fox"}).status_code == 200
    assert client.seen["comfy"][0].model_name == "z_image_turbo-Q6_K.gguf"


def test_comfy_gets_steps_and_seed(client):
    """Until 2026-10-18 the /v1 path forced 4 steps and a random seed on every ComfyUI model -- Z-Image
    (template 8) and Qwen-Image 2.1 (20) came out worse, and nothing could be compared by seed or steps."""
    client.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "x", "steps": 12, "seed": 77})
    client.post("/v1/images/generations", json={"model": "qwen-image-2.1", "prompt": "x"})
    a, b = client.seen["comfy"]
    assert (a.steps, a.seed) == (12, 77)
    assert b.steps is None and 1 <= b.seed   # None = the workflow template's own default


@pytest.mark.parametrize("body", [{"model": "z-image-turbo", "steps": 0}, {"model": "z-image-turbo", "steps": 61}])
def test_steps_out_of_range_are_400(client, body):
    assert client.post("/v1/images/generations", json={"prompt": "x", **body}).status_code == 400
    assert not any(client.seen.values())



def test_models_endpoint_by_task(client):
    rows = client.get("/v1/images/models", params={"task": "style"}).json()["data"]
    assert {r["id"] for r in rows} == {"qwen-edit-2511", "qwen-edit-2511-q3"}
    assert client.get("/v1/images/models", params={"task": "x"}).status_code == 400


def test_an_edit_model_cannot_generate(client):
    r = client.post("/v1/images/generations", json={"prompt": "x", "model": "qwen-edit-2511"})
    assert r.status_code == 400 and "edit" in r.text


def test_edit_endpoints_take_catalog_ids(client, monkeypatch):
    seen = []

    async def fake_edit(req):
        seen.append(("edit", req.model_name))
        return {"task_id": "t"}

    async def fake_style(req):
        seen.append(("style", req.model_name))
        return {"task_id": "t", "status": "queued"}
    monkeypatch.setattr(media, "comfyui_edit", fake_edit)
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG")}, data={"prompt": "rot", "model": "Qwen_Image_Edit-Q4_K_S.gguf"})
    assert r.status_code == 200, r.text
    assert seen[-1] == ("edit", "qwen-image-edit-2511-Q2_K.gguf")   # removed older file -> the 2511 one
    r = client.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG")}, data={"prompt": "rot", "model": "z-image-turbo"})
    assert r.status_code == 400


def test_edits_take_a_seed(client, monkeypatch):
    # the Figurenwerkstatt (2026-10-21) needs reproducible character images; without seed it stays random
    seen = []

    async def fake_edit(req):
        seen.append(req.seed)
        return {"task_id": "t"}
    monkeypatch.setattr(media, "comfyui_edit", fake_edit)
    files = {"image": ("a.png", b"\x89PNG")}
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "model": "qwen-edit-2511", "seed": "4242"}).status_code == 200
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "model": "qwen-edit-2511"}).status_code == 200
    assert seen[0] == 4242 and isinstance(seen[1], int)   # given -> used; missing -> a random one
    assert client.post("/v1/images/edits", files=files, data={"prompt": "x", "seed": "-1"}).status_code == 400
