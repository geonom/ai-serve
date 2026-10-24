"""Edits with several images and a chosen output shape (2026-10-24, for V-Noveller's event pictures: "the creature
from picture 2 in the scene of picture 1", in 16:9; owner: a background and up to four characters, e.g. a group
photo). Qwen-Image-Edit 2511's TextEncodeQwenImageEditPlus takes up to three images ("Picture 1..3" in its prompt).
Four or five images: images 2..5 are stitched left to right into picture 2 -- tried 2026-10-24 with a market square
and four characters: five native references (two via ReferenceLatent) doubled the knight and lost the robot, 2+2
stitched added a fifth person, all four stitched gave exactly the four on three seeds. The output size is the
KSampler's latent -- the first image's by default, an empty latent of Qwen-Image's trained size for an aspect ratio."""
import json

import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import catalog, comfy

client = TestClient(plugin.app, client=("127.0.0.1", 50000))
PNG = b"\x89PNG"


def _template():
    with open(f"{comfy.WORKFLOWS_DIR}/Qwen-Image-Edit.json") as f:
        return json.load(f)


def _by_type(wf, t):
    return [(nid, n) for nid, n in wf.items() if n["class_type"] == t]


def _source(wf, ref):
    """Follow a link back to the LoadImage it starts from (through the scale node)."""
    node = wf[ref[0]]
    while node["class_type"] != "LoadImage":
        node = wf[node["inputs"]["image"][0]]
    return node["inputs"]["image"]


def test_the_catalog_says_how_many_images_a_model_takes():
    rows = {r["id"]: r for r in client.get("/v1/images/models?task=all").json()["data"]}
    assert rows["qwen-edit-2511"]["max_images"] == 5 and rows["qwen-edit-2511-q3"]["max_images"] == 5
    assert rows["birefnet"]["max_images"] == 1 and rows["z-image-turbo"]["max_images"] == 0
    assert "16:9" in rows["qwen-edit-2511"]["aspect_ratios"]


def test_three_images_are_pictures_one_to_three_in_order():
    wf = comfy.patch_edit_workflow(_template(), ["a.png", "b.png", "c.png"], prompt="p", model_name="m.gguf", seed=7)
    encoders = _by_type(wf, "TextEncodeQwenImageEditPlus")
    assert len(encoders) == 2   # positive and negative see the same pictures
    for _, enc in encoders:
        assert [_source(wf, enc["inputs"][f"image{i}"]) for i in (1, 2, 3)] == ["a.png", "b.png", "c.png"]
    positive = next(n for _, n in encoders if "Positive" in n["_meta"]["title"])
    assert positive["inputs"]["prompt"] == "p"
    (_, ks), = _by_type(wf, "KSampler")
    assert ks["inputs"]["seed"] == 7
    # no aspect ratio: the output has the first image's shape, as before
    assert wf[ks["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    json.dumps(wf)


@pytest.mark.parametrize("count", [4, 5])
def test_four_or_five_images_put_the_figures_side_by_side_as_picture_two(count):
    names = ["bg.png", "c1.png", "c2.png", "c3.png", "c4.png"][:count]
    wf = comfy.patch_edit_workflow(_template(), names, prompt="p", model_name="m.gguf", seed=1, aspect_ratio="16:9")
    for _, enc in _by_type(wf, "TextEncodeQwenImageEditPlus"):
        assert _source(wf, enc["inputs"]["image1"]) == "bg.png" and "image3" not in enc["inputs"]
        # follow the stitch chain: image1 of each ImageStitch is everything to its left
        node, right = wf[enc["inputs"]["image2"][0]], []
        while node["class_type"] == "ImageStitch":
            assert node["inputs"]["direction"] == "right" and node["inputs"]["match_image_size"] is True
            right.insert(0, wf[node["inputs"]["image2"][0]]["inputs"]["image"])
            node = wf[node["inputs"]["image1"][0]]
        assert [node["inputs"]["image"], *right] == names[1:]
    json.dumps(wf)


def test_one_image_stays_the_old_workflow():
    wf = comfy.patch_edit_workflow(_template(), ["a.png"], prompt="p", model_name="m.gguf", seed=1)
    for _, enc in _by_type(wf, "TextEncodeQwenImageEditPlus"):
        assert "image2" not in enc["inputs"] and "image3" not in enc["inputs"]
    assert len(_by_type(wf, "LoadImage")) == 1


@pytest.mark.parametrize("aspect,size", [("16:9", (1664, 928)), ("9:16", (928, 1664)), ("1:1", (1328, 1328)),
                                         ("4:3", (1472, 1104)), ("3:4", (1104, 1472))])
def test_an_aspect_ratio_gives_an_empty_latent_of_qwen_images_trained_size(aspect, size):
    wf = comfy.patch_edit_workflow(_template(), ["a.png", "b.png"], prompt="p", model_name="m.gguf", seed=1,
                                   aspect_ratio=aspect)
    (_, ks), = _by_type(wf, "KSampler")
    latent = wf[ks["inputs"]["latent_image"][0]]
    assert latent["class_type"] == "EmptySD3LatentImage"
    assert (latent["inputs"]["width"], latent["inputs"]["height"], latent["inputs"]["batch_size"]) == (*size, 1)
    assert ks["inputs"]["denoise"] == 1   # an empty latent is all noise


def test_patching_refuses_what_the_model_can_not_do():
    with pytest.raises(ValueError):
        comfy.patch_edit_workflow(_template(), ["a", "b", "c", "d", "e", "f"], prompt="p", model_name="m", seed=1)
    with pytest.raises(ValueError):
        comfy.patch_edit_workflow(_template(), [], prompt="p", model_name="m", seed=1)
    with pytest.raises(ValueError):
        comfy.patch_edit_workflow(_template(), ["a"], prompt="p", model_name="m", seed=1, aspect_ratio="2:1")


@pytest.fixture
def edits(monkeypatch):
    seen = []

    async def fake_edit(req):
        seen.append(req)
        return {"task_id": "t1"}

    async def fake_await(pid):
        return "UE5H"
    monkeypatch.setattr(comfy, "comfyui_edit", fake_edit)
    monkeypatch.setattr(comfy, "_await_comfyui_task", fake_await)
    return seen


def test_repeated_image_fields_arrive_in_order(edits):
    files = [("image", ("1.png", b"\x89PNG-one")), ("image", ("2.png", b"\x89PNG-two")), ("image", ("3.png", b"\x89PNG-3"))]
    r = client.post("/v1/images/edits", files=files, data={"prompt": "the creature from picture 2 in picture 1",
                                                           "aspect_ratio": "16:9"})
    assert r.status_code == 200, r.text
    (req,) = edits
    import base64
    assert [base64.b64decode(b) for b in [req.image_base64, *req.extra_images_base64]] == \
        [b"\x89PNG-one", b"\x89PNG-two", b"\x89PNG-3"]
    assert req.aspect_ratio == "16:9"


def test_openai_style_image_brackets_work_too(edits):
    files = [("image[]", ("1.png", PNG)), ("image[]", ("2.png", PNG))]
    assert client.post("/v1/images/edits", files=files, data={"prompt": "p"}).status_code == 200
    assert len(edits[0].extra_images_base64) == 1


def test_wrong_counts_and_shapes_are_refused(edits):
    six = [("image", (f"{i}.png", PNG)) for i in range(6)]
    r = client.post("/v1/images/edits", files=six, data={"prompt": "p"})
    assert r.status_code == 400 and "5" in r.json()["detail"]
    assert client.post("/v1/images/edits", data={"prompt": "p"}).status_code in (400, 422)
    both = [("image", ("1.png", PNG)), ("image[]", ("2.png", PNG))]
    assert client.post("/v1/images/edits", files=both, data={"prompt": "p"}).status_code == 400
    one = [("image", ("1.png", PNG))]
    r = client.post("/v1/images/edits", files=one, data={"prompt": "p", "aspect_ratio": "2:1"})
    assert r.status_code == 400 and "16:9" in r.json()["detail"]
    two = [("image", ("1.png", PNG)), ("image", ("2.png", PNG))]
    for task in ("remove_background", "style"):
        r = client.post("/v1/images/edits", files=two, data={"prompt": "p", "task": task})
        assert r.status_code == 400, task
    assert edits == []


def test_one_image_without_aspect_is_unchanged(edits):
    r = client.post("/v1/images/edits", files={"image": ("a.png", PNG)}, data={"prompt": "p"})
    assert r.status_code == 200
    assert edits[0].extra_images_base64 == [] and edits[0].aspect_ratio is None
    assert catalog.for_task(None, "edit").max_images == 5
