"""MOSS-Image edits with akhaliq's MOSS-Image-Edit-HQ LoRA, v5 (2026-11-23, owner via V-Noveller; Apache-2.0,
trained on MOSS-Image-SFT). ComfyUI's native path, as the model card's "Option A": the MOSS main model, the LoRA in
its ai-toolkit format through LoraLoaderModelOnly, TextEncodeMOSSImageEdit (ComfyUI >= v0.43) with
edit_format = ai-toolkit for the positive and the negative (empty prompt), each reference scaled to ~1 MP first (the
node uses them at the size they come in), SLG on single layer 22, KSampler euler 50 steps cfg 3 (the card's
settings). The output takes the first picture's shape unless an aspect ratio is asked for."""
import pytest

from image_serve import catalog, comfy

LORA = "MOSS/moss-edit-hq-armV5.safetensors"


def _by_type(wf, t):
    return [(nid, n) for nid, n in wf.items() if n["class_type"] == t]


def _one(wf, t):
    (_, n), = _by_type(wf, t)
    return n


def test_catalog_row():
    m = catalog.resolve("moss-edit-v5")
    assert m.tasks == ("edit",) and m.max_images == comfy.MOSS_EDIT_MAX_IMAGES
    assert m.steps_default == 50 and m.steps_adjustable
    assert comfy.is_moss_edit(m.file) and not comfy.is_moss_edit("moss_image_sft_int8_convrot.safetensors")
    assert catalog.DEFAULTS["edit"] == "qwen-edit-2511"     # an extra choice, not the new default
    assert catalog.resolve("moss-edit") is m


def test_available_needs_every_file(tmp_path, monkeypatch):
    from image_serve import config
    monkeypatch.setattr(config, "comfy_ready", lambda: True)
    m = catalog.resolve("moss-edit-v5")
    assert not catalog.available(m, str(tmp_path))
    for rel in m.needs:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    assert catalog.available(m, str(tmp_path))
    (tmp_path / m.needs[0]).unlink()
    assert not catalog.available(m, str(tmp_path))


def test_two_pictures_follow_the_model_card():
    wf = comfy.build_moss_edit_workflow(["a.png", "b.png"], prompt="put them on a beach", lora=LORA, seed=7)
    unet = _one(wf, "UNETLoader")["inputs"]
    assert unet["unet_name"] == comfy.MOSS_EDIT_MAIN_MODEL
    lora = _one(wf, "LoraLoaderModelOnly")["inputs"]
    assert lora["lora_name"] == LORA and lora["strength_model"] == 1.0
    encoders = _by_type(wf, "TextEncodeMOSSImageEdit")
    assert len(encoders) == 2
    prompts = sorted(n["inputs"]["prompt"] for _, n in encoders)
    assert prompts == ["", "put them on a beach"]
    loads = {nid: n["inputs"]["image"] for nid, n in _by_type(wf, "LoadImage")}
    for _, n in encoders:
        i = n["inputs"]
        assert i["edit_format"] == "ai-toolkit" and i["vae"][0] in dict(_by_type(wf, "VAELoader"))
        refs = [i[k] for k in sorted(k for k in i if k.startswith("images.image_"))]
        assert len(refs) == 2
        # each reference is a scaled copy of its upload, in order
        scaled = [wf[r[0]] for r in refs]
        assert all(s["class_type"] == "ImageScaleToTotalPixels" for s in scaled)
        assert [loads[s["inputs"]["image"][0]] for s in scaled] == ["a.png", "b.png"]
    slg = _one(wf, "SkipLayerGuidanceDiTSimple")["inputs"]
    assert slg["single_layers"] == "22" and slg["double_layers"] == ""
    ks = _one(wf, "KSampler")["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["seed"]) == (50, 3.0, "euler", 7)
    assert ks["model"][0] in dict(_by_type(wf, "SkipLayerGuidanceDiTSimple"))
    latent = _one(wf, "EmptyFlux2LatentImage")["inputs"]
    assert wf[latent["width"][0]]["class_type"] == "GetImageSize"   # the first picture's shape


def test_aspect_ratio_and_steps():
    wf = comfy.build_moss_edit_workflow(["a.png"], prompt="p", lora=LORA, seed=1, steps=30, aspect_ratio="16:9")
    latent = _one(wf, "EmptyFlux2LatentImage")["inputs"]
    assert (latent["width"], latent["height"]) == comfy.MOSS_EDIT_SIZES["16:9"]
    assert _one(wf, "KSampler")["inputs"]["steps"] == 30
    assert not _by_type(wf, "GetImageSize")
    assert set(comfy.MOSS_EDIT_SIZES) == set(comfy.EDIT_SIZES)
    # no reference longer than the frame is short (else people keep their size and lose head and feet)
    (_, cap), = _by_type(wf, "ImageScaleToMaxDimension")
    assert cap["inputs"]["largest_size"] == min(comfy.MOSS_EDIT_SIZES["16:9"])   # the edits endpoint checks against EDIT_SIZES


def test_limits():
    with pytest.raises(ValueError):
        comfy.build_moss_edit_workflow([], prompt="p", lora=LORA, seed=1)
    with pytest.raises(ValueError):
        comfy.build_moss_edit_workflow(["a.png"] * (comfy.MOSS_EDIT_MAX_IMAGES + 1), prompt="p", lora=LORA, seed=1)
    with pytest.raises(ValueError):
        comfy.build_moss_edit_workflow(["a.png"], prompt="p", lora=LORA, seed=1, aspect_ratio="2:1")
