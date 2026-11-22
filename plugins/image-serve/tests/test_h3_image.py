"""MiniMax-H3-Image (33B DiT, 2026-11-14, owner: as a GGUF to save disk and VRAM). Built like Comfy-Org's
image_minimax_h3_image_t2i template: MiniMaxH3ImageConditioning (ComfyUI >= v0.45) gives the positive conditioning
and the 24-channel latent; the negative is the zeroed positive; KSampler euler/simple, cfg 1, exactly 6 steps (the
distilled PDD heads are calibrated for 6); VAEDecode with the image VAE. Text encoder: the int8 Qwen3-VL-32B the H3
video model already uses (CLIPLoader type minimax). Edits: the same node with the pictures as image_1.., which the
prompt calls <Picture 1>..; without an aspect ratio the node takes image_1's shape."""
import pytest

from image_serve import catalog, comfy

GGUF = "MiniMax-H3-Image-Pruned-Q4_K_M.gguf"


def _by_type(wf, t):
    return [(nid, n) for nid, n in wf.items() if n["class_type"] == t]


def _one(wf, t):
    (_, n), = _by_type(wf, t)
    return n


def test_catalog_row_generates_and_edits_with_a_fixed_schedule():
    m = catalog.resolve("minimax-h3-image")
    assert m.file == GGUF and m.tasks == ("generate", "edit") and m.max_images >= 2
    assert m.steps_default == 6 and not m.steps_adjustable
    assert catalog.DEFAULTS["generate"] == "z-image-turbo" and catalog.DEFAULTS["edit"] == "qwen-edit-2511"
    with pytest.raises(catalog.CatalogError):
        catalog.check_steps(m, 20)
    catalog.check_steps(m, 6)
    assert comfy.is_h3_image(GGUF) and comfy.is_h3_image("minimax_h3_image_pruned_int8_convrot.safetensors")
    assert not comfy.is_h3_image("minimax_h3_fl2va_pruned-Q4_K.gguf")      # the video model


def test_text_to_image_follows_the_template():
    wf = comfy.build_h3_image_workflow([], prompt="a fox", model_name=GGUF, seed=7, aspect_ratio="16:9")
    cond = _one(wf, "MiniMaxH3ImageConditioning")
    assert cond["inputs"]["prompt"] == "a fox"
    assert (cond["inputs"]["width"], cond["inputs"]["height"]) == comfy.H3_IMAGE_SIZES["16:9"]
    assert not any(k.startswith("images.") for k in cond["inputs"])
    ks = _one(wf, "KSampler")["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"], ks["denoise"], ks["seed"]) == \
        (6, 1, "euler", "simple", 1, 7)
    cid = next(nid for nid, n in wf.items() if n["class_type"] == "MiniMaxH3ImageConditioning")
    assert ks["positive"] == [cid, 0] and ks["latent_image"] == [cid, 1]
    zero = wf[ks["negative"][0]]
    assert zero["class_type"] == "ConditioningZeroOut" and zero["inputs"]["conditioning"] == [cid, 0]
    assert _one(wf, "UnetLoaderGGUF")["inputs"]["unet_name"] == GGUF
    assert _one(wf, "CLIPLoader")["inputs"] == {"clip_name": comfy.H3_IMAGE_TEXT_ENCODER, "type": "minimax",
                                                "device": "default"}
    assert len(_by_type(wf, "SaveImage")) == 1
    assert comfy.comfy_model_files(wf) == {GGUF, comfy.H3_IMAGE_TEXT_ENCODER, comfy.H3_IMAGE_VAE}


def test_sizes_are_about_four_megapixels_on_the_32_px_grid():
    assert set(comfy.H3_IMAGE_SIZES) == set(comfy.EDIT_SIZES) == set(catalog.ASPECTS)
    for w, h in comfy.H3_IMAGE_SIZES.values():
        assert w % 32 == 0 and h % 32 == 0 and 3.6e6 <= w * h <= 4.3e6
    wf = comfy.build_h3_image_workflow([], prompt="p", model_name=GGUF, seed=1)
    cond = _one(wf, "MiniMaxH3ImageConditioning")["inputs"]
    assert (cond["width"], cond["height"]) == (2048, 2048)


def test_an_edit_passes_the_pictures_in_order_and_keeps_the_first_ones_shape():
    wf = comfy.build_h3_image_workflow(["a.png", "b.png"], prompt="put <Picture 2> in <Picture 1>",
                                       model_name=GGUF, seed=3)
    cond = _one(wf, "MiniMaxH3ImageConditioning")["inputs"]
    # Autogrow prefix inputs are named image_0, image_1, ... (ComfyUI's API key: "images.image_0")
    assert [wf[cond[f"images.image_{i}"][0]]["inputs"]["image"] for i in (0, 1)] == ["a.png", "b.png"]
    assert cond["width"] == 0 and cond["height"] == 0           # the node takes image_1's (scaled) shape
    assert cond["vae"][0] == next(nid for nid, n in wf.items() if n["class_type"] == "VAELoader")
    wf = comfy.build_h3_image_workflow(["a.png"], prompt="p", model_name=GGUF, seed=3, aspect_ratio="3:4")
    cond = _one(wf, "MiniMaxH3ImageConditioning")["inputs"]
    assert (cond["width"], cond["height"]) == comfy.H3_IMAGE_SIZES["3:4"]


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        comfy.build_h3_image_workflow([], prompt="p", model_name=GGUF, seed=1, aspect_ratio="2:1")
    with pytest.raises(ValueError):
        comfy.build_h3_image_workflow(["a"] * (comfy.H3_IMAGE_MAX_IMAGES + 1), prompt="p", model_name=GGUF, seed=1)


def test_the_generate_endpoint_routes_to_it():
    req = comfy.ComfyUIGenerateRequest(prompt="p", model_name=GGUF, aspect_ratio="9:16", seed=5)
    wf = comfy._build_patched_generate_workflow(req)
    cond = _one(wf, "MiniMaxH3ImageConditioning")["inputs"]
    assert (cond["width"], cond["height"]) == comfy.H3_IMAGE_SIZES["9:16"]
    assert _one(wf, "KSampler")["inputs"]["seed"] == 5


def test_a_safetensors_checkpoint_uses_the_core_loader():
    wf = comfy.build_h3_image_workflow([], prompt="p", model_name="minimax_h3_image_pruned_int8_convrot.safetensors", seed=1)
    assert _by_type(wf, "UNETLoader") and not _by_type(wf, "UnetLoaderGGUF")
