"""FLUX.2 [klein] 4B as a second edit model (2026-10-26, owner via V-Noveller: an Apache-2.0 alternative to
Qwen-Image-Edit 2511, as a GGUF). Built like ComfyUI's own klein edit template: the text conditioning gets one
ReferenceLatent per input image (scaled to ~1 MP, VAE-encoded) on the positive and on the zeroed negative side,
CFGGuider at cfg 1 (distilled), Flux2Scheduler with 4 steps; the output has the first image's shape unless an aspect
ratio is asked for. Text encoder: the Qwen3-4B GGUF Z-Image Turbo already uses (CLIPLoaderGGUF type flux2)."""
import pytest

from image_serve import bundles, catalog, comfy


def _by_type(wf, t):
    return [(nid, n) for nid, n in wf.items() if n["class_type"] == t]


def _refs(wf, cond):
    """The images a conditioning chain references, innermost first."""
    out = []
    node = wf[cond[0]]
    while node["class_type"] == "ReferenceLatent":
        enc = wf[node["inputs"]["latent"][0]]
        scale = wf[enc["inputs"]["pixels"][0]]
        out.append(wf[scale["inputs"]["image"][0]]["inputs"]["image"])
        node = wf[node["inputs"]["conditioning"][0]]
    return list(reversed(out)), node


def test_catalog_row():
    m = catalog.resolve("flux2-klein-4b")
    assert m.tasks == ("edit",) and m.max_images == 4 and m.steps_default == 4
    assert m.file == "flux-2-klein-4b-Q8_0.gguf" and catalog.BUNDLE_OF[m.id] == "flux2-klein-4b"
    assert catalog.DEFAULTS["edit"] == "qwen-edit-2511"        # the default stays until V-Noveller's tests say otherwise
    names = {f.name for f in bundles.files_of("flux2-klein-4b")}
    assert names == {"flux-2-klein-4b-Q8_0.gguf", "Qwen3-4B-UD-Q6_K_XL.gguf", "flux2-vae.safetensors"}


def test_three_images_are_references_in_order_on_both_sides():
    wf = comfy.build_klein_edit_workflow(["a.png", "b.png", "c.png"], prompt="p", model_name="flux-2-klein-4b-Q8_0.gguf", seed=7)
    (_, guider), = _by_type(wf, "CFGGuider")
    assert guider["inputs"]["cfg"] == 1
    pos, root = _refs(wf, guider["inputs"]["positive"])
    assert pos == ["a.png", "b.png", "c.png"] and root["class_type"] == "CLIPTextEncode" and root["inputs"]["text"] == "p"
    neg, zero = _refs(wf, guider["inputs"]["negative"])
    assert neg == pos and zero["class_type"] == "ConditioningZeroOut"
    (_, clip), = _by_type(wf, "CLIPLoaderGGUF")
    assert clip["inputs"] == {"clip_name": "Qwen3-4B-UD-Q6_K_XL.gguf", "type": "flux2"}
    (_, unet), = _by_type(wf, "UnetLoaderGGUF")
    assert unet["inputs"]["unet_name"] == "flux-2-klein-4b-Q8_0.gguf"
    (_, noise), = _by_type(wf, "RandomNoise")
    assert noise["inputs"]["noise_seed"] == 7
    (_, sched), = _by_type(wf, "Flux2Scheduler")
    assert sched["inputs"]["steps"] == 4
    # no aspect ratio: the first (scaled) image's size
    size = wf[sched["inputs"]["width"][0]]
    assert size["class_type"] == "GetImageSize"
    assert wf[wf[size["inputs"]["image"][0]]["inputs"]["image"][0]]["inputs"]["image"] == "a.png"
    assert len(_by_type(wf, "SaveImage")) == 1
    assert comfy.comfy_model_files(wf) == {"flux-2-klein-4b-Q8_0.gguf", "Qwen3-4B-UD-Q6_K_XL.gguf", "flux2-vae.safetensors"}


def test_aspect_ratio_and_steps():
    wf = comfy.build_klein_edit_workflow(["a.png"], prompt="p", model_name="m.gguf", seed=1, steps=8, aspect_ratio="16:9")
    (_, sched), = _by_type(wf, "Flux2Scheduler")
    (_, latent), = _by_type(wf, "EmptyFlux2LatentImage")
    assert sched["inputs"]["steps"] == 8
    w, h = latent["inputs"]["width"], latent["inputs"]["height"]
    assert (w, h) == comfy.KLEIN_SIZES["16:9"] and w % 16 == 0 and h % 16 == 0
    assert set(comfy.KLEIN_SIZES) == set(comfy.EDIT_SIZES)


@pytest.mark.parametrize("names", [[], ["a"] * 5])
def test_one_to_four_images(names):
    with pytest.raises(ValueError):
        comfy.build_klein_edit_workflow(names, prompt="p", model_name="m.gguf", seed=1)


def test_a_safetensors_checkpoint_uses_the_core_loader():
    wf = comfy.build_klein_edit_workflow(["a.png"], prompt="p", model_name="flux-2-klein-4b.safetensors", seed=1)
    assert _by_type(wf, "UNETLoader") and not _by_type(wf, "UnetLoaderGGUF")
