"""MOSS plugin rules: variants, catalog, untrusted request -> workflow (moved from ai-serve-ui)."""
import pytest

from image_serve import moss as rules


def test_every_variant_is_listed_with_its_steps():
    rows = {r["id"]: r for r in rules.catalog(lambda v: v != "moss-bf16")["data"]}
    assert set(rows) == {"moss", "moss-bf16", "moss-turbo-8", "moss-turbo-8-bf16", "moss-turbo-4", "moss-turbo-4-bf16"}
    assert rows["moss"]["steps"]["default"] == 30 and rows["moss-turbo-4"]["steps"]["default"] == 4
    assert rows["moss"]["layout"] and rows["moss"]["text_in_image"] and not rows["moss-bf16"]["available"]
    assert rows["moss"]["commercial_use"] is True and rows["moss-turbo-4"]["commercial_use"] is True


def test_files_of():
    assert rules.files_of("moss") == ["diffusion_models/moss_image_sft_int8_convrot.safetensors"]
    assert rules.files_of("moss-turbo-4-bf16")[1] == "loras/MOSS/moss_image_sft_4step_v2.0.safetensors"


def test_from_openai_with_aspect_and_layout():
    variant, prompt, elements, w, h, n, seed, steps = rules.from_openai({
        "model": "MOSS-Turbo-8", "prompt": " a poster ", "aspect_ratio": "9:16", "n": 2, "seed": 5, "steps": 8,
        "layout": [{"box": [0.1, 0.1, 0.9, 0.3], "kind": "text", "caption": "title", "text": "Hallo"}]})
    assert (variant, prompt, w, h, n, seed, steps) == ("moss-turbo-8", "a poster", 720, 1280, 2, 5, 8)
    assert len(elements) == 1


@pytest.mark.parametrize("body", [
    {"prompt": "x", "model": "z-image-turbo"}, {"prompt": ""}, {"prompt": "x", "aspect_ratio": "21:9"},
    {"prompt": "x", "aspect_ratio": "1:1", "size": "1024x1024"}, {"prompt": "x", "n": 5}, {"prompt": "x", "n": "2"},
    {"prompt": "x", "steps": 0}, {"prompt": "x", "steps": 61}, {"prompt": "x", "seed": -1},
    {"prompt": "x", "layout": [{"box": [0, 0, 2, 2]}]}, "x",
])
def test_from_openai_refuses(body):
    with pytest.raises(rules.InputError):
        rules.from_openai(body)


def test_workflow_turbo_lora_and_base():
    wf = rules.build_workflow("a fox", [], 1024, 768, seed=7, variant="moss-turbo-4-bf16")
    assert wf["1"]["inputs"]["unet_name"] == rules.BF16 and wf["7"]["inputs"]["steps"] == 4
    assert wf["7"]["inputs"]["cfg"] == 1.0 and wf["7"]["inputs"]["model"] == ["10", 0]
    assert wf["10"]["inputs"]["lora_name"] == "MOSS/moss_image_sft_4step_v2.0.safetensors"
    assert (wf["6"]["inputs"]["width"], wf["6"]["inputs"]["height"], wf["7"]["inputs"]["seed"]) == (1024, 768, 7)
    base = rules.build_workflow("a fox", [], 1024, 1024, variant="moss")
    assert "10" not in base and base["7"]["inputs"]["cfg"] == 3.0 and base["7"]["inputs"]["steps"] == 30


def test_main_model_int8_with_cuda_gguf_without():
    """The compact variants run on int8 with an NVIDIA GPU (fast kernels) and on the Q5 GGUF without one (int8 is
    unusably slow there); when only one of the two files is there, that one."""
    none = lambda f: False
    assert rules.main_model("moss", none, cuda=True) == rules.INT8
    assert rules.main_model("moss-turbo-4", none, cuda=False) == rules.GGUF
    assert rules.main_model("moss", lambda f: f == rules.GGUF, cuda=True) == rules.GGUF
    assert rules.main_model("moss", lambda f: f == rules.INT8, cuda=False) == rules.INT8
    both = lambda f: True
    assert rules.main_model("moss", both, cuda=True) == rules.INT8 and rules.main_model("moss", both, cuda=False) == rules.GGUF
    assert rules.main_model("moss-turbo-8-bf16", both, cuda=False) == rules.BF16
    assert rules.label("moss-turbo-4", rules.GGUF) == "MOSS Turbo 4 v2.0 (GGUF Q5)"
    assert rules.label("moss", rules.INT8) == "MOSS-Image (int8)" and rules.label("moss-bf16", rules.BF16) == "MOSS-Image bf16"
    assert rules.files_of("moss-turbo-4", rules.GGUF)[0] == f"diffusion_models/{rules.GGUF}"


def test_workflow_on_the_gguf_uses_the_gguf_loader():
    wf = rules.build_workflow("a fox", [], 1024, 1024, variant="moss-turbo-4", unet=rules.GGUF)
    assert wf["1"]["class_type"] == "UnetLoaderGGUF" and wf["1"]["inputs"] == {"unet_name": rules.GGUF}
    assert wf["10"]["inputs"]["model"] == ["1", 0] and wf["7"]["inputs"]["model"] == ["10", 0]
    assert rules.build_workflow("a fox", [], 1024, 1024, variant="moss")["1"]["class_type"] == "UNETLoader"
