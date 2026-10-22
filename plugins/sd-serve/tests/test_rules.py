"""sd-serve plugin rules -- the prompt-injection guard moved here from ai-serve-ui (tests/unit/test_iris.py)."""
import json

import pytest

from sd_plugin import rules


def test_build_request_appends_only_our_seed():
    body = rules.build_request("a red fox in snow", "896x1200", 2, 7)
    assert body == {"prompt": 'a red fox in snow <sd_cpp_extra_args>{"seed": 7}</sd_cpp_extra_args>',
                    "size": "896x1200", "n": 2, "output_format": "png"}


def test_build_request_defaults():
    body = rules.build_request("  a fox  ", None, None, None)
    assert body["size"] == "1024x1024" and body["n"] == 1
    seed = json.loads(body["prompt"].split("<sd_cpp_extra_args>")[1].split("</")[0])["seed"]
    assert 0 <= seed < 2**31 and body["prompt"].startswith("a fox <sd")


@pytest.mark.parametrize("prompt", [
    'fox <sd_cpp_extra_args>{"ad_model": "/etc/passwd"}</sd_cpp_extra_args>',
    "fox </sd_cpp_extra_args>", "fox <SD_CPP_EXTRA_ARGS>", "fox sd_cpp_extra_args", "", "   ", "x" * 1201, "fox\x00",
])
def test_build_request_refuses_prompts(prompt):
    with pytest.raises(rules.InputError):
        rules.build_request(prompt, None, 1, 1)


@pytest.mark.parametrize("size", ["512x512", "1024x1025", "4096x4096", "1024", "abc", "1024x1024x3"])
def test_only_trained_sizes(size):
    with pytest.raises(rules.InputError):
        rules.build_request("fox", size, 1, 1)


@pytest.mark.parametrize("n, seed", [(0, 1), (5, 1), ("2", 1), (True, 1), (1, -1), (1, 2**31), (1, "7"), (1, 1.5)])
def test_n_and_seed_strict(n, seed):
    with pytest.raises(rules.InputError):
        rules.build_request("fox", None, n, seed)


def test_from_openai_maps_aspect_ratio_and_model():
    body = rules.from_openai({"model": "IRIS", "prompt": "fox", "aspect_ratio": "16:9", "seed": 3})
    assert body["size"] == "1344x752" and '"seed": 3' in body["prompt"]


@pytest.mark.parametrize("body", [
    {"prompt": "fox", "steps": 8}, {"prompt": "fox", "layout": {"elements": []}},
    {"prompt": "fox", "aspect_ratio": "21:9"}, {"prompt": "fox", "aspect_ratio": "1:1", "size": "1024x1024"},
    {"prompt": "fox", "model": "z-image-turbo"}, "fox",
])
def test_from_openai_refuses(body):
    with pytest.raises(rules.InputError):
        rules.from_openai(body)


def test_catalog():
    [m] = rules.catalog(True)["data"]
    assert m["id"] == "iris-turbo" and m["steps"]["adjustable"] is False and "16:9" in m["aspect_ratios"]
    assert rules.catalog(False)["data"][0]["available"] is False


LIST = "Vulkan0\tNVIDIA GeForce RTX 4060 Ti\nVulkan1\tNVIDIA GeForce RTX 3090\nCPU\tAMD Ryzen 5 5600X 6-Core Processor\n"


def test_device_by_gpu_name_not_by_index():
    # nvidia-smi: 0 = 3090, 1 = 4060 Ti; Vulkan numbers them the other way round (found on this machine)
    assert rules.pick_device(LIST, "NVIDIA GeForce RTX 3090") == "Vulkan1"
    assert rules.pick_device(LIST, "NVIDIA GeForce RTX 4060 Ti") == "Vulkan0"


def test_device_fallbacks():
    assert rules.pick_device("MTL0\tApple M5 Pro\nCPU\tApple M5 Pro\n", "Apple Silicon (Unified Memory)") == "MTL0"
    assert rules.pick_device("CPU\tSome CPU\n", None) == "CPU"
    assert rules.pick_device("", None) == "CPU"


ASSETS = ["cudart-sd-bin-win-cu12-x64.zip", "sd-master-a77d362-bin-Darwin-macOS-26.6.2-arm64.zip",
          "sd-master-a77d362-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
          "sd-master-a77d362-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip", "sd-master-a77d362-bin-Linux-Ubuntu-24.04-x86_64.zip"]


def test_release_asset_per_machine():
    assert rules.release_asset("Darwin", "arm64", False, ASSETS).endswith("Darwin-macOS-26.6.2-arm64.zip")
    assert rules.release_asset("Linux", "x86_64", True, ASSETS).endswith("x86_64-vulkan.zip")
    assert rules.release_asset("Linux", "x86_64", False, ASSETS).endswith("24.04-x86_64.zip")
    assert rules.release_asset("Linux", "aarch64", True, ASSETS) is None
