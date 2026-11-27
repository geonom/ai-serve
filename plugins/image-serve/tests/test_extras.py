"""Local extra modules (image_serve/extra_<kind>_<name>.py, 2026-11-27): models the owner keeps on this machine only
(their licence doesn't allow publishing them) live in files of their own that the public tree leaves out. The core
loads whatever is there and works the same without them."""
import sys
import types

import pytest

from image_serve import catalog, comfy, extras, video


def test_load_imports_the_matching_modules_in_name_order(monkeypatch):
    names = ["extra_video_b", "extra_image_a", "extra_video_a", "video"]
    monkeypatch.setattr(extras, "_names", lambda: names)
    for n in names:
        monkeypatch.setitem(sys.modules, f"image_serve.{n}", types.ModuleType(n))
    assert [m.__name__ for m in extras.load("extra_video_")] == ["extra_video_a", "extra_video_b"]
    assert extras.load("extra_audio_") == []


def test_the_package_lists_its_own_modules():
    assert "video" in extras._names() and "catalog" in extras._names()


def _fake_family(monkeypatch):
    m = video.VideoModel("fake-1", "Fake", "fake", ("text_to_video",), (("unet", "fake.gguf"),), "a fake", "MIT",
                         24, 1, 4, 2, ("640x360",))
    monkeypatch.setattr(video, "CATALOG", video.CATALOG + (m,))
    monkeypatch.setitem(video.FAMILIES, "fake", video.Family(frames=lambda m, s: 99,
                                                             workflow=lambda r, prefix, sage, image: {"fake": prefix}))
    return m


def test_a_family_from_an_extra_module_brings_its_frames_and_graph(monkeypatch, tmp_path):
    m = _fake_family(monkeypatch)
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "fake.gguf").write_bytes(b"x")
    assert video.frame_count(m, 2) == 99
    r = video.parse_request({"prompt": "x", "model": "fake-1"}, str(tmp_path))
    assert video.workflow(r, None, "p") == {"fake": "p"}
    assert video.frame_count(video.resolve("ltx-2.5"), 4) == 97          # the built-in families are unchanged


def test_an_image_model_with_its_own_graph(monkeypatch):
    seen = {}

    class Graph:
        purpose, label, max_images = "fake-image", "Fake Image", 2

        @staticmethod
        def matches(name):
            return (name or "").startswith("fake-image")

        @staticmethod
        def build(image_names, prompt, model_name, seed, aspect_ratio=None):
            seen.update(images=image_names, prompt=prompt, seed=seed)
            return {"g": 1}

    monkeypatch.setattr(comfy, "EXTRA_IMAGE_GRAPHS", [Graph])
    assert comfy.extra_image_graph("fake-image.gguf") is Graph and comfy.extra_image_graph("z-image") is None
    req = comfy.ComfyUIGenerateRequest(prompt="a cat", model_name="fake-image.gguf", seed=5)
    assert comfy._build_patched_generate_workflow(req) == {"g": 1} and seen == {"images": [], "prompt": "a cat", "seed": 5}

