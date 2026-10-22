"""Model bundles (image_serve/bundles.py) and the install endpoint: pinned files, checked by sha256 while downloading,
nothing kept on a mismatch; only a known bundle id starts a download, once."""
import hashlib
import io
import time

import pytest
from fastapi.testclient import TestClient

from image_serve import bundles


def test_every_model_bundle_is_pinned():
    for b in bundles.BUNDLES:
        fs = bundles.files_of(b)
        assert fs and all(len(f.sha256) == 64 and f.size > 0 and f.url.startswith("https://huggingface.co/") for f in fs)
    assert bundles.size_mb("z-image-turbo") > 9000            # model + text encoder + VAE


def _fake(content: bytes):
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    return lambda url: R(content)


def test_download_checks_the_hash(tmp_path, monkeypatch):
    data = b"model bytes"
    f = bundles.BundleFile("t", "o/r", "m.gguf", "unet", "m.gguf", len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(bundles, "FILES", (f,))
    got = []
    bundles.download("t", str(tmp_path), got.append, opener=_fake(data))
    assert (tmp_path / "unet" / "m.gguf").read_bytes() == data and sum(got) == len(data)
    assert bundles.present("t", str(tmp_path))


def test_a_wrong_file_is_never_kept(tmp_path, monkeypatch):
    f = bundles.BundleFile("t", "o/r", "m.gguf", "unet", "m.gguf", 5, hashlib.sha256(b"right").hexdigest())
    monkeypatch.setattr(bundles, "FILES", (f,))
    with pytest.raises(ValueError):
        bundles.download("t", str(tmp_path), opener=_fake(b"wrong"))
    assert list((tmp_path / "unet").iterdir()) == []


def test_install_endpoint(monkeypatch):
    from image_serve import app as plugin
    started = []
    monkeypatch.setattr(bundles, "present", lambda b, root: b == "z-image-turbo")
    monkeypatch.setattr(plugin, "_run_install", lambda b: started.append(b))
    monkeypatch.setattr(plugin, "_install", {})
    c = TestClient(plugin.app)
    items = {i["id"]: i for i in c.get("/plugin/install").json()["items"]}
    assert items["z-image-turbo"]["state"] == "installed" and items["qwen-edit-2511"]["state"] == "missing"
    assert items["qwen-edit-2511"]["size_mb"] > 15000 and items["qwen-edit-2511"]["license"] == "Apache-2.0"
    assert c.post("/plugin/install", json={"id": "../etc"}).status_code == 404
    assert c.post("/plugin/install", json={"id": "z-image-turbo"}).json()["state"] == "installed"   # nothing to do
    assert c.post("/plugin/install", json={"id": "qwen-edit-2511"}).json()["state"] == "downloading"
    c.post("/plugin/install", json={"id": "qwen-edit-2511"})          # already running
    time.sleep(0.1)
    assert started == ["qwen-edit-2511"]


def test_without_comfyui_it_is_offered_and_models_wait_for_it(monkeypatch):
    from image_serve import app as plugin, catalog, config
    monkeypatch.setattr(config, "comfy_ready", lambda: False)
    monkeypatch.setattr(plugin, "_install", {})
    started = []
    monkeypatch.setattr(plugin, "_run_comfy_setup", lambda: started.append("comfyui"))
    c = TestClient(plugin.app)
    items = {i["id"]: i for i in c.get("/plugin/install").json()["items"]}
    assert items["comfyui"]["state"] == "missing" and items["comfyui"]["license"] == "GPL-3.0"
    assert all(r["available"] is False for r in catalog.listing(task="all"))   # files alone can't run
    assert c.post("/plugin/install", json={"id": "comfyui"}).json()["state"] == "downloading"
    time.sleep(0.1)
    assert started == ["comfyui"]


def test_a_model_download_sets_up_comfyui_first(monkeypatch):
    from image_serve import app as plugin, config
    order = []
    monkeypatch.setattr(config, "comfy_ready", lambda: "setup" in order)
    monkeypatch.setattr(plugin, "_setup_comfy", lambda: order.append("setup"))
    monkeypatch.setattr(bundles, "download", lambda b, root, progress=None: order.append(f"download {b}"))
    monkeypatch.setattr(plugin, "_install", {"z-image-turbo": {"state": "downloading", "done": 0}})
    plugin._run_install("z-image-turbo")
    assert order == ["setup", "download z-image-turbo"] and plugin._install["z-image-turbo"]["state"] == "installed"
