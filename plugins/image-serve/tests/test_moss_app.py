"""MOSS-Image inside image-serve (moved from the comfyui-moss plugin, 2026-10-22): its variants appear in the catalog
only where this ComfyUI can run them (v0.42+, or a ComfyUI setup.sh is still to clone), a generate request with a MOSS
model (and `layout`) becomes its own workflow, and its files are a download bundle."""
import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import comfy, config, moss
from image_serve.state import state

client = TestClient(plugin.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def nothing_running(monkeypatch):
    monkeypatch.setattr(state, "comfyui_process", None)
    monkeypatch.setattr(state, "comfyui_loaded_models", None)
    monkeypatch.setattr(comfy, "find_comfyui_process", lambda: None)


def _version(monkeypatch, v):
    monkeypatch.setattr(config, "comfy_version", lambda: v)


def test_version_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMFYUI_DIR", str(tmp_path / "none"))
    assert config.moss_possible()                       # setup.sh will clone the pinned (new enough) ComfyUI
    d = tmp_path / "ComfyUI"
    d.mkdir()
    (d / "main.py").write_text("")
    monkeypatch.setattr(config, "COMFYUI_DIR", str(d))
    (d / "comfyui_version.py").write_text('__version__ = "0.37.0"\n')
    assert config.comfy_version() == (0, 37, 0) and not config.moss_possible()
    (d / "comfyui_version.py").write_text('__version__ = "0.42.4"\n')
    assert config.moss_possible()
    (d / "comfyui_version.py").write_text("garbage")
    assert config.comfy_version() is None and not config.moss_possible()


def test_catalog_lists_moss_only_where_it_can_run(monkeypatch):
    _version(monkeypatch, (0, 42, 4))
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    rows = {r["id"]: r for r in client.get("/v1/images/models").json()["data"]}
    assert set(moss.VARIANTS) <= set(rows)
    r = rows["moss-turbo-4"]
    assert r["layout"] and r["text_in_image"] and r["tasks"] == ["generate"] and r["license"] == "Apache-2.0"
    assert r["default"] is False and isinstance(r["aliases"], list)
    edit_ids = {r["id"] for r in client.get("/v1/images/models?task=edit").json()["data"]}
    assert not edit_ids & set(moss.VARIANTS)
    monkeypatch.setattr(config, "moss_possible", lambda: False)
    rows = {r["id"] for r in client.get("/v1/images/models").json()["data"]}
    assert not rows & set(moss.VARIANTS)


def test_moss_counts_as_installed_only_with_every_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(plugin, "_models_root", lambda: str(tmp_path))
    assert not plugin.moss_installed("moss")
    for rel in moss.needed_files("moss"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert plugin.moss_installed("moss")
    assert not plugin.moss_installed("moss-bf16")       # its main model is another file
    assert "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors" in moss.needed_files("moss-turbo-4")


def test_generate_with_moss_builds_its_workflow(monkeypatch):
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(plugin, "moss_installed", lambda v: True)
    queued = []

    async def fake_queue(workflow, label):
        queued.append((workflow, label))
        return {"task_id": f"p{len(queued)}"}

    async def fake_await(pid):
        return "aW1n"
    monkeypatch.setattr(comfy, "queue_workflow", fake_queue)
    monkeypatch.setattr(comfy, "_await_comfyui_task", fake_await)
    r = client.post("/v1/images/generations", json={
        "prompt": "a poster", "model": "moss-turbo-4", "aspect_ratio": "16:9", "n": 2, "seed": 7,
        "layout": [{"box": [0.1, 0.1, 0.9, 0.3], "kind": "text", "caption": "title", "text": "HELLO"}]})
    assert r.status_code == 200, r.text
    assert [d["b64_json"] for d in r.json()["data"]] == ["aW1n", "aW1n"]
    wf, label = queued[0]
    assert label == "moss-turbo-4"
    assert wf["6"]["inputs"]["width"] == 1280 and wf["6"]["inputs"]["height"] == 720
    assert wf["7"]["inputs"]["steps"] == 4 and wf["7"]["inputs"]["seed"] == 7 and queued[1][0]["7"]["inputs"]["seed"] == 8
    assert wf["7"]["inputs"]["model"] == ["10", 0] and "HELLO" in wf["4"]["inputs"]["text"]


@pytest.mark.parametrize("body,why", [
    ({"prompt": "x", "model": "moss", "aspect_ratio": "2:1"}, "aspect"),
    ({"prompt": "x", "model": "moss", "size": "1024x1024", "aspect_ratio": "1:1"}, "either"),
    ({"prompt": "x", "model": "moss", "steps": 999}, "steps"),
    ({"prompt": "x", "model": "moss", "layout": [{"box": [0, 0, 2, 1]}]}, "box"),
])
def test_moss_requests_are_checked(monkeypatch, body, why):
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(plugin, "moss_installed", lambda v: True)
    r = client.post("/v1/images/generations", json=body)
    assert r.status_code == 400 and why in r.json()["detail"].lower()


def test_moss_refused_when_missing_or_unsupported(monkeypatch):
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(plugin, "moss_installed", lambda v: False)
    r = client.post("/v1/images/generations", json={"prompt": "x", "model": "moss"})
    assert r.status_code == 400 and "not installed" in r.json()["detail"]
    monkeypatch.setattr(config, "moss_possible", lambda: False)
    monkeypatch.setattr(config, "comfy_version", lambda: (0, 37, 0))
    r = client.post("/v1/images/generations", json={"prompt": "x", "model": "moss"})
    assert r.status_code == 400 and "0.42" in r.json()["detail"]


def test_moss_bundles_are_offered_where_moss_can_run(monkeypatch):
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    items = {i["id"]: i for i in client.get("/plugin/install").json()["items"]}
    assert {"moss", "moss-bf16"} <= set(items) and items["moss"]["state"] in ("missing", "installed")


def test_an_older_comfyui_of_someone_elses_is_named_not_hidden(monkeypatch):
    """COMFYUI_DIR points at a ComfyUI image-serve doesn't own and it's too old: MOSS isn't hidden (the page would then
    only say "not available"), it's listed with the reason -- and never upgraded behind the owner's back."""
    monkeypatch.setattr(config, "moss_possible", lambda: False)
    monkeypatch.setattr(config, "comfy_version", lambda: (0, 37, 0))
    items = {i["id"]: i for i in client.get("/plugin/install").json()["items"]}
    assert items["moss"]["state"] == "failed"
    assert "v0.42" in items["moss"]["error"] and "COMFYUI_DIR" in items["moss"]["error"] and "0.37" in items["moss"]["error"]
    assert client.post("/plugin/install", json={"id": "moss"}).status_code == 409


def test_own_older_comfyui_counts_as_upgradable(tmp_path, monkeypatch):
    """image-serve's own ComfyUI (./ComfyUI, cloned by setup.sh) is brought to the pinned release by setup.sh."""
    own = tmp_path / "ComfyUI"
    own.mkdir()
    (own / "main.py").write_text("")
    (own / "comfyui_version.py").write_text('__version__ = "0.37.0"\n')
    monkeypatch.setattr(config, "HERE", tmp_path)
    monkeypatch.setattr(config, "COMFYUI_DIR", str(own))
    assert config.own_comfy() and config.moss_possible() and config.moss_needs_upgrade()
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "main.py").write_text("")
    (elsewhere / "comfyui_version.py").write_text('__version__ = "0.37.0"\n')
    monkeypatch.setattr(config, "COMFYUI_DIR", str(elsewhere))
    assert not config.own_comfy() and not config.moss_possible() and not config.moss_needs_upgrade()


def test_moss_download_upgrades_an_own_old_comfyui_first(monkeypatch):
    from image_serve import bundles
    steps = []
    monkeypatch.setattr(config, "moss_needs_upgrade", lambda: True)
    monkeypatch.setattr(plugin, "_setup_comfy", lambda: steps.append("setup"))
    monkeypatch.setattr(plugin, "_restart_comfy_if_idle", lambda: steps.append("restart"))
    monkeypatch.setattr(bundles, "present", lambda b, root: False)
    monkeypatch.setattr(bundles, "download", lambda b, root, progress=None, opener=None: steps.append(b))
    plugin._install["moss"] = {"state": "downloading", "done": 0}
    plugin._run_install("moss")
    assert steps == ["setup", "restart", "moss"] and plugin._install["moss"]["state"] == "installed"


def test_bf16_download_brings_the_shared_files_first(monkeypatch):
    from image_serve import bundles
    got = []
    monkeypatch.setattr(bundles, "present", lambda b, root: False)
    monkeypatch.setattr(bundles, "download", lambda b, root, progress=None, opener=None: got.append(b))
    plugin._install["moss-bf16"] = {"state": "downloading", "done": 0}
    plugin._run_install("moss-bf16")
    assert got == ["moss", "moss-bf16"] and plugin._install["moss-bf16"]["state"] == "installed"


def test_without_nvidia_moss_runs_on_the_gguf(monkeypatch, tmp_path):
    """A Mac (no CUDA): the compact MOSS variants use the Q5 GGUF through ComfyUI-GGUF's loader, are listed as such,
    and the download offered is moss-gguf, not int8."""
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(plugin, "_models_root", lambda: str(tmp_path))
    for rel in moss.needed_files("moss-turbo-4", moss.GGUF):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert plugin.moss_unet("moss-turbo-4") == moss.GGUF and plugin.moss_installed("moss-turbo-4")
    rows = {r["id"]: r for r in client.get("/v1/images/models").json()["data"]}
    assert rows["moss-turbo-4"]["available"] and rows["moss-turbo-4"]["label"] == "MOSS Turbo 4 v2.0 (GGUF Q5)"
    monkeypatch.setattr(config, "gguf_knows_moss", lambda: False)       # ComfyUI-GGUF not patched yet
    assert not plugin.moss_installed("moss-turbo-4")
    seen = {}

    async def run(fn, *a, **kw):
        seen["wf"] = fn.__defaults__[0] if fn.__defaults__ else None
        return "aGk="
    monkeypatch.setattr(config, "gguf_knows_moss", lambda: True)
    monkeypatch.setattr(plugin, "_run", run)
    r = client.post("/v1/images/generations", json={"prompt": "a sign", "model": "moss-turbo-4"})
    assert r.status_code == 200, r.text
    assert seen["wf"]["1"]["class_type"] == "UnetLoaderGGUF"
    from image_serve import bundles
    monkeypatch.setattr(bundles, "present", lambda b, root: False)
    ids = {i["id"] for i in client.get("/plugin/install").json()["items"]}
    assert "moss-gguf" in ids and "moss" not in ids and "moss-bf16" in ids


def test_with_nvidia_the_int8_download_is_offered(monkeypatch):
    from image_serve import bundles
    monkeypatch.setattr(config, "moss_possible", lambda: True)
    monkeypatch.setattr(bundles, "present", lambda b, root: False)
    ids = {i["id"] for i in client.get("/plugin/install").json()["items"]}
    assert "moss" in ids and "moss-gguf" not in ids


def test_gguf_download_patches_comfyui_gguf_and_bf16_builds_on_it(monkeypatch):
    from image_serve import bundles
    steps = []
    monkeypatch.setattr(config, "has_cuda", lambda: False)
    monkeypatch.setattr(config, "moss_needs_upgrade", lambda: False)
    monkeypatch.setattr(config, "teach_gguf_moss", lambda: steps.append("patch") or True)
    monkeypatch.setattr(plugin, "_restart_comfy_if_idle", lambda: steps.append("restart"))
    monkeypatch.setattr(bundles, "present", lambda b, root: False)
    monkeypatch.setattr(bundles, "download", lambda b, root, progress=None, opener=None: steps.append(b))
    plugin._install["moss-gguf"] = {"state": "downloading", "done": 0}
    plugin._run_install("moss-gguf")
    assert steps == ["patch", "restart", "moss-gguf"] and plugin._install["moss-gguf"]["state"] == "installed"
    steps.clear()
    plugin._install["moss-bf16"] = {"state": "downloading", "done": 0}
    plugin._run_install("moss-bf16")
    assert steps == ["moss-gguf", "moss-bf16"]          # the shared files come with this machine's MOSS bundle


def test_gguf_patch_applies_on_top_of_the_qwen_patch(tmp_path, monkeypatch):
    """patches/comfyui-gguf-moss-image.patch applies to ComfyUI-GGUF at the pinned commit + the Qwen-Image 2.1 patch,
    and only to our own ComfyUI. Needs the pinned ComfyUI-GGUF checkout from setup.sh; skipped without network."""
    import subprocess
    own = tmp_path / "ComfyUI"
    g = own / "custom_nodes" / "ComfyUI-GGUF"
    r = subprocess.run(["git", "clone", "-q", "https://github.com/city96/ComfyUI-GGUF.git", str(g)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("no network")
    subprocess.run(["git", "-C", str(g), "checkout", "-q", "6ea2651e7"], check=True)
    subprocess.run(["git", "-C", str(g), "apply", str(config.HERE / "patches" / "comfyui-gguf-qwen-image21.patch")], check=True)
    monkeypatch.undo()                                   # the real gguf_knows_moss / own_comfy
    monkeypatch.setattr(config, "HERE", tmp_path)
    monkeypatch.setattr(config, "COMFYUI_DIR", str(own))
    monkeypatch.setattr(config, "GGUF_MOSS_PATCH", config.GGUF_MOSS_PATCH)
    assert not config.gguf_knows_moss()
    assert config.teach_gguf_moss() is True and config.gguf_knows_moss()
    assert config.teach_gguf_moss() is False             # already in
    assert "ModelMossImage" in (g / "tools" / "convert.py").read_text()
