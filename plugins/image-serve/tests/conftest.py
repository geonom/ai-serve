import pytest

from image_serve import catalog as _catalog

_REAL_ON_DISK = _catalog._on_disk   # taken before any test patches it


@pytest.fixture(autouse=True)
def _image_models_count_as_installed(monkeypatch):
    """A fresh checkout has no image checkpoints; tests that care about "not installed" patch _on_disk themselves."""
    from image_serve import bundles, config
    monkeypatch.setattr(config, "comfy_ready", lambda: True)            # ComfyUI counts as set up
    monkeypatch.setattr(_catalog, "_on_disk", lambda root, rel: True)
    monkeypatch.setattr(bundles, "present", lambda bundle, root: True)   # download bundles too
    monkeypatch.setattr(config, "has_cuda", lambda: True)               # an NVIDIA machine unless a test says not
    monkeypatch.setattr(config, "gguf_knows_moss", lambda: True)


@pytest.fixture
def real_on_disk():
    return _REAL_ON_DISK


@pytest.fixture(autouse=True)
def _no_gateway(monkeypatch):
    """Unit tests never reach a real ai-serve: every GPU-manager call fails to connect unless a test fakes it."""
    import httpx
    from image_serve import gpu

    async def down(method, path, **kw):
        raise httpx.ConnectError("no ai-serve in unit tests")
    monkeypatch.setattr(gpu, "_call", down)


@pytest.fixture(autouse=True)
def _room_for_jobs(monkeypatch):
    """Video / audio jobs ask for GPU room before they are accepted (comfy.check_room); tests that care fake a no."""
    from image_serve import comfy

    async def yes(workflow, label):
        return None
    monkeypatch.setattr(comfy, "check_room", yes)


@pytest.fixture(autouse=True)
def _no_live_comfyui(monkeypatch):
    """Unit tests never see the ComfyUI running on this machine (2026-11-27: a release test failed whenever someone
    was making a picture -- the busy check asked the live ComfyUI's /queue). Its port points at nothing, and the
    process search finds nothing unless a test fakes it."""
    from image_serve import comfy
    from image_serve.state import state
    monkeypatch.setattr(state, "comfyui_port", 9)          # discard port: refused at once
    monkeypatch.setattr(comfy, "find_comfyui_process", lambda *a, **kw: None)
