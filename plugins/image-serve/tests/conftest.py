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
