import os

# Before anything imports backend.main: unit tests must not write into the live
# logs/backend.log (their deliberate bad requests look like real incidents there).
os.environ.setdefault("LLAMA_SERVE_LOG_TO_FILE", "0")

import pytest


@pytest.fixture(autouse=True)
def _plugins_unreachable(monkeypatch):
    """Unit tests never talk to (or start) a real plugin on this machine (tts-serve, stt-serve, the door ...):
    every plugin call fails to connect and nothing is started. Tests that exercise a plugin install their own
    transport (registry._transport) or fakes on top."""
    import httpx

    from backend.plugins import registry

    def _down(request):
        raise httpx.ConnectError("plugins are unreachable in unit tests")

    async def _no_start(*a, **k):
        return False

    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(_down))
    monkeypatch.setattr(registry, "ensure_running", _no_start)
    from backend.plugins import external     # never this machine's own added model servers (model_servers.json)
    monkeypatch.setattr(external, "FILE", os.devnull + ".none")


@pytest.fixture(autouse=True)
def _no_cluster_devices(monkeypatch):
    """No enrolled devices unless a test says so -- otherwise a real node enrolled on
    this machine (cluster_nodes.json) would take requests in unrelated tests."""
    from backend.cluster import service
    monkeypatch.setattr(service.SERVICE, "has_nodes", lambda: False)


@pytest.fixture(autouse=True)
def _gpu_manager_never_touches_real_gpus(monkeypatch):
    """The real GPU manager measures this machine's GPUs and may unload real
    consumers (llama-servers, ComfyUI, TTS engines). Unit tests get one that grants
    every reservation and never unloads anything; tests of the manager itself build
    their own LocalGpuManager."""
    from backend.cluster import local
    from backend.cluster.consumers import Reservation

    async def reserve(gpu, need_mb, purpose):
        return Reservation(id="test", gpu=gpu, mb=need_mb, purpose=purpose, created=0), [], ""

    async def make_room(gpu, need_mb, purpose):
        return True, [], ""

    async def evict(*a, **kw):
        raise AssertionError("unit tests must not unload real GPU consumers")
    monkeypatch.setattr(local.MANAGER, "reserve", reserve)
    monkeypatch.setattr(local.MANAGER, "make_room", make_room)
    monkeypatch.setattr(local.MANAGER, "evict", evict)
    monkeypatch.setattr(local.MANAGER, "confirm", lambda rid, pid: True)

    async def reserve_exclusive(gpus, purpose):
        return [Reservation(id=f"x{g}", gpu=g, mb=0, purpose=purpose, created=0, exclusive=True) for g in gpus], [], ""

    async def raise_if_exclusive(gpu, purpose):
        return None
    monkeypatch.setattr(local.MANAGER, "reserve_exclusive", reserve_exclusive)

    async def snapshot():
        return [], [], {}
    monkeypatch.setattr(local.MANAGER, "snapshot", snapshot)
    monkeypatch.setattr(local.MANAGER, "raise_if_exclusive", raise_if_exclusive)


