"""GPUs by number or, optionally, by UUID (backend/gpu_ids.py, 2026-10-30: a third card may shift the numbers)."""
import pytest
from fastapi.testclient import TestClient

from backend import gpu_ids

T = {0: "GPU-ec716b3f-07d0-4f1c-0101-fb724ee0e98c", 1: "GPU-ae73ed6b-a062-c1a1-e400-3704740b6016"}


def test_a_number_stays_a_number():
    assert gpu_ids.resolve(1, T) == 1 and gpu_ids.resolve("1", T) == 1 and gpu_ids.resolve(None, T) is None


def test_a_uuid_or_a_unique_prefix_names_its_card():
    assert gpu_ids.resolve(T[1], T) == 1
    assert gpu_ids.resolve("gpu-ae73ed6b", T) == 1                 # case doesn't matter
    with pytest.raises(ValueError) as e:
        gpu_ids.resolve("GPU-ffff0000", T)
    assert T[0] in str(e.value)                                     # says which GPUs exist
    with pytest.raises(ValueError):
        gpu_ids.resolve("GPU-", T)                                  # too short to mean anything
    with pytest.raises(ValueError):
        gpu_ids.resolve(True, T)
    with pytest.raises(ValueError) as e:
        gpu_ids.resolve("GPU-eeee", {0: "GPU-eeee1", 1: "GPU-eeee2"})
    assert "more than one" in str(e.value)


def test_processes_are_pinned_by_uuid_and_read_back():
    assert gpu_ids.visible_devices(1, T) == T[1]
    assert gpu_ids.visible_devices(1, {}) == "1"                    # no nvidia-smi: the number
    assert gpu_ids.from_visible_devices(T[0], T) == 0
    assert gpu_ids.from_visible_devices("1", T) == 1
    assert gpu_ids.from_visible_devices("", T) is None and gpu_ids.from_visible_devices("0,1", T) is None
    assert gpu_ids.key(0, T) == T[0] and gpu_ids.key(3, T) == "gpu3"


def test_requests_may_name_a_gpu_by_uuid(monkeypatch):
    from backend.main import app
    from backend.routers import coordination
    monkeypatch.setattr(gpu_ids, "table", lambda: dict(T))
    seen = {}

    async def reserve(gpu, need, purpose):
        seen["gpu"] = gpu
        return None, [], "no room (test)"
    monkeypatch.setattr(coordination.MANAGER, "reserve", reserve)

    async def status():
        return {"gpus": []}
    monkeypatch.setattr(coordination, "fetch_gpu_status", status)
    c = TestClient(app, client=("127.0.0.1", 50000))
    r = c.post("/api/coordination/gpu-preflight", json={"purpose": "t", "gpu": T[1][:12], "need_mb": 10})
    assert r.status_code == 507 and seen["gpu"] == 1 and r.json()["gpu_uuid"] == T[1]
    r = c.post("/api/coordination/gpu-preflight", json={"purpose": "t", "gpu": "GPU-ffffffff", "need_mb": 10})
    assert r.status_code == 422 and T[0] in r.text
