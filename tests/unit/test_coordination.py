"""POST /api/coordination/gpu-preflight -- asked by tts-serve (and future
out-of-process backends) before a cold start: the gateway stays the GPU owner
(Strata guard). tts-serve evicts its own idle engines itself."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import coordination

client = TestClient(app, client=("127.0.0.1", 50000))  # this machine (access policy)


@pytest.fixture
def gpus(monkeypatch):
    state = {"free": 9000}

    async def fetch():
        return {"gpus": [{"index": 0, "free": state["free"], "total": 24576}]}

    monkeypatch.setattr(coordination, "fetch_gpu_status", fetch)
    return state


def _pre(**body):
    body = {"purpose": "tts:cosyvoice", "gpu": 0, "need_mb": 4212, **body}
    return client.post("/api/coordination/gpu-preflight", json=body)


def test_reports_free_vram(gpus):
    r = _pre()
    assert r.status_code == 200 and r.json()["free_mb"] == 9000 and r.json()["granted"]


def test_strata_refusal_is_507(gpus, monkeypatch):
    """While Strata holds the GPUs exclusively the manager answers 507 -- passed on."""
    async def held(gpu, need, purpose):
        raise HTTPException(status_code=507, detail=f"Strata holds the GPUs exclusively -- {purpose} can't load now")
    monkeypatch.setattr(coordination.MANAGER, "reserve", held)
    r = _pre()
    assert r.status_code == 507 and "tts:cosyvoice" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"gpu": "x"}, {"need_mb": -1}, {"purpose": ""}, {"purpose": "x" * 200}])
def test_invalid(gpus, body):
    assert _pre(**body).status_code == 422


def test_not_reachable_through_a_tunnel(gpus):
    r = client.post("/api/coordination/gpu-preflight", json={"purpose": "x", "gpu": 0, "need_mb": 1},
                    headers={"Tailscale-Funnel-Request": "?1"})
    assert r.status_code == 404


def test_preflight_reserves_and_confirm_binds_to_pid(gpus, monkeypatch):
    """Every preflight is a reservation (fast yes/no, user decision 2026-10-16): the
    backend gets an id it confirms with its worker's PID within 5 s; older backends
    that only read free_mb keep working (their hold just expires)."""
    from backend.cluster.consumers import Reservation
    calls, confirmed = [], []

    async def reserve(gpu, need, purpose):
        calls.append((gpu, need, purpose))
        return Reservation(id="r7", gpu=gpu, mb=need, purpose=purpose, created=0), ["comfyui"], ""
    monkeypatch.setattr(coordination.MANAGER, "reserve", reserve)
    monkeypatch.setattr(coordination.MANAGER, "confirm", lambda rid, pid: confirmed.append((rid, pid)) or True)
    r = _pre(purpose="stt:whisperx", need_mb=6000)
    body = r.json()
    assert r.status_code == 200 and body["granted"] and body["reservation_id"] == "r7" and body["hold_s"] == 5
    assert "free_mb" in body and calls == [(0, 6000, "stt:whisperx")]
    r = client.post("/api/coordination/reservations/r7/confirm", json={"pid": 4242})
    assert r.status_code == 200 and confirmed == [("r7", 4242)]


def test_refused_preflight_says_why(gpus, monkeypatch):
    async def reserve(gpu, need, purpose):
        return None, [], "GPU 0: held by gemma (busy, 9000 MB)"
    monkeypatch.setattr(coordination.MANAGER, "reserve", reserve)
    body = _pre(need_mb=20000).json()
    assert body["granted"] is False and "gemma" in body["reason"] and body["reservation_id"] is None


def test_confirm_late_or_unknown(gpus, monkeypatch):
    monkeypatch.setattr(coordination.MANAGER, "confirm", lambda rid, pid: False)
    assert client.post("/api/coordination/reservations/r9/confirm", json={"pid": 4242}).status_code == 410
    assert client.post("/api/coordination/reservations/r9/confirm", json={"pid": "x"}).status_code == 422


def test_release(gpus, monkeypatch):
    released = []
    monkeypatch.setattr(coordination.MANAGER, "release", lambda rid: released.append(rid))
    assert client.delete("/api/coordination/reservations/r7").status_code == 200 and released == ["r7"]


def test_exclusive_preflight_takes_every_gpu(gpus, monkeypatch):
    """A plugin with gpu "exclusive" (e.g. Strata through its adapter) asks for every GPU at once (docs/plugins.md):
    yes = one reservation per GPU, confirmed with the started process' PID, lasting as long as it lives."""
    from backend.cluster.consumers import Reservation
    asked = []

    async def excl(gpu_list, purpose):
        asked.append((gpu_list, purpose))
        return [Reservation(id=f"r{g}", gpu=g, mb=0, purpose=purpose, created=0.0, exclusive=True)
                for g in gpu_list], ["comfyui"], ""
    monkeypatch.setattr(coordination.MANAGER, "reserve_exclusive", excl)
    r = client.post("/api/coordination/gpu-exclusive", json={"purpose": "Strata"})
    assert r.status_code == 200, r.text
    assert r.json() == {"granted": True, "reservation_ids": ["r0"], "unloaded": ["comfyui"], "reason": None}
    assert asked == [([0], "Strata")]


def test_exclusive_preflight_refusal_is_a_fast_no(gpus, monkeypatch):
    async def excl(gpu_list, purpose):
        return None, [], "GPU 0: CosyVoice 3 is busy"
    monkeypatch.setattr(coordination.MANAGER, "reserve_exclusive", excl)
    r = client.post("/api/coordination/gpu-exclusive", json={"purpose": "Strata"})
    assert r.status_code == 200 and r.json()["granted"] is False and "busy" in r.json()["reason"]
    assert client.post("/api/coordination/gpu-exclusive", json={"purpose": ""}).status_code == 422


def test_preflight_can_leave_the_gpu_choice_to_the_manager(gpus, monkeypatch):
    """A plugin that picks its GPU at process start (sd-server, ComfyUI) asks without `gpu`: the manager picks the GPU
    with room (choose_gpu: start / stay / move) and reserves there; `mode` "move" = restart on the new GPU."""
    from backend.cluster.consumers import Reservation
    asked = []

    async def room():
        return {0: 20000, 1: 3000}

    async def reserve(gpu, need, purpose):
        asked.append((gpu, need, purpose))
        return Reservation(id="r9", gpu=gpu, mb=need, purpose=purpose, created=0.0), [], ""
    monkeypatch.setattr(coordination.MANAGER, "room", room)
    monkeypatch.setattr(coordination.MANAGER, "reserve", reserve)
    r = client.post("/api/coordination/gpu-preflight", json={"purpose": "iris", "need_mb": 7000, "current_gpu": 1})
    assert r.status_code == 200, r.text
    assert (r.json()["gpu"], r.json()["mode"], r.json()["granted"]) == (0, "move", True)
    assert asked == [(0, 7000, "iris")]
    r = client.post("/api/coordination/gpu-preflight", json={"purpose": "iris", "need_mb": 7000})
    assert (r.json()["gpu"], r.json()["mode"]) == (0, "start")
    r = client.post("/api/coordination/gpu-preflight", json={"purpose": "iris", "need_mb": 2000, "current_gpu": 1,
                                                              "movable": False})
    assert (r.json()["gpu"], r.json()["mode"]) == (1, "stay")


def test_room_per_gpu(gpus, monkeypatch):
    async def room():
        return {0: 20000, 1: 3000}
    monkeypatch.setattr(coordination.MANAGER, "room", room)
    assert client.get("/api/coordination/room").json() == {"room": {"0": 20000, "1": 3000}}
