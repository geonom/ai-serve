"""Local GPU manager (backend/cluster/local.py) with faked measurements: it unloads
in the planned order, waits until the VRAM actually shows up as free, refuses busy
consumers, and keeps the owner's pins across restarts."""
import asyncio

import pytest

from backend.cluster import local
from backend.cluster.consumers import Consumer


def make(tmp_path, monkeypatch, free_after):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 2000}]
    cons = [Consumer(id="comfyui", kind="image", label="ComfyUI", root_pids={1}, last_used=1),
            Consumer(id="llm:8081", kind="llm", label="gemma", root_pids={2}, last_used=5),
            Consumer(id="tts:cosy", kind="tts", label="CosyVoice", root_pids={3}, busy=True)]
    att = {("comfyui", 0): 10000, ("llm:8081", 0): 9000, ("tts:cosy", 0): 3000}

    async def snapshot():
        return gpus, cons, att
    evicted = []

    async def evict(cid, consumers=None):
        evicted.append(cid)
    frees = iter(free_after)

    async def gpu_status():
        return {"gpus": [{"index": 0, "free": next(frees)}]}
    monkeypatch.setattr(m, "snapshot", snapshot)
    monkeypatch.setattr(m, "evict", evict)
    import backend.model_manager as mm
    monkeypatch.setattr(mm, "fetch_gpu_status", gpu_status)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(local.asyncio, "sleep", lambda s: real_sleep(0))
    return m, evicted


def test_make_room_unloads_lru_and_waits_for_vram(tmp_path, monkeypatch):
    m, evicted = make(tmp_path, monkeypatch, free_after=[2000, 5000, 12000])
    ok, done, _ = asyncio.run(m.make_room(0, 9000, "test"))
    assert ok and done == ["comfyui"] and evicted == ["comfyui"]


def test_pinned_is_skipped(tmp_path, monkeypatch):
    m, evicted = make(tmp_path, monkeypatch, free_after=[11000])
    m.set_pin("comfyui", True)
    ok, done, _ = asyncio.run(m.make_room(0, 9000, "test"))
    assert ok and done == ["llm:8081"]


def test_cannot_make_room_evicts_nothing(tmp_path, monkeypatch):
    m, evicted = make(tmp_path, monkeypatch, free_after=[])
    ok, done, reason = asyncio.run(m.make_room(0, 23000, "test"))
    assert not ok and evicted == [] and "CosyVoice" in reason


def test_pins_persist(tmp_path):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    m.set_pin("llm:8081", True)
    assert local.LocalGpuManager(str(tmp_path / "pins.json")).pins() == frozenset({"llm:8081"})
    m.set_pin("llm:8081", False)
    assert m.pins() == frozenset()


def test_evict_refuses_busy(tmp_path):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    busy = [Consumer(id="tts:cosy", kind="tts", label="CosyVoice", busy=True)]
    with pytest.raises(PermissionError):
        asyncio.run(m.evict("tts:cosy", busy))
    with pytest.raises(PermissionError):
        asyncio.run(m.evict("training:1", [Consumer(id="training:1", kind="training", label="t",
                                                    evictable_kind=False)]))


def test_two_loads_cannot_both_get_the_same_room(tmp_path, monkeypatch):
    """The overbooking case: 12 GB free, two loads of 8 GB arrive before either has
    allocated anything. The first gets a reservation, the second a fast no."""
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))

    async def snapshot():
        return [{"index": 0, "name": "3090", "total": 24576, "free": 12000}], [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)

    async def both():
        return await asyncio.gather(m.reserve(0, 8000, "a"), m.reserve(0, 8000, "b"))
    (r1, _, _), (r2, _, why) = asyncio.run(both())
    assert r1 is not None and r2 is None and "promised" in why


def test_confirmed_reservation_counts_until_allocated(tmp_path, monkeypatch):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    free = {"v": 12000}

    async def snapshot():
        return [{"index": 0, "name": "3090", "total": 24576, "free": free["v"]}], [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)
    monkeypatch.setattr(local.psutil, "pid_exists", lambda pid: True)
    res, _, _ = asyncio.run(m.reserve(0, 8000, "a"))
    assert m.confirm(res.id, 4242)
    # the process has allocated 6 GB of its 8 GB so far -> 2 GB still promised
    m._procs, m._parents = [(4242, 0, 6000)], {}
    free["v"] = 6000
    assert m._outstanding(0) == 2000
    r2, _, _ = asyncio.run(m.reserve(0, 3000, "b"))   # 6000 free - 2000 promised = 4000 >= 3000
    assert r2 is not None


def test_loading_llama_server_counts_as_busy(tmp_path, monkeypatch):
    """A llama-server still loading has no requests yet but must not be unloaded by
    a second load (it would look idle)."""
    import types
    from backend.state import state
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    proc = types.SimpleNamespace(pid=777, poll=lambda: None)
    monkeypatch.setitem(state.managed_processes, 39999, {"process": proc, "model": "/m/gemma.gguf",
                                                          "active_requests": 0, "loading": True})
    monkeypatch.setattr(local, "_cmdline_pid", lambda frag, cwd=None: None)
    cons = asyncio.run(m.consumers())
    c = next(c for c in cons if c.id == "llm:39999")
    assert c.busy


def _fixed(m, monkeypatch, gpus, cons, att):
    async def snapshot():
        return gpus, cons, att
    monkeypatch.setattr(m, "snapshot", snapshot)
    monkeypatch.setattr(local.psutil, "pid_exists", lambda pid: True)


def test_consumer_with_open_reservation_counts_as_loading(tmp_path, monkeypatch):
    """Any process still loading under a reservation (llama-server via /api/start, a
    TTS worker, Strata) is busy -- the next request can't unload it mid-load."""
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 3000}]
    cons = [Consumer(id="llm:9001", kind="llm", label="loading-model", root_pids={555}, last_used=1)]
    _fixed(m, monkeypatch, gpus, cons, {("llm:9001", 0): 8000})
    res, _, _ = asyncio.run(m.reserve(0, 1000, "first"))
    m.confirm(res.id, 555)
    r2, _, why = asyncio.run(m.reserve(0, 9000, "second"))
    assert r2 is None and "loading-model" in why


def test_exclusive_reservation_unloads_idle_and_blocks_everyone_else(tmp_path, monkeypatch):
    from fastapi import HTTPException
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 3000},
            {"index": 1, "name": "4060", "total": 16380, "free": 5000}]
    cons = [Consumer(id="comfyui", kind="image", label="ComfyUI", root_pids={1}),
            Consumer(id="llm:9", kind="llm", label="gemma", root_pids={2})]
    _fixed(m, monkeypatch, gpus, cons, {("comfyui", 1): 10000, ("llm:9", 0): 20000})
    unloaded = []

    async def evict(cid, consumers=None):
        unloaded.append(cid)
    monkeypatch.setattr(m, "evict", evict)
    held, done, why = asyncio.run(m.reserve_exclusive([0, 1], "Strata"))
    assert held and sorted(done) == ["comfyui", "llm:9"] and why == ""
    for r in held:
        m.confirm(r.id, 900)
    with pytest.raises(HTTPException) as e:
        asyncio.run(m.reserve(1, 100, "tts:cosyvoice"))
    assert e.value.status_code == 507 and "Strata" in e.value.detail


def test_exclusive_refused_when_busy(tmp_path, monkeypatch):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 3000}]
    cons = [Consumer(id="tts:cosy", kind="tts", label="CosyVoice", root_pids={3}, busy=True)]
    _fixed(m, monkeypatch, gpus, cons, {("tts:cosy", 0): 4000})
    held, done, why = asyncio.run(m.reserve_exclusive([0], "Strata"))
    assert held is None and done == [] and "CosyVoice" in why


def test_running_strata_blocks_even_before_it_allocates(tmp_path, monkeypatch):
    """A Strata started elsewhere and still loading shows up as a consumer with no
    VRAM yet -- it still holds every GPU."""
    from fastapi import HTTPException
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 24000}]
    cons = [Consumer(id="strata", kind="exclusive", label="Strata", root_pids={77}, busy=True, evictable_kind=False)]
    _fixed(m, monkeypatch, gpus, cons, {})
    with pytest.raises(HTTPException) as e:
        asyncio.run(m.reserve(0, 100, "llm:x"))
    assert e.value.status_code == 507



def test_cmdline_pid_can_require_the_working_directory(tmp_path):
    """comfyui-moss runs as plain `main.py ...` from its ComfyUI folder -- the cmdline alone never named
    the project, so MOSS counted as "other programs" and was never unloaded. Match cmdline + cwd."""
    import subprocess
    import sys
    import time
    marker = f"cwd-probe-{time.time_ns()}"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", marker], cwd=tmp_path)
    try:
        for _ in range(50):
            if local._cmdline_pid(marker) == proc.pid:
                break
            time.sleep(0.05)
        assert local._cmdline_pid(marker, cwd=str(tmp_path)) == proc.pid
        assert local._cmdline_pid(marker, cwd=str(tmp_path / "elsewhere")) is None
    finally:
        proc.kill()
        proc.wait()


def test_room_is_free_minus_promised_and_zero_under_an_exclusive_holder(tmp_path, monkeypatch):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    gpus = [{"index": 0, "name": "3090", "total": 24576, "free": 20000},
            {"index": 1, "name": "4060 Ti", "total": 16380, "free": 9000}]

    async def snapshot():
        return gpus, [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)
    m.reservations.add(1, 4000, "image:x", __import__("time").time())     # promised, nothing allocated yet
    assert asyncio.run(m.room()) == {0: 20000, 1: 5000}

    async def strata_snapshot():
        return gpus, [Consumer(id="strata", kind="exclusive", label="Strata", root_pids={9})], {}
    monkeypatch.setattr(m, "snapshot", strata_snapshot)
    assert asyncio.run(m.room()) == {0: 0, 1: 0}
