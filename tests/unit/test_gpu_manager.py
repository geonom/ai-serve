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


def test_when_an_unload_frees_less_than_counted_the_next_idle_one_goes_too(tmp_path, monkeypatch):
    """2026-10-26: Qwen3.8-27B needed 22213 MB, 22098 free; the plan unloaded stt-serve (318 MB counted), but its
    CUDA context stays, nothing came free, and the load was refused -- with an idle 1.7 GB MiniCPM next to it."""
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    cons = [Consumer(id="stt-serve", kind="stt", label="stt-serve", root_pids={1}, last_used=0.0),
            Consumer(id="llm:46689", kind="llm", label="MiniCPM5-1B", root_pids={2}, last_used=5)]
    evicted = []

    def free():
        return 22098 + (1680 if "llm:46689" in evicted else 0)

    async def snapshot():
        att = {("stt-serve", 0): 318}
        if "llm:46689" not in evicted:
            att[("llm:46689", 0)] = 1680
        return [{"index": 0, "name": "3090", "total": 24576, "free": free()}], cons, att

    async def evict(cid, consumers=None):
        evicted.append(cid)

    async def gpu_status():
        return {"gpus": [{"index": 0, "free": free()}]}
    monkeypatch.setattr(m, "snapshot", snapshot)
    monkeypatch.setattr(m, "evict", evict)
    import backend.model_manager as mm
    monkeypatch.setattr(mm, "fetch_gpu_status", gpu_status)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(local.asyncio, "sleep", lambda s: real_sleep(0))
    res, done, why = asyncio.run(m.reserve(0, 22213, "llm:Qwen3.8-27B"))
    assert res is not None, why
    assert done == ["stt-serve", "llm:46689"] and evicted == done


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



def test_llama_server_used_directly_on_its_port_counts_as_busy(tmp_path, monkeypatch):
    """llm-benchy talks to an /api/start-ed instance on its own port: no active_requests here, but /slots says it
    is generating (2026-10-31: both GPUs at 95 % from benchy runs, the manager listed them as idle = evictable)."""
    import types
    from backend.state import state
    import backend.model_manager as mm
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    proc = types.SimpleNamespace(pid=778, poll=lambda: None)
    monkeypatch.setitem(state.managed_processes, 39998, {"process": proc, "model": "/m/granite.gguf",
                                                          "active_requests": 0})
    monkeypatch.setattr(local, "_cmdline_pid", lambda frag, cwd=None: None)
    asked = []

    async def busy(port, info):
        asked.append(port)
        return True
    monkeypatch.setattr(mm, "is_port_busy", busy)
    c = next(c for c in asyncio.run(m.consumers()) if c.id == "llm:39998")
    assert c.busy and asked == [39998]

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


def test_snapshot_on_unified_memory_measures_the_consumers(tmp_path, monkeypatch):
    """Apple Silicon: no nvidia-smi -- the consumers' own processes are measured (Iris rendering showed "nothing
    loaded" on a Mac, 2026-10-22)."""
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    iris = Consumer(id="sd-serve:sd-server", kind="image", label="Iris", root_pids={4242})

    async def consumers():
        return [iris]

    async def gpu_status():
        return {"gpus": [{"index": 0, "name": "Apple Silicon (Unified Memory)", "total": 24576, "free": 9300,
                          "gpu_limit_mb": 16384}]}
    import backend.model_manager as mm
    monkeypatch.setattr(mm, "fetch_gpu_status", gpu_status)
    monkeypatch.setattr(m, "consumers", consumers)
    monkeypatch.setattr(local, "_unified_memory", lambda: True)
    monkeypatch.setattr(local, "_nvidia_procs", lambda: pytest.fail("no nvidia-smi on a Mac"))
    monkeypatch.setattr(local, "_footprint_mb", {4242: 5400}.get)
    monkeypatch.setattr(local, "_children", lambda pid: [])
    monkeypatch.setattr(local, "_parents", lambda pids: {})
    _, cons, att = asyncio.run(m.snapshot())
    assert cons == [iris] and att == {("sd-serve:sd-server", 0): 5400}


def roomy(tmp_path, monkeypatch):
    m = local.LocalGpuManager(str(tmp_path / "pins.json"), needs_file=str(tmp_path / "needs.json"))

    async def snapshot():
        return [{"index": 0, "name": "3090", "total": 24576, "free": 20000}], [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)
    m._procs, m._parents = [], {}
    return m


def test_a_load_is_reserved_what_its_purpose_needed_before(tmp_path, monkeypatch):
    """2026-10-24: Whisper medium announced 1500 MB and used 2690 -- the manager now reserves the learned need."""
    import json
    (tmp_path / "needs.json").write_text(json.dumps({"stt:whisperx:medium": [2690]}))
    m = roomy(tmp_path, monkeypatch)
    m.load_needs()
    res, _, _ = asyncio.run(m.reserve(0, 1500, "stt:whisperx:medium"))
    assert res is not None and res.mb == 2959
    assert m.need("stt:whisperx:medium", 1500) == 2959 and m.need("other", 1500) == 1500


def test_without_room_for_the_headroom_the_learned_peak_is_reserved(tmp_path, monkeypatch):
    """2026-10-26: MOSS bf16 learned 14770 MB (+10 % = 16247) and was refused for minutes on a 16 GB card with
    15.2 GB free -- the peak itself fits, so it gets the peak; with less room than the peak it is still a no."""
    import json
    (tmp_path / "needs.json").write_text(json.dumps({"image:moss-turbo-4-bf16": [14770]}))
    m = local.LocalGpuManager(str(tmp_path / "pins.json"), needs_file=str(tmp_path / "needs.json"))
    free = {"mb": 15200}

    async def snapshot():
        return [{"index": 1, "name": "4060 Ti", "total": 16380, "free": free["mb"]}], [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)
    m._procs, m._parents = [], {}
    m.load_needs()
    res, _, why = asyncio.run(m.reserve(1, 9000, "image:moss-turbo-4-bf16"))
    assert res is not None and res.mb == 14770, why
    m.release(res.id)
    free["mb"] = 8000                                          # not even the declared need fits: a no
    res, _, why = asyncio.run(m.reserve(1, 9000, "image:moss-turbo-4-bf16"))
    assert res is None and "16247" in why


def test_a_peak_learned_on_a_roomier_gpu_does_not_refuse_forever(tmp_path, monkeypatch):
    """2026-10-28 (V-Noveller): Qwen-Image-Edit learned 16754 MB on the 3090; on the 4060 Ti (15.3 GB at most) every
    edit got a 507 forever. Without a peak of its own there, the declared need is tried; once this GPU has its own
    peak, that one counts."""
    import json
    (tmp_path / "needs.json").write_text(json.dumps({"image:qwen-image-edit": [16754],
                                                     "image:qwen-image-edit|gpu0": [16754]}))
    m = local.LocalGpuManager(str(tmp_path / "pins.json"), needs_file=str(tmp_path / "needs.json"))
    free = {"mb": 15057}

    async def snapshot():
        return [{"index": 1, "name": "4060 Ti", "total": 16380, "free": free["mb"]}], [], {}
    monkeypatch.setattr(m, "snapshot", snapshot)
    m._procs, m._parents = [], {}
    m.load_needs()
    res, _, why = asyncio.run(m.reserve(1, 10500, "image:qwen-image-edit"))
    assert res is not None and res.mb == 10500, why
    m.release(res.id)
    m.needs.observe("image:qwen-image-edit", 15500, gpu="gpu1")  # it really took 15.5 GB here
    free["mb"] = 14000
    res, _, why = asyncio.run(m.reserve(1, 10500, "image:qwen-image-edit"))
    assert res is None


def test_confirm_measures_the_baseline_and_release_learns_the_peak(tmp_path, monkeypatch):
    import json
    m = roomy(tmp_path, monkeypatch)
    m._procs, m._parents = [(70, 0, 300)], {}                 # stt-serve holds its CUDA context already
    res, _, _ = asyncio.run(m.reserve(0, 1500, "stt:whisperx:medium"))
    assert m.confirm(res.id, 70)
    assert m.reservations.get(res.id).baseline_mb == 300

    async def grown():                                        # loaded and transcribed once: 2990 MB in all
        m._procs = [(70, 0, 2990)]
        return [], [], {}
    monkeypatch.setattr(m, "snapshot", grown)
    asyncio.run(m.release_measured(res.id))
    assert json.loads((tmp_path / "needs.json").read_text()) == {"stt:whisperx:medium": [2690],
                                                                  "stt:whisperx:medium|gpu0": [2690]}
    assert m.need("stt:whisperx:medium", 1500) == 2959


def _residue_case(tmp_path, monkeypatch, busy_after_release=False):
    """2026-10-28, llm-benchy: Sketch-Cydonia-24B needed 15208 MB on the 4060 Ti, 14973 free; unloading stt-serve and
    ComfyUI freed nothing -- both processes kept their CUDA context (158 + 244 MB) -- and the load got a 507 that
    called them "pinned" (nobody had pinned them). Owner: stop such idle processes in a last round."""
    m = local.LocalGpuManager(str(tmp_path / "pins.json"))
    released, exited = [], []

    def cons():
        stt = Consumer(id="stt-serve", kind="stt", label="stt-serve", root_pids={1}, plugin="stt-serve")
        comfy = Consumer(id="comfyui:comfyui", kind="image", label="ComfyUI", root_pids={2}, plugin="comfyui",
                         last_used=5, busy=busy_after_release and bool(released))
        return [c for c in (stt, comfy) if c.plugin not in exited]

    def att():
        out = {}
        if "stt-serve" not in exited:
            out[("stt-serve", 1)] = 158
        if "comfyui" not in exited:
            out[("comfyui:comfyui", 1)] = 244
        return out

    def free():
        return 14973 + (158 if "stt-serve" in exited else 0) + (244 if "comfyui" in exited else 0)

    async def snapshot():
        return [{"index": 1, "name": "4060 Ti", "total": 16380, "free": free()}], cons(), att()

    async def evict(cid, consumers=None):
        released.append(cid)                       # models unloaded, process (and its context) stays

    async def exit_process(cid, consumers=None):
        c = next(c for c in (consumers or cons()) if c.id == cid)
        exited.append(c.plugin)

    async def gpu_status():
        return {"gpus": [{"index": 1, "free": free()}]}
    monkeypatch.setattr(m, "snapshot", snapshot)
    monkeypatch.setattr(m, "evict", evict)
    monkeypatch.setattr(m, "exit_process", exit_process)
    import backend.model_manager as mm
    monkeypatch.setattr(mm, "fetch_gpu_status", gpu_status)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(local.asyncio, "sleep", lambda s: real_sleep(0))
    return m, released, exited


def test_idle_plugin_processes_whose_context_is_all_that_is_missing_are_stopped(tmp_path, monkeypatch):
    m, released, exited = _residue_case(tmp_path, monkeypatch)
    res, done, why = asyncio.run(m.reserve(1, 15208, "llm:Sketch-Cydonia"))
    assert res is not None, why
    assert sorted(released) == ["comfyui:comfyui", "stt-serve"] and sorted(exited) == ["comfyui", "stt-serve"]


def test_a_process_that_became_busy_is_not_stopped_and_the_reason_says_what_is_left(tmp_path, monkeypatch):
    m, released, exited = _residue_case(tmp_path, monkeypatch, busy_after_release=True)
    res, done, why = asyncio.run(m.reserve(1, 15208, "llm:Sketch-Cydonia"))
    assert res is None and exited == []
    assert "pinned" not in why
    assert "stt-serve (already unloaded; only its CUDA context is left (process still running), 158 MB)" in why
    assert "ComfyUI (busy, 244 MB)" in why
