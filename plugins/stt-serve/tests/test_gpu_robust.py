"""Speech recognition must not fail because another model fills the GPU (2026-10-24: Whisper medium announced 1500 MB
to ai-serve's GPU manager, always on GPU 0, released its reservation right after loading, and ran out of memory on
its first transcription while Qwen3.8-27B filled the 3090 -- GPU 1 had 15 GB free; the spoken input was lost).
Rules:
- the need is per model size (medium ~3 GB), the manager picks the GPU (no fixed GPU 0);
- the reservation lasts until the first transcription is done, not just the load;
- the manager says no -> load on the CPU instead of failing; a CPU model moves to a GPU once there is room;
- out of memory while transcribing -> ask again (the manager may unload an idle model or pick the other GPU),
  reload, retry once; still out of memory -> the CPU; out of memory there too -> 507 (retryable), never a bare 500;
- the last use is reported, so the GPU manager doesn't unload STT first every time."""
import numpy as np
import pytest
from fastapi import HTTPException

from stt_serve import coordination, engine

OOM = RuntimeError("CUDA failed with error out of memory")


class Model:
    def __init__(self, device, fail_with=None):
        self.device, self.fail_with, self.calls, self.threads = device, list(fail_with or []), 0, []

    def transcribe(self, audio, batch_size, language, task):
        import threading
        self.calls += 1
        self.threads.append(threading.current_thread().name)
        if self.fail_with:
            raise self.fail_with.pop(0)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": f" hallo ({self.device})"}], "language": language}


@pytest.fixture
def gpu(monkeypatch):
    """A fake GPU manager + loader. grants: list of GPU indices (None = the manager says no) handed out in order;
    fails: per device, the errors its next model raises while transcribing."""
    log, ctl = [], {"grants": [], "fails": {}, "gpu_possible": True}

    def reserve(purpose, need_mb):
        log.append(("reserve", purpose, need_mb))
        g = ctl["grants"].pop(0) if ctl["grants"] else None
        if g is None:
            raise HTTPException(status_code=507, detail="no room")
        return coordination.Grant(f"r{len(log)}", g)

    def load(name, device, device_index, compute_type):
        where = "cpu" if device == "cpu" else f"cuda:{device_index}"
        log.append(("load", where, compute_type))
        return Model(where, ctl["fails"].pop(where, None))

    monkeypatch.setattr(engine, "reserve_sync", reserve)
    monkeypatch.setattr(engine, "release_sync", lambda rid: rid and log.append(("release", rid)))
    monkeypatch.setattr(engine, "_load_model", load)
    monkeypatch.setattr(engine, "_gpu_possible", lambda: ctl["gpu_possible"])
    monkeypatch.setattr(engine, "_release_cuda_memory", lambda: None)
    monkeypatch.setattr(engine, "WHISPER_MODEL", "medium")
    monkeypatch.setattr(engine.state, "whisperx_model", None)
    monkeypatch.setattr(engine.state, "whisperx_device", None)
    monkeypatch.setattr(engine.state, "whisperx_reservation", None)
    monkeypatch.setattr(engine.state, "whisperx_status", "idle")
    return log, ctl


def run():
    engine._ensure_whisperx_loaded()
    return engine._whisper_transcribe(np.zeros(16000, dtype=np.float32), "de")


def test_needs_follow_the_model_size():
    assert coordination.whisper_need_mb("medium") == 3200
    assert coordination.whisper_need_mb("small") == 1800
    assert coordination.whisper_need_mb("large-v3") == 5000
    assert coordination.whisper_need_mb("Systran/faster-whisper-medium") == 3200
    assert coordination.whisper_need_mb("/models/my-whisper") == 5000   # unknown: assume the largest


def test_the_manager_picks_the_gpu_and_the_reservation_covers_the_first_transcription(gpu):
    log, ctl = gpu
    ctl["grants"] = [1]
    out = run()
    assert "cuda:1" in out["segments"][0]["text"]
    assert log == [("reserve", "stt:whisperx:medium", 3200), ("load", "cuda:1", "float16"), ("release", "r1")]
    assert engine.state.whisperx_reservation is None
    run()                                                    # loaded: nothing more to ask
    assert len(log) == 3


def test_a_no_from_the_manager_means_the_cpu_not_a_failure(gpu):
    log, ctl = gpu
    ctl["grants"] = [None]
    out = run()
    assert "cpu" in out["segments"][0]["text"] and ("load", "cpu", "int8") in log


def test_a_cpu_model_moves_to_a_gpu_once_there_is_room(gpu):
    log, ctl = gpu
    ctl["grants"] = [None, 0]
    run()
    out = run()
    assert "cuda:0" in out["segments"][0]["text"]
    assert [e for e in log if e[0] == "load"] == [("load", "cpu", "int8"), ("load", "cuda:0", "float16")]


def test_out_of_memory_asks_again_and_retries(gpu):
    log, ctl = gpu
    ctl["grants"], ctl["fails"] = [0, 1], {"cuda:0": [OOM]}
    out = run()
    assert "cuda:1" in out["segments"][0]["text"]
    assert [e[0] for e in log] == ["reserve", "load", "release", "reserve", "load", "release"]


def test_out_of_memory_twice_goes_to_the_cpu(gpu):
    log, ctl = gpu
    ctl["grants"], ctl["fails"] = [0, 0], {"cuda:0": [OOM]}

    def load(name, device, device_index, compute_type):        # every GPU model runs out of memory
        where = "cpu" if device == "cpu" else f"cuda:{device_index}"
        log.append(("load", where, compute_type))
        return Model(where, [] if where == "cpu" else [OOM])
    engine._load_model = load
    out = run()
    assert "cpu" in out["segments"][0]["text"]
    assert sum(1 for e in log if e[0] == "release") == 2        # both reservations given back


def test_out_of_memory_everywhere_is_a_retryable_507(gpu, monkeypatch):
    log, ctl = gpu
    ctl["grants"] = [0, 0]

    def load(name, device, device_index, compute_type):
        return Model(device, [OOM, OOM, OOM])
    monkeypatch.setattr(engine, "_load_model", load)
    with pytest.raises(HTTPException) as e:
        run()
    assert e.value.status_code == 507 and "try again" in e.value.detail.lower()


def test_other_errors_are_not_retried(gpu):
    log, ctl = gpu
    ctl["grants"], ctl["fails"] = [0], {"cuda:0": [ValueError("broken audio")]}
    with pytest.raises(ValueError):
        run()
    assert [e[0] for e in log] == ["reserve", "load", "release"]


def test_without_a_gpu_it_never_asks(gpu):
    log, ctl = gpu
    ctl["gpu_possible"] = False
    run()
    assert log == [("load", "cpu", "int8")]


def test_the_last_use_is_reported(gpu):
    log, ctl = gpu
    ctl["grants"] = [0]
    before = engine.state.whisperx_last_used or 0
    run()
    assert engine.state.whisperx_last_used > before


def test_reserve_sync_lets_the_manager_choose(monkeypatch):
    import httpx
    sent = []

    def post(url, json=None, timeout=None):
        sent.append((url.split("/api/")[1], json))
        if url.endswith("gpu-preflight"):
            return httpx.Response(200, json={"granted": True, "reservation_id": "r3", "gpu": 1},
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))
    monkeypatch.setattr(coordination.httpx, "post", post)
    g = coordination.reserve_sync("stt:whisperx:medium", 3200)
    assert (g.rid, g.gpu) == ("r3", 1)
    assert sent[0] == ("coordination/gpu-preflight", {"purpose": "stt:whisperx:medium", "need_mb": 3200})
    assert sent[1][0] == "coordination/reservations/r3/confirm"
    monkeypatch.setattr(coordination.httpx, "post", lambda url, json=None, timeout=None: httpx.Response(
        507, json={"detail": "no room", "granted": False}, request=httpx.Request("POST", url)))
    with pytest.raises(HTTPException) as e:
        coordination.reserve_sync("x", 1)
    assert e.value.status_code == 507

    def down(url, json=None, timeout=None):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(coordination.httpx, "post", down)
    assert coordination.reserve_sync("x", 1) == coordination.Grant(None, None)


# --- the workers (MOSS-Transcribe-Diarize, Nemotron diarization): same rules, they run as child processes ----------

def test_a_worker_gets_the_gpu_the_manager_picked_or_the_cpu():
    assert engine._worker_env(coordination.Grant("r1", 1))["CUDA_VISIBLE_DEVICES"] == "1"
    assert engine._worker_env(None)["CUDA_VISIBLE_DEVICES"] == ""          # the manager said no: the CPU
    assert "CUDA_VISIBLE_DEVICES" not in engine._worker_env(coordination.Grant(None, None))  # standalone


def test_worker_needs_and_purposes():
    assert engine._worker_purpose(engine._MOSS_CFG) == "stt:moss_transcribe"
    assert coordination.NEEDS_MB["moss_transcribe"] >= 6000 and coordination.NEEDS_MB["nemotron_diarize"] >= 2000


@pytest.fixture
def moss(monkeypatch):
    calls = {"ensure": [], "jobs": 0, "unloads": 0}
    script = []

    async def ensure(cfg, cpu_only=False):
        calls["ensure"].append("cpu" if cpu_only else "gpu")
        return 9000

    async def job(port, wav_path, prompt, max_new_tokens, duration_s):
        calls["jobs"] += 1
        err = script.pop(0) if script else None
        if err:
            raise err
        return {"segments": [], "device": "cpu" if calls["ensure"][-1] == "cpu" else "cuda:0"}
    monkeypatch.setattr(engine, "_ensure_worker", ensure)
    monkeypatch.setattr(engine, "_moss_worker_transcribe", job)
    monkeypatch.setattr(engine, "_unload_moss_transcribe", lambda: calls.__setitem__("unloads", calls["unloads"] + 1))
    monkeypatch.setattr(engine, "_reset_moss_keep_alive", lambda s: None)
    return calls, script


MOSS_OOM = HTTPException(status_code=500, detail="MOSS-Transcribe-Diarize: CUDA out of memory. Tried to allocate 2 GiB")


def test_moss_out_of_memory_restarts_it_where_there_is_room(moss):
    import asyncio
    calls, script = moss
    script.append(MOSS_OOM)
    data = asyncio.run(engine._moss_job("w.wav", "p", 100, 10.0))
    assert data["device"] == "cuda:0" and calls["ensure"] == ["gpu", "gpu"] and calls["unloads"] == 1


def test_moss_out_of_memory_twice_runs_on_the_cpu(moss):
    import asyncio
    calls, script = moss
    script.extend([MOSS_OOM, MOSS_OOM])
    data = asyncio.run(engine._moss_job("w.wav", "p", 100, 10.0))
    assert data["device"] == "cpu" and calls["ensure"] == ["gpu", "gpu", "cpu"]


def test_moss_out_of_memory_everywhere_is_a_507(moss):
    import asyncio
    calls, script = moss
    script.extend([MOSS_OOM, MOSS_OOM, MOSS_OOM])
    with pytest.raises(HTTPException) as e:
        asyncio.run(engine._moss_job("w.wav", "p", 100, 10.0))
    assert e.value.status_code == 507


def test_moss_other_errors_pass_through(moss):
    import asyncio
    calls, script = moss
    script.append(HTTPException(status_code=500, detail="MOSS-Transcribe-Diarize: bad audio"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(engine._moss_job("w.wav", "p", 100, 10.0))
    assert e.value.status_code == 500 and calls["ensure"] == ["gpu"]


# --- found live 2026-10-24 18:01: after a load on GPU 1 and a later one on GPU 0 in the same process, every
# transcription failed with "CUDA failed with error invalid argument". Reproduced outside the service: CTranslate2
# keeps per-thread CUDA state, and asyncio's executor ran the GPU 0 model on a thread that had worked on GPU 1
# (setting torch's device in that thread isn't enough). Each device gets its own worker thread that never switches.

def test_each_device_transcribes_on_its_own_thread(gpu):
    log, ctl = gpu
    ctl["grants"] = [1, 0]
    run()
    first = engine.state.whisperx_model
    engine._drop_whisperx_model()
    run()
    second = engine.state.whisperx_model
    assert first.threads and second.threads
    assert "cuda:1" in first.threads[0] and "cuda:0" in second.threads[0]
    run()                                                    # same device again: the same thread
    assert second.threads[-1] == second.threads[0]


def test_another_cuda_error_reloads_and_retries(gpu):
    log, ctl = gpu
    ctl["grants"], ctl["fails"] = [0, 0], {"cuda:0": [RuntimeError("CUDA failed with error invalid argument")]}
    out = run()
    assert "cuda:0" in out["segments"][0]["text"] and [e[0] for e in log].count("load") == 2


def test_a_cuda_error_that_stays_is_a_retryable_503(gpu, monkeypatch):
    log, ctl = gpu
    ctl["grants"] = [0, 0]
    bad = RuntimeError("CUDA failed with error invalid argument")

    def load(name, device, device_index, compute_type):
        return Model("cpu" if device == "cpu" else f"cuda:{device_index}", [bad, bad, bad])
    monkeypatch.setattr(engine, "_load_model", load)
    with pytest.raises(HTTPException) as e:
        run()
    assert e.value.status_code == 503 and "try again" in e.value.detail.lower()
