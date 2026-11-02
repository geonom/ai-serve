"""When a gateway request finds its model already running (model_manager._load_locked's reuse loop).

Found live 2026-11-01: V-Noveller's gateway chats (no ctx_size given, so the gateway's default 32000) found
llm-benchy's Qwen3.8 on port 8092, started by benchy through /api/start with ctx 8192 and driven on its own port.
The reuse loop counted it idle (no gateway requests in flight) and restarted it with the larger context -- twice, the
second time while it was still loading -- and benchy's trials failed. Rules now:
- no ctx_size given -> any running context is fine (the default is only for new loads);
- an instance someone started through /api/start, or one adopted after a restart, is never restarted by the gateway;
- an instance that is busy (its own /slots, not only gateway requests) or still loading is never restarted.
If such an instance can't hold an explicitly asked ctx_size, the request gets an instance of its own (or the usual
507 when there is no room) -- never squeezed into it (V-Noveller: it would overflow and skew benchy's timings).
Asking only for more slots is served by the running instance as it is."""
import asyncio

import pytest

from backend import model_manager as mm
from backend.state import state

MODEL = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"


class Alive:
    pid = 4242

    def poll(self):
        return None


class LoadedNew(Exception):
    """Raised where a fresh load would start: the reuse loop decided not to reuse."""


@pytest.fixture
def instance(monkeypatch):
    stopped = []
    info = {"process": Alive(), "model": MODEL, "port": 8092, "ctx_size": 8192, "parallel": 1, "gpu_id": 0,
            "active_requests": 0, "origin": "gateway"}
    monkeypatch.setattr(state, "managed_processes", {8092: info})

    async def stop(port):
        stopped.append(port)
        state.managed_processes.pop(port, None)

    async def idle(port, info):
        return False

    def no_new_load(path, label):
        raise LoadedNew()
    monkeypatch.setattr(mm, "stop_server_by_port", stop)
    monkeypatch.setattr(mm, "is_port_busy", idle)
    monkeypatch.setattr(mm, "check_model_file_exists", no_new_load)
    monkeypatch.setattr(mm, "reset_keep_alive", lambda port, keep_alive: None)
    info["stopped"] = stopped
    return info


def load(ctx_size, parallel=None):
    return asyncio.run(mm._load_locked(MODEL, None, ctx_size, 8, -1, 300, None, False, parallel=parallel))


def test_no_ctx_size_given_reuses_whatever_context_runs(instance):
    assert load(None) == 8092 and instance["stopped"] == []


def test_an_explicit_bigger_context_restarts_an_idle_gateway_instance(instance):
    with pytest.raises(LoadedNew):
        load(32000)
    assert instance["stopped"] == [8092]


@pytest.mark.parametrize("origin", ["api_start", "adopted", None])
def test_an_instance_someone_else_started_is_never_restarted(instance, origin):
    instance["origin"] = origin
    with pytest.raises(LoadedNew):                       # its own load (or a 507 from there), not 8092
        load(32000)
    assert load(8192, parallel=4) == 8092 and load(None) == 8092
    assert instance["stopped"] == []


def test_a_busy_instance_is_never_restarted_also_when_only_its_slots_say_so(instance, monkeypatch):
    async def busy(port, info):
        return True
    monkeypatch.setattr(mm, "is_port_busy", busy)
    with pytest.raises(LoadedNew):
        load(32000)
    assert load(None) == 8092 and instance["stopped"] == []


def test_a_loading_instance_is_never_restarted(instance):
    instance["loading"] = True
    with pytest.raises(LoadedNew):
        load(32000)
    assert instance["stopped"] == []


def test_a_smaller_or_equal_context_reuses(instance):
    assert load(4096) == 8092 and load(8192) == 8092 and instance["stopped"] == []


def test_a_loading_instance_counts_as_busy_for_manual_stops_too():
    """2026-11-01: benchy's unload_all?gpu_id=0 stopped the gateway's Qwen3.8 load halfway (its /slots don't answer
    while loading, so it looked idle) -- "crashed during startup" for V-Noveller's request."""
    assert asyncio.run(mm.is_port_busy(1, {"loading": True, "active_requests": 0})) is True
