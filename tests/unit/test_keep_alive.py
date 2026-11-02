"""Idle stops and adoption (model_manager.reset_keep_alive, remember_origin, scan_existing_llama_servers).

Found live 2026-11-01: after a gateway restart, llm-benchy's /api/start-ed Qwen3.8 (port 8092, driven on its own port)
was adopted -- and adoption gives every instance a 300 s idle timer, which /api/start never does. The timer waited
while /slots said busy and then stopped it at the first instant between two trials; 34 trials failed. Rules now:
- an idle stop needs the instance idle for the whole keep-alive time (busy seen at any poll starts it over);
- who started an instance survives a restart (instance_origins.json by PID), and an /api/start-ed one gets no idle
  timer after adoption either."""
import asyncio

import pytest

from backend import model_manager as mm
from backend.state import state


class Alive:
    pid = 777

    def poll(self):
        return None


@pytest.fixture
def stops(monkeypatch):
    stopped = []

    async def stop(port):
        stopped.append(port)
    monkeypatch.setattr(mm, "stop_server_by_port", stop)
    monkeypatch.setattr(mm, "KEEP_ALIVE_POLL_S", 0.01)
    return stopped


def run_keep_alive(monkeypatch, busy_pattern, keep_alive, wait_s):
    seen = iter(busy_pattern)

    async def busy(port, info):
        return next(seen, False)
    monkeypatch.setattr(mm, "is_port_busy", busy)

    async def main():
        state.managed_processes[5555] = {"process": Alive(), "model": "/m/x.gguf", "active_requests": 0}
        mm.reset_keep_alive(5555, keep_alive)
        await asyncio.sleep(wait_s)
        task = state.managed_processes[5555].get("keep_alive_task")
        if task and not task.done():
            task.cancel()
    monkeypatch.setattr(state, "managed_processes", {})
    asyncio.run(main())


def test_short_gaps_between_busy_spells_do_not_count_as_idle(monkeypatch, stops):
    # busy, a gap, busy, a gap ... for ~0.5 s: the gaps are each far shorter than the keep-alive time
    run_keep_alive(monkeypatch, [True, False, False, True] * 12, keep_alive=0.1, wait_s=0.45)
    assert stops == []


def test_idle_for_the_whole_keep_alive_time_stops(monkeypatch, stops):
    run_keep_alive(monkeypatch, [], keep_alive=0.05, wait_s=0.3)
    assert stops == [5555]


def test_origins_survive_a_restart_and_dead_pids_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "ORIGINS_FILE", str(tmp_path / "instance_origins.json"))
    monkeypatch.setattr(mm.psutil, "pid_exists", lambda pid: pid != 2)
    mm.remember_origin(1, "api_start")
    mm.remember_origin(2, "gateway")
    mm.remember_origin(3, "gateway")
    assert mm.recall_origin(1) == "api_start" and mm.recall_origin(3) == "gateway"
    assert mm.recall_origin(2) is None and mm.recall_origin(99) is None
    (tmp_path / "instance_origins.json").write_text("not json")
    assert mm.recall_origin(1) is None                     # a broken file is no origin, never an error
