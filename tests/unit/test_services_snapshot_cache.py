"""Read-only service views share one process snapshot for a moment (2026-11-03: the overview polls each app's
status ~50x a minute, every call walked all ~680 processes -- the gateway ran at 200-240 % CPU and /api/version
once took over 5 s). Start/stop keep taking a fresh one."""
import threading
import time

from backend.services import runtime


def test_calls_within_the_age_share_one_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "snapshot", lambda: calls.append(1) or [f"snap{len(calls)}"])
    runtime._recent.update(at=0.0, value=None)
    assert runtime.recent_snapshot(max_age=5) == ["snap1"]
    assert runtime.recent_snapshot(max_age=5) == ["snap1"]
    assert len(calls) == 1
    runtime._recent["at"] -= 10
    assert runtime.recent_snapshot(max_age=5) == ["snap2"]


def test_concurrent_callers_wait_for_one_walk(monkeypatch):
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return ["x"]
    monkeypatch.setattr(runtime, "snapshot", slow)
    runtime._recent.update(at=0.0, value=None)
    out = []
    threads = [threading.Thread(target=lambda: out.append(runtime.recent_snapshot(max_age=5))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and out == [["x"]] * 5
