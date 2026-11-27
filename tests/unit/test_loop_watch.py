"""The event-loop watch (backend/loop_watch.py): when the loop stops answering, where the main thread is stuck."""
import asyncio
import sys
import threading
import time

from backend import loop_watch as lw


def test_a_loop_is_blocked_only_past_the_threshold():
    assert not lw.blocked(last_beat=10.0, now=10.4, threshold=0.5)
    assert lw.blocked(last_beat=10.0, now=10.6, threshold=0.5)
    assert not lw.blocked(last_beat=0.0, now=99.0, threshold=0.5)   # not started yet


def test_the_innermost_frames_of_our_own_code_name_the_place():
    stack = [("/usr/lib/python3.12/asyncio/base_events.py", 1990, "_run_once"),
             ("/x/backend/routers/services.py", 306, "list_services"),
             ("/x/backend/services/domain.py", 210, "discover_instances"),
             ("/x/venv/lib/python3.12/site-packages/psutil/__init__.py", 500, "children")]
    assert lw.place(stack) == "backend/services/domain.py:210 discover_instances <- backend/routers/services.py:306 list_services"
    assert lw.place([("/usr/lib/python3.12/selectors.py", 468, "select")]) == "selectors.py:468 select"
    assert lw.place([]) == "?"


def test_a_minute_is_summed_up_most_frequent_first():
    samples = ["a", "b", "a", "a", "c", "b"]
    line = lw.summary(samples, interval=0.25, period_s=60)
    assert line.startswith("event loop blocked ~1.5 s in the last 60 s")
    assert line.index("3x a") < line.index("2x b") < line.index("1x c")
    assert lw.summary([], interval=0.25, period_s=60) is None


def test_it_catches_a_blocking_call_in_a_real_loop():
    seen = []
    watch = lw.LoopWatch(threshold=0.2, interval=0.05, period_s=3600, report=seen.append)

    def busy():                    # blocks the loop, as a CPU-heavy handler would
        end = time.time() + 0.6
        while time.time() < end:
            pass

    async def main():
        watch.start(asyncio.get_running_loop())
        await asyncio.sleep(0.2)
        busy()
        await asyncio.sleep(0.2)
        watch.flush()
        watch.stop()

    asyncio.run(main())
    assert seen and "busy" in seen[0]


def test_waiting_in_select_is_idle_anything_else_is_work():
    assert lw.idle([("/x/asyncio/base_events.py", 1990, "_run_once"), ("/usr/lib/python3.12/selectors.py", 468, "select")])
    assert not lw.idle([("/x/asyncio/base_events.py", 1990, "_run_once"), ("/x/backend/routers/services.py", 306, "list_services")])
    assert lw.idle([])            # no frame: the thread is gone or not started
    # uvloop waits in C: the innermost Python frame is asyncio's runner (seen in the gateway, 2026-11-27)
    assert lw.idle([("/x/backend/main.py", 207, "<module>"), ("/usr/lib/python3.12/asyncio/runners.py", 118, "run")])


def test_busy_share_and_top_places():
    line = lw.busy_summary(["a", "a", "b"], total=10, period_s=60)
    assert line.startswith("event loop busy 30% of the last 60 s")
    assert line.index("2x a") < line.index("1x b")
    assert lw.busy_summary([], total=10, period_s=60) is None
    assert lw.busy_summary(["a"], total=0, period_s=60) is None


def test_it_finds_where_a_busy_loop_spends_its_time():
    seen = []
    watch = lw.LoopWatch(threshold=10, interval=0.05, period_s=3600, report=seen.append, sample_s=0.005)

    def crunch():                  # many short pieces of work, none long enough to count as "blocked"
        end = time.time() + 0.02
        while time.time() < end:
            pass

    async def main():
        watch.start(asyncio.get_running_loop())
        for _ in range(40):
            crunch()
            await asyncio.sleep(0.005)
        watch.flush()
        watch.stop()

    asyncio.run(main())
    busy = [s for s in seen if s.startswith("event loop busy")]
    assert busy and "crunch" in busy[0]
