"""Where the event loop gets stuck (2026-11-26: the gateway answered /health in 0.6 s and /api/services in 6 s while
the same call took 0.33 s in a fresh process; its main thread ran at ~100 % CPU and nothing could attach to it).

A task on the loop beats every `interval` s; a daemon thread checks the beat, and while it is older than
`threshold` s it samples the main thread's Python stack (sys._current_frames) and keeps where it was. Besides, every
`sample_s` it notes where the main thread works when it isn't waiting in select -- many short pieces of work never
count as "blocked" but can still fill the loop (the gateway ran at ~90 % CPU with no single block). Once per
`period_s` both go to the log as one line each, most frequent place first. Costs one sleep per interval on the loop
and a frame lookup per sample in the thread. Tests: tests/unit/test_loop_watch.py."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)
Frame = Tuple[str, int, str]   # (file, line, function), outermost first

_OWN = os.sep + "backend" + os.sep


def blocked(last_beat: float, now: float, threshold: float) -> bool:
    return bool(last_beat) and now - last_beat > threshold


def _short(path: str) -> str:
    i = path.rfind(_OWN)
    return path[i + 1:] if i >= 0 else os.path.basename(path)


def place(stack: Sequence[Frame]) -> str:
    """The two innermost frames of our own code (backend/...), else the innermost frame at all."""
    own = [f for f in stack if _OWN in f[0]]
    pick = list(reversed(own))[:2] if own else list(reversed(stack))[:1]
    return " <- ".join(f"{_short(p)}:{line} {fn}" for p, line, fn in pick) or "?"


def summary(samples: Sequence[str], interval: float, period_s: float) -> Optional[str]:
    if not samples:
        return None
    top = Counter(samples).most_common(5)
    return (f"event loop blocked ~{len(samples) * interval:.1f} s in the last {period_s:.0f} s: "
            + "; ".join(f"{n}x {where}" for where, n in top))


def idle(stack: Sequence[Frame]) -> bool:
    """The loop waits for I/O (innermost frame: selectors' select, or asyncio's runner under uvloop), or no stack."""
    if not stack:
        return True
    path, _, fn = stack[-1]
    base = os.path.basename(path)
    if fn in ("select", "poll", "epoll") and base == "selectors.py":
        return True
    # uvloop waits in C: nothing below asyncio's runner means the loop is waiting
    return fn in ("run", "run_until_complete", "run_forever") and base in ("runners.py", "base_events.py")


def busy_summary(samples: Sequence[str], total: int, period_s: float) -> Optional[str]:
    if not samples or total <= 0:
        return None
    top = Counter(samples).most_common(6)
    return (f"event loop busy {round(100 * len(samples) / total)}% of the last {period_s:.0f} s: "
            + "; ".join(f"{n}x {where}" for where, n in top))


def _stack_of(thread_id: int) -> List[Frame]:
    frame = sys._current_frames().get(thread_id)
    out: List[Frame] = []
    while frame is not None:
        out.append((frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name))
        frame = frame.f_back
    return list(reversed(out))


class LoopWatch:
    def __init__(self, threshold: float = 0.5, interval: float = 0.25, period_s: float = 60.0,
                 report: Callable[[str], None] = logger.warning, sample_s: float = 0.02):
        self.threshold, self.interval, self.period_s, self.report = threshold, interval, period_s, report
        self.sample_s = sample_s
        self._busy: List[str] = []
        self._total = 0
        self._beat = 0.0
        self._samples: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_thread = 0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop_thread = threading.get_ident()
        self._beat = time.monotonic()
        self._task = loop.create_task(self._beater())
        self._thread = threading.Thread(target=self._watch, name="loop-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    async def _beater(self) -> None:
        while not self._stop.is_set():
            self._beat = time.monotonic()
            await asyncio.sleep(self.interval)

    def _watch(self) -> None:
        next_report = time.monotonic() + self.period_s
        next_check = time.monotonic() + self.interval
        while not self._stop.wait(self.sample_s):
            now = time.monotonic()
            stack = _stack_of(self._loop_thread)
            with self._lock:
                self._total += 1
                if not idle(stack):
                    self._busy.append(place(stack))
            if now >= next_check:
                next_check = now + self.interval
                if blocked(self._beat, now, self.threshold):
                    with self._lock:
                        self._samples.append(place(stack))
            if now >= next_report:
                next_report = now + self.period_s
                self.flush()

    def flush(self) -> None:
        with self._lock:
            samples, self._samples = self._samples, []
            busy, total, self._busy, self._total = self._busy, self._total, [], 0
        for line in (summary(samples, self.interval, self.period_s), busy_summary(busy, total, self.period_s)):
            if line:
                self.report(line)
