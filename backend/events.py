"""Recent happenings for the overview page ("Zuletzt passiert"): an in-memory ring
buffer of short plain-text lines -- models loaded/unloaded by the GPU manager, apps
started/stopped, requests answered by another device. Lost on restart by design; the
logs in logs/ stay the full record."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

MAX_EVENTS = 200
_TEXT_MAX = 300

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTS)


def record(kind: str, text: str, level: str = "info") -> None:
    """kind: model | gpu | service | device | strata ...; level: info | warn | error."""
    text = " ".join(str(text).split())
    if len(text) > _TEXT_MAX:
        text = text[:_TEXT_MAX - 1] + "…"
    with _lock:
        _events.append({"t": time.time(), "kind": kind, "level": level, "text": text})


def recent(limit: int = 20) -> List[Dict[str, Any]]:
    with _lock:
        items = list(_events)
    return list(reversed(items))[:max(0, limit)]


def clear() -> None:
    with _lock:
        _events.clear()
