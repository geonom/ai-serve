"""Recent happenings for the overview page ("Recent events"): an in-memory ring
buffer of short plain-text lines -- models loaded/unloaded by the GPU manager, apps
started/stopped, requests answered by another device. Lost on restart by design; the
logs in logs/ stay the full record.

Texts are English templates with {placeholders}; each event keeps `template` and `vars` too, so the dashboard can
show it in the viewer's language (frontend/i18n.de.js has the templates)."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

MAX_EVENTS = 200
_TEXT_MAX = 300

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTS)


def record(kind: str, text: str, level: str = "info", **values: Any) -> None:
    """kind: model | gpu | service | device | strata ...; level: info | warn | error. `text` is the English template;
    `values` fill its {placeholders} (str.format_map, missing ones stay as they are)."""
    template = " ".join(str(text).split())
    vars_ = {k: str(v) for k, v in values.items()}

    class _Keep(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    try:
        text = template.format_map(_Keep(vars_)) if vars_ else template
    except (ValueError, IndexError):
        text = template
    if len(text) > _TEXT_MAX:
        text = text[:_TEXT_MAX - 1] + "…"
    event = {"t": time.time(), "kind": kind, "level": level, "text": text}
    if vars_:
        event.update(template=template, vars=vars_)
    with _lock:
        _events.append(event)


def recent(limit: int = 20) -> List[Dict[str, Any]]:
    with _lock:
        items = list(_events)
    return list(reversed(items))[:max(0, limit)]


def clear() -> None:
    with _lock:
        _events.clear()
