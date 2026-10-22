"""Overview page layout -- pure rules (no FastAPI, no I/O).

The owner arranges the overview's boxes (order, width, hidden) and pins apps to it.
It is plain preference, but it is written through the API and rendered back into
the page, so only known widget ids, sizes and well-formed service ids get through;
anything else is refused rather than stored.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

WIDGETS = ("stats", "compute", "autostart", "pinned", "events")
SIZES = ("narrow", "half", "wide", "full")
MAX_PINNED = 30
_SERVICE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")  # same as a manifest id

_DEFAULT_SIZES = {"stats": "full", "compute": "wide", "autostart": "narrow", "pinned": "half", "events": "half"}
DEFAULT_LAYOUT: Dict[str, Any] = {
    "widgets": [{"id": w, "size": _DEFAULT_SIZES[w], "hidden": False} for w in WIDGETS],
    "pinned": [],
}


class LayoutError(ValueError):
    pass


def _widget(w: Any) -> Dict[str, Any]:
    if not isinstance(w, dict) or set(w) - {"id", "size", "hidden"}:
        raise LayoutError("each widget is {id, size, hidden}")
    wid, size, hidden = w.get("id"), w.get("size"), w.get("hidden", False)
    if wid not in WIDGETS:
        raise LayoutError(f"unknown widget {wid!r}")
    if size not in SIZES:
        raise LayoutError(f"size must be one of {', '.join(SIZES)}")
    if not isinstance(hidden, bool):
        raise LayoutError("hidden must be true or false")
    return {"id": wid, "size": size, "hidden": hidden}


def parse_layout(data: Any) -> Dict[str, Any]:
    """Stored/submitted layout -> normalised layout. None means the default."""
    if data is None:
        return copy.deepcopy(DEFAULT_LAYOUT)
    if not isinstance(data, dict) or set(data) - {"widgets", "pinned"}:
        raise LayoutError("layout is {widgets, pinned}")
    raw = data.get("widgets", [])
    if not isinstance(raw, list):
        raise LayoutError("widgets must be a list")
    widgets: List[Dict[str, Any]] = [_widget(w) for w in raw]
    ids = [w["id"] for w in widgets]
    if len(set(ids)) != len(ids):
        raise LayoutError("a widget appears twice")
    # widgets added after the layout was saved go to the end, with their defaults
    widgets += [dict(d) for d in DEFAULT_LAYOUT["widgets"] if d["id"] not in ids]

    pinned = data.get("pinned", [])
    if not isinstance(pinned, list) or len(pinned) > MAX_PINNED:
        raise LayoutError(f"pinned must be a list of at most {MAX_PINNED} service ids")
    if not all(isinstance(p, str) and _SERVICE_ID.match(p) for p in pinned):
        raise LayoutError("pinned holds service ids only")
    if len(set(pinned)) != len(pinned):
        raise LayoutError("a service is pinned twice")
    return {"widgets": widgets, "pinned": list(pinned)}
