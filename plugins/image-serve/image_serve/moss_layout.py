"""Positions for MOSS-Image (layout generation) -- pure rules, no I/O.

Vocabulary: a *layout* is a list of *elements*; each has a relative *box* [left, top, right, bottom]
in 0..1 (independent of the image size -- easy for people and language models), a *kind*
(object / character / text), a *caption* describing it and, for text, the exact *text* to render;
optionally a colour *palette*. `moss_prompt()` turns prompt + layout into MOSS-Image's layout JSON
exactly like ComfyUI's MOSSImageLayoutPrompt node (comfy_extras/nodes_moss_image.py): pixel edges
enclosing the box, origin at the canvas centre, yxyx.

Input is untrusted (other apps, MCP tools, later /public): at most 20 elements, boxes inside the
canvas and at least 2 % wide/high, bounded caption/text lengths, no control characters, no unknown
fields. Tests: tests/test_layout.py. (Moved here from ai-serve-ui backend/images/layout.py, 2026-10-21.)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

MAX_ELEMENTS = 20
MAX_CAPTION = 400
MAX_TEXT = 200
MIN_EXTENT = 0.02
KINDS = ("object", "character", "text")
_FIELDS = {"box", "kind", "caption", "text", "palette"}
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SIZE_MIN, SIZE_MAX, SIZE_STEP = 256, 2048, 16


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Element:
    box: Tuple[float, float, float, float]
    kind: str
    caption: str
    text: Optional[str] = None
    palette: Tuple[str, ...] = ()


def _clean(value: Any, field: str, limit: int, i: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LayoutError(f"element {i}: {field} missing")
    if len(value) > limit or _CONTROL.search(value):
        raise LayoutError(f"element {i}: {field} too long or contains control characters")
    return value.strip()


def _box(value: Any, i: int) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LayoutError(f"element {i}: box must be [left, top, right, bottom] in 0..1")
    nums = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise LayoutError(f"element {i}: box values must be numbers")
        nums.append(float(v))
    left, top, right, bottom = nums
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise LayoutError(f"element {i}: box must lie inside the image (0..1, left<right, top<bottom)")
    if right - left < MIN_EXTENT or bottom - top < MIN_EXTENT:
        raise LayoutError(f"element {i}: box too small")
    return left, top, right, bottom


def parse_layout(raw: Any) -> List[Element]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LayoutError("layout must be a list of elements")
    if len(raw) > MAX_ELEMENTS:
        raise LayoutError(f"at most {MAX_ELEMENTS} layout elements")
    out = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise LayoutError(f"element {i}: object expected")
        unknown = set(e) - _FIELDS
        if unknown:
            raise LayoutError(f"element {i}: unknown field(s) {', '.join(sorted(unknown))}")
        kind = e.get("kind", "object")
        if kind not in KINDS:
            raise LayoutError(f"element {i}: kind must be one of {', '.join(KINDS)}")
        text = e.get("text")
        if kind == "text":
            text = _clean(text, "text", MAX_TEXT, i)
        elif text is not None:
            raise LayoutError(f"element {i}: text only for kind 'text'")
        palette = e.get("palette") or []
        if not isinstance(palette, list) or len(palette) > 6 or not all(isinstance(c, str) and _HEX.match(c) for c in palette):
            raise LayoutError(f"element {i}: palette must be up to 6 colours like #aa5500")
        out.append(Element(box=_box(e.get("box"), i), kind=kind, caption=_clean(e.get("caption"), "caption", MAX_CAPTION, i),
                           text=text, palette=tuple(c.lower() for c in palette)))
    return out


def parse_size(size: Optional[str]) -> Tuple[int, int]:
    """'WIDTHxHEIGHT' (multiples of 16, 256..2048); missing = 1024x1024."""
    if size is None:
        return 1024, 1024
    m = re.fullmatch(r"(\d{3,4})x(\d{3,4})", str(size))
    if not m:
        raise LayoutError("size must be WIDTHxHEIGHT, e.g. 1024x768")
    w, h = int(m.group(1)), int(m.group(2))
    for v in (w, h):
        if not (SIZE_MIN <= v <= SIZE_MAX) or v % SIZE_STEP:
            raise LayoutError(f"width/height must be multiples of {SIZE_STEP} between {SIZE_MIN} and {SIZE_MAX}")
    return w, h


def moss_prompt(prompt: str, elements: List[Element], width: int, height: int) -> str:
    """MOSS-Image layout JSON (same arithmetic as ComfyUI's MOSSImageLayoutPrompt); plain prompt without layout."""
    if not elements:
        return prompt
    items = []
    for e in elements:
        left, top, right, bottom = e.box
        item: dict = {"label": e.kind, "caption": e.caption}
        if e.kind == "text":
            item["text"] = e.text
        if e.palette:
            item["palette"] = list(e.palette)
        item["bbox"] = [math.floor(round(top * height, 9)) - height // 2, math.floor(round(left * width, 9)) - width // 2,
                        math.ceil(round(bottom * height, 9)) - height // 2, math.ceil(round(right * width, 9)) - width // 2]
        items.append(item)
    layout = {"summary": {"caption": prompt.strip()}, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
