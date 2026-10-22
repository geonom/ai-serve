"""backend/images/layout.py -- positions for MOSS-Image (layout generation).

Callers (other apps, MCP tools, later people via /public) send elements with a relative box
[left, top, right, bottom] in 0..1, a kind (object / character / text), a caption and -- for text --
the exact text to render. The gateway converts them into MOSS-Image's layout JSON exactly like
ComfyUI's MOSSImageLayoutPrompt node (pixel edges, origin at the canvas centre, yxyx). Input is
untrusted: bounded counts and lengths, boxes inside the canvas, no control characters."""
import json
import math

import pytest

from image_serve.moss_layout import LayoutError, moss_prompt, parse_layout, parse_size


def el(**over):
    d = {"box": [0.1, 0.2, 0.5, 0.9], "kind": "object", "caption": "ein roter Ball"}
    d.update(over)
    return d


def test_parse_valid():
    [e] = parse_layout([el()])
    assert (e.box, e.kind, e.caption, e.text) == ((0.1, 0.2, 0.5, 0.9), "object", "ein roter Ball", None)
    [t] = parse_layout([el(kind="text", text="Willkommen", caption="Holzschild")])
    assert t.text == "Willkommen"
    assert parse_layout(None) == [] and parse_layout([]) == []


@pytest.mark.parametrize("bad", [
    el(box=[0.5, 0.2, 0.1, 0.9]),           # left > right
    el(box=[0.1, 0.9, 0.5, 0.2]),           # top > bottom
    el(box=[-0.1, 0.2, 0.5, 0.9]),          # outside
    el(box=[0.1, 0.2, 1.5, 0.9]),
    el(box=[0.1, 0.2, 0.5]),                # wrong length
    el(box=[0.1, 0.2, 0.5, "x"]),
    el(box=[0.1, 0.2, 0.5, float("nan")]),
    el(box=[0.1, 0.2, 0.105, 0.9]),         # too thin to mean anything
    el(box=[0.1, 0.2, True, 0.9]),
    el(kind="monster"),
    el(caption=""),
    el(caption="x" * 401),
    el(caption="a\x00b"),
    el(kind="text"),                        # text box without text
    el(kind="text", text="x" * 201),
    el(kind="object", text="nur bei Text"),
    el(palette=["#ff0000", "red"]),
    el(evil=1),                             # unknown field
    "not an object",
])
def test_parse_refuses(bad):
    with pytest.raises(LayoutError):
        parse_layout([bad])


def test_bounded_count():
    with pytest.raises(LayoutError):
        parse_layout([el()] * 21)
    with pytest.raises(LayoutError):
        parse_layout("x")


def test_palette():
    [e] = parse_layout([el(palette=["#FF0000", "#00ff00"])])
    assert e.palette == ("#ff0000", "#00ff00")


def test_moss_prompt_matches_the_comfyui_node():
    # same arithmetic as comfy_extras/nodes_moss_image.py (0-1000 boxes there, 0-1 here)
    elements = parse_layout([el(box=[0.1, 0.2, 0.5, 0.9], kind="character", caption="Bäckerin"),
                             el(box=[0.55, 0.05, 0.95, 0.25], kind="text", text="Frisch", caption="Schild",
                                palette=["#aa5500"])])
    w, h = 1024, 768
    out = json.loads(moss_prompt("Bäckerei am Morgen", elements, w, h))
    assert out["summary"] == {"caption": "Bäckerei am Morgen"}
    assert out["bbox_coord_space"] == "abs_centralized" and out["bbox_format"] == "yxyx"
    first, second = out["items"]
    assert first == {"label": "character", "caption": "Bäckerin",
                     "bbox": [math.floor(0.2 * h) - h // 2, math.floor(0.1 * w) - w // 2,
                              math.ceil(0.9 * h) - h // 2, math.ceil(0.5 * w) - w // 2]}
    assert second["label"] == "text" and second["text"] == "Frisch" and second["palette"] == ["#aa5500"]


def test_no_layout_is_the_plain_prompt():
    assert moss_prompt("ein Hund", [], 1024, 1024) == "ein Hund"


@pytest.mark.parametrize("size,expected", [("1024x1024", (1024, 1024)), ("1024x768", (1024, 768)),
                                           ("768x1344", (768, 1344)), (None, (1024, 1024))])
def test_sizes(size, expected):
    assert parse_size(size) == expected


@pytest.mark.parametrize("size", ["1000x1000", "4096x4096", "128x128", "abc", "1024x", "1024x1024x2"])
def test_sizes_refused(size):
    with pytest.raises(LayoutError):
        parse_size(size)
