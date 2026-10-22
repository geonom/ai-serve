"""Overview page layout (backend/dashboard_layout.py) -- pure rules.

The layout is plain owner preference, but it is written through the API and read back
into the page, so only known widgets, sizes and service ids get through."""
import pytest

from backend.dashboard_layout import DEFAULT_LAYOUT, LayoutError, SIZES, WIDGETS, parse_layout


def test_default_has_every_widget_once():
    layout = parse_layout(None)
    assert [w["id"] for w in layout["widgets"]] == list(WIDGETS)
    assert layout == DEFAULT_LAYOUT and layout["pinned"] == []


def test_valid_layout_round_trips():
    data = {"widgets": [{"id": "events", "size": "full", "hidden": False},
                        {"id": "stats", "size": "half", "hidden": True}],
            "pinned": ["storytime", "munin"]}
    layout = parse_layout(data)
    assert [w["id"] for w in layout["widgets"]][:2] == ["events", "stats"]
    # widgets missing from a stored layout (e.g. added later) are appended
    assert {w["id"] for w in layout["widgets"]} == set(WIDGETS)
    assert layout["widgets"][1] == {"id": "stats", "size": "half", "hidden": True}
    assert layout["pinned"] == ["storytime", "munin"]


@pytest.mark.parametrize("bad", [
    {"widgets": [{"id": "evil", "size": "half"}]},
    {"widgets": [{"id": "stats", "size": "huge"}]},
    {"widgets": [{"id": "stats", "size": "half", "hidden": "yes"}]},
    {"widgets": [{"id": "stats", "size": "half"}, {"id": "stats", "size": "full"}]},
    {"widgets": [{"id": "stats", "size": "half", "x": 1}]},
    {"widgets": "stats"},
    {"pinned": ["<script>"]},
    {"pinned": ["a", "a"]},
    {"pinned": [f"s{i}" for i in range(31)]},
    {"pinned": "storytime"},
    {"other": 1},
    [],
])
def test_refuses_garbage(bad):
    with pytest.raises(LayoutError):
        parse_layout(bad)


def test_sizes_are_known():
    assert set(SIZES) == {"narrow", "half", "wide", "full"}
