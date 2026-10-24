"""Overview page backend: event feed (backend/events.py) and the saved layout
(GET/PUT /api/dashboard/layout, rules in backend/dashboard_layout.py)."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import events
from backend.routers import overview


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(overview, "DASHBOARD_LAYOUT_FILE", str(tmp_path / "layout.json"))
    events.clear()
    app = FastAPI()
    app.include_router(overview.router)
    yield TestClient(app)
    events.clear()


def test_events_newest_first_and_limited(client):
    for i in range(5):
        events.record("model", f"loaded m{i}")
    events.record("gpu", "refused", level="warn")
    data = client.get("/api/events?limit=3").json()["events"]
    assert [e["text"] for e in data] == ["refused", "loaded m4", "loaded m3"]
    assert data[0]["level"] == "warn" and data[0]["kind"] == "gpu" and data[0]["t"] > 0


def test_events_ring_buffer_and_text_cut():
    events.clear()
    for i in range(events.MAX_EVENTS + 10):
        events.record("x", "y" * 1000)
    got = events.recent(10_000)
    assert len(got) == events.MAX_EVENTS and len(got[0]["text"]) <= 300
    events.clear()


def test_layout_default_then_saved(client, tmp_path):
    d = client.get("/api/dashboard/layout").json()
    assert [w["id"] for w in d["widgets"]][0] == "stats" and d["pinned"] == []
    new = {"widgets": [{"id": "events", "size": "full", "hidden": False}], "pinned": ["storytime"]}
    r = client.put("/api/dashboard/layout", json=new)
    assert r.status_code == 200 and r.json()["widgets"][0]["id"] == "events"
    assert client.get("/api/dashboard/layout").json()["pinned"] == ["storytime"]
    stored = json.loads((tmp_path / "layout.json").read_text())
    assert stored["pinned"] == ["storytime"]


def test_layout_refuses_garbage(client, tmp_path):
    r = client.put("/api/dashboard/layout", json={"widgets": [{"id": "evil", "size": "full"}]})
    assert r.status_code == 400
    assert not (tmp_path / "layout.json").exists()


def test_broken_layout_file_falls_back_to_default(client, tmp_path):
    (tmp_path / "layout.json").write_text("{not json")
    assert client.get("/api/dashboard/layout").json()["pinned"] == []


def test_events_keep_template_and_values_for_the_ui_language():
    events.clear()
    events.record("service", "{name} started", name="Playground")
    events.record("gpu", "plain {text} without values")
    got = events.recent(2)
    assert got[1] == {**got[1], "text": "Playground started", "template": "{name} started", "vars": {"name": "Playground"}}
    assert got[0]["text"] == "plain {text} without values" and "template" not in got[0]
