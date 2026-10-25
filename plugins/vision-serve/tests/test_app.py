"""The plugin server (vision_plugin/app.py): contract endpoints, request checks, and a real run through the worker
process -- here on the CPU, because ai-serve's GPU manager is unreachable in tests (that is also the fallback when
it says no)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vision_plugin import app as plugin

FIX = Path(__file__).parent / "fixtures"
client = TestClient(plugin.app)


@pytest.fixture(autouse=True)
def no_gateway(monkeypatch):
    monkeypatch.setattr(plugin, "GATEWAY", "http://127.0.0.1:9")      # nothing listens: no GPU -> CPU


def img(name="portrait.jpg"):
    return ("image", (name, (FIX / name).read_bytes(), "image/jpeg"))


def test_catalog():
    row = client.get("/v1/vision/models").json()["data"][0]
    assert row["id"] == "auraface" and row["tasks"] == ["faces"] and row["default"] is True
    assert row["embedding"]["dim"] == 512 and row["embedding"]["model"] == "auraface-v1"
    assert row["commercial_use"] is True


@pytest.mark.parametrize("data,n,why", [
    ({"model": "owl"}, 1, "unknown model"),
    ({"tasks": "objects"}, 1, "faces only"),
    ({"tasks": "faces,objects"}, 1, "faces only"),
    ({}, 17, "1 to 16"),
])
def test_requests_are_checked(data, n, why):
    r = client.post("/v1/vision/analyze", files=[img()] * n, data=data)
    assert r.status_code == 400 and why in r.json()["detail"]


def test_not_installed_is_said(monkeypatch):
    monkeypatch.setattr(plugin, "installed", lambda: False)
    r = client.post("/v1/vision/analyze", files=[img()])
    assert r.status_code == 400 and "not installed" in r.json()["detail"]


def test_contract_endpoints():
    assert client.get("/health").json() == {"ok": True}
    s = client.get("/plugin/status").json()
    assert s["workers"] == [] and s["facts"]
    assert client.get("/plugin/install").json()["items"][0]["id"] == "auraface"
    assert client.post("/plugin/install", json={"id": "other"}).status_code == 404
    assert client.post("/plugin/release", json={"worker": "x"}).status_code == 404


@pytest.mark.skipif(not plugin.installed(), reason="models not installed")
def test_a_batch_through_the_worker_on_the_cpu():
    with TestClient(plugin.app) as c:
        r = c.post("/v1/vision/analyze", files=[img("portrait.jpg"), img("group_bw.jpg"), img("landscape.jpg"),
                                                ("image", ("x.jpg", b"garbage", "image/jpeg"))],
                   data={"embed": "false"})
        assert r.status_code == 200, r.text
        rows = r.json()["images"]
        assert [len(x["faces"]) for x in rows] == [1, 6, 0, 0]
        assert "error" in rows[3] and "embedding" not in rows[0]["faces"][0]
        assert rows[0]["width"] == 384
        assert plugin.WORKER.gpu is None and plugin.WORKER.pid
        assert c.post("/plugin/release", json={}).json() == {"released": ["faces"]}
        assert plugin.WORKER.pid is None
