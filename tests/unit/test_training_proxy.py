"""/api/training/* is forwarded to train-serve (its own repo since 2026-10-17), so remote
clients keep their URL. train-serve checks the bearer token itself; the gateway passes
it on unchanged and forwards nothing outside that prefix (the token-free monitor API
used by train-serve's own page stays on loopback)."""
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import training


@pytest.fixture
def seen(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(201 if request.method == "POST" else 200, json={"path": request.url.path,
                              "query": request.url.query.decode()}, headers={"Content-Disposition": "attachment; x"})

    monkeypatch.setattr(training, "_transport", httpx.MockTransport(handler))
    return calls


@pytest.fixture
def client(seen):
    app = FastAPI()
    app.include_router(training.router)
    return TestClient(app)


def test_forwards_path_query_body_and_token(client, seen):
    r = client.post("/api/training/jobs?x=1", content=b"dataset-bytes",
                    headers={"Authorization": "Bearer abc", "Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 201 and r.json() == {"path": "/api/training/jobs", "query": "x=1"}
    req = seen[-1]
    assert req.method == "POST" and req.content == b"dataset-bytes"
    assert req.headers["authorization"] == "Bearer abc"
    assert req.headers["content-type"].startswith("multipart/form-data")
    assert r.headers["content-disposition"] == "attachment; x"


def test_get_and_delete(client, seen):
    assert client.get("/api/training/jobs/abc/log?tail=5").json()["path"] == "/api/training/jobs/abc/log"
    assert client.delete("/api/training/jobs/abc").status_code == 200


@pytest.mark.parametrize("path", ["/api/training/../services", "/api/training/jobs/%2e%2e/x",
                                  "/api/training/jobs/a%2Fb"])
def test_traversal_refused(client, seen, path):
    before = len(seen)
    assert client.get(path).status_code in (400, 404)
    assert len(seen) == before


def test_spoofed_forwarding_headers_not_passed(client, seen):
    client.get("/api/training/health", headers={"X-Forwarded-For": "1.2.3.4", "Cookie": "a=b"})
    h = seen[-1].headers
    assert "cookie" not in h and h.get("x-forwarded-for") != "1.2.3.4"


def test_unreachable_is_503(client, monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(training, "_transport", httpx.MockTransport(down))

    async def no_start():
        return False
    monkeypatch.setattr(training.train_client, "ensure_running", no_start)
    r = client.get("/api/training/jobs")
    assert r.status_code == 503 and "train-serve" in r.json()["detail"]
