"""Wiring of the access policy (backend/access/middleware.py): peer from the ASGI
scope, owner + whois looked up via Tailscale (cached), 403 on refusal, websockets
closed. Rules themselves: test_access_policy.py."""
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from backend.access.middleware import AccessMiddleware
from backend.access.policy import AccessPolicy, ApiKey

OWNER = "owner@github"


def make(whois=None, owner=OWNER, key=""):
    calls = []

    def _whois(ip):
        calls.append(ip)
        return (whois or {}).get(ip)

    app = FastAPI()

    @app.get("/api/status")
    def status():
        return {"ok": True}

    @app.get("/public/v1/models")
    def models():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_text("hi")
        await sock.close()

    app.add_middleware(AccessMiddleware, policy=AccessPolicy(ApiKey(key)),
                       owner=lambda: owner, whois=_whois)
    return app, calls


def client(app, host):
    return TestClient(app, client=(host, 40000))


def test_loopback_ok_without_whois():
    app, calls = make()
    assert client(app, "127.0.0.1").get("/api/status").status_code == 200
    assert calls == []


def test_owner_device_ok_and_whois_cached():
    app, calls = make(whois={"100.64.7.11": OWNER})
    c = client(app, "100.64.7.11")
    assert c.get("/api/status").status_code == 200
    assert c.get("/api/status").status_code == 200
    assert calls == ["100.64.7.11"]


def test_lan_refused_with_reason():
    app, _ = make()
    r = client(app, "192.168.1.40").get("/api/status")
    assert r.status_code == 403 and "API key" in r.json()["detail"]


def test_public_untouched():
    app, calls = make()
    assert client(app, "85.13.165.151").get("/public/v1/models").status_code == 200


def test_whois_failure_refuses():
    def boom(ip):
        raise RuntimeError("tailscale not running")
    app = FastAPI()

    @app.get("/api/status")
    def status():
        return {}
    app.add_middleware(AccessMiddleware, policy=AccessPolicy(ApiKey("")), owner=lambda: OWNER, whois=boom)
    assert client(app, "100.64.7.11").get("/api/status").status_code == 403


def test_websocket_refused_for_lan():
    app, _ = make()
    with pytest.raises(WebSocketDisconnect):
        with client(app, "192.168.1.40").websocket_connect("/ws") as ws:
            ws.receive_text()


def test_websocket_ok_for_loopback():
    app, _ = make()
    with client(app, "127.0.0.1").websocket_connect("/ws") as ws:
        assert ws.receive_text() == "hi"


def test_real_app_refuses_lan_and_strangers(monkeypatch):
    """The production app (backend.main), with Tailscale faked."""
    from backend import network
    from backend.services import tailscale
    from backend.main import app as real_app
    monkeypatch.setattr(network, "_ACTIVE", network.load_provider("tailscale", ""))   # not the default any more
    monkeypatch.setattr(tailscale, "info", lambda: tailscale.TailscaleInfo(owner=OWNER, dns_name="my-pc.x.ts.net"))
    monkeypatch.setattr(tailscale, "whois", lambda ip: {"100.64.7.11": OWNER}.get(ip, "guest@example.com"))
    for path in ("/api/version", "/", "/api/training-monitor/jobs", "/v1/models"):
        assert client(real_app, "192.168.1.40").get(path).status_code == 403, path
        assert client(real_app, "100.90.1.2").get(path).status_code == 403, path
    assert client(real_app, "100.64.7.11").get("/api/version").status_code == 200
    assert client(real_app, "127.0.0.1").get("/api/version").status_code == 200


def test_the_real_app_refuses_rebinding_and_foreign_pages():
    """main.py switches the Host/Origin check on (backend/access/hosts.py): a page that points its own name at this
    machine, or a foreign page posting to it, is refused even from loopback; this machine's own pages pass."""
    from backend.main import app as real
    c = TestClient(real, client=("127.0.0.1", 50000))
    assert c.get("/api/version", headers={"host": "evil.example:8030"}).status_code == 403
    r = c.post("/api/models/unload_all", headers={"host": "localhost:8030", "origin": "https://evil.example"})
    assert r.status_code == 403 and "foreign page" in r.json()["detail"]
    assert c.get("/api/version", headers={"host": "localhost:8030", "origin": "http://localhost:8030"}).status_code == 200
