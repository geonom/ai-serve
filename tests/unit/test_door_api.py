"""backend/routers/door.py wiring: shares stored, rules enforced, routes sent to the door,
changes only from this machine / the owner's devices, identity env given to shared apps."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import httpx

from backend.door.policy import Share
from backend.plugins import registry
from backend.plugins.contracts import parse_provides
from backend.routers import door as door_router
from backend.services.domain import ServiceManifest, ServiceRegistry


def m(id, identity, port, ui=True, provides=None):
    return ServiceManifest(id=id, name=id.title(), description="d", project_dir=Path("/x"), command=("./run.sh",),
                           env={}, port=port, bind="127.0.0.1", match=("x",), identity=identity, ui=ui,
                           provides=parse_provides(provides) if provides else {})


class FakeDoor:
    """The access plugin (edge-door's shape), reached only through the plugin registry."""
    def __init__(self):
        self.routes = {"routes": []}
        self.puts = 0
        self.down = False
        self.seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down")
        assert request.url.host == "127.0.0.1" and request.url.port == 8046
        path, method = request.url.path, request.method
        self.seen.append((method, path))
        if path == "/api/routes" and method == "GET":
            return httpx.Response(200, json=self.routes)
        if path == "/api/routes" and method == "PUT":
            self.routes, self.puts = json.loads(request.content), self.puts + 1
            return httpx.Response(200, json={"ok": True})
        if path == "/status":
            return httpx.Response(200, json={"facts": ["Apps: x"], "overview": None, "domain": "apps.example.org"})
        if path == "/api/people":
            return httpx.Response(200, json={"people": [{"sub": "anna", "name": "Anna"}]})
        if path == "/api/security":
            self.security_query = dict(request.url.params)
            return httpx.Response(200, json={"days": [{"day": "2026-10-19", "total": 10, "leak": 0}], "leak_total": 0,
                                             "top_probes": [["/.env", 3]]})
        raise AssertionError(path)


DOOR = m("door", "none", 8046, ui=False, provides={"access": {"shares": "/api/routes", "status": "/status",
                                                               "security": "/api/security", "people": "/api/people"}})


@pytest.fixture
def client(monkeypatch, tmp_path):
    from backend.main import app
    reg = ServiceRegistry([m("pv-calc", "multi", 5180), m("story", "single", 8008), m("tts", "none", 8050, ui=False),
                           DOOR], [], {})
    monkeypatch.setattr(door_router.services_router, "_registry", lambda: reg)
    monkeypatch.setattr(door_router, "DOOR_SHARES_FILE", str(tmp_path / "shares.json"))
    fake = FakeDoor()
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(fake))
    monkeypatch.setenv("LLAMA_SERVE_OWNER_SUB", "alex")
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.get("/api/door")          # the domain comes from the door's own status
    c.fake = fake
    c.shares = tmp_path / "shares.json"
    c.reg = reg
    return c


def test_share_multi_app(client):
    r = client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"], "users": ["anna"]})
    assert r.status_code == 200, r.text
    assert r.json()["share"]["url"] == "https://sonne.apps.example.org/"
    assert client.fake.routes["routes"][0]["upstream"] == "http://127.0.0.1:5180"
    assert json.loads(client.shares.read_text())["pv-calc"]["users"] == ["anna"]


def test_single_app_only_owner(client):
    r = client.put("/api/door/shares/story", json={"name": "story", "groups": ["apps"]})
    assert r.status_code == 400 and "only be shared with you" in r.json()["detail"]
    assert client.put("/api/door/shares/story", json={"name": "story", "users": ["alex"]}).status_code == 200


def test_no_api_service_and_strict_types(client):
    assert client.put("/api/door/shares/tts", json={"name": "tts", "groups": ["apps"]}).status_code == 400
    assert client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": "apps"}).status_code == 422
    assert client.put("/api/door/shares/nope", json={"name": "x", "groups": ["a"]}).status_code == 404


def test_not_from_the_lan(client, monkeypatch):
    from backend.main import app
    monkeypatch.setattr(door_router.tailscale, "info", lambda: None)
    lan = TestClient(app, client=("192.168.1.50", 50000), headers={})
    # the dashboard access policy refuses the LAN before (403); even if it let it through, the
    # share endpoint itself only accepts this machine / the owner's devices
    assert lan.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]}).status_code == 403


def test_delete_and_overview(client):
    client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]})
    o = client.get("/api/door").json()
    assert o["available"] and o["owner"] == "alex" and o["people"][0]["sub"] == "anna"
    assert o["shares"]["pv-calc"]["url"] == "https://sonne.apps.example.org/"
    assert client.delete("/api/door/shares/pv-calc").status_code == 200
    assert client.fake.routes == {"routes": []}


def test_manifest_turned_single_drops_out(client, monkeypatch):
    client.shares.write_text(json.dumps({"story": {"name": "story", "groups": ["apps"]}}))
    o = client.get("/api/door").json()
    assert "only be shared with you" in o["shares"]["story"]["error"]
    assert client.fake.routes == {"routes": []}


def test_door_down_is_reported(client):
    client.fake.down = True
    r = client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]})
    assert r.status_code == 200 and "unreachable" in r.json()["door_error"]


def test_identity_env_for_shared_app(client):
    client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]})
    assert door_router.identity_env_for("pv-calc")["LLAMA_SERVE_IDENTITY_AUDIENCE"] == "sonne.apps.example.org"
    assert door_router.identity_env_for("story") == {}



def test_security_overview_is_passed_through(client):
    """What the door turned away, per day (edge-door's /api/security)."""
    r = client.get("/api/door/security?days=7")
    assert r.status_code == 200 and r.json()["available"] is True
    assert r.json()["leak_total"] == 0 and r.json()["days"][0]["total"] == 10
    assert client.fake.security_query == {"days": "7"}
    assert client.get("/api/door/security?days=abc").status_code == 422


def test_security_overview_when_the_door_is_down(client):
    client.fake.down = True
    r = client.get("/api/door/security")
    assert r.status_code == 200 and r.json()["available"] is False


def test_without_an_access_plugin_the_door_is_simply_not_set_up(client, monkeypatch):
    # the door is an optional extension: no app provides "access" -> nothing is contacted, no error shown
    reg = ServiceRegistry([s for s in client.reg.services if s.id != "door"], [], {})
    monkeypatch.setattr(door_router.services_router, "_registry", lambda: reg)
    seen = len(client.fake.seen)
    o = client.get("/api/door").json()
    assert o["configured"] is False and o["available"] is False and o["door_error"] is None
    assert client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]}).status_code == 409
    assert len(client.fake.seen) == seen


def test_domain_is_remembered_while_the_door_is_down(client):
    client.put("/api/door/shares/pv-calc", json={"name": "sonne", "groups": ["apps"]})
    client.fake.down = True
    assert door_router.identity_env_for("pv-calc")["LLAMA_SERVE_IDENTITY_AUDIENCE"] == "sonne.apps.example.org"
    assert client.get("/api/door").json()["domain"] == "apps.example.org"
