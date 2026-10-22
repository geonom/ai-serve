"""Downloads a plugin offers (contract endpoint `install`, docs/plugins.md): the gateway lists them across plugins
and starts one on the owner's request. Owner, 2026-10-22: missing models should be downloadable from the Spielwiese
and the dashboard instead of empty model lists -- only by the owner (this machine / the owner's devices; the
Spielwiese passes it on only for its owner).

Rules: the request names only an item id (the plugin decides which files, with checksums); unknown plugins, plugins
without `install`, and ids that aren't plain item ids are refused; whatever the plugin answers is checked field by
field."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import registry
from backend.plugins.contracts import ContractError, normalize_install, parse_provides
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8066


class Fake:
    def __init__(self):
        self.started = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert (request.url.host, request.url.port) == ("127.0.0.1", PORT)
        if request.url.path == "/plugin/install" and request.method == "GET":
            return httpx.Response(200, json={"items": [
                {"id": "iris-turbo", "label": "Iris-3B Turbo", "size_mb": 5500, "state": "missing",
                 "license": "Apache-2.0", "description": "Images from text", "evil": "dropped"},
                {"id": "bad id!", "label": "x", "size_mb": 1, "state": "missing"},
                {"id": "voxcpm", "label": "VoxCPM2", "size_mb": 4000, "state": "downloading", "done_mb": 1200}]})
        if request.url.path == "/plugin/install" and request.method == "POST":
            body = json.loads(request.content)
            self.started.append(body)
            return httpx.Response(202, json={"id": body["id"], "state": "downloading", "size_mb": 5500})
        return httpx.Response(404, json={"detail": request.url.path})


def manifest(pid, provides):
    return ServiceManifest(id=pid, name=pid, description="d", project_dir=Path("/x"), command=("./run.sh",), env={},
                           port=PORT, bind="127.0.0.1", match=(pid,), identity="none", ui=False,
                           provides=parse_provides(provides))


@pytest.fixture
def fake(monkeypatch):
    from backend.routers import services as svc
    f = Fake()
    ms = [manifest("sd-serve", {"image": {"catalog": "/c", "generate": "/g", "install": "/plugin/install"}}),
          manifest("other", {"image": {"catalog": "/c", "generate": "/g"}})]
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry(ms, [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(f))

    async def up(p, timeout_s=40.0):
        return True
    monkeypatch.setattr(registry, "ensure_running", up)
    return f


# ------------------------------------------------------------------ rules (pure)

def test_install_list_is_checked_field_by_field():
    items, errors = normalize_install({"items": [
        {"id": "a", "label": "A", "size_mb": 10, "state": "installed", "x": 1},
        {"id": "b", "size_mb": -5, "state": "weird"},
        {"id": "../c", "state": "missing"},
        "not an object"]})
    assert items[0] == {"id": "a", "label": "A", "description": "", "size_mb": 10, "done_mb": None, "state": "installed",
                        "error": "", "license": ""}
    assert items[1]["state"] == "missing" and items[1]["size_mb"] is None    # unknown state -> missing, bad size -> None
    assert len(items) == 2 and len(errors) == 2


def test_install_endpoint_is_a_common_endpoint():
    p = parse_provides({"tts": {"catalog": "/c", "generate": "/g", "install": "/i"}})
    assert p["tts"].endpoints["install"] == "/i"
    with pytest.raises(ContractError):
        parse_provides({"tts": {"catalog": "/c", "generate": "/g", "install": "http://elsewhere/x"}})


# ------------------------------------------------------------------ gateway

def test_list_across_plugins(fake):
    r = client.get("/api/plugins/installs")
    assert r.status_code == 200
    items = r.json()["items"]
    assert [(i["plugin"], i["id"], i["state"]) for i in items] == [("sd-serve", "iris-turbo", "missing"),
                                                                    ("sd-serve", "voxcpm", "downloading")]
    assert "evil" not in items[0] and items[1]["done_mb"] == 1200
    assert items[0]["capability"] == "image"           # so a page shows only its own downloads
    assert any("bad id" in e for e in r.json()["errors"])


def test_start_a_download(fake):
    r = client.post("/api/plugins/sd-serve/install", json={"id": "iris-turbo"})
    assert r.status_code == 202 and r.json()["state"] == "downloading"
    assert fake.started == [{"id": "iris-turbo"}]          # only the item id goes to the plugin


@pytest.mark.parametrize("plugin, body, status", [
    ("nope", {"id": "iris-turbo"}, 404),                    # unknown plugin
    ("other", {"id": "iris-turbo"}, 404),                   # plugin without install
    ("sd-serve", {"id": "../../etc"}, 400),
    ("sd-serve", {"id": "x", "url": "https://evil/x"}, 400),  # nothing but the id
    ("sd-serve", {}, 400),
    ("sd-serve", ["iris-turbo"], 400),
])
def test_refused(fake, plugin, body, status):
    assert client.post(f"/api/plugins/{plugin}/install", json=body).status_code == status
    assert fake.started == []


def test_not_from_the_lan(fake):
    lan = TestClient(app, client=("192.168.1.77", 50000))
    assert lan.post("/api/plugins/sd-serve/install", json={"id": "iris-turbo"}).status_code == 403
