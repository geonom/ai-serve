"""/api/services over HTTP + the real process runtime (backend/services/runtime.py).
Domain rules are covered in test_services_domain.py; here: wiring, and that start/
stop really act only on what the rules allow."""
import json
import socket
import stat
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import services as svc
from backend.services import runtime
from backend.services.domain import MANIFEST_NAME, ObservedProcess


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


SERVER = """#!{python}
import http.server, os, socketserver, sys
class S(http.server.HTTPServer):   # no getfqdn(): it can take ~35 s on a Mac (CI)
    def server_bind(self):
        socketserver.TCPServer.server_bind(self); self.server_name, self.server_port = "127.0.0.1", self.server_address[1]
port = int(os.environ["DEMO_PORT"])
open("started_env.txt", "w").write(os.environ.get("LLAMA_SERVE_LOCAL_HOSTS", "-"))
S(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler).serve_forever()
"""


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "demo"
    d.mkdir()
    script = d / "run.py"
    script.write_text(SERVER.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    port = _free_port()
    (d / MANIFEST_NAME).write_text(json.dumps({
        "id": "demo", "name": "Demo", "description": "test service",
        "command": ["./run.py"], "env": {"DEMO_PORT": "{port}"}, "port": port, "match": ["run.py"],
        "identity": "multi",   # the exposure tests use the whole-tailnet level
    }))
    return d, port


@pytest.fixture
def client(tmp_path, project, monkeypatch):
    d, _ = project  # tmp_path/demo -> discovered as a child of the root tmp_path
    reg = tmp_path / "services.json"
    reg.write_text(json.dumps({"include": [str(tmp_path / 'missing')]}))
    monkeypatch.setattr(svc, "SERVICES_FILE", str(reg))
    monkeypatch.setattr(svc, "SERVICE_ROOTS", [str(tmp_path)])
    monkeypatch.setattr(svc, "SERVICE_LOGS_DIR", str(tmp_path / "logs"))
    app = FastAPI()
    app.include_router(svc.router)
    yield TestClient(app)
    # never leave a demo server behind
    for proc in list(svc._launched.values()):
        if proc.poll() is None:
            proc.kill()
    svc._launched.clear()


def _wait(pred, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _service(client):
    data = client.get("/api/services").json()
    return next(s for s in data["services"] if s["id"] == "demo"), data


def test_list_reports_services_and_errors(client, tmp_path):
    s, data = _service(client)
    assert s["status"] == "stopped" and s["instances"] == [] and s["port"]
    assert s["source"] == f"root {tmp_path}"
    assert len(data["errors"]) == 1 and "missing" in data["errors"][0]["dir"]
    assert data["roots"] == [str(tmp_path)]


def test_works_without_services_json(client, monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "SERVICES_FILE", str(tmp_path / "absent.json"))
    s, data = _service(client)
    assert s["id"] == "demo" and data["errors"] == []


def test_manifest_schema_endpoint(client):
    schema = client.get("/api/services/manifest-schema").json()
    assert schema["title"] and "command" in schema["properties"]


def test_start_open_stop_roundtrip(client, project):
    _, port = project
    r = client.post("/api/services/demo/start")
    assert r.status_code == 200, r.text
    assert _wait(lambda: _service(client)[0]["status"] == "running")
    s, _ = _service(client)
    inst = s["instances"][0]
    assert inst["ports"] == [port] and inst["open"] == {"port": port, "path": "/", "local_only": True}
    assert inst["managed"] is True

    assert client.post("/api/services/demo/start").status_code == 409  # already running

    r = client.post("/api/services/demo/stop", json={"pid": inst["pid"]})
    assert r.status_code == 200, r.text
    assert _wait(lambda: _service(client)[0]["status"] == "stopped")
    assert "log" in client.get("/api/services/demo/log").json()


def test_stop_refuses_foreign_pid(client, monkeypatch):
    killed = []
    monkeypatch.setattr(svc.runtime, "terminate", lambda pids, timeout=15: killed.append(pids))
    monkeypatch.setattr(svc.runtime, "snapshot", lambda: [
        ObservedProcess(pid=4242, ppid=1, cwd="/home/me", cmdline=("firefox",),
                        listening_ports=frozenset({5555}))])
    r = client.post("/api/services/demo/stop", json={"pid": 4242})
    assert r.status_code == 404 and killed == []


def test_start_refuses_occupied_port(client, project, monkeypatch):
    _, port = project
    monkeypatch.setattr(svc.runtime, "snapshot", lambda: [
        ObservedProcess(pid=7, ppid=1, cwd="/x", cmdline=("nginx",), listening_ports=frozenset({port}))])
    launched = []
    monkeypatch.setattr(svc.runtime, "launch", lambda plan, log_path: launched.append(plan))
    r = client.post("/api/services/demo/start")
    assert r.status_code == 409 and str(port) in r.json()["detail"] and launched == []


def test_unknown_service_404(client):
    assert client.post("/api/services/nope/start").status_code == 404
    assert client.post("/api/services/..%2f..%2fetc/start").status_code == 404
    assert client.get("/api/services/nope/log").status_code == 404


def test_launch_uses_no_shell_and_own_session(project, tmp_path):
    d, port = project
    from backend.services.domain import LaunchPlan
    proc = runtime.launch(LaunchPlan(argv=[str(d / "run.py")], cwd=d, env={"DEMO_PORT": str(port)}),
                          str(tmp_path / "x.log"))
    try:
        import os
        assert os.getsid(proc.pid) == proc.pid  # new session: survives a backend restart
        assert _wait(lambda: any(port in p.listening_ports for p in runtime.snapshot()))
    finally:
        runtime.terminate(frozenset({proc.pid}), timeout=5)
    assert proc.wait(timeout=5) is not None


def test_terminate_skips_pids_that_are_gone():
    assert runtime.terminate(frozenset({2 ** 22 + 12345}), timeout=1) == {"terminated": [], "killed": []}


# --------------------------------------------------------------------------- exposure

class FakeTailscale:
    def __init__(self, owner="owner@github", dns="my-pc.example.ts.net", logins=None):
        self.owner, self.dns = owner, dns
        self.serve = {}
        self.calls = []
        self.logins = logins or {}

    def info(self):
        from backend.services.tailscale import TailscaleInfo
        return None if self.owner is None else TailscaleInfo(owner=self.owner, dns_name=self.dns)

    def serve_status(self):
        return {"Web": {f"{self.dns}:{p}": {"Handlers": {"/": {"Proxy": t}}} for p, t in self.serve.items()}}

    def serve_set(self, port, target):
        self.calls.append(("set", port, target))
        self.serve[port] = target

    def serve_off(self, port):
        self.calls.append(("off", port))
        self.serve.pop(port, None)

    def whois(self, addr):
        return self.logins.get(addr)


class FakeGates:
    def __init__(self):
        self.specs = {}

    async def ensure(self, spec):
        self.specs[spec.listen_port] = spec

    async def stop(self, port):
        self.specs.pop(port, None)

    def running_ports(self):
        return set(self.specs)


TPORT = 31000  # fixed: the demo's random OS port + offsets could exceed 65535


@pytest.fixture
def exposure(client, project, tmp_path, monkeypatch):
    import json as _json
    d, _ = project
    m = _json.loads((d / MANIFEST_NAME).read_text())
    m["tailnet_port"] = TPORT
    (d / MANIFEST_NAME).write_text(_json.dumps(m))
    ts = FakeTailscale(logins={"100.70.0.2": "owner@github", "100.70.0.9": "friend@github"})
    gates = FakeGates()
    monkeypatch.setattr(svc, "tailscale", ts)
    monkeypatch.setattr(svc, "GATES", gates)
    monkeypatch.setattr(svc, "SERVICE_EXPOSURE_FILE", str(tmp_path / "exposure.json"))
    monkeypatch.setattr(svc, "_peer_address", lambda request: "127.0.0.1")
    return ts, gates


def _set(client, level):
    return client.post("/api/services/demo/exposure", json={"level": level})


def test_exposure_listed(client, exposure):
    s, data = _service(client)
    assert s["exposure"]["level"] == "local" and s["exposure"]["url"] is None
    assert data["tailscale"] == {"available": True, "owner": "owner@github", "dns_name": "my-pc.example.ts.net"}


def test_tailnet_then_owner_then_local(client, project, exposure, tmp_path):
    ts, gates = exposure
    _, port = project
    tport = TPORT

    r = _set(client, "tailnet")
    assert r.status_code == 200, r.text
    assert ts.serve == {tport: f"http://127.0.0.1:{port}"} and gates.specs == {}
    s, _ = _service(client)
    assert s["exposure"]["level"] == "tailnet"
    assert s["exposure"]["url"] == f"https://my-pc.example.ts.net:{tport}/"

    assert _set(client, "owner").status_code == 200
    assert ts.serve == {tport: f"http://127.0.0.1:{tport + 20000}"}
    assert gates.specs[tport + 20000].upstream_port == port

    assert _set(client, "local").status_code == 200
    assert ts.serve == {} and gates.specs == {}
    import json as _json
    assert _json.loads((tmp_path / "exposure.json").read_text()) == {"services": {}}


def test_unmanaged_serve_entries_untouched(client, exposure):
    ts, _ = exposure
    ts.serve = {443: "http://127.0.0.1:8000", 19999: "http://127.0.0.1:1"}
    assert _set(client, "tailnet").status_code == 200
    assert ts.serve[443] == "http://127.0.0.1:8000" and ts.serve[19999] == "http://127.0.0.1:1"


@pytest.mark.parametrize("peer,expected", [
    ("100.70.0.2", 200),    # owner's other device
    ("100.70.0.9", 403),    # another tailnet user
    ("192.168.1.20", 403),  # LAN
])
def test_who_may_change_exposure(client, exposure, monkeypatch, peer, expected):
    ts, _ = exposure
    monkeypatch.setattr(svc, "_peer_address", lambda request: peer)
    r = _set(client, "tailnet")
    assert r.status_code == expected
    if expected == 403:
        assert ts.serve == {} and ts.calls == []


def test_invalid_level(client, exposure):
    assert _set(client, "public").status_code == 422
    assert _set(client, "funnel").status_code == 422


def test_hand_made_route_to_service_removed_on_apply(client, project, exposure):
    ts, _ = exposure
    _, port = project
    ts.serve = {443: f"http://127.0.0.1:{port}", 444: "http://127.0.0.1:1"}
    assert _set(client, "tailnet").status_code == 200
    assert 443 not in ts.serve and ts.serve[444] == "http://127.0.0.1:1"
    assert ts.serve[TPORT] == f"http://127.0.0.1:{port}"


def test_tailscale_missing(client, exposure, monkeypatch):
    ts, _ = exposure
    ts.owner = None
    assert _set(client, "tailnet").status_code == 503
    assert _service(client)[1]["tailscale"]["available"] is False


def test_apply_on_startup_restores_gates_and_routes(client, project, exposure, tmp_path):
    ts, gates = exposure
    _, port = project
    (tmp_path / "exposure.json").write_text('{"services": {"demo": "owner"}}')
    import asyncio
    asyncio.run(svc.apply_exposure())
    assert gates.specs and ts.serve == {TPORT: f"http://127.0.0.1:{TPORT + 20000}"}


# --------------------------------------------------------------------------- built-in ComfyUI

def test_autostart_toggle_persists_and_lists(client, tmp_path, monkeypatch):
    f = tmp_path / "autostart.json"
    monkeypatch.setattr(svc, "SERVICE_AUTOSTART_FILE", str(f))
    assert _service(client)[0]["autostart"] is False
    assert client.post("/api/services/demo/autostart", json={"enabled": True}).json()["autostart"] is True
    assert json.loads(f.read_text()) == {"services": {"demo": True}}
    assert _service(client)[0]["autostart"] is True
    assert client.post("/api/services/nope/autostart", json={"enabled": True}).status_code == 404
    assert client.post("/api/services/demo/autostart", json={"enabled": "yes"}).status_code == 422


def test_run_autostart_starts_enabled_once(client, project, tmp_path, monkeypatch):
    import asyncio
    _, port = project
    f = tmp_path / "autostart.json"
    f.write_text(json.dumps({"services": {"demo": True, "gone": True}}))
    monkeypatch.setattr(svc, "SERVICE_AUTOSTART_FILE", str(f))
    first = asyncio.run(svc.run_autostart())
    assert list(first) == ["demo"] and first["demo"]["status"] == "starting"
    assert _wait(lambda: _service(client)[0]["status"] == "running")
    assert asyncio.run(svc.run_autostart()) == {}   # already running -> nothing


# --------------------------------------------------------------------------- presentation (redesign)

def _rewrite_manifest(d, **extra):
    data = json.loads((d / MANIFEST_NAME).read_text())
    data.update(extra)
    (d / MANIFEST_NAME).write_text(json.dumps(data))


PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


def test_presentation_fields_listed(client, project):
    d, _ = project
    (d / "shot.png").write_bytes(PNG)
    _rewrite_manifest(d, category="stories", icon="book", screenshot="./shot.png")
    s, _ = _service(client)
    assert (s["category"], s["icon"], s["has_screenshot"], s["has_status"]) == ("stories", "book", True, False)


def test_screenshot_served_as_image_only(client, project):
    d, _ = project
    (d / "shot.png").write_bytes(PNG)
    _rewrite_manifest(d, screenshot="./shot.png")
    r = client.get("/api/services/demo/screenshot")
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png" and r.headers["x-content-type-options"] == "nosniff"


def test_screenshot_404_when_absent_or_swapped(client, project, tmp_path):
    assert client.get("/api/services/demo/screenshot").status_code == 404
    d, _ = project
    (d / "shot.png").write_bytes(PNG)
    _rewrite_manifest(d, screenshot="./shot.png")
    # after the manifest is read the file becomes a link to something outside
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    real_load = svc._registry

    def swapped():
        reg = real_load()
        (d / "shot.png").unlink()
        (d / "shot.png").symlink_to(outside)
        return reg
    svc._registry = swapped
    try:
        assert client.get("/api/services/demo/screenshot").status_code == 404
    finally:
        svc._registry = real_load


def test_status_facts_from_the_running_app(client, project):
    d, _ = project
    (d / "status.json").write_text(json.dumps({"facts": [{"label": "Aufgaben", "value": 3}, {"label": "x"}],
                                               "secret": "not passed on"}))
    _rewrite_manifest(d, status_path="/status.json")
    assert client.get("/api/services/demo/status").json() == {"id": "demo", "running": False, "facts": []}
    assert client.post("/api/services/demo/start").status_code == 200
    assert _wait(lambda: _service(client)[0]["status"] == "running")
    assert _wait(lambda: client.get("/api/services/demo/status").json()["facts"] != [])
    assert client.get("/api/services/demo/status").json() == {
        "id": "demo", "running": True, "facts": [{"label": "Aufgaben", "value": "3"}]}
    s, _ = _service(client)
    assert client.post("/api/services/demo/stop", json={"pid": s["instances"][0]["pid"]}).status_code == 200


def test_status_without_status_path(client):
    assert client.get("/api/services/demo/status").json() == {"id": "demo", "running": False, "facts": []}


def test_api_service_without_page_cannot_be_exposed(client, project, exposure):
    d, _ = project
    _rewrite_manifest(d, ui=False)
    s, _ = _service(client)
    assert s["ui"] is False
    r = _set(client, "owner")
    assert r.status_code == 400 and "no page" in r.json()["detail"]
    assert _set(client, "local").status_code == 200


def test_single_app_refused_for_whole_tailnet(client, project, exposure):
    d, _ = project
    _rewrite_manifest(d, identity="single")
    s, _ = _service(client)
    assert s["identity"] == "single"
    r = _set(client, "tailnet")
    assert r.status_code == 400 and "Only me" in r.json()["detail"]
    assert _set(client, "owner").status_code == 200



def test_a_started_app_gets_this_machines_names(client, project):
    """Found live 2026-10-23: the names were put into the manifest's env, where child_environment() (rightly) drops
    them – the apps never got LLAMA_SERVE_LOCAL_HOSTS and the identity SDK would refuse Tailscale Serve's host name."""
    d, port = project
    assert client.post("/api/services/demo/start").status_code == 200
    try:
        for _ in range(50):
            if (d / "started_env.txt").exists():
                break
            time.sleep(0.1)
        names = (d / "started_env.txt").read_text().split(",")
        assert "localhost" in names and "127.0.0.1" in names
    finally:
        pid = next(i["pid"] for x in client.get("/api/services").json()["services"] if x["id"] == "demo"
                   for i in x["instances"]) if any(x["instances"] for x in client.get("/api/services").json()["services"] if x["id"] == "demo") else None
        if pid:
            client.post("/api/services/demo/stop", json={"pid": pid})
