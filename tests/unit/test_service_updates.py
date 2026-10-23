"""Restarting apps and plugins that still run old code (backend/services/updates.py)."""
import os
import time

from backend.services import updates


def test_only_apps_and_plugins_of_this_installation(tmp_path):
    root = tmp_path / "ai-serve"
    for d in ("apps/spielwiese", "plugins/image-serve"):
        (root / d).mkdir(parents=True)
    other = tmp_path / "Projekte" / "storytime"
    other.mkdir(parents=True)
    assert updates.part_of_installation(str(root / "apps/spielwiese"), str(root))
    assert updates.part_of_installation(str(root / "plugins/image-serve"), str(root))
    assert not updates.part_of_installation(str(other), str(root))          # someone else's project
    assert not updates.part_of_installation(str(root / "plugins"), str(root))
    assert not updates.part_of_installation(str(root), str(root))


def test_outdated_means_code_newer_than_the_process(tmp_path):
    root = tmp_path / "ai-serve"
    a, b = root / "plugins/a", root / "apps/b"
    newest = {str(a): 2000.0, str(b): 1000.0}
    got = updates.outdated([("a", str(a), 11, 1500.0), ("b", str(b), 12, 1500.0),
                            ("x", str(tmp_path / "elsewhere"), 13, 0.0)], newest, str(root))
    assert got == [("a", 11)]


def test_only_code_counts_not_environments_or_data(tmp_path):
    p = tmp_path / "plugin"
    (p / ".venv/lib").mkdir(parents=True)
    (p / "piper-venv").mkdir()
    (p / "logs").mkdir()
    (p / "app.py").write_text("x")
    old = time.time() - 1000
    os.utime(p / "app.py", (old, old))
    for f in (p / ".venv/lib/site.py", p / "piper-venv/x.py", p / "logs/run.log", p / "voice.onnx"):
        f.write_text("new")
    assert abs(updates.newest_code_mtime(str(p)) - old) < 1
    (p / "app.py").write_text("changed")
    assert updates.newest_code_mtime(str(p)) > old + 500


def test_endpoint_restarts_outdated_idle_ones_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import backend.config as config
    from backend.main import app
    from backend.routers import services as svc

    root = tmp_path / "ai-serve"
    dirs = {"idle": root / "plugins/idle", "busy": root / "plugins/busy", "foreign": tmp_path / "Projekte/foreign"}
    for d in dirs.values():
        d.mkdir(parents=True)
        (d / "app.py").write_text("new code")
    from backend.services.domain import ServiceManifest   # real manifests: found a bug that SimpleNamespace hid
    ms = [ServiceManifest(id=k, name=k, description="d", project_dir=v, command=("./run.sh",), env={}, port=9000 + i,
                          bind="127.0.0.1", match=(k,), identity="none", ui=False,
                          status_path="/status" if k == "busy" else None)
          for i, (k, v) in enumerate(dirs.items())]
    monkeypatch.setattr(config, "_PROJECT_DIR", str(root))
    monkeypatch.setattr(svc, "_registry", lambda: SimpleNamespace(services=ms))
    monkeypatch.setattr(svc.runtime, "snapshot", lambda: [])
    monkeypatch.setattr(svc, "discover_instances",
                        lambda m, procs: [SimpleNamespace(root_pid=100 + len(m.id), started_at=0.0, ports={m.port})])

    async def busy(m, inst):
        return m.id == "busy"
    calls = []

    async def stop(sid, req=None):
        calls.append(("stop", sid, req.pid))
        return {}

    async def start(sid):
        calls.append(("start", sid))
        return {}
    monkeypatch.setattr(svc, "_reports_busy", busy)
    monkeypatch.setattr(svc, "stop_service", stop)
    monkeypatch.setattr(svc, "start_service", start)
    r = TestClient(app, client=("127.0.0.1", 50000)).post("/api/services/restart-outdated")
    assert r.status_code == 200, r.text
    assert r.json() == {"restarted": ["idle"], "busy": ["busy"]}
    assert calls == [("stop", "idle", 104), ("start", "idle")]
