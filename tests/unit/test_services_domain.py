"""Service management domain (backend/services/domain.py) -- pure rules.

Threat model: the dashboard's /api/services can start and stop processes. Anyone
who can reach the dashboard (LAN, tailnet) can call it, so it must not become a way
to run arbitrary commands or to kill processes that don't belong to a registered
service. Trust anchor: services.json in this repo names the project directories;
each project's own llama-serve.service.json describes how to run it."""
import os
import stat
from pathlib import Path

import pytest

from backend.services.domain import (
    MANIFEST_NAME,
    ManifestError,
    ObservedProcess,
    ServiceConflict,
    ServiceManifest,
    ProjectRef,
    ServiceRegistry,
    ServicesConfig,
    find_project_dirs,
    UnknownService,
    child_environment,
    discover_instances,
    open_target,
    plan_start,
    plan_stop,
)

RESERVED = frozenset({8030, 8189})


def _project(tmp_path, name="StoryTime", script="run.sh", executable=True):
    d = tmp_path / name
    d.mkdir()
    s = d / script
    s.write_text("#!/bin/sh\nexec sleep 100\n")
    if executable:
        s.chmod(s.stat().st_mode | stat.S_IXUSR)
    return d


def _manifest_dict(**over):
    d = {
        "id": "storytime",
        "name": "StoryTime",
        "description": "Interactive stories",
        "command": ["./run.sh"],
        "env": {"STORYTIME_PORT": "{port}"},
        "port": 8008,
        "match": ["main.py"],
    }
    d.update(over)
    return d


def _manifest(tmp_path, **over):
    return ServiceManifest.from_dict(_manifest_dict(**over), _project(tmp_path), reserved_ports=RESERVED)


def _proc(pid, cwd, cmdline, ports=(), ppid=1, started=1000.0, addrs=None):
    addrs = frozenset(addrs) if addrs is not None else frozenset(("127.0.0.1", p) for p in ports)
    return ObservedProcess(pid=pid, ppid=ppid, cwd=str(cwd), cmdline=tuple(cmdline),
                           listening_ports=frozenset(ports), create_time=started, listen_addresses=addrs)


# --------------------------------------------------------------------------- manifest

class TestManifest:
    def test_valid_manifest(self, tmp_path):
        m = _manifest(tmp_path)
        assert (m.id, m.name, m.port, m.bind, m.open_path) == ("storytime", "StoryTime", 8008, "127.0.0.1", "/")
        assert m.argv() == [str(tmp_path / "StoryTime" / "run.sh")]
        assert m.environment() == {"STORYTIME_PORT": "8008"}

    def test_port_placeholder_in_args(self, tmp_path):
        m = _manifest(tmp_path, command=["./run.sh", "--port", "{port}"], env={})
        assert m.argv()[1:] == ["--port", "8008"]

    @pytest.mark.parametrize("command", [
        [], "./run.sh", ["bash", "-c", "rm -rf ~"], ["/bin/sh", "run.sh"], ["run.sh"],
        ["./../other/run.sh"], ["./missing.sh"], ["./run.sh", 5], ["./"],
    ])
    def test_command_must_be_a_script_inside_the_project(self, tmp_path, command):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, command=command)

    def test_symlink_escaping_the_project_rejected(self, tmp_path):
        d = _project(tmp_path)
        outside = tmp_path / "evil.sh"
        outside.write_text("#!/bin/sh\n")
        outside.chmod(0o755)
        (d / "link.sh").symlink_to(outside)
        with pytest.raises(ManifestError):
            ServiceManifest.from_dict(_manifest_dict(command=["./link.sh"]), d, reserved_ports=RESERVED)

    def test_non_executable_script_rejected(self, tmp_path):
        d = _project(tmp_path, executable=False)
        with pytest.raises(ManifestError):
            ServiceManifest.from_dict(_manifest_dict(), d, reserved_ports=RESERVED)

    @pytest.mark.parametrize("env", [
        {"LD_PRELOAD": "/tmp/x.so"}, {"LD_LIBRARY_PATH": "/tmp"}, {"PATH": "/tmp"},
        {"BASH_ENV": "/tmp/x"}, {"PYTHONSTARTUP": "/tmp/x"}, {"lower": "x"}, {"A B": "x"}, {"X": 5},
    ])
    def test_dangerous_or_malformed_env_rejected(self, tmp_path, env):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, env=env)

    @pytest.mark.parametrize("port", [0, 80, 1023, 65536, "8008", 8030, 8189, True])
    def test_invalid_or_reserved_port_rejected(self, tmp_path, port):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, port=port)

    @pytest.mark.parametrize("over", [
        {"id": "Story Time"}, {"id": ""}, {"id": "../x"}, {"name": ""}, {"bind": "8.8.8.8"},
        {"bind": "192.168.1.40"},
        {"open_path": "http://evil.example/"}, {"open_path": "//evil.example"}, {"health_path": "x"},
        {"match": "main.py"}, {"match": [""]}, {"match": []},
    ])
    def test_other_invalid_fields_rejected(self, tmp_path, over):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, **over)

    @pytest.mark.parametrize("bind", ["0.0.0.0", "::"])
    def test_binding_all_interfaces_is_refused(self, tmp_path, bind):
        # Reachability beyond this machine is the dashboard's decision (Tailscale
        # levels) -- an app on all interfaces would decide it by itself.
        with pytest.raises(ManifestError) as e:
            _manifest(tmp_path, bind=bind)
        assert "127.0.0.1" in str(e.value) and "dashboard" in str(e.value)

    @pytest.mark.parametrize("bind", ["127.0.0.1", "localhost", "::1"])
    def test_loopback_binds_allowed(self, tmp_path, bind):
        assert _manifest(tmp_path, bind=bind).bind == bind


# --------------------------------------------------------------------------- registry

def _write_manifest(d, data):
    import json
    (d / MANIFEST_NAME).write_text(json.dumps(data))


class TestServicesConfig:
    def test_empty_and_missing(self):
        cfg = ServicesConfig.from_dict(None)
        assert cfg.include == () and cfg.exclude == ()

    def test_include_exclude(self, tmp_path):
        cfg = ServicesConfig.from_dict({"include": [str(tmp_path / "a")], "exclude": ["storytime", str(tmp_path / "b")]})
        assert cfg.include == (str(tmp_path / "a"),) and cfg.exclude == ("storytime", str(tmp_path / "b"))

    def test_legacy_services_list_counts_as_include(self, tmp_path):
        cfg = ServicesConfig.from_dict({"services": [{"dir": str(tmp_path / "a")}]})
        assert cfg.include == (str(tmp_path / "a"),)

    @pytest.mark.parametrize("data", [[], {"include": "x"}, {"include": [5]}, {"exclude": [None]},
                                      {"include": ["relative/dir"]}, {"services": [5]}, {"bogus": []}])
    def test_malformed(self, data):
        with pytest.raises(ManifestError):
            ServicesConfig.from_dict(data)


class TestFindProjectDirs:
    def test_children_of_roots_with_manifest(self, tmp_path):
        root = tmp_path / "Projekte"
        root.mkdir()
        a = _project(root, "a")
        _write_manifest(a, {})
        _project(root, "no-manifest")
        nested = root / "group" / "deep"
        nested.mkdir(parents=True)
        _write_manifest(nested, {})
        hidden = root / ".hidden"
        hidden.mkdir()
        _write_manifest(hidden, {})
        refs, errors = find_project_dirs([str(root)], ServicesConfig.from_dict(None))
        assert [(r.dir, r.source) for r in refs] == [(a, f"root {root}")] and errors == []

    def test_symlinked_project_is_ignored(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = _project(tmp_path, "outside")
        _write_manifest(outside, {})
        (root / "link").symlink_to(outside, target_is_directory=True)
        refs, _ = find_project_dirs([str(root)], ServicesConfig.from_dict(None))
        assert refs == []

    def test_missing_root_is_reported(self, tmp_path):
        refs, errors = find_project_dirs([str(tmp_path / "nope")], ServicesConfig.from_dict(None))
        assert refs == [] and len(errors) == 1 and "nope" in errors[0]["dir"]

    def test_include_and_exclude(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        a = _project(root, "a")
        b = _project(root, "b")
        for d in (a, b):
            _write_manifest(d, {})
        extra = _project(tmp_path, "extra")
        cfg = ServicesConfig.from_dict({"include": [str(extra), str(a)], "exclude": [str(b)]})
        refs, _ = find_project_dirs([str(root)], cfg)
        assert [(r.dir.name, r.source) for r in refs] == [("a", f"root {root}"), ("extra", "services.json")]

    def test_relative_root_rejected(self):
        with pytest.raises(ManifestError):
            find_project_dirs(["Projekte"], ServicesConfig.from_dict(None))


class TestRegistry:
    def _refs(self, *dirs):
        return [ProjectRef(dir=d, source="test") for d in dirs]

    def test_loads_and_keeps_source(self, tmp_path):
        d = _project(tmp_path)
        _write_manifest(d, _manifest_dict())
        reg = ServiceRegistry.load(self._refs(d), reserved_ports=RESERVED)
        assert [s.id for s in reg.services] == ["storytime"] and reg.errors == []
        assert reg.get("storytime").project_dir == d and reg.source("storytime") == "test"

    def test_broken_manifest_reported_not_fatal(self, tmp_path):
        good = _project(tmp_path, "Good")
        _write_manifest(good, _manifest_dict(id="good"))
        bad = _project(tmp_path, "Bad")
        _write_manifest(bad, _manifest_dict(id="bad", command=["bash", "-c", "x"]))
        missing = tmp_path / "Missing"
        reg = ServiceRegistry.load(self._refs(good, bad, missing), reserved_ports=RESERVED)
        assert [s.id for s in reg.services] == ["good"]
        assert len(reg.errors) == 2 and all({"dir", "error", "source"} <= set(e) for e in reg.errors)

    def test_duplicate_ids_and_ports_rejected(self, tmp_path):
        a, b, c = (_project(tmp_path, n) for n in "ABC")
        _write_manifest(a, _manifest_dict())
        _write_manifest(b, _manifest_dict())  # same id
        _write_manifest(c, _manifest_dict(id="other"))  # same port
        reg = ServiceRegistry.load(self._refs(a, b, c), reserved_ports=RESERVED)
        assert [s.id for s in reg.services] == ["storytime"] and len(reg.errors) == 2

    def test_exclude_by_id(self, tmp_path):
        d = _project(tmp_path)
        _write_manifest(d, _manifest_dict())
        reg = ServiceRegistry.load(self._refs(d), reserved_ports=RESERVED, exclude_ids=("storytime",))
        assert reg.services == [] and reg.errors == []

    def test_unknown_service(self):
        with pytest.raises(UnknownService):
            ServiceRegistry.load([], reserved_ports=RESERVED).get("storytime")


class TestManifestSpec:
    def test_unknown_field_is_an_error_not_ignored(self, tmp_path):
        with pytest.raises(ManifestError) as e:
            _manifest(tmp_path, comand=["./run.sh"])
        assert "comand" in str(e.value)

    def test_comment_and_schema_keys_allowed(self, tmp_path):
        _manifest(tmp_path, _comment="hi", **{"$schema": "x"})

    def test_schema_matches_domain(self):
        import json
        from backend.services.domain import MANIFEST_FIELDS, MANIFEST_REQUIRED, manifest_schema
        schema = manifest_schema()
        assert set(schema["properties"]) == set(MANIFEST_FIELDS)
        assert set(schema["required"]) == set(MANIFEST_REQUIRED)
        json.dumps(schema)

    def test_schema_accepts_valid_and_rejects_typo(self):
        import jsonschema
        from backend.services.domain import manifest_schema
        schema = manifest_schema()
        jsonschema.validate(_manifest_dict(), schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**_manifest_dict(), "comand": []}, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**_manifest_dict(), "command": ["bash", "-c", "x"]}, schema)


# --------------------------------------------------------------------------- discovery

class TestDiscovery:
    def test_finds_instances_by_cwd_and_match(self, tmp_path):
        m = _manifest(tmp_path)
        backend = m.project_dir / "backend"
        procs = [
            _proc(100, backend, ["python3", "main.py"], ports=[8008]),
            # uvicorn reload: spawn child shares the listening socket
            _proc(101, backend, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], ports=[8008], ppid=100),
            _proc(200, backend, ["python3", "main.py"], ports=[8018], started=2000.0),
            _proc(300, "/somewhere/else", ["python3", "main.py"], ports=[9000]),
        ]
        inst = discover_instances(m, procs)
        assert [(i.root_pid, sorted(i.ports), sorted(i.pids)) for i in inst] == \
            [(100, [8008], [100, 101]), (200, [8018], [200])]

    def test_helper_processes_in_project_dir_are_not_instances(self, tmp_path):
        m = _manifest(tmp_path, id="llm-benchy", match=["benchy serve"])
        procs = [_proc(5, m.project_dir, [".venv/bin/python", "scripts/label_images.py", "--port", "8765"], ports=[8765])]
        assert discover_instances(m, procs) == []

    def test_sibling_dir_with_same_prefix_is_not_inside(self, tmp_path):
        m = _manifest(tmp_path)
        procs = [_proc(7, str(m.project_dir) + "2", ["python3", "main.py"], ports=[8008])]
        assert discover_instances(m, procs) == []

    @pytest.mark.parametrize("shell", ["/bin/bash", "bash", "/usr/bin/zsh", "sh", "/bin/dash"])
    def test_wrapper_shell_mentioning_the_command_is_not_part_of_the_instance(self, tmp_path, shell):
        # e.g. another tool's `bash -c "... && python3 main.py"` that started the server
        m = _manifest(tmp_path)
        backend = m.project_dir / "backend"
        procs = [
            _proc(50, backend, [shell, "-c", "source snap.sh && eval 'python3 main.py'"]),
            _proc(51, backend, ["python3", "main.py"], ports=[8008], ppid=50),
        ]
        inst = discover_instances(m, procs)
        assert len(inst) == 1 and inst[0].root_pid == 51 and inst[0].pids == frozenset({51})
        assert sorted(plan_stop(m, inst, pid=51)) == [51]

    def test_shell_running_the_start_script_is_part_of_the_instance(self, tmp_path):
        m = _manifest(tmp_path)
        procs = [
            _proc(60, m.project_dir, ["/bin/bash", str(m.project_dir / "run.sh")]),
            _proc(61, m.project_dir, ["python3", "main.py"], ports=[8008], ppid=60),
        ]
        assert discover_instances(m, procs)[0].root_pid == 60

    def test_instance_listening_beyond_loopback_is_flagged(self, tmp_path):
        # e.g. started by hand with HOST=0.0.0.0 -- bypasses the dashboard's control
        m = _manifest(tmp_path)
        procs = [_proc(5, m.project_dir, ["python3", "main.py"], ports=[8008],
                       addrs=[("0.0.0.0", 8008), ("127.0.0.1", 9100)])]
        assert discover_instances(m, procs)[0].exposed == (("0.0.0.0", 8008),)

    def test_loopback_instance_not_flagged(self, tmp_path):
        m = _manifest(tmp_path)
        procs = [_proc(5, m.project_dir, ["python3", "main.py"], ports=[8008], addrs=[("::1", 8008), ("127.0.0.1", 8008)])]
        assert discover_instances(m, procs)[0].exposed == ()

    def test_instance_nested_in_another_instance_tree_is_merged(self, tmp_path):
        # dynasty: serve.sh -> npx -> `sh -c vite` -> node vite. The `sh -c` wrapper
        # stops the climb, but node is still inside serve.sh's tree: one instance.
        m = _manifest(tmp_path, match=["vite", "image_studio/server.py"])
        d = m.project_dir
        procs = [
            _proc(1, d, ["/bin/bash", str(d / "run.sh")]),
            _proc(2, d, ["python3", "tools/image_studio/server.py"], ports=[8035], ppid=1),
            _proc(3, d, ["npm", "exec", "vite", "--port", "3000"], ppid=1),
            _proc(4, d, ["sh", "-c", "vite --port 3000"], ppid=3),
            _proc(5, d, ["node", "node_modules/.bin/vite", "--port", "3000"], ports=[3000], ppid=4),
        ]
        inst = discover_instances(m, procs)
        assert [(i.root_pid, sorted(i.ports)) for i in inst] == [(1, [3000, 8035])]
        assert inst[0].pids == frozenset({1, 2, 3, 4, 5})

    def test_non_listening_tree_member_included(self, tmp_path):
        m = _manifest(tmp_path)
        procs = [
            _proc(10, m.project_dir, ["bash", "./run.sh"]),
            _proc(11, m.project_dir / "backend", ["python3", "main.py"], ports=[8008], ppid=10),
            _proc(12, m.project_dir / "backend", ["python3", "worker"], ppid=11),
        ]
        inst = discover_instances(m, procs)
        assert len(inst) == 1 and inst[0].root_pid == 10 and inst[0].pids == frozenset({10, 11, 12})


# --------------------------------------------------------------------------- start / stop / open

class TestPlanStart:
    def test_plan(self, tmp_path):
        m = _manifest(tmp_path)
        plan = plan_start(m, processes=[])
        assert plan.argv == [str(m.project_dir / "run.sh")] and plan.cwd == m.project_dir
        assert plan.env == {"STORYTIME_PORT": "8008"}

    def test_port_taken_by_anything_refused(self, tmp_path):
        m = _manifest(tmp_path)
        with pytest.raises(ServiceConflict) as e:
            plan_start(m, processes=[_proc(9, "/x", ["nginx"], ports=[8008])])
        assert "8008" in e.value.message

    def test_already_running_refused(self, tmp_path):
        m = _manifest(tmp_path)
        with pytest.raises(ServiceConflict):
            plan_start(m, processes=[_proc(9, m.project_dir, ["python3", "main.py"], ports=[8008])])


class TestPlanStop:
    def _setup(self, tmp_path):
        m = _manifest(tmp_path)
        procs = [
            _proc(100, m.project_dir, ["python3", "main.py"], ports=[8008]),
            _proc(101, m.project_dir, ["python3", "spawn"], ppid=100),
            _proc(200, m.project_dir, ["python3", "main.py"], ports=[8018]),
            _proc(999, "/home/me", ["firefox"], ports=[5555]),
        ]
        return m, discover_instances(m, procs)

    def test_stop_by_root_pid_signals_whole_tree(self, tmp_path):
        m, inst = self._setup(tmp_path)
        assert sorted(plan_stop(m, inst, pid=100)) == [100, 101]

    def test_ambiguous_without_pid(self, tmp_path):
        m, inst = self._setup(tmp_path)
        with pytest.raises(ServiceConflict):
            plan_stop(m, inst, pid=None)

    def test_single_instance_without_pid(self, tmp_path):
        m, inst = self._setup(tmp_path)
        assert sorted(plan_stop(m, inst[:1], pid=None)) == [100, 101]

    @pytest.mark.parametrize("pid", [999, 101, 1, 0, -1])
    def test_foreign_or_non_root_pid_refused(self, tmp_path, pid):
        m, inst = self._setup(tmp_path)
        with pytest.raises(UnknownService):
            plan_stop(m, inst, pid=pid)

    def test_nothing_running(self, tmp_path):
        m = _manifest(tmp_path)
        with pytest.raises(UnknownService):
            plan_stop(m, [], pid=None)


class TestOpenTarget:
    def test_loopback_bind_is_local_only(self, tmp_path):
        t = open_target(_manifest(tmp_path, open_path="/app"), port=8018)
        assert t == {"port": 8018, "path": "/app", "local_only": True}

    @pytest.mark.parametrize("bind", ["localhost", "::1"])
    def test_every_allowed_bind_is_local_only(self, tmp_path, bind):
        assert open_target(_manifest(tmp_path, bind=bind), port=8008)["local_only"] is True


class TestChildEnvironment:
    BASE = {
        "HOME": "/home/me", "LANG": "de_DE.UTF-8",
        "PATH": "/opt/ai-serve/venv/bin:/usr/local/bin:/usr/bin",
        "VIRTUAL_ENV": "/opt/ai-serve/venv",
        "PYTHONPATH": "/opt/ai-serve",
        "LLAMA_SERVE_PUBLIC_API_KEY": "secret1", "LLAMA_SERVE_API_KEY": "secret2",
        "TRAINING_SERVICE_TOKEN": "secret3", "HF_TOKEN": "hf_x",
    }

    def test_secrets_of_this_backend_are_not_inherited(self):
        env = child_environment(self.BASE, {})
        assert not any(v.startswith("secret") for v in env.values())

    def test_our_python_environment_is_not_inherited(self):
        env = child_environment(self.BASE, {})
        assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
        assert env["PATH"] == "/usr/local/bin:/usr/bin"

    def test_ordinary_variables_kept_and_manifest_env_wins(self):
        env = child_environment(self.BASE, {"LANG": "C", "STORYTIME_PORT": "8008"})
        assert env["HOME"] == "/home/me" and env["LANG"] == "C" and env["STORYTIME_PORT"] == "8008"
        assert env["HF_TOKEN"] == "hf_x"  # not ours -- services may need it


# --------------------------------------------------------------------------- built-in services

def _comfy(tmp_path):
    d = tmp_path / "ComfyUI"
    d.mkdir(exist_ok=True)
    return ServiceManifest.builtin(
        id="comfyui", name="ComfyUI", description="Image generation", project_dir=d,
        port=8189, match=("main.py",), controller="comfyui")


class TestBuiltinService:
    def test_builtin_has_no_command_and_a_controller(self, tmp_path):
        m = _comfy(tmp_path)
        assert m.controller == "comfyui" and m.command == () and m.bind == "127.0.0.1"

    def test_builtin_may_use_a_reserved_port(self, tmp_path):
        assert _comfy(tmp_path).port == 8189  # reserved for manifests, not for the backend's own

    def test_generic_start_refused_for_builtin(self, tmp_path):
        with pytest.raises(ServiceConflict):
            plan_start(_comfy(tmp_path), processes=[])

    def test_unknown_controller_rejected(self, tmp_path):
        with pytest.raises(ManifestError):
            ServiceManifest.builtin(id="x", name="X", description="", project_dir=tmp_path, port=9000,
                                    match=("x",), controller="rm -rf")

    def test_discovered_like_any_service(self, tmp_path):
        m = _comfy(tmp_path)
        procs = [_proc(77, m.project_dir, ["/venv/bin/python", "main.py", "--port", "8189"], ports=[8189]),
                 _proc(78, m.project_dir, ["bash"], ports=[])]
        inst = discover_instances(m, procs)
        assert [(i.root_pid, sorted(i.ports)) for i in inst] == [(77, [8189])]

    def test_registry_lists_builtins_first_and_guards_their_id(self, tmp_path):
        d = _project(tmp_path)
        _write_manifest(d, _manifest_dict(id="comfyui", port=8100))
        other = _project(tmp_path, "Other")
        _write_manifest(other, _manifest_dict(id="storytime"))
        refs = [ProjectRef(dir=d, source="t"), ProjectRef(dir=other, source="t")]
        reg = ServiceRegistry.load(refs, reserved_ports=RESERVED, builtins=[_comfy(tmp_path)])
        assert [s.id for s in reg.services] == ["comfyui", "storytime"]
        assert reg.source("comfyui") == "built-in" and len(reg.errors) == 1


# --------------------------------------------------------------------------- presentation (redesign 2026-10-16)

class TestPresentation:
    """Category, icon, screenshot and status facts shown in the dashboard. Text comes
    from other repos, so nothing here may smuggle markup, foreign files or remote URLs
    into the page: icons are names from a fixed set, screenshots are image files inside
    the project, status facts are short plain strings fetched from the app's own port."""

    def _shot(self, d, name="docs/shot.png", data=b"\x89PNG\r\n\x1a\n" + b"0" * 32):
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_defaults(self, tmp_path):
        m = _manifest(tmp_path)
        assert m.category == "tools" and m.icon is None and m.screenshot is None and m.status_path is None

    def test_valid_presentation(self, tmp_path):
        d = _project(tmp_path)
        self._shot(d)
        m = ServiceManifest.from_dict(_manifest_dict(category="stories", icon="book", screenshot="./docs/shot.png",
                                                     status_path="/api/status"), d, reserved_ports=RESERVED)
        assert (m.category, m.icon, m.status_path) == ("stories", "book", "/api/status")
        assert m.screenshot == d / "docs/shot.png"

    @pytest.mark.parametrize("category", ["games", "", 3, None])
    def test_unknown_category(self, tmp_path, category):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, category=category)

    @pytest.mark.parametrize("icon", ["<svg onload=x>", "M0 0h24", "rocket", 1])
    def test_icon_only_from_fixed_set(self, tmp_path, icon):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, icon=icon)

    def test_screenshot_must_stay_inside_project(self, tmp_path):
        d = _project(tmp_path)
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"x")
        for value in ["../secret.png", "./../secret.png", str(outside), "/etc/passwd"]:
            with pytest.raises(ManifestError):
                ServiceManifest.from_dict(_manifest_dict(screenshot=value), d, reserved_ports=RESERVED)

    def test_screenshot_symlink_out_of_project_refused(self, tmp_path):
        d = _project(tmp_path)
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"x")
        (d / "shot.png").symlink_to(outside)
        with pytest.raises(ManifestError):
            ServiceManifest.from_dict(_manifest_dict(screenshot="./shot.png"), d, reserved_ports=RESERVED)

    @pytest.mark.parametrize("name", ["docs/shot.svg", "docs/shot.html", "docs/shot"])
    def test_screenshot_raster_images_only(self, tmp_path, name):
        d = _project(tmp_path)
        self._shot(d, name)
        with pytest.raises(ManifestError):
            ServiceManifest.from_dict(_manifest_dict(screenshot="./" + name), d, reserved_ports=RESERVED)

    def test_screenshot_must_exist(self, tmp_path):
        with pytest.raises(ManifestError):
            _manifest(tmp_path, screenshot="./docs/missing.png")

    def test_status_path_like_health_path(self, tmp_path):
        d = _project(tmp_path)
        for bad in ["http://evil/x", "//evil/x", "status", "/a\\b"]:
            with pytest.raises(ManifestError):
                ServiceManifest.from_dict(_manifest_dict(status_path=bad), d, reserved_ports=RESERVED)

    def test_screenshot_file_rechecked_when_served(self, tmp_path):
        from backend.services.domain import screenshot_file
        d = _project(tmp_path)
        shot = self._shot(d)
        m = ServiceManifest.from_dict(_manifest_dict(screenshot="./docs/shot.png"), d, reserved_ports=RESERVED)
        assert screenshot_file(m) == shot
        # swapped for a symlink to somewhere else after the manifest was read
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"x")
        shot.unlink()
        shot.symlink_to(outside)
        assert screenshot_file(m) is None

    def test_screenshot_file_size_limit(self, tmp_path):
        from backend.services.domain import MAX_SCREENSHOT_BYTES, screenshot_file
        d = _project(tmp_path)
        shot = self._shot(d)
        m = ServiceManifest.from_dict(_manifest_dict(screenshot="./docs/shot.png"), d, reserved_ports=RESERVED)
        shot.write_bytes(b"0" * (MAX_SCREENSHOT_BYTES + 1))
        assert screenshot_file(m) is None
        other = ServiceManifest.from_dict(_manifest_dict(id="other"), _project(tmp_path, name="Other"),
                                          reserved_ports=RESERVED)
        assert screenshot_file(other) is None

    def test_status_facts(self):
        from backend.services.domain import parse_status_facts
        facts = parse_status_facts({"facts": [
            {"label": "Zugang", "value": "aus dem Internet"},
            {"label": "Aufgaben", "value": 3},
            {"label": "ok", "value": True},
            {"label": "", "value": "x"},             # no label
            {"label": "x" * 200, "value": "y"},      # label too long -> cut
            {"label": "nested", "value": {"a": 1}},  # not a scalar
            "junk",
        ]})
        assert facts[:3] == [{"label": "Zugang", "value": "aus dem Internet"}, {"label": "Aufgaben", "value": "3"},
                             {"label": "ok", "value": "ja"}]
        assert len(facts) == 4 and len(facts[3]["label"]) <= 40

    def test_status_facts_limits_and_garbage(self):
        from backend.services.domain import parse_status_facts
        assert parse_status_facts(None) == [] and parse_status_facts([1]) == [] and parse_status_facts({"facts": "x"}) == []
        many = parse_status_facts({"facts": [{"label": f"l{i}", "value": "v" * 500} for i in range(20)]})
        assert len(many) == 6 and all(len(f["value"]) <= 80 for f in many)

    def test_schema_lists_presentation_fields(self):
        import jsonschema
        from backend.services.domain import manifest_schema
        schema = manifest_schema()
        jsonschema.validate({**_manifest_dict(), "category": "learning", "icon": "bubbles",
                             "screenshot": "./docs/a.webp", "status_path": "/status"}, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**_manifest_dict(), "icon": "rocket"}, schema)


class TestUiFlag:
    def test_ui_defaults_true_and_must_be_boolean(self, tmp_path):
        assert _manifest(tmp_path).ui is True
        d = _project(tmp_path, name="Api")
        assert ServiceManifest.from_dict(_manifest_dict(ui=False), d, reserved_ports=RESERVED).ui is False
        for bad in ["no", 0, None]:
            with pytest.raises(ManifestError):
                ServiceManifest.from_dict(_manifest_dict(ui=bad), d, reserved_ports=RESERVED)


class TestStartedHere:
    """An instance counts as started from the dashboard when the process it launched is
    the instance's root or an ancestor of it -- `npm exec` -> `sh -c vite` -> node:
    the shell rule stops the instance at node, the launched pid is npm."""

    def test_launched_pid_is_an_ancestor(self, tmp_path):
        from backend.services.domain import descends_from
        procs = [_proc(10, "/", ["npm", "exec", "vite"], ppid=1),
                 _proc(11, "/", ["sh", "-c", "vite"], ppid=10),
                 _proc(12, "/", ["node", "vite"], ppid=11),
                 _proc(20, "/", ["other"], ppid=1)]
        assert descends_from(12, 10, procs) and descends_from(12, 12, procs)
        assert not descends_from(12, 20, procs) and not descends_from(20, 10, procs)

    def test_cycles_and_unknown_pids_terminate(self):
        from backend.services.domain import descends_from
        procs = [_proc(5, "/", ["a"], ppid=6), _proc(6, "/", ["b"], ppid=5)]
        assert not descends_from(5, 99, procs) and not descends_from(42, 1, procs)


class TestIdentityField:
    """docs/identity.md section 4: how an app handles people. Missing = single (assume one
    person's data until the app says otherwise)."""

    def test_default_is_single(self, tmp_path):
        assert _manifest(tmp_path).identity == "single"

    @pytest.mark.parametrize("value", ["none", "single", "multi"])
    def test_known_values(self, tmp_path, value):
        d = _project(tmp_path, name=f"P{value}")
        assert ServiceManifest.from_dict(_manifest_dict(identity=value), d, reserved_ports=RESERVED).identity == value

    @pytest.mark.parametrize("value", ["multiple", "", None, True, "MULTI"])
    def test_unknown_values_refused(self, tmp_path, value):
        with pytest.raises(ManifestError, match="identity"):
            _manifest(tmp_path, identity=value)

    def test_schema(self):
        import jsonschema
        from backend.services.domain import manifest_schema
        jsonschema.validate({**_manifest_dict(), "identity": "multi"}, manifest_schema())
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**_manifest_dict(), "identity": "many"}, manifest_schema())


class TestOwnerSubForApps:
    """Apps learn the owner's stable id (docs/identity.md 2.2) from the environment."""

    def test_owner_sub_passed_on(self):
        env = child_environment({"PATH": "/usr/bin", "LLAMA_SERVE_OWNER_SUB": "u-123"}, {})
        assert env["LLAMA_SERVE_OWNER_SUB"] == "u-123"

    def test_default_owner(self):
        assert child_environment({"PATH": "/usr/bin"}, {})["LLAMA_SERVE_OWNER_SUB"] == "owner"

    def test_manifest_cannot_pose_as_someone_else(self):
        env = child_environment({"PATH": "/usr/bin", "LLAMA_SERVE_OWNER_SUB": "u-123"},
                                {"LLAMA_SERVE_OWNER_SUB": "evil"})
        assert env["LLAMA_SERVE_OWNER_SUB"] == "u-123"


def test_owner_tailscale_login_passed_on_and_not_overridable():
    env = child_environment({"PATH": "/usr/bin", "LLAMA_SERVE_OWNER_TAILSCALE_LOGIN": "owner@github"},
                            {"LLAMA_SERVE_OWNER_TAILSCALE_LOGIN": "evil@example.com"})
    assert env["LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"] == "owner@github"
    assert "LLAMA_SERVE_OWNER_TAILSCALE_LOGIN" not in child_environment({"PATH": "/usr/bin"}, {})


class TestMaturityField:
    """How far an app is (owner 2026-10-17): alpha / beta / stable (= fully tested). Optional --
    missing means not rated yet."""

    def test_default_is_unrated(self, tmp_path):
        assert _manifest(tmp_path).maturity is None

    @pytest.mark.parametrize("value", ["alpha", "beta", "stable"])
    def test_known_values(self, tmp_path, value):
        d = _project(tmp_path, name=f"M{value}")
        assert ServiceManifest.from_dict(_manifest_dict(maturity=value), d, reserved_ports=RESERVED).maturity == value

    @pytest.mark.parametrize("value", ["gamma", "", True, "Beta", 1])
    def test_unknown_values_refused(self, tmp_path, value):
        with pytest.raises(ManifestError, match="maturity"):
            _manifest(tmp_path, maturity=value)

    def test_schema(self):
        import jsonschema
        from backend.services.domain import manifest_schema
        jsonschema.validate({**_manifest_dict(), "maturity": "beta"}, manifest_schema())
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**_manifest_dict(), "maturity": "rc"}, manifest_schema())


# --------------------------------------------------------------------------- plugins (2026-10-21)
# An app may declare optional capabilities it provides to the gateway (rules: backend/plugins/contracts.py).

def test_manifest_with_provides(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    m = _manifest(tmp_path / "a", provides={"image": {"catalog": "/v1/images/models", "generate": "/v1/images/generations",
                                                      "status": "/v1/status", "release": "/v1/release", "gpu": "shared"}})
    assert m.provides["image"].endpoints["generate"] == "/v1/images/generations" and m.provides["image"].gpu == "shared"
    assert _manifest(tmp_path / "b").provides == {}


def test_bad_provides_makes_the_manifest_invalid(tmp_path):
    with pytest.raises(ManifestError, match="provides"):
        _manifest(tmp_path, provides={"image": {"catalog": "http://evil.example/models", "generate": "/x"}})


def test_apps_get_this_machines_names_and_a_manifest_cannot_widen_them():
    """LLAMA_SERVE_LOCAL_HOSTS (identity SDK: Host/Origin must name this machine without a door identity) comes from
    ai-serve; a manifest may not add names – that would re-open DNS rebinding for that app."""
    env = child_environment({"PATH": "/usr/bin", "LLAMA_SERVE_LOCAL_HOSTS": "my-pc,100.1.2.3"},
                            {"LLAMA_SERVE_LOCAL_HOSTS": "evil.example"})
    assert env["LLAMA_SERVE_LOCAL_HOSTS"] == "my-pc,100.1.2.3"
    assert "LLAMA_SERVE_LOCAL_HOSTS" not in child_environment({"PATH": "/usr/bin"}, {"LLAMA_SERVE_LOCAL_HOSTS": "x"})
