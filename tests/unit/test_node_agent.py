"""Node agent (node_agent/llama_serve_node.py): runs on other devices (Mac, Pi,
PC, Android/Termux) so the coordinator (this machine) can use their accelerators.
One file, Python stdlib only -- install = Python + Tailscale + a llama.cpp build.

Security rules under test (it starts processes on someone's computer):
- Never listens beyond loopback/tailnet: the listen address must be a loopback or
  Tailscale address (no 0.0.0.0, no LAN).
- Every request needs the node token (>= 32 chars, constant-time) AND must come from
  loopback or from a tailnet peer whose `tailscale whois` login is the configured
  coordinator login. Token alone (stolen) or tailnet identity alone is not enough.
- It only ever starts the configured llama-server binary (absolute, executable,
  named llama-server*), with models resolved inside the configured model dirs
  (.gguf, symlinks resolved, no traversal), and only these knobs: ctx_size,
  parallel, gpu index (must exist). No free-form arguments.
- llama-server itself binds 127.0.0.1; the coordinator reaches it only through the
  agent's authenticated proxy, and only on inference paths."""
import json
import os
import stat
import sys
import threading
import time
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "node_agent"))
import llama_serve_node as agent  # noqa: E402

TOKEN = "t" * 40
COORD = "owner@github"


@pytest.fixture
def setup(tmp_path):
    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    (models / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 100)
    (models / "sub" / "other.gguf").write_bytes(b"GGUF")
    (models / "notes.txt").write_text("x")
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"GGUF")
    (models / "escape.gguf").symlink_to(outside)
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_text(FAKE_LLAMA_SERVER.replace("PYTHON", sys.executable))
    binary.chmod(0o755)
    cfg = {"node_name": "test-node", "listen": "127.0.0.1", "port": 0, "token": TOKEN,
           "coordinator_login": COORD, "llama_server": str(binary), "model_dirs": [str(models)],
           "log_dir": str(tmp_path / "logs")}
    return cfg, tmp_path


def conf(cfg, **over):
    return agent.AgentConfig.from_dict({**cfg, **over})


class TestConfig:
    def test_valid(self, setup):
        cfg, _ = setup
        assert conf(cfg).node_name == "test-node"

    @pytest.mark.parametrize("listen", ["0.0.0.0", "192.168.1.40", "::", "example.com", ""])
    def test_listen_only_loopback_or_tailnet(self, setup, listen):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, listen=listen)

    def test_tailnet_listen_ok(self, setup):
        cfg, _ = setup
        assert conf(cfg, listen="100.64.7.11").listen == "100.64.7.11"

    def test_short_token(self, setup):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, token="short")

    def test_binary_rules(self, setup, tmp_path):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, llama_server="llama-server")             # not absolute
        sh = tmp_path / "bin" / "sh"
        sh.write_text("#!/bin/sh\n")
        sh.chmod(0o755)
        with pytest.raises(agent.ConfigError):
            conf(cfg, llama_server=str(sh))                     # not a llama-server
        noexec = tmp_path / "bin" / "llama-server-noexec"
        noexec.write_text("x")
        with pytest.raises(agent.ConfigError):
            conf(cfg, llama_server=str(noexec))

    def test_model_dirs_absolute_and_existing(self, setup):
        cfg, tmp = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, model_dirs=["models"])
        with pytest.raises(agent.ConfigError):
            conf(cfg, model_dirs=[str(tmp / "missing")])

    def test_unknown_field(self, setup):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, extra_args=["--rpc"])


class TestAuthorize:
    def test_loopback_with_token(self):
        assert agent.authorize("127.0.0.1", f"Bearer {TOKEN}", None, TOKEN, COORD)

    def test_coordinator_with_token(self):
        assert agent.authorize("100.64.7.10", f"Bearer {TOKEN}", COORD, TOKEN, COORD)

    def test_token_alone_not_enough(self):
        assert not agent.authorize("100.90.1.2", f"Bearer {TOKEN}", "thief@x", TOKEN, COORD)
        assert not agent.authorize("192.168.1.9", f"Bearer {TOKEN}", None, TOKEN, COORD)

    def test_identity_alone_not_enough(self):
        assert not agent.authorize("100.64.7.10", None, COORD, TOKEN, COORD)
        assert not agent.authorize("100.64.7.10", f"Bearer {TOKEN}x", COORD, TOKEN, COORD)


class TestModels:
    def test_resolve(self, setup):
        cfg, _ = setup
        c = conf(cfg)
        assert agent.resolve_model(c, "tiny.gguf").endswith("models/tiny.gguf")
        assert agent.resolve_model(c, "sub/other.gguf").endswith("sub/other.gguf")

    @pytest.mark.parametrize("name", ["../outside.gguf", "/etc/passwd", "notes.txt", "missing.gguf",
                                      "escape.gguf", "", "sub/../../outside.gguf"])
    def test_refused(self, setup, name):
        cfg, _ = setup
        with pytest.raises(agent.RequestError):
            agent.resolve_model(conf(cfg), name)

    def test_list(self, setup):
        cfg, _ = setup
        names = [m["name"] for m in agent.list_models(conf(cfg))]
        assert names == ["sub/other.gguf", "tiny.gguf"]


class TestStartRequest:
    accels = [{"index": 0, "kind": "cuda"}, {"index": 1, "kind": "cuda"}]

    def test_defaults(self):
        r = agent.StartRequest.from_dict({"model": "tiny.gguf"}, self.accels)
        assert (r.ctx_size, r.parallel, r.gpu) == (8192, 1, None)

    @pytest.mark.parametrize("body", [
        {"model": "tiny.gguf", "ctx_size": 100}, {"model": "tiny.gguf", "ctx_size": 10_000_000},
        {"model": "tiny.gguf", "parallel": 0}, {"model": "tiny.gguf", "gpu": 7},
        {"model": "tiny.gguf", "args": ["--rpc", "evil:1"]}, {"model": 5}, {},
    ])
    def test_refused(self, body):
        with pytest.raises(agent.RequestError):
            agent.StartRequest.from_dict(body, self.accels)


class TestProxyPaths:
    @pytest.mark.parametrize("path", ["v1/chat/completions", "v1/embeddings", "health", "props", "v1/models"])
    def test_allowed(self, path):
        assert agent.proxy_path_allowed(path)

    @pytest.mark.parametrize("path", ["slots/0?action=save", "../health", "v1/../slots", "metrics", ""])
    def test_refused(self, path):
        assert not agent.proxy_path_allowed(path)


# --------------------------------------------------------------------------- end to end

FAKE_LLAMA_SERVER = '''#!PYTHON
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
if "--version" in args:
    print("version: 9999 (fake)"); sys.exit(0)
port = int(args[args.index("--port") + 1]); host = args[args.index("--host") + 1]
assert host == "127.0.0.1"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send({"status": "ok"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
        self._send({"choices": [{"message": {"content": "echo " + req["messages"][0]["content"]}}],
                    "args": args})
import socketserver
class S(HTTPServer):   # no getfqdn(): it can take ~35 s on a Mac (CI)
    def server_bind(self):
        socketserver.TCPServer.server_bind(self); self.server_name, self.server_port = "127.0.0.1", self.server_address[1]
S(("127.0.0.1", port), H).serve_forever()
'''


def call(base, method, path, body=None, token=TOKEN):
    req = urllib.request.Request(base + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def test_end_to_end(setup, monkeypatch):
    cfg, _ = setup
    monkeypatch.setattr(agent, "detect_accelerators", lambda: [
        {"index": 0, "kind": "cuda", "name": "Fake GPU", "total_mb": 8000, "free_mb": 7000}])
    server = agent.make_server(conf(cfg))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert call(base, "GET", "/v1/node", token=None)[0] == 401
        assert call(base, "GET", "/v1/node", token="w" * 40)[0] == 401
        code, node = call(base, "GET", "/v1/node")
        assert code == 200 and node["name"] == "test-node" and node["accelerators"][0]["kind"] == "cuda"
        assert "9999" in node["runtime"]["llama_server"]
        assert call(base, "POST", "/v1/instances", {"model": "../outside.gguf"})[0] == 400

        code, inst = call(base, "POST", "/v1/instances", {"model": "tiny.gguf", "ctx_size": 4096, "gpu": 0})
        assert code == 201
        iid = inst["id"]
        for _ in range(100):
            if call(base, "GET", f"/v1/instances/{iid}")[1]["state"] == "ready":
                break
            time.sleep(0.1)
        code, out = call(base, "POST", f"/v1/instances/{iid}/proxy/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hi"}]})
        assert code == 200 and out["choices"][0]["message"]["content"] == "echo hi"
        assert out["args"][out["args"].index("-c") + 1] == "4096"
        assert call(base, "GET", f"/v1/instances/{iid}/proxy/slots")[0] == 403
        assert call(base, "DELETE", f"/v1/instances/{iid}")[0] == 200
        assert call(base, "GET", f"/v1/instances/{iid}")[0] == 404
    finally:
        server.shutdown()
        agent.stop_all(server.agent)


class TestStaticNetwork:
    """Open-source overlays (WireGuard, Nebula, NetBird, ZeroTier, Headscale without
    whois): private networks + address -> identity map in the agent config."""
    NET = {"provider": "static", "networks": ["10.8.0.0/24"], "peers": {"10.8.0.1": COORD}}

    def test_listen_on_overlay_address(self, setup):
        cfg, _ = setup
        assert conf(cfg, listen="10.8.0.5", network=self.NET).listen == "10.8.0.5"

    def test_listen_outside_overlay_refused(self, setup):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, listen="10.9.0.5", network=self.NET)
        with pytest.raises(agent.ConfigError):
            conf(cfg, listen="10.8.0.5")   # tailscale default doesn't cover it

    @pytest.mark.parametrize("net", [
        {"provider": "static", "networks": ["0.0.0.0/0"], "peers": {}},
        {"provider": "static", "networks": ["8.8.8.0/24"], "peers": {}},
        {"provider": "static", "networks": ["10.8.0.0/24"], "peers": {"192.168.1.40": COORD}},
        {"provider": "magic"},
        {"provider": "static", "networks": ["10.8.0.0/24"], "peers": {}, "trust": "all"},
    ])
    def test_bad_network(self, setup, net):
        cfg, _ = setup
        with pytest.raises(agent.ConfigError):
            conf(cfg, network=net)

    def test_authorize_by_map(self, setup):
        cfg, _ = setup
        c = conf(cfg, listen="10.8.0.5", network=self.NET)
        net = c.network
        assert agent.authorize("10.8.0.1", f"Bearer {TOKEN}", net.identify("10.8.0.1"), TOKEN, COORD, net.is_private)
        assert not agent.authorize("10.8.0.7", f"Bearer {TOKEN}", net.identify("10.8.0.7"), TOKEN, COORD, net.is_private)
        assert not agent.authorize("100.64.7.11", f"Bearer {TOKEN}", COORD, TOKEN, COORD, net.is_private)


class TestEnroll:
    """`--enroll <coordinator> --code <code>`: the agent makes its own token, checks
    that the coordinator really belongs to the identity it claims (whois of the
    coordinator's address, independently of the coordinator's answer), then writes
    a 0600 config. No runtime yet -- provisioning installs it."""

    def run(self, tmp_path, whois_login=COORD, answer_login=COORD, status=201):
        sent = {}

        def post(url, body):
            sent["url"], sent["body"] = url, body
            return status, {"node_id": body["node_name"], "coordinator_login": answer_login}

        path = tmp_path / "cfg" / "config.json"
        result = agent.enroll("http://100.64.7.10:8030", "CODE123", listen="100.64.7.11", name="macbook",
                              model_dirs=[str(tmp_path)], config_path=str(path), port=8040,
                              whois=lambda ip: whois_login if ip == "100.64.7.10" else None, post=post)
        return result, sent, path

    def test_happy_path(self, tmp_path):
        result, sent, path = self.run(tmp_path)
        assert sent["url"] == "http://100.64.7.10:8030/api/cluster/enroll"
        body = sent["body"]
        assert body["code"] == "CODE123" and body["agent_url"] == "http://100.64.7.11:8040"
        assert len(body["token"]) >= 32
        cfg = json.loads(path.read_text())
        assert cfg["token"] == body["token"] and cfg["coordinator_login"] == COORD
        assert cfg["llama_server"] is None
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert agent.AgentConfig.from_dict(cfg).llama_server is None

    def test_coordinator_identity_mismatch(self, tmp_path):
        with pytest.raises(agent.ConfigError, match="identity"):
            self.run(tmp_path, answer_login="mallory@evil")
        with pytest.raises(agent.ConfigError, match="identity"):
            self.run(tmp_path, whois_login=None)

    def test_refused_enrollment(self, tmp_path):
        with pytest.raises(agent.ConfigError, match="403"):
            self.run(tmp_path, status=403)

    def test_existing_config_not_overwritten(self, tmp_path):
        _, _, path = self.run(tmp_path)
        with pytest.raises(agent.ConfigError, match="exists"):
            self.run(tmp_path)


def test_start_without_runtime_refused(setup, monkeypatch):
    cfg, _ = setup
    monkeypatch.setattr(agent, "detect_accelerators", lambda: [{"index": 0, "kind": "cpu", "name": "cpu", "total_mb": 8000, "free_mb": 4000}])
    a = agent.Agent(conf(cfg, llama_server=None))
    with pytest.raises(agent.RequestError) as e:
        a.start(agent.StartRequest(model="tiny.gguf"))
    assert e.value.status == 409 and "runtime" in str(e.value)
    assert a.node_info()["runtime"]["llama_server"] is None


# --------------------------------------------------------------------------- runtime install

import hashlib
import io
import tarfile
from http.server import HTTPServer, SimpleHTTPRequestHandler

SHA = "a" * 64
GH = "https://github.com/ggml-org/llama.cpp/releases/download/"


def req(**over):
    body = {"tag": "b11681", "variant": "cpu", "files": [
        {"name": "llama-b11681-bin-ubuntu-x64.tar.gz", "url": GH + "b11681/llama-b11681-bin-ubuntu-x64.tar.gz",
         "sha256": SHA, "size": 1000}]}
    body.update(over)
    return body


class TestInstallRequest:
    def test_valid(self):
        r = agent.InstallRequest.from_dict(req())
        assert r.tag == "b11681" and r.files[0].sha256 == SHA

    @pytest.mark.parametrize("change", [
        {"tag": "../x"},
        {"files": [{"name": "llama-b11681-bin-ubuntu-x64.tar.gz", "url": "https://evil.example/x.tar.gz",
                    "sha256": SHA, "size": 1}]},
        {"files": [{"name": "llama-b11681-bin-ubuntu-x64.tar.gz",
                    "url": GH + "b99999/llama-b11681-bin-ubuntu-x64.tar.gz", "sha256": SHA, "size": 1}]},
        {"files": [{"name": "../../evil.tar.gz", "url": GH + "b11681/../../evil.tar.gz", "sha256": SHA, "size": 1}]},
        {"files": [{"name": "llama-b11681-bin-ubuntu-x64.tar.gz",
                    "url": GH + "b11681/llama-b11681-bin-ubuntu-x64.tar.gz", "sha256": "nothex", "size": 1}]},
        {"files": [{"name": "llama-b11681-bin-ubuntu-x64.tar.gz",
                    "url": GH + "b11681/llama-b11681-bin-ubuntu-x64.tar.gz", "sha256": SHA, "size": 5 * 2**30}]},
        {"files": []},
        {"variant": "cpu; rm -rf /"},
    ])
    def test_refused(self, change):
        with pytest.raises(agent.RequestError):
            agent.InstallRequest.from_dict(req(**change))


def _tar(members):
    """members: list of (name, kind, data_or_target)"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, kind, payload in members:
            ti = tarfile.TarInfo(name)
            if kind == "file":
                data = payload.encode() if isinstance(payload, str) else payload
                ti.size, ti.mode = len(data), 0o755
                t.addfile(ti, io.BytesIO(data))
            elif kind == "symlink":
                ti.type, ti.linkname = tarfile.SYMTYPE, payload
                t.addfile(ti)
            elif kind == "dev":
                ti.type = tarfile.CHRTYPE
                t.addfile(ti)
    return buf.getvalue()


class TestSafeExtract:
    def test_ok_with_relative_symlink(self, tmp_path):
        p = tmp_path / "a.tar.gz"
        p.write_bytes(_tar([("llama-b1/llama-server", "file", "x"), ("llama-b1/libx.so.0", "file", "y"),
                            ("llama-b1/libx.so", "symlink", "libx.so.0")]))
        agent.safe_extract(str(p), str(tmp_path / "out"))
        assert (tmp_path / "out" / "llama-b1" / "libx.so").resolve().name == "libx.so.0"

    @pytest.mark.parametrize("members", [
        [("../evil", "file", "x")],
        [("/etc/evil", "file", "x")],
        [("d/link", "symlink", "../../../etc/passwd")],
        [("d/link", "symlink", "/etc/passwd")],
        [("d/dev", "dev", None)],
    ])
    def test_refused(self, tmp_path, members):
        p = tmp_path / "a.tar.gz"
        p.write_bytes(_tar(members))
        with pytest.raises(agent.RequestError):
            agent.safe_extract(str(p), str(tmp_path / "out"))
        assert not (tmp_path / "evil").exists()


def _serve_dir(directory):
    handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=str(directory), **kw)  # noqa: E731
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_install_end_to_end(setup, tmp_path, monkeypatch):
    cfg, _ = setup
    dl = tmp_path / "dl" / "b11681"
    dl.mkdir(parents=True)
    fake = FAKE_LLAMA_SERVER.replace("PYTHON", sys.executable)
    data = _tar([("llama-b11681/llama-server", "file", fake), ("llama-b11681/libggml.so", "file", "lib")])
    name = "llama-b11681-bin-ubuntu-x64.tar.gz"
    (dl / name).write_bytes(data)
    srv = _serve_dir(tmp_path / "dl")
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    monkeypatch.setattr(agent, "DOWNLOAD_PREFIX", base)
    monkeypatch.setattr(agent, "detect_accelerators", lambda: [
        {"index": 0, "kind": "cpu", "name": "cpu", "total_mb": 8000, "free_mb": 4000}])
    cfg_path = tmp_path / "config.json"
    c = {**cfg, "llama_server": None, "data_dir": str(tmp_path / "data")}
    cfg_path.write_text(json.dumps(c))
    a = agent.Agent(conf(c), config_path=str(cfg_path))
    try:
        good = {"tag": "b11681", "variant": "cpu",
                "files": [{"name": name, "url": base + "b11681/" + name,
                           "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}]}
        bad = json.loads(json.dumps(good))
        bad["files"][0]["sha256"] = "0" * 64
        a.install_runtime(agent.InstallRequest.from_dict(bad), wait=True)
        assert a.install_state["state"] == "failed" and "sha256" in a.install_state["error"]
        assert a.cfg.llama_server is None
        a.install_runtime(agent.InstallRequest.from_dict(good), wait=True)
        assert a.install_state["state"] == "installed", a.install_state
        assert a.cfg.llama_server.endswith("llama-server") and "9999" in a.runtime_version()
        assert json.loads(cfg_path.read_text())["llama_server"] == a.cfg.llama_server  # persisted
        inst = a.start(agent.StartRequest(model="tiny.gguf"))
        assert inst["state"] in ("loading", "ready")
    finally:
        agent.stop_all(a)
        srv.shutdown()


# --------------------------------------------------------------------------- model downloads

COORD_URL = "http://100.64.7.10:8030"
HF = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"


def dreq(**over):
    body = {"name": "Qwen3-0.6B-Q4_K_M.gguf", "sha256": SHA, "size": 1000,
            "sources": [COORD_URL + "/api/cluster/models/Qwen3-0.6B-Q4_K_M.gguf", HF]}
    body.update(over)
    return body


class TestDownloadRequest:
    def test_valid(self):
        r = agent.DownloadRequest.from_dict(dreq(), COORD_URL)
        assert r.name == "Qwen3-0.6B-Q4_K_M.gguf" and len(r.sources) == 2

    @pytest.mark.parametrize("change", [
        {"name": "../x.gguf"}, {"name": "sub/x.gguf"}, {"name": "x.bin"},
        {"sources": ["https://evil.example/Qwen3-0.6B-Q4_K_M.gguf"]},
        {"sources": ["http://100.1.1.1:8030/api/cluster/models/Qwen3-0.6B-Q4_K_M.gguf"]},   # not our coordinator
        {"sources": [COORD_URL + "/api/other/Qwen3-0.6B-Q4_K_M.gguf"]},
        {"sources": ["https://huggingface.co/u/r/resolve/main/other.gguf"]},                # different file
        {"sources": ["http://huggingface.co/u/r/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"]},     # not https
        {"sources": []}, {"sha256": "x"}, {"size": 0}, {"size": 10 ** 15},
    ])
    def test_refused(self, change):
        with pytest.raises(agent.RequestError):
            agent.DownloadRequest.from_dict(dreq(**change), COORD_URL)

    def test_no_coordinator_url_means_hf_only(self):
        with pytest.raises(agent.RequestError):
            agent.DownloadRequest.from_dict(dreq(), None)
        assert agent.DownloadRequest.from_dict(dreq(sources=[HF]), None)


class TestQuota:
    """No automatic deletion (user, 2026-10-16: freeing space needs the user's
    decision). Too little room -> refused, with what's there and how big, so the
    user can delete explicitly."""

    def _files(self, d):
        d.mkdir()
        for i, name in enumerate(["old.gguf", "mid.gguf", "new.gguf"]):
            p = d / name
            p.write_bytes(b"x" * 1000)
            os.utime(p, (100 + i, 100 + i))

    def test_fits(self, tmp_path):
        self._files(tmp_path / "dl")
        agent.check_room(str(tmp_path / "dl"), need_bytes=500, quota_bytes=4000)

    def test_refused_lists_candidates_and_deletes_nothing(self, tmp_path):
        d = tmp_path / "dl"
        self._files(d)
        with pytest.raises(agent.RequestError) as e:
            agent.check_room(str(d), need_bytes=1500, quota_bytes=3000)
        msg = str(e.value)
        assert "old.gguf" in msg and "delete" in msg.lower()
        assert sorted(os.listdir(d)) == ["mid.gguf", "new.gguf", "old.gguf"]

    def test_larger_than_quota(self, tmp_path):
        with pytest.raises(agent.RequestError):
            agent.check_room(str(tmp_path), need_bytes=5000, quota_bytes=3000)


class TestDeleteDownload:
    def test_only_managed_downloads_and_not_in_use(self, setup):
        cfg, tmp = setup
        a = agent.Agent(conf(cfg))
        d = os.path.join(conf(cfg).model_dirs[0], agent.DOWNLOAD_SUBDIR)
        os.makedirs(d)
        open(os.path.join(d, "x.gguf"), "wb").write(b"x")
        with pytest.raises(agent.RequestError):
            a.delete_download("tiny.gguf")            # the user's own file, not a download
        with pytest.raises(agent.RequestError):
            a.delete_download("../tiny.gguf")
        a.instances["i"] = {"model": f"{agent.DOWNLOAD_SUBDIR}/x.gguf", "proc": type("P", (), {"poll": lambda s: None})(),
                            "state": "ready"}
        with pytest.raises(agent.RequestError):
            a.delete_download("x.gguf")              # in use
        del a.instances["i"]
        a.delete_download("x.gguf")
        assert not os.path.exists(os.path.join(d, "x.gguf"))
        assert os.path.exists(os.path.join(conf(cfg).model_dirs[0], "tiny.gguf"))

    def test_list_downloads(self, setup):
        cfg, _ = setup
        a = agent.Agent(conf(cfg))
        d = os.path.join(conf(cfg).model_dirs[0], agent.DOWNLOAD_SUBDIR)
        os.makedirs(d)
        open(os.path.join(d, "x.gguf"), "wb").write(b"x" * 2048)
        assert [(m["name"], m["size_bytes"]) for m in a.list_downloads()] == [("x.gguf", 2048)]


class _Slow(SimpleHTTPRequestHandler):
    delay = 0.0

    def copyfile(self, source, outputfile):
        while True:
            buf = source.read(4096)
            if not buf:
                break
            time.sleep(self.delay)
            outputfile.write(buf)


def _range_server(directory, delay):
    handler_cls = type("H", (_Slow,), {"delay": delay})

    class Ranged(handler_cls):
        def send_head(self):  # minimal Range support for the probe
            rng = self.headers.get("Range")
            path = self.translate_path(self.path)
            if not rng or not os.path.isfile(path):
                return super().send_head()
            start, end = [int(x) for x in rng.split("=")[1].split("-")]
            data = open(path, "rb").read()[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return io.BytesIO(data)

    h = lambda *a, **kw: Ranged(*a, directory=str(directory), **kw)  # noqa: E731
    srv = HTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_download_picks_faster_source_and_falls_back(setup, tmp_path, monkeypatch):
    cfg, _ = setup
    payload = os.urandom(300_000)
    good, bad = tmp_path / "good", tmp_path / "bad"
    for d in (good, bad):
        (d / "api" / "cluster" / "models").mkdir(parents=True)
    (good / "api/cluster/models/m.gguf").write_bytes(payload)
    (bad / "api/cluster/models/m.gguf").write_bytes(b"corrupt" + payload[7:])
    slow_good = _range_server(good, 0.002)
    fast_bad = _range_server(bad, 0.0)
    monkeypatch.setattr(agent, "PROBE_BYTES", 64 * 1024)
    try:
        a = agent.Agent(conf(cfg))
        req = agent.DownloadRequest(name="m.gguf", sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
                                    sources=(f"http://127.0.0.1:{slow_good.server_address[1]}/api/cluster/models/m.gguf",
                                             f"http://127.0.0.1:{fast_bad.server_address[1]}/api/cluster/models/m.gguf"))
        a.download_model(req, wait=True)
        st = a.downloads["m.gguf"]
        assert st["state"] == "done", st
        assert st["tried"][0].startswith(f"http://127.0.0.1:{fast_bad.server_address[1]}")  # faster first
        assert "sha256" in st["errors"][0]                                                   # ... was corrupt
        target = os.path.join(conf(cfg).model_dirs[0], agent.DOWNLOAD_SUBDIR, "m.gguf")
        assert open(target, "rb").read() == payload
        assert agent.resolve_model(conf(cfg), f"{agent.DOWNLOAD_SUBDIR}/m.gguf") == os.path.realpath(target)
    finally:
        slow_good.shutdown()
        fast_bad.shutdown()


class TestRuntimeKinds:
    """Which accelerator kinds the installed runtime can drive -- the coordinator
    must not route a model to a CUDA GPU through a CPU-only build."""

    @pytest.mark.parametrize("variant,kinds", [
        ("cpu", {"cpu"}), ("cuda-12.8", {"cuda", "cpu"}), ("rocm-10.0", {"rocm", "cpu"}),
        ("vulkan", {"vulkan", "cuda", "rocm", "cpu"}), ("metal", {"metal", "cpu"}),
    ])
    def test_managed_runtime(self, setup, tmp_path, variant, kinds):
        cfg, _ = setup
        data = tmp_path / "data"
        b = data / "runtimes" / f"b1-{variant}" / "files" / "llama-b1" / "llama-server"
        b.parent.mkdir(parents=True)
        b.write_text("#!/bin/sh\n")
        b.chmod(0o755)
        a = agent.Agent(conf(cfg, data_dir=str(data), llama_server=str(b)))
        assert set(a.runtime_kinds()) == kinds

    def test_none_without_runtime(self, setup):
        cfg, _ = setup
        assert agent.Agent(conf(cfg, llama_server=None)).runtime_kinds() == []

    def test_own_binary_uses_detected(self, setup, monkeypatch):
        cfg, _ = setup
        monkeypatch.setattr(agent, "detect_accelerators", lambda: [{"index": 0, "kind": "cuda"}])
        assert set(agent.Agent(conf(cfg)).runtime_kinds()) == {"cuda", "cpu"}


@pytest.mark.parametrize("variant,kind,var", [
    ("vulkan", "cuda", "GGML_VK_VISIBLE_DEVICES"),   # found live: Vulkan ignores CUDA_VISIBLE_DEVICES
    ("cuda-12.8", "cuda", "CUDA_VISIBLE_DEVICES"),
    ("rocm-10.0", "rocm", "HIP_VISIBLE_DEVICES"),
])
def test_gpu_pinning_per_runtime(setup, tmp_path, monkeypatch, variant, kind, var):
    cfg, _ = setup
    data = tmp_path / "data"
    b = data / "runtimes" / f"b1-{variant}" / "files" / "llama-b1" / "llama-server"
    b.parent.mkdir(parents=True)
    b.write_text('#!/bin/sh\necho "  Vulkan0: g0 (1 MiB)"\necho "  Vulkan1: g1 (1 MiB)"\n')
    b.chmod(0o755)
    monkeypatch.setattr(agent, "detect_accelerators", lambda: [
        {"index": 0, "kind": kind, "name": "g0", "total_mb": 8000, "free_mb": 8000},
        {"index": 1, "kind": kind, "name": "g1", "total_mb": 8000, "free_mb": 8000}])
    seen = {}

    class P:
        pid = 1

        def __init__(self, argv, **kw):
            seen.update(kw["env"])

        def poll(self):
            return None
    a = agent.Agent(conf(cfg, data_dir=str(data), llama_server=str(b)))
    a.backend_device(1)  # resolve the device mapping with the real subprocess first
    monkeypatch.setattr(agent.subprocess, "Popen", P)
    monkeypatch.setattr(agent.Agent, "_wait_ready", lambda self, inst, timeout=600: None)
    a.start(agent.StartRequest(model="tiny.gguf", gpu=1))
    assert seen.get(var) == "1"


LIST_DEVICES = """#!/bin/sh
echo "Available devices:"
echo "  Vulkan0: NVIDIA GeForce RTX 4060 Ti (16626 MiB, 14622 MiB free)"
echo "  Vulkan1: NVIDIA GeForce RTX 3090 (24822 MiB, 23851 MiB free)"
"""


class TestVulkanDeviceMapping:
    """Found live 2026-10-16: Vulkan numbers the GPUs differently from nvidia-smi
    (here 4060 Ti = Vulkan0, 3090 = Vulkan1, while nvidia-smi says 0 = 3090), so a
    plan for "GPU 0" landed on the 4060 Ti. The agent maps by device name via the
    runtime's own --list-devices and refuses rather than guess."""

    def make(self, setup, tmp_path, script, accels):
        cfg, _ = setup
        data = tmp_path / "data"
        b = data / "runtimes" / "b1-vulkan" / "files" / "llama-b1" / "llama-server"
        b.parent.mkdir(parents=True)
        b.write_text(script)
        b.chmod(0o755)
        return agent.Agent(conf(cfg, data_dir=str(data), llama_server=str(b)))

    def test_maps_by_name(self, setup, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "detect_accelerators", lambda: [
            {"index": 0, "kind": "cuda", "name": "NVIDIA GeForce RTX 3090", "total_mb": 24576, "free_mb": 1},
            {"index": 1, "kind": "cuda", "name": "NVIDIA GeForce RTX 4060 Ti", "total_mb": 16380, "free_mb": 1}])
        a = self.make(setup, tmp_path, LIST_DEVICES, None)
        assert a.backend_device(0) == "1" and a.backend_device(1) == "0"

    def test_same_name_keeps_order(self, setup, tmp_path, monkeypatch):
        twin = '#!/bin/sh\necho "  Vulkan0: RTX A (1 MiB)"\necho "  Vulkan1: RTX A (1 MiB)"\n'
        monkeypatch.setattr(agent, "detect_accelerators", lambda: [
            {"index": 0, "kind": "cuda", "name": "RTX A"}, {"index": 1, "kind": "cuda", "name": "RTX A"}])
        a = self.make(setup, tmp_path, twin, None)
        assert a.backend_device(1) == "1"

    def test_unmappable_refuses(self, setup, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "detect_accelerators", lambda: [
            {"index": 0, "kind": "cuda", "name": "Some Other GPU"}])
        a = self.make(setup, tmp_path, LIST_DEVICES, None)
        with pytest.raises(agent.RequestError):
            a.backend_device(0)


class TestBudgets:
    """User, 2026-10-16: on memory shared with the system (Apple Silicon, CPU-only
    devices like a Pi) the default budget is HALF the RAM, adjustable; dedicated GPU
    memory (CUDA/ROCm) defaults to the whole VRAM -- it isn't the system's RAM."""

    def agent_with(self, setup, monkeypatch, accels, **over):
        cfg, _ = setup
        monkeypatch.setattr(agent, "detect_accelerators", lambda: accels)
        return agent.Agent(conf(cfg, **over))

    def test_unified_memory_defaults_to_half(self, setup, monkeypatch):
        a = self.agent_with(setup, monkeypatch, [{"index": 0, "kind": "metal", "name": "M3", "total_mb": 16384,
                                                  "free_mb": None}])
        assert a.accelerators()[0]["budget_mb"] == 8192

    def test_cpu_defaults_to_half(self, setup, monkeypatch):
        a = self.agent_with(setup, monkeypatch, [{"index": 0, "kind": "cpu", "name": "arm", "total_mb": 8000,
                                                  "free_mb": 6000}])
        assert a.accelerators()[0]["budget_mb"] == 4000

    def test_dedicated_vram_defaults_to_full(self, setup, monkeypatch):
        a = self.agent_with(setup, monkeypatch, [{"index": 0, "kind": "cuda", "name": "4060", "total_mb": 8188,
                                                  "free_mb": 8000}])
        assert a.accelerators()[0]["budget_mb"] == 8188

    def test_configured_budget_wins_capped_at_total(self, setup, monkeypatch):
        a = self.agent_with(setup, monkeypatch, [{"index": 0, "kind": "metal", "name": "M3", "total_mb": 16384,
                                                  "free_mb": None}], budget_mb={"0": 12000})
        assert a.accelerators()[0]["budget_mb"] == 12000

    def test_set_budget_persists(self, setup, monkeypatch, tmp_path):
        cfg, _ = setup
        monkeypatch.setattr(agent, "detect_accelerators", lambda: [
            {"index": 0, "kind": "metal", "name": "M3", "total_mb": 16384, "free_mb": None}])
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        a = agent.Agent(conf(cfg), config_path=str(path))
        a.set_budgets({"0": 10000})
        assert a.accelerators()[0]["budget_mb"] == 10000
        assert json.loads(path.read_text())["budget_mb"] == {"0": 10000}
        a.set_budgets({"0": None})          # back to the default
        assert a.accelerators()[0]["budget_mb"] == 8192

    @pytest.mark.parametrize("bad", [{"0": 0}, {"0": 20000}, {"7": 1000}, {"0": "lots"}, {"0": True}, []])
    def test_set_budget_refused(self, setup, monkeypatch, bad):
        a = self.agent_with(setup, monkeypatch, [{"index": 0, "kind": "metal", "name": "M3", "total_mb": 16384,
                                                  "free_mb": None}])
        with pytest.raises(agent.RequestError):
            a.set_budgets(bad)


class TestService:
    """`--install-service`: start the agent with the device and restart it if it
    dies (user, 2026-10-16). Per-user only (no root): launchd LaunchAgent on macOS,
    systemd --user on Linux, Termux:Boot on Android. `--uninstall-service` undoes it."""

    def files(self, platform, tmp_path):
        return agent.service_plan(platform, python="/usr/bin/python3", script=str(tmp_path / "agent.py"),
                                  config=str(tmp_path / "config.json"), home=str(tmp_path), uid=501)

    def test_macos(self, tmp_path):
        plan = self.files("darwin", tmp_path)
        path, content = plan["files"][0]
        assert path.endswith("Library/LaunchAgents/net.llama-serve.node.plist")
        assert "<string>/usr/bin/python3</string>" in content and "--config" in content
        assert "<key>KeepAlive</key>" in content and "<key>RunAtLoad</key>" in content
        # found live: re-installing over a loaded agent failed ("Bootstrap failed: 5")
        # and the old script kept running -- unload first (errors ignored), then load
        assert plan["unload"] == ["launchctl", "bootout", "gui/501/net.llama-serve.node"]
        assert plan["loaded"] == ["launchctl", "print", "gui/501/net.llama-serve.node"]
        assert plan["start"] == [["launchctl", "bootstrap", "gui/501", path]]

    def test_macos_reinstall_waits_until_unloaded(self, tmp_path):
        # found live: bootout is asynchronous -- bootstrapping right after it failed
        # ("Bootstrap failed: 5") and left no agent running at all
        plan = self.files("darwin", tmp_path)
        still_loaded = [0, 0, 1]          # `launchctl print` succeeds twice, then the job is gone
        ran = []

        def run(cmd):
            ran.append(cmd[:2])
            return still_loaded.pop(0) if cmd[1] == "print" else 0
        agent.install_service(plan, run=run, sleep=lambda s: None)
        assert ran == [["launchctl", "bootout"], ["launchctl", "print"], ["launchctl", "print"],
                       ["launchctl", "print"], ["launchctl", "bootstrap"]]

    def test_linux(self, tmp_path):
        plan = self.files("linux", tmp_path)
        path, content = plan["files"][0]
        assert path.endswith(".config/systemd/user/llama-serve-node.service")
        assert "ExecStart=/usr/bin/python3 " in content and "Restart=on-failure" in content
        assert ["systemctl", "--user", "enable", "--now", "llama-serve-node.service"] in plan["start"]
        assert "linger" in plan["hint"]

    def test_android(self, tmp_path):
        plan = self.files("android", tmp_path)
        path, content = plan["files"][0]
        assert path.endswith(".termux/boot/llama-serve-node") and content.startswith("#!")
        assert "Termux:Boot" in plan["hint"]

    def test_paths_are_quoted(self, tmp_path):
        plan = agent.service_plan("linux", python="/usr/bin/python3", script="/home/a b/agent.py",
                                  config="/home/a b/c.json", home=str(tmp_path), uid=1)
        assert '"/home/a b/agent.py"' in plan["files"][0][1]

    def test_install_and_uninstall(self, tmp_path):
        ran = []
        plan = self.files("linux", tmp_path)
        agent.install_service(plan, run=lambda cmd: ran.append(cmd) or 0)
        path = plan["files"][0][0]
        assert os.path.exists(path) and ran == plan["start"]
        ran.clear()
        agent.uninstall_service(plan, run=lambda cmd: ran.append(cmd) or 0)
        assert not os.path.exists(path) and ran == plan["stop"]


class TestCoordinatorByAddress:
    """Found live 2026-10-16 on the owner's Mac: enrollment worked (whois from the
    terminal), but the running agent answered every coordinator call with 401 --
    its runtime whois failed. The coordinator's overlay address (from the enrolled
    coordinator_url, checked by whois once at enrollment) now identifies it too;
    on Tailscale/WireGuard an overlay address is bound to one device key. The token
    stays mandatory either way."""

    def test_pinned_address_accepted_without_whois(self):
        assert agent.authorize("100.64.7.10", f"Bearer {TOKEN}", None, TOKEN, COORD,
                               coordinator_ip="100.64.7.10")

    def test_other_address_still_needs_identity(self):
        assert not agent.authorize("100.90.1.2", f"Bearer {TOKEN}", None, TOKEN, COORD,
                                   coordinator_ip="100.64.7.10")

    def test_pinned_address_still_needs_token(self):
        assert not agent.authorize("100.64.7.10", "Bearer wrong", None, TOKEN, COORD,
                                   coordinator_ip="100.64.7.10")

    def test_pinned_address_must_be_overlay(self):
        # a LAN address is never trusted, even if a config names it
        assert not agent.authorize("192.168.1.40", f"Bearer {TOKEN}", None, TOKEN, COORD,
                                   coordinator_ip="192.168.1.40")

    def test_coordinator_ip_from_config(self, setup):
        cfg, _ = setup
        c = conf(cfg, coordinator_url="http://100.64.7.10:8030")
        assert agent.Agent(c).coordinator_ip() == "100.64.7.10"
        assert agent.Agent(conf(cfg)).coordinator_ip() is None

    def test_refusal_reason(self):
        assert agent.refusal_reason("100.90.1.2", f"Bearer {TOKEN}", None, TOKEN, COORD) \
            == "identity: whois gave nothing for 100.90.1.2"
        assert agent.refusal_reason("100.90.1.2", None, None, TOKEN, COORD) == "missing or wrong token"
        assert "not the coordinator" in agent.refusal_reason("100.90.1.2", f"Bearer {TOKEN}", "x@y", TOKEN, COORD)


class TestStartModes:
    """Embedding and reranking servers on devices (2026-10-16). A mode is one of a
    fixed set mapped to fixed llama-server flags; pooling only from llama.cpp's own
    list. Still no free-form arguments."""
    accels = [{"index": 0, "kind": "cuda"}]

    def test_defaults_to_chat(self):
        r = agent.StartRequest.from_dict({"model": "m.gguf"}, self.accels)
        assert r.mode == "chat" and r.pooling is None and r.flags() == []

    def test_embedding(self):
        r = agent.StartRequest.from_dict({"model": "m.gguf", "mode": "embedding", "pooling": "cls"}, self.accels)
        assert r.flags() == ["--embedding", "--pooling", "cls"]

    def test_reranking(self):
        r = agent.StartRequest.from_dict({"model": "m.gguf", "mode": "reranking"}, self.accels)
        assert r.flags() == ["--reranking"]

    @pytest.mark.parametrize("body", [
        {"model": "m.gguf", "mode": "--rpc"}, {"model": "m.gguf", "mode": "embedding", "pooling": "max; rm"},
        {"model": "m.gguf", "mode": "chat", "pooling": "cls"},        # pooling only for embedding/reranking
    ])
    def test_refused(self, body):
        with pytest.raises(agent.RequestError):
            agent.StartRequest.from_dict(body, self.accels)


def test_embedding_instance_end_to_end(setup, monkeypatch):
    cfg, _ = setup
    monkeypatch.setattr(agent, "detect_accelerators", lambda: [
        {"index": 0, "kind": "cuda", "name": "Fake GPU", "total_mb": 8000, "free_mb": 7000}])
    a = agent.Agent(conf(cfg))
    try:
        inst = a.start(agent.StartRequest(model="tiny.gguf", mode="embedding", pooling="mean"))
        assert inst["mode"] == "embedding" and inst["pooling"] == "mean"
        for _ in range(100):
            if a.public(a.instances[inst["id"]])["state"] == "ready":
                break
            time.sleep(0.1)
        args = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{a.instances[inst['id']]['port']}/v1/embeddings", data=b'{"messages":[{"content":"x"}]}',
            headers={"Content-Type": "application/json"}), timeout=10).read())["args"]
        assert "--embedding" in args and args[args.index("--pooling") + 1] == "mean"
    finally:
        agent.stop_all(a)


def test_the_agent_server_does_not_resolve_its_own_name(monkeypatch, setup):
    """http.server calls socket.getfqdn() while binding -- ~35 s on a Mac whose name resolution hangs (CI macOS)."""
    import socket
    cfg, _ = setup
    monkeypatch.setattr(socket, "getfqdn", lambda *a: pytest.fail("getfqdn called"))
    server = agent.make_server(conf(cfg))
    server.server_close()
