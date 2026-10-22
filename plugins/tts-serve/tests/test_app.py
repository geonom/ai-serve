"""tts-serve over HTTP with a real (fake) worker process: spawn on demand, health
wait, adapter payload, keep-alive unload, busy guard, eviction, gateway
coordination, error mapping, temp-file cleanup."""
import asyncio
import base64
import json
import os
import stat
import sys
import time
from pathlib import Path

import httpx
import pytest

from tts_serve import app as app_module
from tts_serve.workers import CoordinationRefused

WORKER = r'''
import base64, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
if os.environ.get("FAKE_CRASH"):
    sys.exit(3)
class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, {"status": "ok", "device": "cuda:0"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if req.get("text") == "worker-400":
            return self._send(400, {"error": "bad combination"})
        if req.get("text") == "worker-500":
            return self._send(500, {"error": "CUDA out of memory at /secret/path"})
        if req.get("text") == "slow":
            time.sleep(1.5)
        ref = req.get("ref")
        req["ref_exists"] = bool(ref and os.path.exists(ref))
        req["env"] = os.environ.get("FAKE_ENV")
        self._send(200, {"audio_base64": base64.b64encode(json.dumps(req).encode()).decode(), "device": "cuda:0"})
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

ADAPTER = '''
from tts_serve.domain import RequestError
def build_payload(req, ref_audio_path):
    if req.variant == "bad":
        raise RequestError("variant bad is not supported")
    return {"text": req.text, "ref": ref_audio_path, "language": req.language}
'''


def _make_provider(root: Path, pid: str, gpu=0, vram_mb=1000, keep_alive_s=60, env=None, extra=None):
    d = root / pid
    (d / "venv" / "bin").mkdir(parents=True)
    (d / "venv" / "bin" / "python").symlink_to(sys.executable)
    (d / "worker.py").write_text(WORKER)
    (d / "adapter.py").write_text(ADAPTER)
    (d / "llama-serve.provider.json").write_text(json.dumps({
        "id": pid, "name": pid.title(), "capability": "tts", "gpu": gpu, "vram_mb": vram_mb,
        "command": ["./venv/bin/python", "worker.py", "--port", "{port}"],
        "env": env or {"FAKE_ENV": "{dir}"}, "keep_alive_s": keep_alive_s, "startup_timeout_s": 15,
        "default_reference": True, "variants": ["a", "b"], **(extra or {}),
    }))
    return d


class FakeGpu:
    def __init__(self, free=20000, total=24576):
        self.free, self.total = free, total

    def __call__(self):
        return {0: {"free": self.free, "total": self.total}}


class FakeCoordinator:
    def __init__(self, refuse=None):
        self.refuse, self.calls, self.confirmed = refuse, [], []
        self.last_reservation = None

    async def preflight(self, purpose, gpu, need_mb):
        self.calls.append((purpose, gpu, need_mb))
        if self.refuse:
            raise CoordinationRefused(self.refuse)
        self.last_reservation = f"r-{purpose}"
        return None

    async def confirm(self, reservation_id, pid):
        self.confirmed.append((reservation_id, pid))


@pytest.fixture
def env(tmp_path, monkeypatch):
    providers = tmp_path / "providers"
    providers.mkdir()
    _make_provider(providers, "alpha")
    _make_provider(providers, "beta", extra={
        "languages": ["de", "en"],
        "modes": [{"variant": "a", "label": "Klonen", "inputs": {"reference": "required"},
                   "params": [{"name": "temperature", "min": 0.1, "max": 1.5, "default": 0.8}]},
                  {"variant": "b", "label": "Sprecher", "inputs": {"speaker": "required"}, "speakers": ["Vivian"]}]})
    gpu, coord = FakeGpu(), FakeCoordinator()
    app = app_module.create_app(providers_dir=providers, logs_dir=tmp_path / "logs", gpu_probe=gpu,
                                coordinator=coord, default_engine="alpha")
    return app, providers, gpu, coord


async def _with_client(app, fn):
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30) as c:
            return await fn(c)


def run(app, fn):
    return asyncio.run(_with_client(app, fn))


def _decode(resp):
    return json.loads(base64.b64decode(resp.json()["audio_base64"]))


# --------------------------------------------------------------------------- generate

def test_generate_spawns_worker_and_uses_adapter(env):
    app, providers, _, coord = env
    ref = base64.b64encode(b"RIFFfake").decode()

    async def fn(c):
        r = await c.post("/v1/generate", json={"text": "Hallo", "language": "de", "ref_audio_base64": ref})
        engines = (await c.get("/v1/engines")).json()
        return r, engines

    r, engines = run(app, fn)
    assert r.status_code == 200, r.text
    payload = _decode(r)
    assert payload["text"] == "Hallo" and payload["language"] == "de"
    assert payload["ref_exists"] is True and payload["env"] == str(providers / "alpha")
    assert not os.path.exists(payload["ref"])  # temp reference removed afterwards
    alpha = next(e for e in engines["engines"] if e["id"] == "alpha")
    assert alpha["status"] == "idle" and alpha["loaded"] and alpha["device"] == "cuda:0"
    assert alpha["default_reference"] is True and alpha["variants"] == ["a", "b"]
    assert coord.calls == [("tts:alpha", 0, 1000 + 512)]
    # the reservation from the preflight is bound to the started worker's PID
    assert len(coord.confirmed) == 1 and coord.confirmed[0][0] == "r-tts:alpha" and coord.confirmed[0][1] > 1


def test_engines_describe_their_modes(env):
    app, *_ = env
    engines = run(app, lambda c: c.get("/v1/engines")).json()["engines"]
    beta = next(e for e in engines if e["id"] == "beta")
    assert [m["label"] for m in beta["modes"]] == ["Klonen", "Sprecher"] and beta["languages"] == ["de", "en"]
    assert beta["modes"][1]["speakers"] == ["Vivian"] and beta["params"] == ["temperature"]
    alpha = next(e for e in engines if e["id"] == "alpha")
    assert alpha["modes"] == [] and sorted(alpha["params"]) == ["cfg_weight", "exaggeration", "temperature"]


def test_params_an_engine_does_not_declare_are_refused(env):
    app, *_ = env
    ok = run(app, lambda c: c.post("/v1/generate", json={"text": "x", "engine": "beta", "params": {"temperature": 0.5}}))
    bad = run(app, lambda c: c.post("/v1/generate", json={"text": "x", "engine": "beta", "params": {"exaggeration": 1}}))
    assert ok.status_code == 200, ok.text
    assert bad.status_code == 400 and "exaggeration" in bad.text


def test_wav_response(env):
    app, *_ = env
    r = run(app, lambda c: c.post("/v1/generate", json={"text": "x", "engine": "beta", "response_format": "wav"}))
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"


@pytest.mark.parametrize("body, status", [
    ({"text": ""}, 400), ({"text": "x", "engine": "nope"}, 404), ({"text": "x", "variant": "bad"}, 400),
    ({"text": "x", "ref_audio_path": "/etc/passwd"}, 400), ({"text": "worker-400"}, 400),
])
def test_request_errors(env, body, status):
    app, *_ = env
    r = run(app, lambda c: c.post("/v1/generate", json=body))
    assert r.status_code == status and r.json()["detail"]


def test_worker_failure_is_502_without_internals_leaking_status(env):
    app, *_ = env

    async def fn(c):
        r = await c.post("/v1/generate", json={"text": "worker-500"})
        return r, (await c.get("/v1/engines")).json()

    r, engines = run(app, fn)
    assert r.status_code == 502
    assert next(e for e in engines["engines"] if e["id"] == "alpha")["status"] == "error"


def test_worker_crash_on_startup(env):
    app, providers, *_ = env
    manifest = json.loads((providers / "beta" / "llama-serve.provider.json").read_text())
    manifest["env"] = {"FAKE_CRASH": "1"}
    (providers / "beta" / "llama-serve.provider.json").write_text(json.dumps(manifest))
    r = run(app, lambda c: c.post("/v1/generate", json={"text": "x", "engine": "beta"}))
    assert r.status_code == 500 and "exited during startup" in r.json()["detail"]


def test_gateway_refusal_is_507_and_nothing_starts(env):
    app, _, _, coord = env
    coord.refuse = "Strata uses the GPUs"

    async def fn(c):
        r = await c.post("/v1/generate", json={"text": "x"})
        return r, (await c.get("/v1/engines")).json()

    r, engines = run(app, fn)
    assert r.status_code == 507 and "Strata" in r.json()["detail"]
    assert all(not e["loaded"] for e in engines["engines"])


# --------------------------------------------------------------------------- lifecycle

def test_unload_and_busy_guard(env):
    app, *_ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x"})
        slow = asyncio.create_task(c.post("/v1/generate", json={"text": "slow"}))
        await asyncio.sleep(0.5)
        busy = await c.post("/v1/engines/alpha/unload")
        await slow
        ok = await c.post("/v1/engines/alpha/unload")
        again = await c.post("/v1/engines/alpha/unload")
        engines = (await c.get("/v1/engines")).json()
        return busy, ok, again, engines

    busy, ok, again, engines = run(app, fn)
    assert busy.status_code == 409
    assert ok.json()["status"] == "unloaded" and again.json()["status"] == "already unloaded"
    assert not next(e for e in engines["engines"] if e["id"] == "alpha")["loaded"]


def test_keep_alive_unloads_idle_worker(env):
    app, *_ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x", "keep_alive": 1})
        await asyncio.sleep(2.5)
        return (await c.get("/v1/engines")).json()

    engines = run(app, fn)
    assert not next(e for e in engines["engines"] if e["id"] == "alpha")["loaded"]


def test_eviction_makes_room_with_idle_lru_engine(env):
    app, _, gpu, _ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x", "engine": "alpha"})
        gpu.free = 100  # beta (1000 MB + margin) won't fit next to alpha
        r = await c.post("/v1/generate", json={"text": "x", "engine": "beta"})
        return r, (await c.get("/v1/engines")).json()

    r, engines = run(app, fn)
    loaded = {e["id"]: e["loaded"] for e in engines["engines"]}
    assert r.status_code == 200 and loaded == {"alpha": False, "beta": True}


def test_unload_idle_endpoint_and_status(env):
    app, *_ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x"})
        before = (await c.get("/v1/status")).json()
        r = await c.post("/v1/unload_idle", params={"gpu": 0})
        after = (await c.get("/v1/status")).json()
        return before, r, after

    before, r, after = run(app, fn)
    assert before["status"] == "idle" and r.json()["unloaded"] == ["alpha"] and after["status"] == "unloaded"


def test_logs(env):
    app, *_ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x"})
        return await c.get("/v1/logs", params={"engine": "alpha", "tail": 50})

    r = run(app, fn)
    assert r.status_code == 200 and "alpha" in r.json()["logs"]


def test_shutdown_terminates_workers(env):
    app, *_ = env
    pids = []

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x"})
        pids.append(next(e for e in (await c.get("/v1/engines")).json()["engines"] if e["id"] == "alpha")["pid"])

    run(app, fn)
    time.sleep(0.5)
    with pytest.raises(OSError):
        os.kill(pids[0], 0)


# --------------------------------------------------------------------------- plugin contract (2026-10-21)
# ai-serve-ui's GPU manager learns from /v1/status.workers which processes hold GPU memory and frees them with
# POST /v1/release -- it no longer knows tts-serve by name (ai-serve-ui docs/plugins.md).

def test_status_lists_workers_and_release_frees_them(env):
    app, *_ = env

    async def fn(c):
        await c.post("/v1/generate", json={"text": "x"})
        st = (await c.get("/v1/status")).json()
        slow = asyncio.create_task(c.post("/v1/generate", json={"text": "slow"}))
        await asyncio.sleep(0.5)
        busy = await c.post("/v1/release", json={"worker": "alpha"})
        await slow
        one = await c.post("/v1/release", json={"worker": "alpha"})
        await c.post("/v1/generate", json={"text": "x", "engine": "beta"})
        idle = await c.post("/v1/release", json={})
        unknown = await c.post("/v1/release", json={"worker": "nope"})
        bad = await c.post("/v1/release", json={"worker": 5})
        return st, busy, one, idle, unknown, bad

    st, busy, one, idle, unknown, bad = run(app, fn)
    w = {x["id"]: x for x in st["workers"]}
    assert set(w) == {"alpha"} and w["alpha"]["pid"] > 1 and w["alpha"]["loaded"] is True and w["alpha"]["label"] == "Alpha"
    assert busy.status_code == 409
    assert one.status_code == 200 and one.json()["released"] == ["alpha"]
    assert idle.status_code == 200 and idle.json()["released"] == ["beta"]
    assert unknown.status_code == 404 and bad.status_code == 400


def test_default_engine_falls_back_to_an_installed_one(tmp_path):
    """DEFAULT engine not set up (e.g. cosyvoice on a fresh machine) -> the first installed engine, and the
    engine list says which ones are installed."""
    from tts_serve.workers import WorkerPool
    providers = tmp_path / "providers"
    _make_provider(providers, "alpha")
    _make_provider(providers, "beta")
    (providers / "alpha" / "venv" / "bin" / "python").unlink()      # alpha: only the folder
    pool = WorkerPool(providers, tmp_path / "logs", FakeGpu(), FakeCoordinator(), default_engine="alpha")
    view = pool.engines()
    assert {e["id"]: e["installed"] for e in view["engines"]} == {"beta": True}   # alpha: not set up -> errors
    assert any(err["dir"].endswith("alpha") and "setup" in err["error"] for err in view["errors"])
    assert view["default_engine"] == "beta"
