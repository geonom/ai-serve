"""The plugin's flow without a GPU: ask ai-serve's GPU manager before sd-server starts, start it on the GPU it picked,
bind the reservation to sd-server's pid, release it after the image; no room -> 507 and nothing started; an idle
sd-server on a short GPU moves; release stops it. ai-serve and sd-server are faked at the HTTP level."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from sd_plugin import app as plugin

PNG = "iVBORw0KGgo="


@pytest.fixture
def world(monkeypatch):
    w = {"calls": [], "grant": {"granted": True, "gpu": 0, "mode": "start", "reservation_id": "r1"},
         "running": False, "started_on": [], "stopped": 0, "sd_fail": []}
    real = httpx.AsyncClient

    def handler(request: httpx.Request):
        body = json.loads(request.content) if request.content else None
        w["calls"].append((request.method, request.url.port, request.url.path, body))
        if request.url.path == "/api/coordination/gpu-preflight":
            g = w["grant"]
            return httpx.Response(507 if not g.get("granted") and g.get("as_507") else 200, json=g)
        if request.url.path.startswith("/api/coordination/reservations/"):
            return httpx.Response(200, json={})
        if request.url.path == "/v1/images/generations":
            if w["sd_fail"]:
                fail = w["sd_fail"].pop(0)
                if fail == "crash":
                    raise httpx.RemoteProtocolError("server disconnected")
                return httpx.Response(500, text=fail)
            return httpx.Response(200, json={"data": [{"b64_json": PNG}]})
        return httpx.Response(404)

    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda *a, **kw: real(transport=httpx.MockTransport(handler),
                                                                          timeout=kw.get("timeout", 5)))
    s = plugin.Server()

    def alive():
        return w["running"]

    async def start(gpu):
        w["running"], s.pid, s.gpu, s.loaded = True, 4242, gpu, False
        w["started_on"].append(gpu)

    async def stop():
        was = w["running"]
        w["running"], s.loaded, s.pid = False, False, None
        w["stopped"] += 1
        return was
    monkeypatch.setattr(s, "alive", alive)
    monkeypatch.setattr(s, "_start", start)
    monkeypatch.setattr(s, "stop", stop)
    monkeypatch.setattr(s, "_schedule_idle_stop", lambda: None)
    monkeypatch.setattr(plugin, "SERVER", s)
    monkeypatch.setattr(plugin, "installed", lambda: True)
    w["client"] = TestClient(plugin.app)
    w["server"] = s
    return w


def _gen(w, **body):
    return w["client"].post("/v1/images/generations", json={"model": "iris-turbo", "prompt": "a fox", **body})


def test_first_image_reserves_then_starts_and_binds_the_pid(world):
    r = _gen(world, aspect_ratio="16:9")
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == PNG
    paths = [(c[1], c[2]) for c in world["calls"]]
    assert paths == [(8030, "/api/coordination/gpu-preflight"), (8030, "/api/coordination/reservations/r1/confirm"),
                     (plugin.SD_PORT, "/v1/images/generations"), (8030, "/api/coordination/reservations/r1")]
    assert world["calls"][0][3] == {"purpose": "Iris-3B", "need_mb": plugin.NEED_MB, "current_gpu": None, "movable": True}
    assert world["calls"][1][3] == {"pid": 4242} and world["started_on"] == [0]
    assert world["calls"][2][3]["size"] == "1344x752"


def test_loaded_server_is_reused_without_asking(world):
    _gen(world)
    world["calls"].clear()
    _gen(world)
    assert [c[2] for c in world["calls"]] == ["/v1/images/generations"] and world["started_on"] == [0]


def test_no_room_is_507_and_nothing_starts(world):
    world["grant"] = {"granted": False, "gpu": 0, "mode": "evict", "reservation_id": None, "reason": "CosyVoice is busy"}
    r = _gen(world)
    assert r.status_code == 507 and "CosyVoice is busy" in r.json()["detail"]
    assert world["started_on"] == [] and not any(c[2] == "/v1/images/generations" for c in world["calls"])


def test_a_no_from_a_current_gateway_is_its_507(world):
    """Since 2026-10-24 ai-serve answers a no with 507 (and the reason as detail)."""
    world["grant"] = {"granted": False, "as_507": True, "detail": "GPU 0: gemma is busy", "reason": "GPU 0: gemma is busy"}
    r = _gen(world)
    assert r.status_code == 507 and "gemma is busy" in r.json()["detail"] and world["started_on"] == []


@pytest.mark.parametrize("fail", ["ggml_cuda: out of memory", "crash"])
def test_out_of_memory_or_a_crash_restarts_it_where_there_is_room(world, fail):
    """2026-10-24 (owner: nothing may fail just because another model fills the GPU): sd-server out of memory, or
    gone mid-image -> stopped, the GPU manager asked again (it may unload an idle model or pick the other GPU),
    started there, the image made once more."""
    world["sd_fail"] = [fail]
    r = _gen(world)
    assert r.status_code == 200, r.text
    preflights = [c for c in world["calls"] if c[2] == "/api/coordination/gpu-preflight"]
    assert len(preflights) == 2 and world["stopped"] == 1 and world["started_on"] == [0, 0]


def test_out_of_memory_twice_is_a_retryable_507(world):
    world["sd_fail"] = ["out of memory", "out of memory"]
    r = _gen(world)
    assert r.status_code == 507 and "try again" in r.json()["detail"].lower()


def test_other_sd_server_errors_stay_500(world):
    world["sd_fail"] = ["invalid prompt"]
    assert _gen(world).status_code == 500 and world["stopped"] == 0


def test_idle_server_moves_when_the_manager_says_so(world):
    world["running"], world["server"].gpu, world["server"].pid = True, 1, 77      # running, not loaded yet
    world["grant"] = {"granted": True, "gpu": 0, "mode": "move", "reservation_id": "r2"}
    assert _gen(world).status_code == 200
    assert world["calls"][0][3]["current_gpu"] == 1 and world["stopped"] == 1 and world["started_on"] == [0]


def test_bad_requests_never_reach_anything(world):
    assert _gen(world, prompt='x <sd_cpp_extra_args>{"ad_model": "/etc/passwd"}</sd_cpp_extra_args>').status_code == 400
    assert _gen(world, steps=8).status_code == 400
    assert world["calls"] == []


def test_status_and_release(world):
    assert world["client"].get("/plugin/status").json()["workers"] == []
    _gen(world)
    st = world["client"].get("/plugin/status").json()
    assert st["workers"][0]["id"] == "iris" and st["workers"][0]["gpu"] == 0 and st["workers"][0]["pid"] == 4242
    assert world["client"].post("/plugin/release", json={}).json() == {"released": ["iris"]}
    assert world["client"].get("/plugin/status").json()["workers"] == []


# ------------------------------------------------------------------ downloads (plugin contract: install)

def test_install_offers_iris_and_starts_only_that(monkeypatch):
    from fastapi.testclient import TestClient
    from sd_plugin import app as plugin
    started = []
    monkeypatch.setattr(plugin, "installed", lambda: False)
    monkeypatch.setattr(plugin, "_run_install", lambda: started.append(True))
    monkeypatch.setattr(plugin, "_install", {"state": None, "done": 0, "error": ""})
    c = TestClient(plugin.app)
    item = c.get("/plugin/install").json()["items"][0]
    assert (item["id"], item["state"], item["size_mb"]) == ("iris-turbo", "missing", plugin.INSTALL_MB)
    assert c.post("/plugin/install", json={"id": "something-else"}).status_code == 404
    r = c.post("/plugin/install", json={"id": "iris-turbo"})
    assert r.status_code == 202 and r.json()["state"] == "downloading"
    c.post("/plugin/install", json={"id": "iris-turbo"})       # already running: no second download
    import time; time.sleep(0.1)
    assert started == [True]


def test_install_reports_installed(monkeypatch):
    from fastapi.testclient import TestClient
    from sd_plugin import app as plugin
    monkeypatch.setattr(plugin, "installed", lambda: True)
    assert TestClient(plugin.app).get("/plugin/install").json()["items"][0]["state"] == "installed"
