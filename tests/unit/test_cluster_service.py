"""Executing routing plans (backend/cluster/service.py) against a fake agent.

- reuse/start return an upstream through the agent's proxy (token header) and count
  the request; start waits until the instance is ready.
- copy runs in the background and the caller gets 503 + Retry-After at once; a node
  that refuses the download is skipped for the next candidate (user: "wenn ein
  Download nicht möglich ist, wird auf ein anderes ausgewichen"); only one copy per
  model at a time.
- Instances the coordinator started are stopped after their idle time; ones it
  didn't start are never touched. Files are never deleted."""
import asyncio

import pytest
from fastapi import HTTPException

from backend.cluster import service
from backend.cluster.enrollment import NodeRecord


class FakeAgent:
    def __init__(self, nodes):
        self.nodes = nodes          # id -> dict(models, instances, accelerators, runtime_kinds, refuse_download)
        self.calls = []

    async def __call__(self, rec, method, path, body=None, timeout=10.0):
        self.calls.append((rec.id, method, path))
        n = self.nodes[rec.id]
        if path == "/v1/node":
            return {"accelerators": n["accelerators"], "instances": n["instances"],
                    "runtime_kinds": n["runtime_kinds"]}
        if path == "/v1/models":
            return {"models": [{"name": m, "size_mb": 7000} for m in n["models"]]}
        if path == "/v1/instances" and method == "POST":
            inst = {"id": "new1", "model": body["model"], "state": "ready", "gpu": body.get("gpu"),
                    "ctx_size": body["ctx_size"]}
            n["instances"].append(inst)
            return inst
        if path.startswith("/v1/instances/") and method == "GET":
            return next(i for i in n["instances"] if i["id"] == path.rsplit("/", 1)[1])
        if path.startswith("/v1/instances/") and method == "DELETE":
            n["instances"] = [i for i in n["instances"] if i["id"] != path.rsplit("/", 1)[1]]
            return {}
        if path == "/v1/downloads":
            return {"downloads": n.get("downloads", {}), "files": []}
        raise AssertionError(path)


GPU = [{"index": 0, "kind": "metal", "name": "M3", "total_mb": 24000, "budget_mb": 16000, "free_mb": None}]


@pytest.fixture
def svc(monkeypatch, tmp_path):
    recs = {n: NodeRecord(id=n, url=f"http://100.100.0.{i}:8040", token="t" * 40, enrolled_at=0,
                          coordinator_url="http://my-pc:8030") for i, n in enumerate(["mac", "pi"], 1)}

    class Reg:
        def all(self):
            return list(recs.values())

        def get(self, i):
            return recs.get(i)
    monkeypatch.setattr(service, "registry", lambda: Reg())
    s = service.ClusterService(snapshot_ttl_s=0)
    return s


def run(coro):
    return asyncio.run(coro)


def test_start_and_upstream(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": ["m.gguf"], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
                      "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    monkeypatch.setattr(service, "agent_call", fake)
    up = run(svc.upstream_for("m.gguf", local_path=None, ctx_size=8192, keep_alive=300))
    assert up.base == "http://100.100.0.1:8040/v1/instances/new1/proxy"
    assert up.headers == {"Authorization": "Bearer " + "t" * 40}
    assert ("mac", "POST", "/v1/instances") in fake.calls


def test_copy_falls_back_and_answers_503(svc, monkeypatch):
    fake = FakeAgent({
        "mac": {"models": [], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
        "pi": {"models": [], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]}})
    monkeypatch.setattr(service, "agent_call", fake)
    sent = []

    async def send(rec, name, local_path):
        sent.append(rec.id)
        if rec.id == "mac":
            raise service.CopyRefused("not enough room in the model quota")
        return {"state": "downloading"}

    async def wait_download(rec, name, timeout_s):
        return "done"
    monkeypatch.setattr(service, "send_model_to", send)
    monkeypatch.setattr(service, "wait_download", wait_download)

    async def scenario():
        with pytest.raises(HTTPException) as e:
            await svc.upstream_for("m.gguf", local_path="/models/m.gguf", ctx_size=8192, keep_alive=300, local_size_mb=7000)
        assert e.value.status_code == 503 and e.value.headers["Retry-After"]
        with pytest.raises(HTTPException):   # second call: same copy, not a new one
            await svc.upstream_for("m.gguf", local_path="/models/m.gguf", ctx_size=8192, keep_alive=300, local_size_mb=7000)
        await svc.copies["m.gguf"]["task"]
    run(scenario())
    assert sent == ["mac", "pi"]
    assert svc.copies["m.gguf"]["state"] == "done" and svc.copies["m.gguf"]["node"] == "pi"


def test_no_node_no_local_file(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": [], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
                      "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    monkeypatch.setattr(service, "agent_call", fake)
    with pytest.raises(HTTPException) as e:
        run(svc.upstream_for("m.gguf", local_path=None, ctx_size=8192, keep_alive=300))
    assert e.value.status_code == 507


def test_idle_reaper_only_own_instances(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": ["m.gguf"], "instances": [
        {"id": "user1", "model": "x.gguf", "state": "ready", "gpu": 0, "ctx_size": 4096}],
        "accelerators": GPU, "runtime_kinds": ["metal"]},
        "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    monkeypatch.setattr(service, "agent_call", fake)
    up = run(svc.upstream_for("m.gguf", local_path=None, ctx_size=8192, keep_alive=60))
    svc.begin(up)
    svc.end(up, keep_alive=60)
    svc.sessions[up.session]["last_used"] -= 61
    run(svc.reap_idle())
    ids = [i["id"] for i in fake.nodes["mac"]["instances"]]
    assert ids == ["user1"]       # ours stopped, the user's untouched


def test_place_model_best_device_with_fallback(svc, monkeypatch):
    fake = FakeAgent({
        "mac": {"models": [], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
        "pi": {"models": [], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]}})
    monkeypatch.setattr(service, "agent_call", fake)
    tried = []

    async def send_plan(rec, plan):
        tried.append(rec.id)
        return {}

    async def wait_download(rec, name, timeout_s):
        return "not enough room in the model quota" if rec.id == "mac" else "done"
    monkeypatch.setattr(service, "send_plan_to", send_plan)
    monkeypatch.setattr(service, "wait_download", wait_download)

    async def scenario():
        st = await svc.place_model({"name": "h.gguf", "size": 7000 << 20, "sha256": "a" * 64,
                                    "sources": ["https://huggingface.co/u/r/resolve/main/h.gguf"]})
        assert st["state"] == "copying"
        await svc.copies["h.gguf"]["task"]
    run(scenario())
    assert tried == ["mac", "pi"] and svc.copies["h.gguf"]["node"] == "pi"


def test_place_model_already_there(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": ["h.gguf"], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
                      "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    monkeypatch.setattr(service, "agent_call", fake)
    st = run(svc.place_model({"name": "h.gguf", "size": 7000 << 20, "sha256": "a" * 64, "sources": []}))
    assert st == {"state": "present", "node": "mac"}


def test_start_passes_mode_and_pooling(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": ["bge.gguf"], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
                      "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    sent = []
    orig = fake.__call__

    async def spy(rec, method, path, body=None, timeout=10.0):
        if path == "/v1/instances" and method == "POST":
            sent.append(body)
        return await orig(rec, method, path, body, timeout)
    monkeypatch.setattr(service, "agent_call", spy)
    run(svc.upstream_for("bge.gguf", local_path=None, ctx_size=8192, keep_alive=300, mode="embedding", pooling="cls"))
    assert sent[0]["mode"] == "embedding" and sent[0]["pooling"] == "cls"


def test_chat_start_body_stays_compatible_with_older_agents(svc, monkeypatch):
    fake = FakeAgent({"mac": {"models": ["m.gguf"], "instances": [], "accelerators": GPU, "runtime_kinds": ["metal"]},
                      "pi": {"models": [], "instances": [], "accelerators": [], "runtime_kinds": []}})
    sent = []
    orig = fake.__call__

    async def spy(rec, method, path, body=None, timeout=10.0):
        if path == "/v1/instances" and method == "POST":
            sent.append(body)
        return await orig(rec, method, path, body, timeout)
    monkeypatch.setattr(service, "agent_call", spy)
    run(svc.upstream_for("m.gguf", local_path=None, ctx_size=8192, keep_alive=300))
    assert "mode" not in sent[0] and "pooling" not in sent[0]


# --------------------------------------------------------------------------- snapshots without waiting
# Found 2026-10-22: with an enrolled Mac asleep, every expired snapshot cache made the next /v1/llm/models wait the
# full 5 s connect timeout (every other call took 5.3 s; the Playground stalled loading).

class SlowOffline:
    """mac: unreachable (counts the attempts); pi: answers."""
    def __init__(self):
        self.tries = {"mac": 0, "pi": 0}

    async def __call__(self, rec, method, path, body=None, timeout=10.0):
        self.tries[rec.id] += 1
        if rec.id == "mac":
            raise service.NodeUnreachable("asleep")
        if path == "/v1/node":
            return {"accelerators": GPU, "instances": [], "runtime_kinds": ["metal"]}
        return {"models": []}


def test_an_unreachable_device_is_not_asked_again_for_a_while(svc, monkeypatch):
    agent = SlowOffline()
    monkeypatch.setattr(service, "agent_call", agent)
    clock = [1000.0]
    monkeypatch.setattr(service.time, "monotonic", lambda: clock[0])
    s = service.ClusterService(snapshot_ttl_s=0, offline_backoff_s=30)
    for _ in range(3):
        snaps = {n.id: n for n in run(s.snapshots())}
        assert snaps["mac"].online is False and snaps["pi"].online is True
    assert agent.tries["mac"] == 1                      # asked once, then left alone
    clock[0] += 31
    run(s.snapshots())
    assert agent.tries["mac"] == 2                      # after the pause: asked again


def test_an_expired_but_recent_snapshot_is_answered_at_once_and_refreshed_behind(svc, monkeypatch):
    agent = SlowOffline()
    monkeypatch.setattr(service, "agent_call", agent)
    clock = [1000.0]
    monkeypatch.setattr(service.time, "monotonic", lambda: clock[0])
    s = service.ClusterService(snapshot_ttl_s=5, offline_backoff_s=0)

    async def scenario():
        first = await s.snapshots()                     # nothing cached yet: waits
        clock[0] += 6                                   # expired, but not old
        again = await s.snapshots()                     # answered from the cache ...
        assert again is first
        await asyncio.sleep(0)                          # ... while the refresh runs behind it
        await asyncio.sleep(0)
        return agent.tries["pi"]
    assert run(scenario()) == 4                          # 2 calls first, 2 for the background refresh
