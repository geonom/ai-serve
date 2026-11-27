"""One model on several GPUs (2026-11-27, owner: "ein großes Modell, das auf beiden Grakas läuft" -- only when asked:
`gpu_id: "all"`). backend/gpu_split.py: what each card can give, how the need is shared, llama-server's flags."""
import pytest
from pydantic import BaseModel

from backend import gpu_split as gs
from backend.gpu_ids import ALL_GPUS, GpuRef, from_visible_devices, visible_devices_list

TABLE = {0: "GPU-ec716b3f-07d0", 1: "GPU-ae73ed6b-a062"}


def _gpu(index, total, consumers=(), reserved=0, free=None):
    free = total - sum(c["mb"] for c in consumers) if free is None else free
    return {"index": index, "total_mb": total, "free_mb": free, "reserved_mb": reserved, "consumers": list(consumers)}


def test_a_card_gives_what_is_free_plus_what_idle_models_hold():
    """2026-11-27, live: the 4060 Ti had 16380 MB, 6139 MB of other programs and 8523 MB free -- ~1.2 GB belongs to
    nobody nvidia-smi names (driver, display). Counting from the size promised 9.7 GB the card couldn't give."""
    view = [_gpu(1, 16380, [{"mb": 6139, "busy": False, "pinned": False, "evictable": False},
                            {"mb": 158, "busy": False, "pinned": False, "evictable": True}], free=8523)]
    assert gs.capacities(view) == {1: 8523 + 158 - gs.SAFETY_MB}


def test_a_card_gives_what_isnt_held_by_busy_pinned_or_other_programs():
    view = [_gpu(0, 24576, [{"mb": 9000, "busy": False, "pinned": False, "evictable": True},     # idle: may go
                            {"mb": 3000, "busy": True, "pinned": False, "evictable": True}]),    # working: stays
            _gpu(1, 16380, [{"mb": 6139, "busy": False, "pinned": False, "evictable": False},    # other programs
                            {"mb": 1000, "busy": False, "pinned": True, "evictable": True}], reserved=500)]
    assert gs.capacities(view) == {0: 24576 - 3000 - gs.SAFETY_MB, 1: 16380 - 6139 - 1000 - 500 - gs.SAFETY_MB}


def test_the_need_is_shared_by_what_each_card_can_give():
    s = gs.shares(30000, {0: 24000, 1: 12000})
    assert set(s) == {0, 1}
    assert s[0] + s[1] == 30000 + 2 * gs.OVERHEAD_MB          # each card also holds its own compute buffers
    net = {g: s[g] - gs.OVERHEAD_MB for g in s}               # the model's part, shared like the room left
    room = {0: 24000 - gs.OVERHEAD_MB, 1: 12000 - gs.OVERHEAD_MB}
    assert abs(net[0] / net[1] - room[0] / room[1]) < 0.01
    assert s[0] <= 24000 and s[1] <= 12000


def test_no_share_when_even_both_cards_together_are_short():
    assert gs.shares(40000, {0: 24000, 1: 12000}) is None


def test_a_card_with_too_little_to_give_is_left_out():
    s = gs.shares(20000, {0: 23000, 1: 300, 2: 9000})
    assert set(s) == {0, 2}
    assert gs.shares(20000, {0: 23000, 1: 300}) is None       # one card alone is no split


def test_the_card_giving_most_comes_first_and_leads():
    s = {1: 3000 + gs.OVERHEAD_MB, 0: 7000 + gs.OVERHEAD_MB}   # the model's part 7:3
    assert gs.order(s) == [0, 1]
    assert gs.tensor_split(s, [0, 1]) == "0.700,0.300"
    assert gs.flags(s, [0, 1]) == ["--split-mode", "layer", "--tensor-split", "0.700,0.300", "--main-gpu", "0"]


def test_gpu_id_all_means_every_card():
    class Req(BaseModel):
        gpu_id: GpuRef = None
    assert Req(gpu_id="all").gpu_id == ALL_GPUS
    assert Req(gpu_id="ALL").gpu_id == ALL_GPUS
    assert Req(gpu_id=1).gpu_id == 1 and Req().gpu_id is None


def test_a_split_process_is_pinned_to_its_cards_by_uuid_in_order():
    assert visible_devices_list([1, 0], TABLE) == "GPU-ae73ed6b-a062,GPU-ec716b3f-07d0"
    assert gs.devices_of("GPU-ec716b3f-07d0,GPU-ae73ed6b-a062", TABLE) == [0, 1]   # adoption after a restart
    assert gs.devices_of("GPU-ec716b3f-07d0", TABLE) == [0]
    assert gs.devices_of("", TABLE) == [] and gs.devices_of("GPU-nope", TABLE) == []
    assert from_visible_devices("GPU-ec716b3f-07d0,GPU-ae73ed6b-a062", TABLE) is None   # unchanged: one card only


@pytest.mark.parametrize("asked,info,reuse", [
    (None, {"gpu_id": 0, "gpu_ids": [0, 1]}, True),           # no wish: any instance serves
    (ALL_GPUS, {"gpu_id": 0, "gpu_ids": [0, 1]}, True),
    (ALL_GPUS, {"gpu_id": 0}, False),                         # asked for both cards, this one runs on one
    (0, {"gpu_id": 0, "gpu_ids": [0, 1]}, False),             # asked for card 0 alone
    (0, {"gpu_id": 0}, True),
    (1, {"gpu_id": 0}, False),
])
def test_which_running_instance_answers_a_gpu_wish(asked, info, reuse):
    assert gs.serves(asked, info) is reuse


# ------------------------------------------------------------------ the gateway's load path (model_manager._load_locked)

class FakeManager:
    def __init__(self, view, refuse=()):
        self._view, self.refuse = view, set(refuse)
        self.reserved, self.confirmed, self.released = [], [], []

    async def view(self):
        return self._view

    async def reserve(self, gpu, need_mb, purpose, requester=None):
        if gpu in self.refuse:
            return None, [], "gemma (busy)"
        r = type("R", (), {"id": f"r{gpu}", "gpu": gpu, "pid": None})()
        self.reserved.append((gpu, need_mb, purpose))
        return r, [], ""

    def confirm(self, rid, pid):
        self.confirmed.append((rid, pid))

    def release(self, rid):
        self.released.append(rid)


@pytest.fixture
def load(monkeypatch, tmp_path):
    import httpx
    from backend import model_manager as mm
    from backend.cluster import local
    from backend.state import state
    spawned = {}

    class Proc:
        pid = 4242

        def poll(self):
            return None

    def popen(cmd, env=None, **kw):
        spawned.update(cmd=cmd, env=env)
        return Proc()

    async def port():
        return 8123

    async def fast(_):
        return None

    monkeypatch.setattr(state, "managed_processes", {})
    monkeypatch.setattr(mm, "check_model_file_exists", lambda *a, **k: None)
    monkeypatch.setattr(mm, "get_model_size_mb", lambda p: 27000)
    monkeypatch.setattr(mm, "estimate_kv_overhead_mb", lambda ctx: 1000)
    monkeypatch.setattr(mm, "reserve_free_port", port)
    monkeypatch.setattr(mm, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(mm, "remember_origin", lambda *a: None)
    monkeypatch.setattr(mm, "reset_keep_alive", lambda *a: None)
    monkeypatch.setattr(mm, "get_model_traits", lambda p: {})
    monkeypatch.setattr(mm.subprocess, "Popen", popen)
    monkeypatch.setattr(mm.asyncio, "sleep", fast)
    monkeypatch.setattr(mm.httpclient, "AsyncClient",
                        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    monkeypatch.setattr(mm, "visible_devices_list", lambda ids: ",".join(f"GPU-{i}" for i in ids))

    def run(manager, gpu_id=ALL_GPUS):
        import asyncio
        monkeypatch.setattr(local, "MANAGER", manager)
        port = asyncio.run(mm._load_locked("/m/Qwen3.8-27B-Q8_0.gguf", gpu_id, 8192, 8, -1, 300, None, False))
        return port, spawned, state.managed_processes
    return run


VIEW = [_gpu(0, 24576), _gpu(1, 16380, [{"mb": 6139, "busy": False, "pinned": False, "evictable": False}])]


def test_all_spreads_the_model_reserving_each_cards_share_for_one_process(load):
    m = FakeManager(VIEW)
    port, spawned, procs = load(m)
    assert [g for g, _, _ in m.reserved] == [0, 1]                          # the card giving most leads
    assert sum(mb for _, mb, _ in m.reserved) == 28000 + 2 * gs.OVERHEAD_MB
    assert all(p == "llm:Qwen3.8-27B-Q8_0.gguf@8192+split" for _, _, p in m.reserved)
    assert m.confirmed == [("r0", 4242), ("r1", 4242)] and m.released == ["r0", "r1"]
    assert spawned["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1"
    cmd = spawned["cmd"]
    assert cmd[cmd.index("--split-mode") + 1] == "layer" and cmd[cmd.index("--main-gpu") + 1] == "0"
    a, b = (float(x) for x in cmd[cmd.index("--tensor-split") + 1].split(","))
    assert a > b and abs(a + b - 1) < 0.01
    assert procs[port]["gpu_ids"] == [0, 1] and procs[port]["gpu_id"] == 0


def test_a_card_refusing_its_share_gives_the_others_back_and_says_why(load):
    from fastapi import HTTPException
    m = FakeManager(VIEW, refuse={1})
    with pytest.raises(HTTPException) as e:
        load(m)
    assert e.value.status_code == 507 and "GPU 1" in e.value.detail and "gemma (busy)" in e.value.detail
    assert m.released == ["r0"] and m.confirmed == []


def test_too_big_even_for_both_is_a_507_naming_what_each_card_can_give(load):
    from fastapi import HTTPException
    m = FakeManager([_gpu(0, 24576, [{"mb": 20000, "busy": True, "pinned": False, "evictable": True}]),
                     _gpu(1, 16380)])
    with pytest.raises(HTTPException) as e:
        load(m)
    assert e.value.status_code == 507 and "GPU 0" in e.value.detail and m.reserved == []


def test_status_says_which_cards_a_spread_instance_uses(monkeypatch):
    """2026-11-27, live: /api/status filtered gpu_ids out (its response model didn't have it)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.state import state

    class Proc:
        pid = 4242

        def poll(self):
            return None
    monkeypatch.setattr(state, "managed_processes", {
        1: {"process": Proc(), "model": "/m/big.gguf", "port": 1, "gpu_id": 0, "gpu_ids": [0, 1], "ctx_size": 4096}})
    rows = TestClient(app, client=("127.0.0.1", 50000)).get("/api/status").json()["instances"]
    row = next(r for r in rows if r["port"] == 1)
    assert row["gpu_id"] == 0 and row["gpu_ids"] == [0, 1]
