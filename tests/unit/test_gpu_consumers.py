"""Local GPU management across all consumers (backend/cluster/consumers.py) -- pure.

Architecture review 2026-10-16: every subsystem freed VRAM only for itself
(tts-serve its engines, ComfyUI after 180 s, llama-server loads refused with 507
even when an idle ComfyUI held 10 GB next to them). Now one view per GPU:

- Who holds memory is MEASURED: nvidia-smi's per-process VRAM, attributed to a
  consumer by walking each process's parent chain to a known root pid (a
  llama-server, ComfyUI, a tts-serve engine worker, stt-serve, a training job,
  Strata). Unknown processes (other programs) count as "other": they take room
  but are never touched.
- A consumer is evictable only when idle and not pinned; training and Strata never
  are (Strata = exclusive). Eviction means unloading from VRAM -- never deleting
  files.
- Making room on a GPU: evict idle consumers least-recently-used first, whatever
  their kind, until the need fits the hardware-free memory; if that can't work,
  nothing is evicted and the blockers are named."""
import pytest

from backend.cluster.consumers import Consumer, attribute_vram, plan_room, gpu_view

GPUS = [{"index": 0, "name": "RTX 3090", "total": 24576, "free": 2000},
        {"index": 1, "name": "RTX 4060 Ti", "total": 16380, "free": 14000}]


def consumers():
    return [
        Consumer(id="llm:8081", kind="llm", label="gemma-12b", root_pids={100}, last_used=50),
        Consumer(id="comfyui", kind="image", label="ComfyUI", root_pids={200}, last_used=10),
        Consumer(id="tts:cosyvoice", kind="tts", label="CosyVoice", root_pids={300}, last_used=30, busy=True),
        Consumer(id="training:job1", kind="training", label="sft job1", root_pids={400}, busy=True,
                 evictable_kind=False),
    ]


PARENTS = {101: 100, 201: 200, 999: 1, 301: 300}


class TestAttribution:
    def test_by_parent_chain(self):
        procs = [(101, 0, 9000), (201, 0, 10000), (301, 0, 3000), (999, 1, 500), (400, 0, 0)]
        att = attribute_vram(procs, PARENTS, consumers())
        assert att[("llm:8081", 0)] == 9000 and att[("comfyui", 0)] == 10000
        assert att[("tts:cosyvoice", 0)] == 3000 and att[("other", 1)] == 500

    def test_cycle_safe(self):
        att = attribute_vram([(7, 0, 100)], {7: 8, 8: 7}, consumers())
        assert att[("other", 0)] == 100


class TestPlanRoom:
    def plan(self, need, gpu=0, pinned=frozenset(), free=2000):
        gpus = [dict(GPUS[0], free=free), GPUS[1]]
        att = {("llm:8081", 0): 9000, ("comfyui", 0): 10000, ("tts:cosyvoice", 0): 3000}
        return plan_room(gpu, need, gpus, consumers(), att, pinned=pinned)

    def test_fits_without_eviction(self):
        p = self.plan(1500)
        assert p.ok and p.evict == []

    def test_evicts_least_recently_used_idle_first(self):
        p = self.plan(9000)          # ComfyUI (last used 10) goes before the LLM (50)
        assert p.ok and p.evict == ["comfyui"]

    def test_evicts_more_when_needed(self):
        p = self.plan(20000)
        assert p.ok and p.evict == ["comfyui", "llm:8081"]

    def test_busy_and_pinned_never(self):
        p = self.plan(9000, pinned=frozenset({"comfyui"}))
        assert p.ok and p.evict == ["llm:8081"]
        p = self.plan(22000, pinned=frozenset({"comfyui"}))
        assert not p.ok and p.evict == [] and "ComfyUI" in p.reason and "pinned" in p.reason

    def test_nothing_evicted_if_it_cannot_work(self):
        p = self.plan(23000)        # even evicting every idle one leaves too little
        assert not p.ok and p.evict == [] and "CosyVoice" in p.reason

    def test_training_never_evicted(self):
        att = {("training:job1", 0): 20000}
        p = plan_room(0, 5000, [dict(GPUS[0], free=2000)], consumers(), att)
        assert not p.ok and "training" in p.reason.lower() or "sft" in p.reason


def test_gpu_view():
    att = {("llm:8081", 0): 9000, ("comfyui", 0): 10000, ("other", 1): 500}
    view = gpu_view(GPUS, consumers(), att, pinned=frozenset({"llm:8081"}))
    g0 = view[0]
    assert g0["index"] == 0 and g0["free_mb"] == 2000
    rows = {r["id"]: r for r in g0["consumers"]}
    assert rows["llm:8081"]["pinned"] and not rows["llm:8081"]["evictable"]
    assert rows["comfyui"]["evictable"] and rows["comfyui"]["mb"] == 10000
    assert view[1]["consumers"][0]["id"] == "other" and not view[1]["consumers"][0]["evictable"]


def test_llm_pins_follow_the_model_not_the_port():
    """A llama-server's port changes on every load; a pin on "the model" must
    survive that -- pinned by model name (pin_key), unloaded by port (id)."""
    c = Consumer(id="llm:41234", kind="llm", label="gemma.gguf", root_pids={1}, pin_key="model:gemma.gguf")
    att = {("llm:41234", 0): 20000}
    p = plan_room(0, 10000, [{"index": 0, "total": 24576, "free": 2000}], [c], att,
                  pinned=frozenset({"model:gemma.gguf"}))
    assert not p.ok and "pinned" in p.reason
    view = gpu_view([{"index": 0, "total": 24576, "free": 2000}], [c], att, pinned=frozenset({"model:gemma.gguf"}))
    assert view[0]["consumers"][0]["pinned"] and view[0]["consumers"][0]["pin_key"] == "model:gemma.gguf"


# --------------------------------------------------------------------------- reservations

from backend.cluster.consumers import Reservation, Reservations


class TestReservations:
    """User, 2026-10-16: the dispatcher answers yes/no fast (no waiting list; callers
    retry). A yes is a reservation that closes the gap between "room is free" and
    "the new process has allocated it" (found: a 27B model took >10 s to show up in
    nvidia-smi -- two loads in that window would overbook a GPU):
    - held for 5 s until the requester confirms with the PID it started;
    - then bound to that PID: only its not-yet-allocated part counts, so it shrinks
      as the process allocates; it ends when the PID is gone, or after 2 min at most;
    - others see free = measured free - outstanding reservations."""

    def book(self):
        return Reservations(hold_s=5, bound_max_s=120)

    def test_hold_expires_unconfirmed(self):
        r = self.book()
        res = r.add(gpu=0, mb=8000, purpose="llm:x", now=0)
        assert r.outstanding(0, now=4, alive=set(), measured={}) == 8000
        assert r.outstanding(0, now=6, alive=set(), measured={}) == 0

    def test_confirmed_bound_to_pid_shrinks_as_it_allocates(self):
        r = self.book()
        res = r.add(gpu=0, mb=8000, purpose="llm:x", now=0)
        assert r.confirm(res.id, pid=4242, now=1)
        assert r.outstanding(0, now=30, alive={4242}, measured={4242: 3000}) == 5000
        assert r.outstanding(0, now=30, alive={4242}, measured={4242: 9000}) == 0

    def test_ends_when_process_is_gone(self):
        r = self.book()
        res = r.add(gpu=0, mb=8000, purpose="llm:x", now=0)
        r.confirm(res.id, pid=4242, now=1)
        assert r.outstanding(0, now=10, alive=set(), measured={}) == 0
        assert r.get(res.id) is None

    def test_bound_max_age(self):
        r = self.book()
        res = r.add(gpu=0, mb=8000, purpose="llm:x", now=0)
        r.confirm(res.id, pid=1, now=1)
        assert r.outstanding(0, now=122, alive={1}, measured={}) == 0

    def test_confirm_after_hold_expired_fails(self):
        r = self.book()
        res = r.add(gpu=0, mb=8000, purpose="llm:x", now=0)
        assert not r.confirm(res.id, pid=1, now=6)

    def test_per_gpu_and_release(self):
        r = self.book()
        a = r.add(gpu=0, mb=1000, purpose="a", now=0)
        r.add(gpu=1, mb=2000, purpose="b", now=0)
        assert r.outstanding(1, now=1, alive=set(), measured={}) == 2000
        r.release(a.id)
        assert r.outstanding(0, now=1, alive=set(), measured={}) == 0


def test_plan_room_counts_outstanding_reservations():
    gpus = [{"index": 0, "total": 24576, "free": 12000}]
    # 12 GB measured free, but 8 GB already promised to a load that hasn't allocated yet
    assert not plan_room(0, 9000, gpus, [], {}, reserved_mb=8000).ok
    assert plan_room(0, 4000, gpus, [], {}, reserved_mb=8000).ok


# --------------------------------------------------------------------------- exclusive (Strata)

from backend.cluster.consumers import plan_exclusive


class TestExclusive:
    """Strata splits its layers over both GPUs and fills them: it needs them for
    itself (an exclusive reservation) instead of six separate guards. Starting it
    unloads idle, unpinned consumers like any other request; busy/pinned/training --
    or other programs holding more than a bare CUDA context (512 MB) -- block it."""
    gpus = [{"index": 0, "total": 24576, "free": 2000}]

    def test_evicts_every_idle_consumer(self):
        p = plan_exclusive(0, self.gpus, consumers(), {("llm:8081", 0): 9000, ("comfyui", 0): 10000})
        assert p.ok and p.evict == ["comfyui", "llm:8081"]

    def test_busy_blocks(self):
        p = plan_exclusive(0, self.gpus, consumers(), {("comfyui", 0): 10000, ("tts:cosyvoice", 0): 3000})
        assert not p.ok and p.evict == [] and "CosyVoice" in p.reason

    def test_bare_contexts_and_small_other_programs_are_fine(self):
        busy_ctx = [Consumer(id="stt", kind="stt", label="stt-serve", busy=True)]
        p = plan_exclusive(0, self.gpus, busy_ctx, {("stt", 0): 318, ("other", 0): 276})
        assert p.ok

    def test_big_other_program_blocks(self):
        p = plan_exclusive(0, self.gpus, [], {("other", 0): 3000})
        assert not p.ok and "other programs" in p.reason


class TestExclusiveReservation:
    def test_blocks_every_other_reservation_while_its_process_lives(self):
        r = Reservations(hold_s=5, bound_max_s=120)
        ex = r.add(gpu=0, mb=0, purpose="strata", now=0, exclusive=True)
        r.confirm(ex.id, pid=900, now=1)
        assert r.exclusive_holder(0, now=500, alive={900}) == "strata"     # no time limit
        assert r.exclusive_holder(1, now=500, alive={900}) is None
        assert r.exclusive_holder(0, now=501, alive=set()) is None         # process gone


def test_plan_room_refuses_with_exclusive_holder():
    gpus = [{"index": 0, "total": 24576, "free": 20000}]
    p = plan_room(0, 1000, gpus, [], {}, exclusive_holder="Strata")
    assert not p.ok and p.exclusive_holder == "Strata" and "Strata" in p.reason


# --- where an image service should run (2026-10-19: ComfyUI/MOSS/Iris were pinned to GPU 1 while GPU 0 was free)
from backend.cluster.consumers import choose_gpu


def test_stay_when_no_other_gpu_has_clearly_more_room():
    assert choose_gpu({0: 10000, 1: 9000}, need_mb=8000, current=1, movable=True) == (1, "stay")
    assert choose_gpu({0: 9000, 1: 20000}, need_mb=8000, current=1, movable=True) == (1, "stay")


def test_models_load_anyway_so_an_idle_service_goes_where_there_is_most_room():
    # 2026-10-20: ComfyUI sat idle on the 4060 Ti (15 GB free) and an edit loaded there although the 3090
    # was empty (23 GB) -- 71 s instead of ~43 s. Moving costs only the restart; the models load anyway.
    assert choose_gpu({0: 23213, 1: 15600}, need_mb=8500, current=1, movable=True) == (0, "move")


def test_a_busy_service_stays_where_it_fits():
    assert choose_gpu({0: 23213, 1: 15600}, need_mb=8500, current=1, movable=False) == (1, "stay")


def test_move_to_a_free_gpu_instead_of_unloading_others():
    assert choose_gpu({0: 20000, 1: 3000}, need_mb=8000, current=1, movable=True) == (0, "move")


def test_busy_service_doesnt_move():
    assert choose_gpu({0: 20000, 1: 3000}, need_mb=8000, current=1, movable=False) == (1, "evict")


def test_no_gpu_has_room_without_unloading():
    assert choose_gpu({0: 2000, 1: 3000}, need_mb=8000, current=1, movable=True) == (1, "evict")


def test_not_running_yet_starts_where_most_is_free():
    assert choose_gpu({0: 20000, 1: 12000}, need_mb=8000, current=None, movable=True) == (0, "start")
    assert choose_gpu({0: 1000, 1: 12000}, need_mb=8000, current=None, movable=True) == (1, "start")
    assert choose_gpu({0: 1000, 1: 2000}, need_mb=8000, current=None, movable=True) == (1, "start")   # most free, then evict


def test_moving_prefers_the_gpu_with_most_room():
    assert choose_gpu({0: 9000, 1: 1000, 2: 15000}, need_mb=8000, current=1, movable=True) == (2, "move")
