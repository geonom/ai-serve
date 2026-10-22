"""Cluster resource model (backend/cluster/domain.py) -- pure rules, no I/O.

Architecture review 2026-10-16: every subsystem managed VRAM on its own (tts-serve
evicts only its engines, ComfyUI frees after 180 s, llama-server loads refuse with
400 instead of evicting an idle model, concurrent loads can overbook a GPU), and
other devices (Macs, Pis, other PCs, phones) should become usable as optional
resources. This model is node-aware from the start; this machine is just node 1.

Vocabulary:
- Node: a device (this machine or one running the node agent). Has a priority and
  is either available or not (offline, or its conditions -- e.g. "only on power" --
  aren't met right now). An unavailable node's accelerators are never chosen.
- Accelerator: one compute unit of a node (CUDA/ROCm/Vulkan GPU, Apple Metal,
  NPU, CPU) with a memory BUDGET (what the owner lets us use -- on a laptop with
  unified memory far less than its total).
- Lease: memory reserved on one accelerator by one consumer (llm, image, tts, stt,
  training, exclusive). Busy and pinned leases are never evicted; an exclusive
  lease (Strata) owns the whole accelerator.
- Admission: per accelerator, room = budget minus leases, also capped by what the
  hardware reports free (unmanaged processes count). If short, idle unpinned
  leases are evicted least-recently-used first -- whatever their consumer kind.
- Placement: among available nodes' accelerators of a kind the runtime supports,
  prefer one that fits without evicting, then higher node priority, then more room."""
import pytest

from backend.cluster.domain import (
    AccelKind, Accelerator, Admission, ConsumerKind, Ledger, Lease, Node, place,
)


def gpu(node="my-pc", index=0, budget=24000, kind=AccelKind.CUDA, total=None):
    return Accelerator(node=node, index=index, kind=kind, name="RTX", total_mb=total or budget, budget_mb=budget)


def lease(lid, accel, mb, last_used=0.0, busy=False, pinned=False, kind=ConsumerKind.LLM, exclusive=False):
    return Lease(id=lid, accel=accel.id, consumer=kind, label=lid, mb=mb, last_used=last_used,
                 busy=busy, pinned=pinned, exclusive=exclusive)


class TestModel:
    def test_accel_id(self):
        assert gpu(index=1).id == "my-pc/1"

    def test_budget_cannot_exceed_total(self):
        with pytest.raises(ValueError):
            Accelerator(node="mac", index=0, kind=AccelKind.METAL, name="M2", total_mb=16000, budget_mb=17000)

    def test_lease_size_positive(self):
        a = gpu()
        with pytest.raises(ValueError):
            lease("x", a, 0)


class TestAdmission:
    def setup_method(self):
        self.a = gpu(budget=24000)
        self.ledger = Ledger([self.a])

    def test_fits(self):
        adm = self.ledger.admit(self.a.id, 10000, hardware_free_mb=24000)
        assert adm.ok and adm.evict == []

    def test_evicts_lru_idle_first_any_kind(self):
        self.ledger.grant(lease("old-image", self.a, 9000, last_used=10, kind=ConsumerKind.IMAGE))
        self.ledger.grant(lease("newer-llm", self.a, 9000, last_used=50))
        self.ledger.grant(lease("busy-tts", self.a, 4000, last_used=1, busy=True, kind=ConsumerKind.TTS))
        adm = self.ledger.admit(self.a.id, 8000, hardware_free_mb=2000)
        assert adm.ok and adm.evict == ["old-image"]

    def test_evicts_several_until_enough(self):
        self.ledger.grant(lease("a", self.a, 6000, last_used=1))
        self.ledger.grant(lease("b", self.a, 6000, last_used=2))
        self.ledger.grant(lease("c", self.a, 6000, last_used=3))
        adm = self.ledger.admit(self.a.id, 16000, hardware_free_mb=6000)
        assert adm.ok and adm.evict == ["a", "b"]

    def test_never_evicts_busy_or_pinned(self):
        self.ledger.grant(lease("pinned", self.a, 12000, pinned=True))
        self.ledger.grant(lease("busy", self.a, 10000, busy=True))
        adm = self.ledger.admit(self.a.id, 8000, hardware_free_mb=2000)
        assert not adm.ok and "pinned" in adm.reason and "busy" in adm.reason

    def test_hardware_free_caps_budget(self):
        # an unmanaged process holds memory: the budget alone would say "fits"
        adm = self.ledger.admit(self.a.id, 10000, hardware_free_mb=5000)
        assert not adm.ok

    def test_budget_caps_hardware(self):
        mac = Accelerator(node="mac", index=0, kind=AccelKind.METAL, name="M2", total_mb=16000, budget_mb=6000)
        ledger = Ledger([mac])
        assert not ledger.admit(mac.id, 8000, hardware_free_mb=12000).ok
        assert ledger.admit(mac.id, 5000, hardware_free_mb=12000).ok

    def test_exclusive_lease_blocks_everything(self):
        self.ledger.grant(lease("strata", self.a, 1, exclusive=True, kind=ConsumerKind.EXCLUSIVE))
        adm = self.ledger.admit(self.a.id, 100, hardware_free_mb=24000)
        assert not adm.ok and "exclusive" in adm.reason

    def test_exclusive_request_evicts_all_idle(self):
        self.ledger.grant(lease("a", self.a, 6000))
        self.ledger.grant(lease("b", self.a, 6000))
        adm = self.ledger.admit(self.a.id, 1, hardware_free_mb=12000, exclusive=True)
        assert adm.ok and sorted(adm.evict) == ["a", "b"]

    def test_exclusive_request_refused_while_busy(self):
        self.ledger.grant(lease("a", self.a, 6000, busy=True))
        assert not self.ledger.admit(self.a.id, 1, hardware_free_mb=12000, exclusive=True).ok

    def test_unknown_accelerator(self):
        assert not self.ledger.admit("nope/0", 1, hardware_free_mb=1).ok

    def test_grant_release_touch(self):
        self.ledger.grant(lease("x", self.a, 1000, last_used=1))
        self.ledger.touch("x", now=99)
        assert self.ledger.get("x").last_used == 99
        assert self.ledger.used_mb(self.a.id) == 1000
        self.ledger.release("x")
        assert self.ledger.used_mb(self.a.id) == 0

    def test_grant_overbooking_refused(self):
        self.ledger.grant(lease("x", self.a, 20000))
        with pytest.raises(ValueError):
            self.ledger.grant(lease("y", self.a, 8000))


class TestPlacement:
    def nodes(self, **avail):
        return {
            "my-pc": Node(id="my-pc", name="my-pc", priority=10, available=avail.get("pop", True)),
            "mac": Node(id="mac", name="MacBook", priority=5, available=avail.get("mac", True)),
        }

    def test_prefers_fit_without_eviction(self):
        a, b = gpu(index=0, budget=24000), gpu(index=1, budget=16000)
        ledger = Ledger([a, b])
        ledger.grant(lease("big", a, 20000))
        order = place(ledger, self.nodes(), 8000, {AccelKind.CUDA}, hardware_free={a.id: 4000, b.id: 16000})
        assert order[0] == b.id

    def test_unavailable_nodes_excluded(self):
        a = gpu()
        m = Accelerator(node="mac", index=0, kind=AccelKind.METAL, name="M2", total_mb=24000, budget_mb=16000)
        ledger = Ledger([a, m])
        order = place(ledger, self.nodes(pop=False), 8000, {AccelKind.CUDA, AccelKind.METAL},
                      hardware_free={a.id: 24000, m.id: 16000})
        assert order == [m.id]

    def test_runtime_kinds_respected(self):
        a = gpu()
        m = Accelerator(node="mac", index=0, kind=AccelKind.METAL, name="M2", total_mb=24000, budget_mb=16000)
        ledger = Ledger([a, m])
        order = place(ledger, self.nodes(), 8000, {AccelKind.METAL}, hardware_free={a.id: 24000, m.id: 16000})
        assert order == [m.id]

    def test_priority_breaks_ties(self):
        a = gpu(budget=16000)
        m = Accelerator(node="mac", index=0, kind=AccelKind.METAL, name="M2", total_mb=24000, budget_mb=16000)
        ledger = Ledger([a, m])
        order = place(ledger, self.nodes(), 8000, {AccelKind.CUDA, AccelKind.METAL},
                      hardware_free={a.id: 16000, m.id: 16000})
        assert order[0] == a.id

    def test_nothing_fits(self):
        a = gpu(budget=8000)
        ledger = Ledger([a])
        ledger.grant(lease("busy", a, 6000, busy=True))
        assert place(ledger, self.nodes(), 4000, {AccelKind.CUDA}, hardware_free={a.id: 2000}) == []
