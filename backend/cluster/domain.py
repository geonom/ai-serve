"""Cluster resource model -- pure rules, no I/O. Vocabulary and the reasons behind
the rules: tests/unit/test_cluster_domain.py's docstring. In short:

- Node: a device; `available` folds in "online" and its owner-set conditions.
- Accelerator: one compute unit of a node, with a memory budget <= its total.
- Lease: memory one consumer holds on one accelerator. Busy/pinned leases are never
  evicted; an exclusive lease owns the whole accelerator.
- Ledger.admit(): room = min(budget - leases, hardware-reported free); if short,
  evict idle unpinned leases least-recently-used first, whatever their kind.
- place(): available nodes, supported kinds; fits-without-eviction first, then
  node priority, then most room."""
import enum
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Set


class AccelKind(enum.Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    METAL = "metal"
    NPU = "npu"
    CPU = "cpu"


class ConsumerKind(enum.Enum):
    LLM = "llm"
    IMAGE = "image"
    TTS = "tts"
    STT = "stt"
    TRAINING = "training"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    priority: int = 0
    available: bool = True


@dataclass(frozen=True)
class Accelerator:
    node: str
    index: int
    kind: AccelKind
    name: str
    total_mb: int
    budget_mb: int

    def __post_init__(self):
        if self.total_mb <= 0 or self.budget_mb < 0:
            raise ValueError("memory sizes must be positive")
        if self.budget_mb > self.total_mb:
            raise ValueError(f"budget {self.budget_mb} MB exceeds total {self.total_mb} MB")

    @property
    def id(self) -> str:
        return f"{self.node}/{self.index}"


@dataclass(frozen=True)
class Lease:
    id: str
    accel: str
    consumer: ConsumerKind
    label: str
    mb: int
    last_used: float = 0.0
    busy: bool = False
    pinned: bool = False
    exclusive: bool = False

    def __post_init__(self):
        if self.mb <= 0:
            raise ValueError("a lease needs a positive size")

    @property
    def evictable(self) -> bool:
        return not (self.busy or self.pinned)


@dataclass(frozen=True)
class Admission:
    ok: bool
    evict: List[str] = field(default_factory=list)
    reason: str = ""


class Ledger:
    def __init__(self, accelerators: Iterable[Accelerator] = ()):
        self._accels: Dict[str, Accelerator] = {}
        self._leases: Dict[str, Lease] = {}
        for a in accelerators:
            self.add_accelerator(a)

    # --- accelerators
    def add_accelerator(self, a: Accelerator) -> None:
        self._accels[a.id] = a

    def remove_node(self, node: str) -> List[Lease]:
        """A node left: its accelerators and leases are gone (returned for cleanup)."""
        gone = [l for l in self._leases.values() if self._accels.get(l.accel) and self._accels[l.accel].node == node]
        for l in gone:
            del self._leases[l.id]
        for aid in [aid for aid, a in self._accels.items() if a.node == node]:
            del self._accels[aid]
        return gone

    def accelerators(self) -> List[Accelerator]:
        return list(self._accels.values())

    def accelerator(self, accel_id: str) -> Optional[Accelerator]:
        return self._accels.get(accel_id)

    # --- leases
    def leases(self, accel_id: Optional[str] = None) -> List[Lease]:
        return [l for l in self._leases.values() if accel_id is None or l.accel == accel_id]

    def get(self, lease_id: str) -> Optional[Lease]:
        return self._leases.get(lease_id)

    def used_mb(self, accel_id: str) -> int:
        return sum(l.mb for l in self.leases(accel_id))

    def grant(self, lease: Lease) -> None:
        a = self._accels.get(lease.accel)
        if a is None:
            raise ValueError(f"unknown accelerator {lease.accel}")
        others = [l for l in self.leases(lease.accel) if l.id != lease.id]
        if any(l.exclusive for l in others) or (lease.exclusive and others):
            raise ValueError(f"{lease.accel} is held exclusively")
        if not lease.exclusive and sum(l.mb for l in others) + lease.mb > a.budget_mb:
            raise ValueError(f"lease {lease.id} would overbook {lease.accel}")
        self._leases[lease.id] = lease

    def record(self, lease: Lease) -> None:
        """Measured usage (e.g. nvidia-smi per process): recorded as-is, without the
        budget check grant() does -- reality may already exceed a budget."""
        if lease.accel not in self._accels:
            raise ValueError(f"unknown accelerator {lease.accel}")
        self._leases[lease.id] = lease

    def release(self, lease_id: str) -> Optional[Lease]:
        return self._leases.pop(lease_id, None)

    def touch(self, lease_id: str, now: float) -> None:
        if lease_id in self._leases:
            self._leases[lease_id] = replace(self._leases[lease_id], last_used=now)

    def set_busy(self, lease_id: str, busy: bool) -> None:
        if lease_id in self._leases:
            self._leases[lease_id] = replace(self._leases[lease_id], busy=busy)

    def set_pinned(self, lease_id: str, pinned: bool) -> None:
        if lease_id in self._leases:
            self._leases[lease_id] = replace(self._leases[lease_id], pinned=pinned)

    # --- rules
    def room_mb(self, accel_id: str, hardware_free_mb: int) -> int:
        a = self._accels[accel_id]
        return max(0, min(a.budget_mb - self.used_mb(accel_id), hardware_free_mb))

    def admit(self, accel_id: str, mb: int, hardware_free_mb: int, exclusive: bool = False) -> Admission:
        a = self._accels.get(accel_id)
        if a is None:
            return Admission(False, reason=f"unknown accelerator {accel_id}")
        held = self.leases(accel_id)
        if any(l.exclusive for l in held):
            owner = next(l.label for l in held if l.exclusive)
            return Admission(False, reason=f"{accel_id} is held exclusively by {owner}")
        blockers = [l for l in held if not l.evictable]
        if exclusive:
            if blockers:
                return Admission(False, reason=self._blocked(accel_id, blockers))
            return Admission(True, evict=[l.id for l in held])
        if mb > a.budget_mb:
            return Admission(False, reason=f"needs {mb} MB, {accel_id} budget is {a.budget_mb} MB")
        room = self.room_mb(accel_id, hardware_free_mb)
        if room >= mb:
            return Admission(True)
        evict: List[str] = []
        freed = 0
        hw_free = hardware_free_mb
        for l in sorted((l for l in held if l.evictable), key=lambda l: l.last_used):
            evict.append(l.id)
            freed += l.mb
            hw_free += l.mb
            if min(a.budget_mb - self.used_mb(accel_id) + freed, hw_free) >= mb:
                return Admission(True, evict=evict)
        return Admission(False, reason=f"needs {mb} MB on {accel_id}, only {room} MB free"
                         + (f"; {self._blocked(accel_id, blockers)}" if blockers else
                            " (memory held outside this manager)"))

    @staticmethod
    def _blocked(accel_id: str, blockers: List[Lease]) -> str:
        parts = [f"{l.label} ({'pinned' if l.pinned else 'busy'}, {l.mb} MB)" for l in blockers]
        return f"{accel_id} is held by " + ", ".join(parts)


def place(ledger: Ledger, nodes: Mapping[str, Node], mb: int, kinds: Set[AccelKind],
          hardware_free: Mapping[str, int]) -> List[str]:
    """Accelerator ids that can take `mb`, best first."""
    candidates = []
    for a in ledger.accelerators():
        node = nodes.get(a.node)
        if node is None or not node.available or a.kind not in kinds:
            continue
        free = hardware_free.get(a.id, 0)
        adm = ledger.admit(a.id, mb, free)
        if not adm.ok:
            continue
        candidates.append((bool(adm.evict), -node.priority, -ledger.room_mb(a.id, free), a.id))
    return [c[-1] for c in sorted(candidates)]
