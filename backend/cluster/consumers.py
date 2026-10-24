"""Local GPU management across all consumers -- pure. Reasons and rules:
tests/unit/test_gpu_consumers.py. Built on the cluster ledger (domain.py): one
accelerator per local GPU, one lease per (consumer, GPU) sized by MEASURED VRAM."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from backend.cluster.domain import AccelKind, Accelerator, ConsumerKind, Lease, Ledger

OTHER = "other"
_KIND = {"llm": ConsumerKind.LLM, "image": ConsumerKind.IMAGE, "tts": ConsumerKind.TTS, "stt": ConsumerKind.STT,
         "training": ConsumerKind.TRAINING, "exclusive": ConsumerKind.EXCLUSIVE}


@dataclass(frozen=True)
class Consumer:
    id: str                       # llm:<port> / comfyui / <plugin>:<worker> / training:<job>
    kind: str                     # llm / image / tts / stt / training / exclusive (plugins: their capability)
    label: str
    root_pids: Set[int] = field(default_factory=set)
    last_used: float = 0.0
    busy: bool = False
    evictable_kind: bool = True   # training / Strata: never evicted
    pin_key: str = ""             # what a pin refers to (default: id); LLMs: "model:<file>"
    plugin: str = ""              # the plugin it belongs to (backend/plugins/registry.py) -> freed through it

    @property
    def pin(self) -> str:
        return self.pin_key or self.id


@dataclass(frozen=True)
class RoomPlan:
    ok: bool
    evict: List[str]
    reason: str = ""
    exclusive_holder: Optional[str] = None   # refused because this holds the GPU exclusively


UNIFIED_MIN_MB = 32   # below this a process is just an idle runtime, not a model


def unified_procs(consumers: Sequence[Consumer], children: Callable[[int], Optional[Iterable[int]]],
                  size_mb: Callable[[int], Optional[int]]) -> List[Tuple[int, int, int]]:
    """Unified memory (Apple Silicon): there is no per-process GPU list like nvidia-smi's, so measure the processes of
    every known consumer and their children instead -> (pid, 0, MiB), the same shape attribute_vram takes. Other
    programs aren't measured (on unified memory everything uses RAM; only the GPU limit's remainder counts as free)."""
    out, seen = [], set()
    stack = [pid for c in consumers for pid in sorted(c.root_pids)]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        mb = size_mb(pid)
        if mb is not None and mb >= UNIFIED_MIN_MB:
            out.append((pid, 0, int(mb)))
        stack.extend(children(pid) or [])
    return out


def attribute_vram(procs: Iterable[Tuple[int, int, int]], parents: Mapping[int, int],
                   consumers: Sequence[Consumer]) -> Dict[Tuple[str, int], int]:
    """procs: (pid, gpu index, MiB) from nvidia-smi. Each process belongs to the
    consumer whose root pid is on its parent chain; otherwise to "other"."""
    roots = {pid: c.id for c in consumers for pid in c.root_pids}
    out: Dict[Tuple[str, int], int] = {}
    for pid, gpu, mb in procs:
        owner, cur, seen = OTHER, pid, set()
        while cur and cur not in seen:
            if cur in roots:
                owner = roots[cur]
                break
            seen.add(cur)
            cur = parents.get(cur)
        out[(owner, gpu)] = out.get((owner, gpu), 0) + int(mb)
    return out


def _ledger(gpu: Mapping[str, Any], consumers: Sequence[Consumer], att: Mapping[Tuple[str, int], int],
            pinned: FrozenSet[str]) -> Tuple[Ledger, str]:
    acc = Accelerator(node="local", index=int(gpu["index"]), kind=AccelKind.CUDA, name=str(gpu.get("name", "")),
                      total_mb=int(gpu["total"]), budget_mb=int(gpu["total"]))
    ledger = Ledger([acc])
    for c in consumers:
        mb = att.get((c.id, acc.index), 0)
        if mb <= 0:
            continue
        ledger.record(Lease(
            id=c.id, accel=acc.id, consumer=_KIND.get(c.kind, ConsumerKind.LLM), label=c.label, mb=mb,
            last_used=c.last_used, busy=c.busy or not c.evictable_kind, pinned=c.pin in pinned,
            exclusive=c.kind == "exclusive"))
    return ledger, acc.id


def plan_room(gpu_index: int, need_mb: int, gpus: Sequence[Mapping[str, Any]], consumers: Sequence[Consumer],
              att: Mapping[Tuple[str, int], int], pinned: FrozenSet[str] = frozenset(),
              reserved_mb: int = 0, exclusive_holder: Optional[str] = None) -> RoomPlan:
    """reserved_mb: memory promised to loads that haven't allocated it yet;
    exclusive_holder: who holds this GPU exclusively (Strata), if anyone."""
    if exclusive_holder:
        return RoomPlan(False, [], f"GPU {gpu_index} is reserved exclusively for {exclusive_holder}",
                        exclusive_holder=exclusive_holder)
    gpu = next((g for g in gpus if int(g["index"]) == gpu_index), None)
    if gpu is None:
        return RoomPlan(False, [], f"no GPU {gpu_index}")
    ledger, aid = _ledger(gpu, consumers, att, pinned)
    free = int(gpu.get("free") or 0) - int(reserved_mb)
    if free >= need_mb:
        return RoomPlan(True, [])
    held = ledger.leases(aid)
    if any(l.exclusive for l in held):
        owner = next(l.label for l in held if l.exclusive)
        return RoomPlan(False, [], f"GPU {gpu_index} is held exclusively by {owner}", exclusive_holder=owner)
    evict, freed = [], 0
    for l in sorted((l for l in held if l.evictable), key=lambda l: l.last_used):
        evict.append(l.id)
        freed += l.mb
        if free + freed >= need_mb:
            return RoomPlan(True, evict)
    blockers = [l for l in held if not l.evictable]
    why = ", ".join(f"{l.label} ({'pinned' if l.pinned else 'busy'}, {l.mb} MB)" for l in blockers)
    other = att.get((OTHER, gpu_index), 0)
    return RoomPlan(False, [], f"GPU {gpu_index}: needs {need_mb} MB, {free} MB free"
                    + (f" ({reserved_mb} MB promised to loads in progress)" if reserved_mb else "")
                    + (f", even after unloading every idle user only {free + freed} MB" if evict else "")
                    + (f"; held by {why}" if why else "")
                    + (f"; other programs use {other} MB" if other else ""))


CONTEXT_TOLERANCE_MB = 512   # a bare CUDA context; anything more blocks an exclusive start


def plan_exclusive(gpu_index: int, gpus: Sequence[Mapping[str, Any]], consumers: Sequence[Consumer],
                   att: Mapping[Tuple[str, int], int], pinned: FrozenSet[str] = frozenset()) -> RoomPlan:
    """Room for a consumer that needs the whole GPU (Strata): unload every idle,
    unpinned consumer on it; anything else holding more than a bare CUDA context
    blocks -- then nothing is unloaded."""
    gpu = next((g for g in gpus if int(g["index"]) == gpu_index), None)
    if gpu is None:
        return RoomPlan(False, [], f"no GPU {gpu_index}")
    ledger, aid = _ledger(gpu, consumers, att, pinned)
    held = ledger.leases(aid)
    evict = [l.id for l in sorted((l for l in held if l.evictable), key=lambda l: l.last_used)]
    blockers = [l for l in held if not l.evictable and l.mb > CONTEXT_TOLERANCE_MB]
    other = att.get((OTHER, gpu_index), 0)
    if blockers or other > CONTEXT_TOLERANCE_MB:
        why = ", ".join(f"{l.label} ({'pinned' if l.pinned else 'busy'}, {l.mb} MB)" for l in blockers)
        if other > CONTEXT_TOLERANCE_MB:
            why = (why + ", " if why else "") + f"other programs ({other} MB)"
        return RoomPlan(False, [], f"GPU {gpu_index} can't be freed: {why}")
    return RoomPlan(True, evict)


def gpu_view(gpus: Sequence[Mapping[str, Any]], consumers: Sequence[Consumer],
             att: Mapping[Tuple[str, int], int], pinned: FrozenSet[str] = frozenset()) -> List[Dict[str, Any]]:
    by_id = {c.id: c for c in consumers}
    out = []
    for g in gpus:
        idx = int(g["index"])
        rows = []
        for (cid, gi), mb in sorted(att.items(), key=lambda kv: -kv[1]):
            if gi != idx or mb <= 0:
                continue
            c = by_id.get(cid)
            if c is None:
                rows.append({"id": OTHER, "kind": OTHER, "label": "other programs", "mb": mb, "busy": False,
                             "pinned": False, "evictable": False, "last_used": None})
                continue
            rows.append({"id": c.id, "pin_key": c.pin, "kind": c.kind, "label": c.label, "mb": mb,
                         "busy": c.busy, "pinned": c.pin in pinned, "last_used": c.last_used or None,
                         "evictable": c.evictable_kind and not c.busy and c.pin not in pinned,
                         "plugin": c.plugin or None})
        view = {"index": idx, "name": g.get("name"), "total_mb": int(g["total"]),
                "free_mb": int(g.get("free") or 0), "consumers": rows}
        if g.get("gpu_limit_mb"):          # unified memory (Apple): how much of the RAM the GPU may use
            view["gpu_limit_mb"] = int(g["gpu_limit_mb"])
        out.append(view)
    return out


# --------------------------------------------------------------------------- reservations

@dataclass
class Reservation:
    id: str
    gpu: int
    mb: int
    purpose: str
    created: float
    pid: Optional[int] = None
    confirmed_at: Optional[float] = None
    exclusive: bool = False   # the whole GPU, for as long as the process lives (Strata)
    baseline_mb: int = 0      # what the process already held when it confirmed (only growth after that counts)
    peak_mb: int = 0          # the most it grew by since -- handed to NeedBook when the reservation ends


class Reservations:
    """Memory promised to a load that hasn't allocated it yet. Held hold_s until the
    requester confirms with the started PID; then only the not-yet-allocated part
    counts, until the PID is gone (or bound_max_s at most). Rules and reasons:
    tests/unit/test_gpu_consumers.py::TestReservations."""

    def __init__(self, hold_s: float = 5.0, bound_max_s: float = 120.0):
        self.hold_s = hold_s
        self.bound_max_s = bound_max_s
        self._items: Dict[str, Reservation] = {}
        self._n = 0
        self._finished: List[Reservation] = []   # ended bound reservations, for learning (drain_finished)

    def add(self, gpu: int, mb: int, purpose: str, now: float, exclusive: bool = False) -> Reservation:
        self._n += 1
        r = Reservation(id=f"r{self._n}-{int(now * 1000) % 100000}", gpu=gpu, mb=int(mb), purpose=purpose,
                        created=now, exclusive=exclusive)
        self._items[r.id] = r
        return r

    def get(self, rid: str) -> Optional[Reservation]:
        return self._items.get(rid)

    def confirm(self, rid: str, pid: int, now: float, baseline_mb: int = 0) -> bool:
        r = self._items.get(rid)
        if r is None or r.pid is not None or now - r.created > self.hold_s:
            self._items.pop(rid, None)
            return False
        r.pid, r.confirmed_at, r.baseline_mb = int(pid), now, max(0, int(baseline_mb))
        return True

    def _end(self, rid: str) -> None:
        r = self._items.pop(rid, None)
        if r is not None and r.pid is not None and not r.exclusive:
            self._finished.append(r)

    def release(self, rid: str) -> None:
        self._end(rid)

    def drain_finished(self) -> List[Reservation]:
        out, self._finished = self._finished, []
        return out

    def prune(self, now: float, alive: Set[int]) -> None:
        for rid, r in list(self._items.items()):
            if r.pid is None and now - r.created > self.hold_s:
                del self._items[rid]
            elif r.pid is not None and (r.pid not in alive or
                                        (not r.exclusive and now - (r.confirmed_at or now) > self.bound_max_s)):
                self._end(rid)

    def observe(self, measured: Mapping[int, int]) -> None:
        """Remember how far each bound process has grown past its baseline."""
        for r in self._items.values():
            if r.pid is not None and r.pid in measured:
                r.peak_mb = max(r.peak_mb, int(measured[r.pid]) - r.baseline_mb)

    def outstanding(self, gpu: int, now: float, alive: Set[int], measured: Mapping[int, int]) -> int:
        """measured: MiB allocated per PID (the reserving process's own, incl. what it held before confirming)."""
        self.observe(measured)
        self.prune(now, alive)
        total = 0
        for r in self._items.values():
            if r.gpu == gpu and not r.exclusive:
                grown = max(0, int(measured.get(r.pid, 0)) - r.baseline_mb) if r.pid is not None else 0
                total += max(0, r.mb - grown)
        return total

    def exclusive_holder(self, gpu: int, now: float, alive: Set[int]) -> Optional[str]:
        self.prune(now, alive)
        return next((r.purpose for r in self._items.values() if r.gpu == gpu and r.exclusive), None)

    def all(self) -> List[Reservation]:
        return list(self._items.values())


class NeedBook:
    """What each purpose ("stt:whisperx:medium", "tts:cosyvoice", "image:Z-Image Turbo" ...) really needed: the peaks
    of its finished reservations (Reservation.peak_mb). A load is reserved max(declared, largest of the last
    SAMPLES peaks * 1.1) -- a declared need that is too low (found 2026-10-24: Whisper medium announced 1500 MB, used
    2690, ran out of memory) corrects itself after one run, and still follows a model that got smaller."""
    SAMPLES = 5
    HEADROOM_PCT = 10

    def __init__(self, samples: Optional[Dict[str, List[int]]] = None):
        self._samples: Dict[str, List[int]] = samples or {}

    def observe(self, purpose: str, mb: int) -> None:
        if not purpose or not isinstance(mb, int) or mb <= 0:
            return
        self._samples[purpose] = (self._samples.get(purpose, []) + [mb])[-self.SAMPLES:]

    def need(self, purpose: str, declared: int) -> int:
        seen = self._samples.get(purpose)
        if not seen:
            return declared
        peak = max(seen)
        return max(declared, -(-peak * (100 + self.HEADROOM_PCT) // 100))   # ceil, in integers

    def to_dict(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self._samples.items() if v}

    @classmethod
    def from_dict(cls, data: Any) -> "NeedBook":
        out: Dict[str, List[int]] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and k and isinstance(v, list):
                    good = [x for x in v if isinstance(x, int) and not isinstance(x, bool) and x > 0]
                    if good:
                        out[k] = good[-cls.SAMPLES:]
        return cls(out)


MOVE_MARGIN_MB = 2048   # less difference than this isn't worth a restart


def choose_gpu(free_mb: Mapping[int, int], need_mb: int, current: Optional[int], movable: bool) -> Tuple[int, str]:
    """Where a restartable GPU service (ComfyUI, MOSS's ComfyUI, sd-server) should serve the next job.

    `free_mb` = hardware-free minus promised-but-unallocated memory per GPU. Returns (gpu, mode):
    "stay" -- the current GPU has room as it is and none has clearly more; "move" -- the service is idle and
    another GPU has room and clearly more of it (or the own one is short): restart it there (models load
    anyway when this is asked) instead of crowding or unloading others; "evict" -- no GPU has room
    as it is (or the service is busy): make room on the current one; "start" -- not running yet: the GPU
    with the most room (room is then made there if needed). Found 2026-10-19: all three image services
    were pinned to GPU 1 and pushed each other out while GPU 0 was free."""
    if current is None:
        return max(free_mb, key=lambda g: (free_mb[g], -g)), "start"
    # Callers ask only when models have to be loaded anyway, so moving an idle service costs just its restart:
    # go where there is clearly more room (2026-10-20: an edit loaded on the 4060 Ti with the 3090 empty).
    best = max(free_mb, key=lambda g: (free_mb[g], -g))
    if (movable and best != current and free_mb[best] >= need_mb
            and free_mb[best] - free_mb.get(current, 0) >= MOVE_MARGIN_MB):
        return best, "move"
    if free_mb.get(current, 0) >= need_mb:
        return current, "stay"
    if movable:
        roomy = [g for g, f in free_mb.items() if g != current and f >= need_mb]
        if roomy:
            return max(roomy, key=lambda g: (free_mb[g], -g)), "move"
    return current, "evict"
