"""Where a model request goes when this machine can't serve it -- pure. Rules and
reasons: tests/unit/test_cluster_routing.py. Uses the cluster ledger (domain.py)
for room and ordering."""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

from backend.cluster.domain import AccelKind, Accelerator, ConsumerKind, Lease, Ledger, Node, place
from backend.response_processing import estimate_kv_overhead_mb

_KINDS = {k.value: k for k in AccelKind}


@dataclass(frozen=True)
class NodeSnapshot:
    id: str
    online: bool
    priority: int
    runtime_kinds: FrozenSet[str]          # accelerator kinds the installed runtime can use
    accelerators: List[Dict[str, Any]]     # agent's /v1/node accelerators
    models: Dict[str, int]                 # relative model path -> size MB
    instances: List[Dict[str, Any]]        # agent's instances


@dataclass(frozen=True)
class RoutePlan:
    action: str                            # reuse / start / copy / none
    node: Optional[str] = None
    instance: Optional[str] = None
    model_path: Optional[str] = None
    gpu: Optional[int] = None
    candidates: List[str] = field(default_factory=list)
    reason: str = ""
    mode: str = "chat"
    pooling: Optional[str] = None


def need_mb(size_mb: int, ctx_size: int) -> int:
    return int(size_mb + estimate_kv_overhead_mb(ctx_size))


def _ledger(snaps: Sequence[NodeSnapshot]):
    ledger = Ledger()
    free: Dict[str, int] = {}
    for s in snaps:
        for a in s.accelerators:
            kind = _KINDS.get(a.get("kind"))
            if kind is None or not a.get("total_mb"):
                continue
            budget = min(int(a.get("budget_mb") or a["total_mb"]), int(a["total_mb"]))
            acc = Accelerator(node=s.id, index=int(a["index"]), kind=kind, name=str(a.get("name", "")),
                              total_mb=int(a["total_mb"]), budget_mb=budget)
            ledger.add_accelerator(acc)
            free[acc.id] = int(a["free_mb"]) if a.get("free_mb") is not None else budget
        for i in s.instances:
            if i.get("state") not in ("loading", "ready"):
                continue
            accel = f"{s.id}/{i.get('gpu') or 0}"
            if ledger.accelerator(accel) is None:
                continue
            size = s.models.get(i.get("model"), 0)
            mb = need_mb(size, int(i.get("ctx_size") or 8192))
            try:
                # running instances are busy for placement purposes: never evicted here
                ledger.grant(Lease(id=f"{s.id}:{i['id']}", accel=accel, consumer=ConsumerKind.LLM,
                                   label=str(i.get("model")), mb=max(mb, 1), busy=True))
            except ValueError:
                pass
            # hardware "free" from nvidia-smi already accounts for it; for budget-only
            # (unified memory) accelerators the lease is what makes it count
    return ledger, free


def _same_kind(inst: Dict[str, Any], mode: str, pooling: Optional[str]) -> bool:
    """A chat server can't answer /v1/embeddings and vice versa; pooling only
    matters when the caller asks for one. Older agents report no mode = chat."""
    if (inst.get("mode") or "chat") != mode:
        return False
    return pooling is None or inst.get("pooling") == pooling


def plan_route(model_name: str, size_mb: int, ctx_size: int, snaps: Sequence[NodeSnapshot],
               mode: str = "chat", pooling: Optional[str] = None) -> RoutePlan:
    base = os.path.basename(model_name)
    usable = [s for s in snaps if s.online and s.runtime_kinds]
    for s in usable:
        for i in s.instances:
            if i.get("state") == "ready" and os.path.basename(str(i.get("model"))) == base \
                    and _same_kind(i, mode, pooling):
                return RoutePlan("reuse", node=s.id, instance=i["id"], model_path=i["model"],
                                 mode=mode, pooling=pooling)
    ledger, free = _ledger(usable)
    nodes = {s.id: Node(id=s.id, name=s.id, priority=s.priority, available=True) for s in usable}
    mb = need_mb(size_mb, ctx_size)
    with_file, copy_targets = [], []
    for s in usable:
        kinds = {_KINDS[k] for k in s.runtime_kinds if k in _KINDS}
        order = place(ledger, {s.id: nodes[s.id]}, mb, kinds, free)
        order = [a for a in order if not ledger.admit(a, mb, free.get(a, 0)).evict]
        if not order:
            continue
        rel = next((m for m in s.models if os.path.basename(m) == base), None)
        if rel:
            with_file.append((-s.priority, order[0], s.id, rel))
        else:
            copy_targets.append((-s.priority, s.id))
    if with_file:
        _, accel, node, rel = sorted(with_file)[0]
        gpu = ledger.accelerator(accel).index
        kind = ledger.accelerator(accel).kind
        return RoutePlan("start", node=node, model_path=rel,
                         gpu=gpu if kind != AccelKind.CPU else None, mode=mode, pooling=pooling)
    if copy_targets:
        return RoutePlan("copy", candidates=[n for _, n in sorted(copy_targets)])
    return RoutePlan("none", reason=f"no node has room for {base} (~{mb} MB)" if usable else "no node available")
