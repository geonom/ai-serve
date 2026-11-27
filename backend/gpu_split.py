"""One model spread over several GPUs (2026-11-27, owner: "ein großes Modell, das auf beiden Grakas läuft" -- only
when the caller asks: `gpu_id: "all"`; never on its own, since it takes room on every card, also where the owner
plays). Pure rules; the load paths (model_manager._load_locked, /api/start) use them. Tests: tests/unit/test_gpu_split.py.

- What a card can give (`capacities`): its measured free memory plus what idle, unpinned models hold there, minus
  open reservations and SAFETY_MB (not its size minus the users: ~1.2 GB of the 4060 Ti belongs to nobody nvidia-smi
  names, measured 2026-11-27). Idle models count as room: the GPU manager unloads them when
  the share is reserved.
- How the need is shared (`shares`): in proportion to what each card can give, plus OVERHEAD_MB per card (each holds
  its own compute buffers); a card that can give less than MIN_SHARE_MB is left out; at least two cards, else None.
- llama-server (`flags`): --split-mode layer (whole layers per card: only activations cross the PCIe bus, the right
  mode for unlike cards without NVLink) with --tensor-split in CUDA_VISIBLE_DEVICES' order, the card giving most
  first and leading (--main-gpu 0: output layer, intermediate results).
Each share is reserved with the GPU manager on its card and bound to the one process (it measures per process and
card), so the usual rules hold on every card."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from backend import gpu_ids

SAFETY_MB = 512
OVERHEAD_MB = 1024
MIN_SHARE_MB = 2048


def capacities(view: List[Dict[str, Any]]) -> Dict[int, int]:
    """GPU number -> MB a model may get there (from the GPU manager's view())."""
    out: Dict[int, int] = {}
    for g in view:
        may_go = sum(int(c.get("mb") or 0) for c in g.get("consumers", [])
                     if c.get("evictable") and not c.get("busy") and not c.get("pinned"))
        if g.get("free_mb") is not None:   # measured: part of a card belongs to nobody nvidia-smi names
            room = int(g["free_mb"]) + may_go
        else:
            held = sum(int(c.get("mb") or 0) for c in g.get("consumers", [])) - may_go
            room = int(g["total_mb"]) - held
        out[int(g["index"])] = max(0, room - int(g.get("reserved_mb") or 0) - SAFETY_MB)
    return out


def shares(need_mb: int, caps: Mapping[int, int]) -> Optional[Dict[int, int]]:
    """GPU -> MB to reserve there (need shared by room, plus each card's overhead); None if it can't fit."""
    usable = {g: c - OVERHEAD_MB for g, c in caps.items() if c - OVERHEAD_MB >= MIN_SHARE_MB}
    if len(usable) < 2 or sum(usable.values()) < need_mb:
        return None
    total = sum(usable.values())
    out = {g: round(need_mb * c / total) for g, c in usable.items()}
    out[max(usable, key=usable.get)] += need_mb - sum(out.values())     # rounding goes to the biggest
    return {g: mb + OVERHEAD_MB for g, mb in out.items()}


def order(share: Mapping[int, int]) -> List[int]:
    """CUDA_VISIBLE_DEVICES order: the card giving most first (it leads)."""
    return sorted(share, key=lambda g: (-share[g], g))


def tensor_split(share: Mapping[int, int], devices: List[int]) -> str:
    weights = [max(0, share[g] - OVERHEAD_MB) for g in devices]
    total = sum(weights) or 1
    return ",".join(f"{w / total:.3f}" for w in weights)


def flags(share: Mapping[int, int], devices: List[int]) -> List[str]:
    return ["--split-mode", "layer", "--tensor-split", tensor_split(share, devices), "--main-gpu", "0"]


def devices_of(visible: Optional[str], gpus: Optional[Dict[int, str]] = None) -> List[int]:
    """The GPU numbers in a process's CUDA_VISIBLE_DEVICES (adoption after a restart); [] if any is unknown."""
    out: List[int] = []
    for part in (visible or "").split(","):
        if not part.strip():
            continue
        try:
            n = gpu_ids.resolve(part.strip(), gpus)
        except ValueError:
            return []
        out.append(n)
    return out


def serves(asked: Optional[int], info: Mapping[str, Any]) -> bool:
    """May this running instance answer a request with this `gpu_id`? None = any; "all" = a spread one; a number =
    one running on that card alone."""
    spread = len(info.get("gpu_ids") or []) > 1
    if asked is None:
        return True
    if asked == gpu_ids.ALL_GPUS:
        return spread
    return not spread and info.get("gpu_id") == asked
