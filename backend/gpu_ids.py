"""GPUs by number or, optionally, by UUID (2026-10-30, owner: a third card, an RTX 3060, may join the pool; "UUIDs
sollten optional sein").

A GPU's number is its position in nvidia-smi's list, and adding a card can shift it: settings that say "GPU 1"
would then name another card, and learned per-GPU figures would land on the wrong one. Its UUID ("GPU-ae73ed6b-...")
never changes. So:
- wherever a GPU is named (API `gpu_id`, coordination `gpu`, settings), a number still works and a UUID (or a unique
  prefix of one, at least "GPU-" + 4 characters) works too -- `resolve()` turns it into today's number;
- processes are pinned with the UUID in CUDA_VISIBLE_DEVICES (`visible_devices()`): CUDA numbers cards in its own
  order (fastest first unless CUDA_DEVICE_ORDER=PCI_BUS_ID), which matched nvidia-smi's with two cards here but need
  not with three; a UUID means the same card to both;
- what is remembered per GPU is keyed by UUID (`key()`), so it follows the card.
Without nvidia-smi (Macs, CPU-only) there are no UUIDs and everything stays a number."""
import subprocess
import time
from typing import Annotated, Dict, Optional, Union

from pydantic import BeforeValidator

_TTL_S = 60.0
_cache: Dict[str, object] = {"at": 0.0, "table": {}}


def table() -> Dict[int, str]:
    """number -> UUID of this machine's NVIDIA GPUs (cached 60 s; {} without nvidia-smi)."""
    if time.time() - float(_cache["at"]) < _TTL_S:
        return dict(_cache["table"])
    out: Dict[int, str] = {}
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout
        for line in text.strip().splitlines():
            i, _, u = (p.strip() for p in line.partition(","))
            if i.isdigit() and u.startswith("GPU-"):
                out[int(i)] = u
    except (OSError, subprocess.TimeoutExpired):
        pass
    _cache.update(at=time.time(), table=out)
    return dict(out)


def resolve(gpu: Union[int, str, None], gpus: Optional[Dict[int, str]] = None) -> Optional[int]:
    """A GPU given by number or UUID -> its number now. None stays None. ValueError says what exists."""
    if gpu is None:
        return None
    if isinstance(gpu, bool):
        raise ValueError("a GPU is a number or a UUID")
    if isinstance(gpu, int):
        if gpu < 0:
            raise ValueError("a GPU number can't be negative")
        return gpu
    s = str(gpu).strip()
    if s.isdigit():
        return int(s)
    gpus = table() if gpus is None else gpus
    if s.upper().startswith("GPU-") and len(s) >= 8:
        hits = [i for i, u in gpus.items() if u.lower().startswith(s.lower())]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"'{s}' fits more than one GPU -- give more of its UUID")
    known = ", ".join(f"{i} = {u}" for i, u in sorted(gpus.items())) or "none with a UUID on this machine"
    raise ValueError(f"no GPU '{s}' -- give a number or a UUID ({known})")


def uuid_of(index: Optional[int], gpus: Optional[Dict[int, str]] = None) -> Optional[str]:
    if index is None:
        return None
    return (table() if gpus is None else gpus).get(index)


def key(index: int, gpus: Optional[Dict[int, str]] = None) -> str:
    """How something remembered per GPU names it: its UUID, or 'gpu<n>' where there is none."""
    return uuid_of(index, gpus) or f"gpu{index}"


def visible_devices(index: int, gpus: Optional[Dict[int, str]] = None) -> str:
    """CUDA_VISIBLE_DEVICES for one GPU: its UUID (the same card whatever CUDA's own order), else its number."""
    return uuid_of(index, gpus) or str(index)


def from_visible_devices(value: Optional[str], gpus: Optional[Dict[int, str]] = None) -> Optional[int]:
    """The GPU number a process was pinned to (adoption after a restart): a number or a UUID; else None."""
    v = (value or "").strip()
    if not v or "," in v:
        return None
    try:
        return resolve(v, gpus)
    except ValueError:
        return None


def _number(v):
    return None if v is None else resolve(v)


# A request field naming a GPU: a number, or optionally a UUID (resolved to today's number while the request is read;
# an unknown one is a 422 that lists this machine's GPUs)
GpuRef = Annotated[Optional[int], BeforeValidator(_number)]
