"""Kontingente: how much of the GPU a guest of the Spielwiese may use.

Threat model: the Spielwiese is shared through the door, and behind it every gateway call arrives from
loopback -- with the owner's rights (no per-person limits in the gateway). Without this, one guest could
keep both GPUs busy all day, load big models that evict the owner's, or queue hundreds of images. So:
- what is GPU-bound is decided here by method + path (`classify`), never by the client;
- per person and local day: GPU seconds (wall time of their GPU-bound calls) and images, plus at most
  `concurrent` GPU-bound calls in flight; batch runs are the owner's only;
- the owner (llama_serve_identity's is_owner) is counted but never limited;
- limits live in an owner-edited file (`data/quotas.json`), parsed strictly -- an unreadable or invalid
  file falls back to the defaults, never to "unlimited".
The check-and-reserve in `Ledger.begin` is one locked step, so parallel requests can't slip past a limit.
Tests: tests/test_quota.py, tests/test_quota_server.py.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

from spielwiese.gallery import person_dir


class QuotaError(ValueError):
    status = 400


class QuotaExceeded(Exception):
    status = 429

    def __init__(self, message: str, busy: bool = False):
        super().__init__(message)
        self.busy = busy   # only "another request of yours is still running" -- worth waiting for


@dataclass(frozen=True)
class Limits:
    gpu_seconds_per_day: Optional[int]   # None = unlimited
    images_per_day: Optional[int]
    concurrent: Optional[int]


DEFAULT_LIMITS = Limits(gpu_seconds_per_day=1800, images_per_day=60, concurrent=1)
UNLIMITED = Limits(None, None, None)
_FIELDS = ("gpu_seconds_per_day", "images_per_day", "concurrent")

_KINDS = [
    ("POST", re.compile(r"^/(v1|api)/chat/completions$"), "chat"),
    ("POST", re.compile(r"^/v1/images/generations$"), "image"),
    ("POST", re.compile(r"^/api/comfyui/(generate|edit|style-transfer)$"), "image"),
    ("POST", re.compile(r"^/api/comfyui/batch(/.*)?$"), "owner_only"),
    ("POST", re.compile(r"^/api/(tts/generate|stt/transcribe)$"), "speech"),
    ("POST", re.compile(r"^/api/ocr$"), "ocr"),
    ("POST", re.compile(r"^/v1/(embeddings|rerank)$"), "text"),   # "Texte vergleichen": loads a model too
]


def classify(method: str, path: str) -> Optional[str]:
    """chat | image | speech | ocr | text | owner_only for GPU-bound calls, None for everything else."""
    path = path.split("?", 1)[0]
    for m, pattern, kind in _KINDS:
        if method == m and pattern.match(path):
            return kind
    return None


def _limit_value(v: Any, name: str, minimum: int) -> Optional[int]:
    if v is None:
        return None
    if type(v) is not int or v < minimum:
        raise QuotaError(f"{name} must be a whole number >= {minimum}, or null for unlimited")
    return v


def _parse_limits(raw: Any, where: str) -> Dict[str, Optional[int]]:
    if not isinstance(raw, dict):
        raise QuotaError(f"{where} must be an object")
    unknown = set(raw) - set(_FIELDS)
    if unknown:
        raise QuotaError(f"{where}: unknown fields {', '.join(sorted(unknown))}")
    return {k: _limit_value(raw[k], f"{where}.{k}", 1 if k == "concurrent" else 0) for k in raw}


def parse_limits_file(raw: Any) -> Dict[str, Any]:
    """{"defaults": {...}, "people": {sub: {...} | null}} -> checked config (None/missing = all defaults)."""
    if raw is None:
        return {"defaults": {}, "people": {}}
    if not isinstance(raw, dict) or set(raw) - {"defaults", "people"}:
        raise QuotaError("expected {defaults, people}")
    people = raw.get("people", {})
    if people is None:
        people = {}
    if not isinstance(people, dict):
        raise QuotaError("people must be an object")
    out_people = {}
    for sub, lim in people.items():
        if not isinstance(sub, str) or not sub or len(sub) > 200:
            raise QuotaError("people keys must be account ids")
        out_people[sub] = None if lim is None else _parse_limits(lim, f"people.{sub}")
    defaults = raw.get("defaults", {})
    return {"defaults": _parse_limits({} if defaults is None else defaults, "defaults"), "people": out_people}


def load_limits(cfg: Dict[str, Any], sub: str, is_owner: bool) -> Limits:
    if is_owner:
        return UNLIMITED
    lim = replace(DEFAULT_LIMITS, **cfg.get("defaults", {}))
    own = cfg.get("people", {}).get(sub)
    return replace(lim, **own) if own else lim


def read_limits_file(path: str) -> Dict[str, Any]:
    """The owner's file; missing or invalid -> defaults (never unlimited)."""
    try:
        with open(path) as f:
            return parse_limits_file(json.load(f))
    except FileNotFoundError:
        return parse_limits_file(None)
    except (OSError, ValueError, QuotaError):
        return parse_limits_file(None)


def write_limits_file(path: str, raw: Any) -> Dict[str, Any]:
    cfg = parse_limits_file(raw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
    return cfg


def _today() -> str:
    return time.strftime("%Y-%m-%d")


@dataclass
class Ticket:
    sub: str
    kind: str
    images: int


class Ledger:
    """Per person and day: GPU seconds and images used (persisted), GPU calls in flight (memory)."""

    def __init__(self, root: str, today: Callable[[], str] = _today):
        self.root, self.today = root, today
        self._lock = threading.Lock()
        self._in_flight: Dict[str, int] = {}

    def _path(self, sub: str) -> str:
        return os.path.join(person_dir(self.root, sub), "usage.json")

    def _read(self, sub: str) -> Dict[str, Any]:
        day = self.today()
        try:
            with open(self._path(sub)) as f:
                u = json.load(f)
            if u.get("day") == day:
                return {"day": day, "gpu_seconds": u.get("gpu_seconds", 0), "images": u.get("images", 0)}
        except (OSError, ValueError):
            pass
        return {"day": day, "gpu_seconds": 0, "images": 0}

    def _write(self, sub: str, u: Dict[str, Any]) -> None:
        path = self._path(sub)
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(u, f)
        os.replace(tmp, path)

    def usage(self, sub: str) -> Dict[str, Any]:
        with self._lock:
            return self._read(sub)

    def in_flight(self, sub: str) -> int:
        with self._lock:
            return self._in_flight.get(sub, 0)

    def begin(self, sub: str, lim: Limits, kind: str, images: int = 0) -> Ticket:
        """Check and reserve in one step; raises QuotaExceeded (a message for the person)."""
        with self._lock:
            u = self._read(sub)
            busy = self._in_flight.get(sub, 0)
            if lim.concurrent is not None and busy >= lim.concurrent:
                raise QuotaExceeded("Another request of yours is still running – wait until it's done.", busy=True)
            if lim.gpu_seconds_per_day is not None and u["gpu_seconds"] >= lim.gpu_seconds_per_day:
                raise QuotaExceeded(f"Your daily compute time ({lim.gpu_seconds_per_day // 60} minutes) "
                                    "is used up – it continues tomorrow.")
            if images and lim.images_per_day is not None and u["images"] + images > lim.images_per_day:
                left = max(0, lim.images_per_day - u["images"])
                raise QuotaExceeded(f"Only {left} image{'' if left == 1 else 's'} left in your quota today "
                                    f"(of {lim.images_per_day}) – it continues tomorrow.")
            self._in_flight[sub] = busy + 1
            return Ticket(sub, kind, images)

    def end(self, ticket: Ticket, seconds: float, images: int) -> None:
        with self._lock:
            self._in_flight[ticket.sub] = max(0, self._in_flight.get(ticket.sub, 1) - 1)
            u = self._read(ticket.sub)
            u["gpu_seconds"] = round(u["gpu_seconds"] + max(0.0, seconds), 1)
            u["images"] += max(0, images)
            self._write(ticket.sub, u)
