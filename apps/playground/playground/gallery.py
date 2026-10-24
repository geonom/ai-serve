"""Each person's generated images ("Meine Bilder"), stored on this machine.

Threat model: the Playground is shared through the door, so several people may use it; one must never
see, fetch or delete another's images. The person comes from llama_serve_identity (verified door token
or the owner on loopback), never from the request body. Their folder is a hash of the account id, image
ids are generated here (25 hex chars, time-sortable) and checked by pattern before they touch the file
system, only real PNG/JPEG/WebP bytes are stored (no SVG/HTML that a browser would run), and metadata is
an allowlist of plain fields. Deleting is an explicit action of the person; nothing is pruned
automatically. Tests: tests/test_gallery.py.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

MAX_IMAGE_BYTES = 25 * 1024 * 1024
ASPECTS = ("1:1", "4:3", "3:4", "16:9", "9:16")
SOURCES = ("bild", "vergleich", "positionen")
_ID = re.compile(r"^[0-9a-f]{25}$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$")
_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}
_last = [0]
_id_lock = threading.Lock()


class GalleryError(ValueError):
    status = 400


def person_dir(root: str, sub: str) -> str:
    return os.path.join(root, "users", hashlib.sha256(sub.encode()).hexdigest()[:32])


def new_id() -> str:
    """Microseconds (13 hex, strictly increasing within this process) + 12 random hex."""
    with _id_lock:
        t = max(time.time_ns() // 1000, _last[0] + 1)
        _last[0] = t
    return f"{t:013x}{secrets.token_hex(6)}"


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID.match(value))


def decode_image(value: Any) -> Tuple[bytes, str]:
    """base64 (optionally a data: URI) -> (bytes, extension); only PNG, JPEG and WebP."""
    if not isinstance(value, str) or not value:
        raise GalleryError("image missing")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    if len(value) > MAX_IMAGE_BYTES * 4 // 3 + 8:
        raise GalleryError("image too large")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise GalleryError("image is not valid base64")
    if len(data) > MAX_IMAGE_BYTES:
        raise GalleryError("image too large")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data, "png"
    if data.startswith(b"\xff\xd8\xff"):
        return data, "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, "webp"
    raise GalleryError("only PNG, JPEG or WebP images")


def parse_meta(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise GalleryError("meta must be an object")
    allowed = {"prompt", "model", "label", "steps", "seed", "aspect_ratio", "seconds", "source", "comparison"}
    unknown = set(raw) - allowed
    if unknown:
        raise GalleryError(f"unknown fields: {', '.join(sorted(unknown))}")
    out: Dict[str, Any] = {"source": "bild"}
    if raw.get("prompt") is not None:
        p = raw["prompt"]
        if not isinstance(p, str) or len(p) > 4000:
            raise GalleryError("prompt must be text up to 4000 characters")
        out["prompt"] = p.strip()
    v = raw.get("model")
    if v is not None:
        if not isinstance(v, str) or not _MODEL.match(v) or ".." in v:
            raise GalleryError("model must be a plain name")
        out["model"] = v
    v = raw.get("label")   # display text only (the UI sets it via textContent): any printable text
    if v is not None:
        if not isinstance(v, str) or not 1 <= len(v) <= 120 or any(ord(ch) < 32 or ord(ch) == 127 for ch in v):
            raise GalleryError("label must be 1-120 printable characters")
        out["label"] = v
    if raw.get("steps") is not None:
        s = raw["steps"]
        if type(s) is not int or not 1 <= s <= 200:
            raise GalleryError("steps must be 1..200")
        out["steps"] = s
    if raw.get("seed") is not None:
        s = raw["seed"]
        if type(s) is not int or not 0 <= s < 2**63:
            raise GalleryError("seed must be a non-negative integer")
        out["seed"] = s
    if raw.get("aspect_ratio") is not None:
        if raw["aspect_ratio"] not in ASPECTS:
            raise GalleryError(f"aspect_ratio must be one of {', '.join(ASPECTS)}")
        out["aspect_ratio"] = raw["aspect_ratio"]
    if raw.get("seconds") is not None:
        s = raw["seconds"]
        if type(s) not in (int, float) or not 0 <= s <= 86400:
            raise GalleryError("seconds must be 0..86400")
        out["seconds"] = round(float(s), 1)
    if raw.get("source") is not None:
        if raw["source"] not in SOURCES:
            raise GalleryError(f"source must be one of {', '.join(SOURCES)}")
        out["source"] = raw["source"]
    if raw.get("comparison") is not None:
        if not valid_id(raw["comparison"]):
            raise GalleryError("comparison must be an id")
        out["comparison"] = raw["comparison"]
    return out


class Gallery:
    def __init__(self, root: str):
        self.root = root

    def _dir(self, sub: str) -> str:
        return os.path.join(person_dir(self.root, sub), "images")

    def save(self, sub: str, data: bytes, meta: Dict[str, Any]) -> Dict[str, Any]:
        meta = parse_meta(meta)
        ext = decode_image(base64.b64encode(data).decode())[1]
        d = self._dir(sub)
        os.makedirs(d, mode=0o700, exist_ok=True)
        image_id = new_id()
        entry = {"id": image_id, "created": time.time(), "type": ext, **meta}
        with open(os.path.join(d, f"{image_id}.{ext}"), "wb") as f:
            f.write(data)
        tmp = os.path.join(d, f".{image_id}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(entry, f)
        os.replace(tmp, os.path.join(d, f"{image_id}.json"))
        return entry

    def _entry(self, sub: str, image_id: str) -> Optional[Dict[str, Any]]:
        if not valid_id(image_id):
            return None
        try:
            with open(os.path.join(self._dir(sub), f"{image_id}.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def list(self, sub: str, limit: int = 60, before: Optional[str] = None,
             comparison: Optional[str] = None) -> List[Dict[str, Any]]:
        d = self._dir(sub)
        try:
            names = os.listdir(d)
        except OSError:
            return []
        ids = sorted((n[:-5] for n in names if n.endswith(".json") and valid_id(n[:-5])), reverse=True)
        if before is not None:
            ids = [i for i in ids if i < before] if valid_id(before) else []
        out: List[Dict[str, Any]] = []
        for i in ids:
            e = self._entry(sub, i)
            if e is None or (comparison is not None and e.get("comparison") != comparison):
                continue
            out.append(e)
            if len(out) >= max(1, min(limit, 500)):
                break
        return out

    def image(self, sub: str, image_id: str) -> Optional[Tuple[str, str]]:
        e = self._entry(sub, image_id)
        if e is None or e.get("type") not in _TYPES:
            return None
        path = os.path.join(self._dir(sub), f"{image_id}.{e['type']}")
        return (path, _TYPES[e["type"]]) if os.path.isfile(path) else None

    def delete(self, sub: str, image_id: str) -> bool:
        e = self._entry(sub, image_id)
        if e is None:
            return False
        d = self._dir(sub)
        for name in (f"{image_id}.{e.get('type')}", f"{image_id}.json"):
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
        return True
