"""Image analysis from `vision` plugins (docs/plugins.md; 2026-10-25, first caller: Tapestry, the family-tree app):
faces (boxes, 5 landmarks, quality, embeddings for recognition) and objects. Each plugin lists its models with the
`tasks` they do; /v1/vision/analyze takes the tasks an app wants (faces, objects or both) and sends each to the model
for it -- one call when one model does both. Nothing here knows a plugin by name.

Privacy: images and face embeddings are biometric data -- they pass through, never stored, never logged (only
path, status, counts). Tests: tests/unit/test_vision_plugins.py."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from backend.plugins import registry
from backend.plugins.contracts import VISION_TASKS, merge_catalogs, normalize_catalog

CATALOG_TTL_S = 10.0
ANALYZE_TIMEOUT_S = 600.0            # a cold model load + an album page of large scans
_cache: Tuple[float, List[Dict[str, Any]], List[str]] = (0.0, [], [])


class VisionError(ValueError):
    pass


async def _ask(p: registry.Plugin) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not await registry.ensure_running(p, 30.0):
        return [], [f"{p.id}: not reachable"]
    try:
        r = await registry.call(p, "catalog", timeout=10.0)
        if r.status_code != 200:
            return [], [f"{p.id}: catalog answered {r.status_code}"]
        return normalize_catalog("vision", p.id, r.json())
    except (httpx.HTTPError, ValueError) as e:
        return [], [f"{p.id}: {type(e).__name__}"]


async def catalog(fresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    global _cache
    plugins = registry.providers("vision")
    if not plugins:
        return [], []
    at, entries, errors = _cache
    if not fresh and time.monotonic() - at < CATALOG_TTL_S:
        return entries, errors
    parts, errors = [], []
    for p in plugins:
        e, errs = await _ask(p)
        parts.append((p.id, e))
        errors += errs
    entries, clashes = merge_catalogs(parts)
    errors += [f"{c['ignored']}: model id {c['id']!r} already belongs to {c['kept']}" for c in clashes]
    _cache = (time.monotonic(), entries, errors)
    return entries, errors


def parse_tasks(raw: Optional[str]) -> List[str]:
    """"faces", "objects", "faces,objects" (or "both") -> the tasks; default faces."""
    raw = (raw or "faces").strip().lower()
    tasks = list(VISION_TASKS) if raw in ("both", "all") else [t.strip() for t in raw.split(",") if t.strip()]
    if not tasks or any(t not in VISION_TASKS for t in tasks):
        raise VisionError(f"tasks: {', '.join(VISION_TASKS)} (comma-separated) or both")
    return list(dict.fromkeys(tasks))


def pick(entries: Sequence[Dict[str, Any]], task: str, model: Optional[str]) -> Dict[str, Any]:
    """The model for `task`: the one named (id or alias) if it does that task, else the default for it."""
    usable = [e for e in entries if task in e["tasks"] and e["available"]]
    if model:
        key = model.strip().lower()
        named = next((e for e in entries if key == e["id"] or key in (a.lower() for a in e.get("aliases") or [])), None)
        if named is None:
            raise VisionError(f"unknown vision model {model!r} -- GET /v1/vision/models lists them")
        if task not in named["tasks"]:
            raise VisionError(f"{named['label']} doesn't do {task}")
        if not named["available"]:
            raise VisionError(f"{named['label']} is not installed (Compute & models -> Downloads)")
        return named
    if not usable:
        raise VisionError(f"no installed model for {task} -- install a vision plugin's model "
                          f"(GET /v1/vision/models lists what exists)")
    return next((e for e in usable if e["default"]), usable[0])


def plan(entries: Sequence[Dict[str, Any]], tasks: Sequence[str],
         models: Dict[str, Optional[str]]) -> List[Tuple[Dict[str, Any], List[str]]]:
    """[(entry, its tasks)] -- tasks that land on the same model go in one call."""
    out: List[Tuple[Dict[str, Any], List[str]]] = []
    for t in tasks:
        e = pick(entries, t, models.get(t))
        for got, ts in out:
            if got["id"] == e["id"]:
                ts.append(t)
                break
        else:
            out.append((e, [t]))
    return out


def merge(n_images: int, answers: Sequence[Tuple[Dict[str, Any], List[str], Any]]) -> Dict[str, Any]:
    """Each model's per-image results -> one row per image (faces from one, objects from another)."""
    images: List[Dict[str, Any]] = [{"index": i} for i in range(n_images)]
    models: Dict[str, Any] = {}
    for entry, tasks, data in answers:
        rows = data.get("images") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != n_images:
            raise RuntimeError(f"{entry['plugin']}: answered {len(rows) if isinstance(rows, list) else 'no'} "
                               f"results for {n_images} images")
        for img, row in zip(images, rows):
            if not isinstance(row, dict):
                continue
            for k in ("width", "height", "error"):
                if k in row and k not in img:
                    img[k] = row[k]
            for t in tasks:
                if t in row:
                    img[t] = row[t]
        for t in tasks:
            models[t] = {"id": entry["id"], "label": entry["label"], "plugin": entry["plugin"],
                         "license": entry["license"], **({"embedding": entry["embedding"]}
                                                         if t == "faces" and entry.get("embedding") else {})}
    return {"object": "vision.analysis", "images": images, "models": models}


def provider_of(entry: Dict[str, Any]) -> Optional[registry.Plugin]:
    return next((p for p in registry.providers("vision") if p.id == entry["plugin"]), None)


async def analyze(p: registry.Plugin, fields: Dict[str, str], files: List[Any]) -> httpx.Response:
    if not await registry.ensure_running(p):
        raise RuntimeError(f"{p.name} is not reachable -- see Apps & Services, logs/service-{p.id}.log")
    try:
        return await registry.call(p, "analyze", "POST", timeout=ANALYZE_TIMEOUT_S, data=fields, files=files)
    except httpx.HTTPError as e:
        raise RuntimeError(f"{p.name}: {type(e).__name__}")
