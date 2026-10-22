"""Image models from `image` plugins (docs/plugins.md): image backends outside the gateway -- Iris in sd-serve, MOSS
in its own ComfyUI, the main ComfyUI -- each list their models (`catalog`), and /v1/images/generations, /v1/images/edits
and /v1/images/cancel go to the plugin that lists the model. A plugin row takes over a built-in row with the same id
(that is how a backend moves out of the gateway). Nothing here knows a plugin by name.
Tests: tests/unit/test_image_plugins.py."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.plugins import registry
from backend.plugins.contracts import merge_catalogs, normalize_catalog

logger = logging.getLogger("ai-serve-ui.plugins.images")
CATALOG_TTL_S = 10.0
GENERATE_TIMEOUT_S = 1800.0          # a cold model load + a slow model
_cache: Tuple[float, List[Dict[str, Any]], List[str]] = (0.0, [], [])


async def _ask(p: registry.Plugin) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not await registry.ensure_running(p, 30.0):
        return [], [f"{p.id}: not reachable"]
    try:
        r = await registry.call(p, "catalog", timeout=10.0)
        if r.status_code != 200:
            return [], [f"{p.id}: catalog answered {r.status_code}"]
        return normalize_catalog("image", p.id, r.json())
    except (httpx.HTTPError, ValueError) as e:
        return [], [f"{p.id}: {type(e).__name__}"]


async def catalog(fresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    global _cache
    plugins = registry.providers("image")
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


def invalidate() -> None:
    global _cache
    _cache = (0.0, [], [])


def _match(entries: List[Dict[str, Any]], model: str) -> Optional[Dict[str, Any]]:
    key = model.strip().lower()
    return next((e for e in entries if key == e["id"] or key in (a.lower() for a in e.get("aliases") or [])), None)


async def find(model: Optional[str]) -> Optional[Tuple[registry.Plugin, Dict[str, Any]]]:
    """(plugin, entry) when `model` names a plugin's image model (id or alias, case-insensitive)."""
    if not model or not registry.providers("image"):
        return None
    entries, _ = await catalog()
    e = _match(entries, model)
    if e is None:
        return None
    p = next((x for x in registry.providers("image") if x.id == e["plugin"]), None)
    return (p, e) if p else None


def row(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A plugin entry in /v1/images/models' row shape (provider = the plugin)."""
    out = {k: v for k, v in entry.items() if k != "plugin"}
    out["provider"] = entry["plugin"]
    return out


async def forward(p: registry.Plugin, endpoint: str, request_id: Optional[str] = None, **kw) -> httpx.Response:
    """POST to the plugin's endpoint; network errors -> RuntimeError (the caller maps it)."""
    if not await registry.ensure_running(p):
        raise RuntimeError(f"{p.name} is not reachable -- see Apps & Services, logs/service-{p.id}.log")
    headers = {"X-Request-Id": request_id} if request_id else {}
    try:
        return await registry.call(p, endpoint, "POST", timeout=GENERATE_TIMEOUT_S, headers=headers, **kw)
    except httpx.HTTPError as e:
        raise RuntimeError(f"{p.name}: {type(e).__name__}")


async def cancel(request_id: str) -> bool:
    """Ask every image plugin that can cancel to stop this request (only the one running it knows it)."""
    hit = False
    for p in registry.providers("image"):
        if "cancel" not in p.provision.endpoints or not await registry.running(p):
            continue
        try:
            r = await registry.call(p, "cancel", "POST", timeout=10.0, json={"request_id": request_id})
            hit = hit or (r.status_code == 200 and r.json().get("cancelled") is True)
        except (httpx.HTTPError, ValueError, AttributeError):
            continue
    return hit
