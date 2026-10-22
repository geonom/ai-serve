"""Language models from `llm` plugins (docs/plugins.md): other OpenAI-compatible servers next to this machine's
llama.cpp -- Strata through its adapter, later MLX, vLLM, Ollama. Which models exist comes from each plugin's
`{openai}/models`; a request naming one of them (id or alias, case-insensitive) goes to that plugin. Nothing here
knows a plugin by name. Tests: tests/unit/test_llm_plugins.py."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.plugins import registry
from backend.plugins.contracts import merge_catalogs, normalize_catalog

logger = logging.getLogger("ai-serve-ui.plugins.llm")
CATALOG_TTL_S = 10.0
_cache: Tuple[float, List[Dict[str, Any]], List[str]] = (0.0, [], [])


async def _ask(p: registry.Plugin) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not await registry.ensure_running(p, 20.0):
        return [], [f"{p.id}: not reachable"]
    try:
        r = await registry.call(p, "openai", suffix="/models", timeout=5.0)
        if r.status_code != 200:
            return [], [f"{p.id}: models answered {r.status_code}"]
        return normalize_catalog("llm", p.id, r.json())
    except (httpx.HTTPError, ValueError) as e:
        return [], [f"{p.id}: {type(e).__name__}"]


async def catalog(fresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Every llm plugin's models (checked, tagged with the plugin; an id belongs to the first plugin listing it)."""
    global _cache
    at, entries, errors = _cache
    plugins = registry.providers("llm")
    if not plugins:
        return [], []
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
    """Forget the cached catalog (a server was added or removed)."""
    global _cache
    _cache = (0.0, [], [])


def _match(entries: List[Dict[str, Any]], model: str) -> Optional[Dict[str, Any]]:
    """By id or alias, case-insensitive; then by the basename of a path-like handle (a benchmark's /m/x-strata.gguf)
    -- ids may contain '/' themselves (LM Studio's publisher/model), so the whole name is tried first."""
    for key in dict.fromkeys([model.strip().lower(), os.path.basename(model.strip()).lower()]):
        for e in entries:
            if key == e["id"].lower() or key in (a.lower() for a in e.get("aliases") or []):
                return e
    return None


async def find(model: Optional[str]) -> Optional[Tuple[registry.Plugin, Dict[str, Any]]]:
    """(plugin, entry) when `model` names a plugin's model, else None (then it's a local/device GGUF)."""
    if not model or not registry.providers("llm"):
        return None
    entries, _ = await catalog()
    e = _match(entries, model)
    if e is None:                       # a plugin that just came up may list it already
        entries, _ = await catalog(fresh=True)
        e = _match(entries, model)
    if e is None:
        return None
    p = next((x for x in registry.providers("llm") if x.id == e["plugin"]), None)
    return (p, e) if p else None


def base_url(p: registry.Plugin) -> str:
    """The plugin's server root as seen by OpenAI paths (`.../v1/chat/completions` is appended)."""
    return registry.url(p, "openai")[: -len("/v1")]


async def load(p: registry.Plugin, model_id: str, timeout_s: float = registry.TIMEOUT_S) -> None:
    """Make the plugin load `model_id` now and wait until it serves (its optional `load` endpoint; without one the
    first request loads it). Raises RuntimeError with the plugin's reason."""
    if not await registry.ensure_running(p):
        raise RuntimeError(f"{p.name} is not reachable")
    if "load" not in p.provision.endpoints:
        return
    r = await registry.call(p, "load", "POST", timeout=timeout_s, json={"model": model_id})
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail") or r.text
        except (ValueError, AttributeError):
            detail = r.text
        raise RuntimeError(f"{p.name}: {detail}")


def by_port(port: int) -> Optional[registry.Plugin]:
    return next((p for p in registry.providers("llm") if p.port == port), None)
