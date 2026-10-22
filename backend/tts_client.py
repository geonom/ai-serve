"""Speech output through the `tts` plugins (docs/plugins.md; tts-serve, and any other app that provides `tts`). Which app that is comes from Apps &
Services (backend/plugins/registry.py) -- nothing here knows its name, port or URL. Starts it on demand, maps its
errors to this API's, and degrades to "unavailable" (None / []) instead of raising for read-only calls."""
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from backend.plugins import registry

logger = logging.getLogger("ai-serve-ui.tts_client")
_GENERATE_TIMEOUT = 1900.0  # tts-serve's own worker timeout is 1800 s (cold download + load)


def plugins() -> List[registry.Plugin]:
    """Every `tts` plugin, in plugin-id order (that order decides a duplicate engine id)."""
    return registry.providers("tts")


def plugin() -> Optional[registry.Plugin]:
    """The first speech plugin (labels, legacy single-plugin views)."""
    found = plugins()
    return found[0] if found else None


def unavailable_reason() -> str:
    found = plugins()
    if not found:
        return "No speech plugin is set up (an app in Apps & Services that provides \"tts\", e.g. tts-serve)."
    names = ", ".join(p.name for p in found)
    return (f"Speech runs in {names}, which is not reachable. Start it in Apps & Services or check "
            f"logs/service-<id>.log.")


async def health() -> bool:
    for p in plugins():
        if await registry.running(p):
            return True
    return False


async def ensure_running(timeout_s: float = 30.0) -> bool:
    """Start every speech plugin that isn't running; True when at least one is up."""
    up = False
    for p in plugins():
        up = await registry.ensure_running(p, timeout_s) or up
    return up


async def _call_on(p: registry.Plugin, endpoint: str, method: str = "GET", timeout: float = 5.0,
                   **kw) -> Optional[httpx.Response]:
    if endpoint not in p.provision.endpoints:
        return None
    try:
        return await registry.call(p, endpoint, method, timeout=timeout, **kw)
    except httpx.HTTPError:
        return None


_owner: Dict[str, str] = {}   # engine id -> plugin id, from the last merged catalog


async def engines() -> Optional[Dict[str, Any]]:
    """Every speech plugin's catalog merged: {"engines": [... each with "plugin"], "default_engine", "errors"}.
    A duplicate engine id stays with the plugin that sorts first (reported in errors). None when no plugin answers."""
    merged: List[Dict[str, Any]] = []
    errors: List[str] = []
    default = None
    seen: Dict[str, str] = {}
    answered = False
    for p in plugins():
        r = await _call_on(p, "catalog", timeout=3.0)
        try:
            view = r.json() if r is not None and r.status_code == 200 else None
        except ValueError:
            view = None
        if not isinstance(view, dict):
            continue
        answered = True
        errors += [f"{p.id}: {e}" for e in (view.get("errors") or []) if isinstance(e, str)]
        for e in view.get("engines") or []:
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                continue
            if e["id"] in seen:
                errors.append(f"{p.id}: engine {e['id']!r} is already offered by {seen[e['id']]} -- ignored")
                continue
            seen[e["id"]] = p.id
            merged.append({**e, "plugin": p.id})
        d = view.get("default_engine")
        if isinstance(d, str) and seen.get(d) == p.id:
            installed_default = next((e.get("installed") is not False for e in merged if e["id"] == d), False)
            if default is None or (installed_default and not default[1]):
                default = (d, installed_default)   # the first plugin's default -- an installed one wins
    if not answered:
        return None
    _owner.clear()
    _owner.update(seen)
    return {"engines": merged, "default_engine": default[0] if default else None, "errors": errors}


async def _owner_of(engine: str) -> Optional[registry.Plugin]:
    if engine not in _owner:
        await engines()
    pid = _owner.get(engine)
    return next((p for p in plugins() if p.id == pid), None)


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail") or r.text)
    except (ValueError, AttributeError):
        return r.text


async def generate(payload: Dict[str, Any]) -> bytes:
    """payload: the tts contract's generate request with response_format='wav' -- sent to the plugin that lists its
    engine. Returns WAV bytes."""
    p = await _owner_of(str(payload.get("engine") or ""))
    if p is None:
        raise HTTPException(status_code=503, detail=unavailable_reason())
    try:
        r = await registry.call(p, "generate", "POST", timeout=_GENERATE_TIMEOUT, json=payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"{p.name} unreachable ({type(e).__name__})")
    if r.status_code == 200:
        return r.content
    detail = _detail(r)
    if r.status_code == 404:
        raise HTTPException(status_code=400, detail=detail)
    if r.status_code in (400, 409, 422, 507):
        raise HTTPException(status_code=r.status_code, detail=detail)
    raise HTTPException(status_code=502, detail=detail)


async def unload(engine: str, force: bool = False) -> Dict[str, Any]:
    p = await _owner_of(engine)
    r = await _call_on(p, "release", "POST", timeout=60.0, json={"worker": engine, "force": force}) if p else None
    if r is None:
        return {"id": engine, "status": "already unloaded"}  # plugin down -> nothing loaded there
    if r.status_code != 200:
        raise HTTPException(status_code=400 if r.status_code == 404 else r.status_code, detail=_detail(r))
    try:
        released = r.json().get("released") or []
    except (ValueError, AttributeError):
        released = []
    return {"id": engine, "status": "unloaded" if released else "already unloaded"}


async def unload_idle(gpu: Optional[int] = None) -> List[str]:
    """Every speech plugin frees its idle engines (on that GPU)."""
    out: List[str] = []
    for p in plugins():
        r = await _call_on(p, "release", "POST", timeout=60.0, json={} if gpu is None else {"gpu": gpu})
        try:
            out += list(r.json().get("released") or []) if r is not None and r.status_code == 200 else []
        except (ValueError, AttributeError):
            pass
    return out


async def logs(tail: int = 200, engine: Optional[str] = None) -> str:
    """One engine's log from its plugin; without an engine, every speech plugin's log."""
    targets = [await _owner_of(engine)] if engine else plugins()
    parts = []
    for p in targets:
        if p is None:
            continue
        r = await _call_on(p, "logs", params={"tail": tail, **({"engine": engine} if engine else {})})
        try:
            text = str(r.json().get("logs", "")) if r is not None and r.status_code == 200 else ""
        except (ValueError, AttributeError):
            text = ""
        if text:
            parts.append(text if len(targets) == 1 else f"=== {p.name} ===\n{text}")
    return "\n\n".join(parts)
