"""Transcription through the `stt` plugins (docs/plugins.md; stt-serve: WhisperX, diarization, MOSS, live sessions --
and any other app that provides `stt`). Which apps those are comes from Apps & Services (backend/plugins/registry.py)
-- nothing here knows a name, port or URL. Several plugins at once (like speech output, tts_client.py): their engines
are merged into one catalog (a duplicate engine id stays with the plugin that sorts first), a transcription goes to the
plugin that lists its engine, a live session stays with the plugin that started it. Started on demand; HTTP errors
pass through with their status, connection failures and a missing plugin become 503. Tests: test_stt_multi.py."""
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from backend.config import DEFAULT_STT_ENGINE
from backend.plugins import registry

logger = logging.getLogger("ai-serve-ui.stt_client")
# Long recordings + a cold model download/load can take a while.
_TIMEOUT = 3600.0

_owner: Dict[str, str] = {}      # engine id -> plugin id, from the last merged catalog
_default: Optional[str] = None   # the merged catalog's default engine
_sessions: Dict[str, str] = {}   # live session id -> plugin id


def plugins() -> List[registry.Plugin]:
    return registry.providers("stt")


def plugin() -> Optional[registry.Plugin]:
    """The first transcription plugin (labels, legacy single-plugin views)."""
    found = plugins()
    return found[0] if found else None


def _by_id(pid: Optional[str]) -> Optional[registry.Plugin]:
    return next((p for p in plugins() if p.id == pid), None)


def unavailable_reason() -> str:
    found = plugins()
    if not found:
        return "No transcription plugin is set up (an app in Apps & Services that provides \"stt\", e.g. stt-serve)."
    names = ", ".join(p.name for p in found)
    return f"Transcription runs in {names}, which is not reachable. Start it in Apps & Services or check its log."


async def health() -> bool:
    for p in plugins():
        if await registry.running(p):
            return True
    return False


async def ensure_running(timeout_s: float = 40.0) -> bool:
    up = False
    for p in plugins():
        up = await registry.ensure_running(p, timeout_s) or up
    return up


async def _get(p: registry.Plugin, endpoint: str, timeout: float = 5.0, **kw) -> Optional[Any]:
    if endpoint not in p.provision.endpoints:
        return None
    try:
        r = await registry.call(p, endpoint, timeout=timeout, **kw)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def engines() -> Optional[Dict[str, Any]]:
    """Every transcription plugin's catalog merged: {"engines": [... each with "plugin"], "default_engine",
    "errors"}. Default: DEFAULT_STT_ENGINE, else the first plugin default that is available. None: nobody answered."""
    global _default
    merged: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen: Dict[str, str] = {}
    defaults: List[str] = []
    answered = False
    for p in plugins():
        view = await _get(p, "catalog", timeout=5.0)
        if not isinstance(view, dict):
            continue
        answered = True
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
            defaults.append(d)
    if not answered:
        return None
    available = {e["id"] for e in merged if e.get("available") is not False}
    _default = (DEFAULT_STT_ENGINE if DEFAULT_STT_ENGINE in seen else
                next((d for d in defaults if d in available), defaults[0] if defaults else None))
    _owner.clear()
    _owner.update(seen)
    return {"engines": merged, "default_engine": _default, "errors": errors}


async def _owner_of(engine: Optional[str]) -> Optional[registry.Plugin]:
    if not engine or engine not in _owner:
        await engines()
    return _by_id(_owner.get(engine or _default or ""))


async def _pick(endpoint: str, suffix: str, json: Optional[Dict[str, Any]],
                params: Optional[Dict[str, Any]]) -> Optional[registry.Plugin]:
    """Which plugin a call goes to (see the module docstring)."""
    if endpoint == "transcribe":
        if json is not None and not json.get("engine"):
            await engines()
            if _default:
                json["engine"] = _default      # named explicitly: the default may live in another plugin
        owner = await _owner_of((json or {}).get("engine"))
        # an engine no catalog names: the first plugin decides (and refuses it with its own message)
        return owner or next((p for p in plugins() if "transcribe" in p.provision.endpoints), None)
    if endpoint == "live":
        sid = (json or {}).get("session_id") or (params or {}).get("session_id")
        if sid and sid in _sessions:
            return _by_id(_sessions[sid])
        live = [p for p in plugins() if "live" in p.provision.endpoints]
        if not live:
            return None
        if suffix == "/start":     # the default engine's plugin if it has live sessions, else the first that has
            home = await _owner_of(None)
            return home if home in live else live[0]
        return live[0]
    return next((p for p in plugins() if endpoint in p.provision.endpoints), None)


async def call(endpoint: str, method: str = "GET", suffix: str = "", json: Optional[Dict[str, Any]] = None,
               params: Optional[Dict[str, Any]] = None, timeout: float = _TIMEOUT) -> Any:
    """Forward one call to the plugin it belongs to; start that plugin first if needed."""
    if endpoint == "catalog":
        view = await engines() if await ensure_running() else None
        if view is None:
            raise HTTPException(status_code=503, detail=unavailable_reason())
        return view
    if endpoint == "live" and suffix == "/sessions":
        out: List[Any] = []
        for p in plugins():
            if "live" in p.provision.endpoints and await registry.ensure_running(p):
                got = await _get(p, "live", suffix="/sessions")
                out += (got or {}).get("sessions", []) if isinstance(got, dict) else []
        return {"sessions": out}
    p = await _pick(endpoint, suffix, json, params)
    if p is None or not await registry.ensure_running(p):
        raise HTTPException(status_code=503, detail=unavailable_reason())
    if endpoint not in p.provision.endpoints:
        raise HTTPException(status_code=501, detail=f"{p.name} doesn't offer {endpoint!r}")
    try:
        r = await registry.call(p, endpoint, method, suffix=suffix, timeout=timeout, json=json, params=params)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"{p.name} unreachable ({type(e).__name__})")
    try:
        data = r.json()
    except ValueError:
        data = {"detail": r.text}
    if r.status_code >= 400:
        detail = data.get("detail", r.text) if isinstance(data, dict) else r.text
        status = r.status_code if r.status_code in (400, 404, 409, 422, 500, 507) else 502
        raise HTTPException(status_code=status, detail=detail)
    if endpoint == "live" and suffix == "/start" and isinstance(data, dict) and data.get("session_id"):
        _sessions[str(data["session_id"])] = p.id
    return data


async def status() -> Optional[Dict[str, Any]]:
    """Read-only, doesn't start anything: the first running plugin's status with every plugin's workers added.
    None when no plugin is running."""
    merged: Optional[Dict[str, Any]] = None
    for p in plugins():
        st = await _get(p, "status", timeout=3.0)
        if not isinstance(st, dict):
            continue
        if merged is None:
            merged = dict(st, workers=list(st.get("workers") or []))
        else:
            merged["workers"] += list(st.get("workers") or [])
            merged["busy"] = bool(merged.get("busy")) or bool(st.get("busy"))
    return merged


async def _release_on(p: registry.Plugin, body: Dict[str, Any]) -> List[str]:
    if "release" not in p.provision.endpoints:
        return []
    try:
        r = await registry.call(p, "release", "POST", timeout=60.0, json=body)
    except httpx.HTTPError:
        return []        # plugin down -> nothing loaded there
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except (ValueError, AttributeError):
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    try:
        return list(r.json().get("released") or [])
    except (ValueError, AttributeError):
        return []


async def release(worker: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """Free one worker's memory (on the plugin that lists it), or every plugin's."""
    body: Dict[str, Any] = {"force": force, **({"worker": worker} if worker else {})}
    targets = plugins()
    if worker:
        holder = None
        for p in targets:
            st = await _get(p, "status", timeout=3.0)
            if any(isinstance(w, dict) and w.get("id") == worker for w in (st or {}).get("workers") or []):
                holder = p
                break
        targets = [holder or (targets[0] if targets else None)]
    released: List[str] = []
    for p in targets:
        if p is not None:
            released += await _release_on(p, body)
    return {"released": released}


async def unload_all(force: bool = False) -> Dict[str, Any]:
    return await release(None, force)


async def logs(tail: int = 200) -> str:
    found = plugins()
    parts = []
    for p in found:
        got = await _get(p, "logs", params={"tail": tail})
        text = str(got.get("logs", "")) if isinstance(got, dict) else ""
        if text:
            parts.append(text if len(found) == 1 else f"=== {p.name} ===\n{text}")
    return "\n\n".join(parts)
