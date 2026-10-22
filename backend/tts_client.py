"""Speech output through the `tts` plugin (docs/plugins.md; today tts-serve). Which app that is comes from Apps &
Services (backend/plugins/registry.py) -- nothing here knows its name, port or URL. Starts it on demand, maps its
errors to this API's, and degrades to "unavailable" (None / []) instead of raising for read-only calls."""
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from backend.plugins import registry

logger = logging.getLogger("ai-serve-ui.tts_client")
_GENERATE_TIMEOUT = 1900.0  # tts-serve's own worker timeout is 1800 s (cold download + load)


def plugin() -> Optional[registry.Plugin]:
    return registry.provider("tts")


def unavailable_reason() -> str:
    p = plugin()
    if p is None:
        return "No speech plugin is set up (an app in Apps & Services that provides \"tts\", e.g. tts-serve)."
    return (f"Speech runs in {p.name}, which is not reachable. Start it in Apps & Services or check "
            f"logs/service-{p.id}.log.")


async def health() -> bool:
    p = plugin()
    return p is not None and await registry.running(p)


async def ensure_running(timeout_s: float = 30.0) -> bool:
    p = plugin()
    return p is not None and await registry.ensure_running(p, timeout_s)


async def _call(endpoint: str, method: str = "GET", timeout: float = 5.0, **kw) -> Optional[httpx.Response]:
    p = plugin()
    if p is None or endpoint not in p.provision.endpoints:
        return None
    try:
        return await registry.call(p, endpoint, method, timeout=timeout, **kw)
    except httpx.HTTPError:
        return None


async def engines() -> Optional[Dict[str, Any]]:
    r = await _call("catalog", timeout=3.0)
    try:
        return r.json() if r is not None and r.status_code == 200 else None
    except ValueError:
        return None


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail") or r.text)
    except (ValueError, AttributeError):
        return r.text


async def generate(payload: Dict[str, Any]) -> bytes:
    """payload: the tts contract's generate request with response_format='wav'. Returns WAV bytes."""
    p = plugin()
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
    r = await _call("release", "POST", timeout=60.0, json={"worker": engine, "force": force})
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
    r = await _call("release", "POST", timeout=60.0, json={} if gpu is None else {"gpu": gpu})
    try:
        return list(r.json().get("released") or []) if r is not None and r.status_code == 200 else []
    except (ValueError, AttributeError):
        return []


async def logs(tail: int = 200, engine: Optional[str] = None) -> str:
    r = await _call("logs", params={"tail": tail, **({"engine": engine} if engine else {})})
    try:
        return str(r.json().get("logs", "")) if r is not None and r.status_code == 200 else ""
    except (ValueError, AttributeError):
        return ""
