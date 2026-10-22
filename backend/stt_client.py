"""Transcription through the `stt` plugin (docs/plugins.md; today stt-serve: WhisperX, diarization, MOSS, live
sessions). Which app that is comes from Apps & Services (backend/plugins/registry.py) -- nothing here knows its
name, port or URL. Started on demand; its HTTP errors pass through with their status, connection failures and a
missing plugin become 503."""
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from backend.plugins import registry

logger = logging.getLogger("ai-serve-ui.stt_client")
# Long recordings + a cold model download/load can take a while.
_TIMEOUT = 3600.0


def plugin() -> Optional[registry.Plugin]:
    return registry.provider("stt")


def unavailable_reason() -> str:
    p = plugin()
    if p is None:
        return "No transcription plugin is set up (an app in Apps & Services that provides \"stt\", e.g. stt-serve)."
    return (f"Transcription runs in {p.name}, which is not reachable. Start it in Apps & Services or check "
            f"logs/service-{p.id}.log.")


async def health() -> bool:
    p = plugin()
    return p is not None and await registry.running(p)


async def ensure_running(timeout_s: float = 40.0) -> bool:
    p = plugin()
    return p is not None and await registry.ensure_running(p, timeout_s)


async def call(endpoint: str, method: str = "GET", suffix: str = "", json: Optional[Dict[str, Any]] = None,
               params: Optional[Dict[str, Any]] = None, timeout: float = _TIMEOUT) -> Any:
    """Forward one call to a declared endpoint; start the plugin first if needed."""
    p = plugin()
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
    return data


async def status() -> Optional[Dict[str, Any]]:
    """Read-only, raw (the plugin's own fields too); doesn't start anything. None when it isn't running."""
    p = plugin()
    if p is None or "status" not in p.provision.endpoints:
        return None
    try:
        r = await registry.call(p, "status", timeout=3.0)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def release(worker: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """Free one worker's GPU memory, or all of them. Plugin down -> nothing loaded there."""
    p = plugin()
    if p is None or "release" not in p.provision.endpoints:
        return {"released": []}
    body: Dict[str, Any] = {"force": force, **({"worker": worker} if worker else {})}
    try:
        r = await registry.call(p, "release", "POST", timeout=60.0, json=body)
    except httpx.HTTPError:
        return {"released": []}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except (ValueError, AttributeError):
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()


async def unload_all(force: bool = False) -> Dict[str, Any]:
    return await release(None, force)


async def logs(tail: int = 200) -> str:
    p = plugin()
    if p is None or "logs" not in p.provision.endpoints:
        return ""
    try:
        r = await registry.call(p, "logs", timeout=5.0, params={"tail": tail})
        return str(r.json().get("logs", "")) if r.status_code == 200 else ""
    except (httpx.HTTPError, ValueError, AttributeError):
        return ""
