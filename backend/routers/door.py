"""Shares through the door: per app, which groups/people may use it from outside. The door is the `access`
plugin (docs/plugins.md; today edge-door) -- without one nothing here is contacted and the dashboard doesn't
offer sharing. Rules in backend/door/policy.py; this module stores the shares (door_shares.json, the authority)
and sends the resulting routes to the plugin's `shares` endpoint. The public domain is what the plugin's status
reports (remembered in door_shares.domain, so apps started while the door is down still get it)."""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, StrictStr

from backend.config import DOOR_SHARES_FILE
from backend.door.policy import DoorError, Share, check_share, make_share, parse_shares, routes_doc, taken_names
from backend.plugins import registry
from backend.routers import services as services_router
from backend.services import tailscale
from backend.services.exposure import may_change_exposure

logger = logging.getLogger(__name__)
router = APIRouter()


def owner_sub() -> Optional[str]:
    sub = os.environ.get("LLAMA_SERVE_OWNER_SUB")
    return sub if sub and sub != "owner" else None


def load_shares() -> Dict[str, Share]:
    try:
        with open(DOOR_SHARES_FILE, encoding="utf-8") as f:
            shares, errors = parse_shares(json.load(f))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.error(f"{DOOR_SHARES_FILE} unreadable -- nothing shared: {e}")
        return {}
    for e in errors:
        logger.error(f"door share ignored: {e}")
    return shares


def _save_shares(shares: Dict[str, Share]) -> None:
    tmp = DOOR_SHARES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({a: s.to_dict() for a, s in sorted(shares.items())}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DOOR_SHARES_FILE)


def _manifests() -> Dict[str, Any]:
    return {m.id: m for m in services_router._registry().services}


def _domain_file() -> str:
    return os.path.splitext(DOOR_SHARES_FILE)[0] + ".domain"


def _plugin() -> Optional[registry.Plugin]:
    return registry.provider("access")


def _known_domain() -> str:
    try:
        with open(_domain_file(), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _remember_domain(domain: Optional[str]) -> None:
    if domain and domain != _known_domain():
        try:
            with open(_domain_file(), "w", encoding="utf-8") as f:
                f.write(domain)
        except OSError as e:
            logger.warning(f"door domain not remembered: {e}")


async def _door(endpoint: str, method: str = "GET", body: Any = None, params: Optional[Dict[str, Any]] = None) -> Any:
    p = _plugin()
    if p is None:
        raise DoorError("no access plugin")
    if endpoint not in p.provision.endpoints:
        raise DoorError(f"{p.name} doesn't offer {endpoint!r}")
    r = await registry.call(p, endpoint, method, timeout=5.0, json=body, params=params)
    if r.status_code >= 400:
        raise DoorError(r.json().get("error", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text)
    return r.json()


async def _status() -> Optional[Dict[str, Any]]:
    p = _plugin()
    st = await registry.status(p) if p else None
    if st:
        _remember_domain(st.get("domain"))
    return st


async def sync_door() -> Dict[str, Any]:
    """Send the routes to the door if they differ. Returns {"pushed", "errors", "door_error"}."""
    doc, errors = routes_doc(load_shares(), _manifests(), owner=owner_sub())
    out: Dict[str, Any] = {"pushed": False, "errors": errors, "door_error": None}
    try:
        current = await _door("shares")
        if current.get("routes") != doc["routes"]:
            await _door("shares", "PUT", doc)
            out["pushed"] = True
            logger.info(f"door routes updated: {[r['name'] for r in doc['routes']]}")
    except (httpx.HTTPError, DoorError, ValueError) as e:
        out["door_error"] = f"The door is unreachable: {e}" if isinstance(e, httpx.HTTPError) else str(e)
    return out


def identity_env_for(app_id: str) -> Dict[str, str]:
    from backend.door.policy import identity_env
    s = load_shares().get(app_id)
    domain = _known_domain()
    return identity_env(s, domain) if s and domain else {}


async def _require_owner_device(request: Request) -> None:
    info = await asyncio.to_thread(tailscale.info)
    peer = request.client.host if request.client else None
    login = await asyncio.to_thread(tailscale.whois, peer) if peer and info else None
    if not may_change_exposure(peer, login, info.owner if info else None):
        raise HTTPException(status_code=403, detail="Shares can only be changed on this machine or from your own devices.")


@router.get("/api/door/security", summary="What the door turned away, per day -- and whether anything got through unsigned")
async def door_security(days: int = 14):
    days = max(1, min(days, 90))
    if _plugin() is None:
        return {"configured": False, "available": False, "detail": "the door is not set up"}
    try:
        data = await _door("security", params={"days": days})
    except (httpx.HTTPError, DoorError, ValueError) as e:
        return {"available": False, "detail": f"The door is unreachable or too old ({type(e).__name__})"}
    return {"available": True, **data}


@router.get("/api/door", summary="The door: status, shares per app, people who have logged in")
async def door_overview():
    if _plugin() is None:   # the door is an optional extension: not set up, nothing to contact
        return {"configured": False, "available": False, "domain": "", "owner": owner_sub(), "facts": [],
                "overview": None, "people": [], "shares": {}, "door_error": None}
    sync = await sync_door()
    status, people = await _status(), []
    try:
        people = (await _door("people")).get("people", []) if status is not None else []
    except (httpx.HTTPError, DoorError, ValueError):
        pass
    shares = load_shares()
    domain = (status or {}).get("domain") or _known_domain()
    return {
        "configured": True, "available": status is not None, "domain": domain, "owner": owner_sub(),
        "facts": (status or {}).get("facts", []), "overview": (status or {}).get("overview"),
        "people": people,
        "shares": {a: {**s.to_dict(), "url": f"https://{s.name}.{domain}/" if domain else None,
                       "error": sync["errors"].get(a)}
                   for a, s in shares.items()},
        "door_error": sync["door_error"],
    }


class ShareRequest(BaseModel):
    name: StrictStr
    groups: List[StrictStr] = []
    users: List[StrictStr] = []


@router.put("/api/door/shares/{service_id}", summary="Share an app through the door")
async def put_share(service_id: str, req: ShareRequest, request: Request):
    await _require_owner_device(request)
    if _plugin() is None:
        raise HTTPException(status_code=409, detail="the door is not set up (no app provides \"access\", see docs/plugins.md)")
    manifests = _manifests()
    m = manifests.get(service_id)
    if m is None:
        raise HTTPException(status_code=404, detail="unknown service")
    shares = load_shares()
    old = shares.get(m.id)
    # a renamed share keeps its earlier addresses: the door redirects them (links people already have keep working)
    previous = [p for p in ((*old.previous, old.name) if old and old.name != req.name else (old.previous if old else ()))
                if p != req.name]
    try:
        s = make_share(m.id, req.name, req.groups, req.users, previous)
        check_share(m, s, owner=owner_sub(), taken=taken_names(shares.values()))
    except DoorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    restart = m.id not in shares or shares[m.id].name != s.name
    shares[m.id] = s
    _save_shares(shares)
    logger.info(f"door share for {m.id}: {s.to_dict()}")
    sync = await sync_door()
    domain = _known_domain()
    return {"share": {**s.to_dict(), "url": f"https://{s.name}.{domain}/" if domain else None}, "door_error": sync["door_error"],
            "restart_needed": restart and m.identity == "multi"}


@router.delete("/api/door/shares/{service_id}", summary="Stop sharing an app through the door")
async def delete_share(service_id: str, request: Request):
    await _require_owner_device(request)
    shares = load_shares()
    if service_id not in shares:
        raise HTTPException(status_code=404, detail="not shared")
    del shares[service_id]
    _save_shares(shares)
    logger.info(f"door share for {service_id} removed")
    sync = await sync_door()
    return {"removed": service_id, "door_error": sync["door_error"]}
