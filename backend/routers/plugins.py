"""Plugins in the dashboard (docs/plugins.md): which apps provide what, what they hold, and freeing it by hand.
Release goes to the plugin's own `release` endpoint (busy -> 409 unless force). Tests: tests/unit/test_plugins_api.py."""
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.plugins import external, llm, registry
from backend.plugins.contracts import ContractError, normalize_install
import re

router = APIRouter(tags=["plugins"])


@router.get("/api/plugins", summary="Plugins: capabilities, GPU use, status")
async def list_plugins():
    out = []
    seen = set()
    for m in registry._manifests():
        provides = getattr(m, "provides", None) or {}
        if not provides or m.id in seen:
            continue
        seen.add(m.id)
        p = registry.Plugin(m.id, m.name, m.port, next(iter(provides.values())), m)
        st = await registry.status(p) if any("status" in v.endpoints for v in provides.values()) else None
        out.append({"id": m.id, "name": m.name, "port": m.port,
                    "capabilities": {c: {"gpu": v.gpu, "endpoints": sorted(v.endpoints)} for c, v in provides.items()},
                    "running": st is not None, "status": st})
    return {"plugins": sorted(out, key=lambda x: x["id"])}


@router.post("/api/plugins/{plugin_id}/release", summary="Free what a plugin holds (one worker, or everything)")
async def release_plugin(plugin_id: str, worker: Optional[str] = None, force: bool = False):
    try:
        return await registry.release(f"{plugin_id}:{worker}" if worker else plugin_id, force=force)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no GPU plugin {plugin_id!r} with a release endpoint")
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=f"{e} -- retry later or pass force=true")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ------------------------------------------------------------------ downloads plugins offer (models, engines)
# Owner only: the dashboard access policy lets only this machine and the owner's devices reach /api/ (the Spielwiese
# passes these on only for its owner). The request names an item id; the plugin decides which files.

_ITEM_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _installers():
    seen, out = set(), []
    for m in registry._manifests():
        for cap, prov in sorted((getattr(m, "provides", None) or {}).items()):
            if "install" in prov.endpoints and m.id not in seen:
                seen.add(m.id)
                out.append(registry.Plugin(m.id, m.name, m.port, prov, m))
    return out


@router.get("/api/plugins/installs", summary="What the plugins can download (models, engines) and how far it got")
async def list_installs():
    items, errors = [], []
    for p in _installers():
        if not await registry.ensure_running(p):
            errors.append(f"{p.id}: not running")
            continue
        try:
            r = await registry.call(p, "install", timeout=15.0)
            got, errs = normalize_install(r.json() if r.status_code == 200 else None)
        except (httpx.HTTPError, ValueError) as e:
            got, errs = [], [f"unreachable ({type(e).__name__})"]
        cap = next((c for c, prov in (getattr(p.manifest, "provides", None) or {}).items() if prov is p.provision), None)
        items += [{**i, "plugin": p.id, "plugin_name": p.name, "capability": cap} for i in got]
        errors += [f"{p.id}: {e}" for e in errs]
    return {"items": items, "errors": errors}


@router.post("/api/plugins/{plugin_id}/install", status_code=202, summary="Start downloading one item a plugin offers")
async def start_install(plugin_id: str, body: Any = Body(...)):
    if not isinstance(body, dict) or set(body) != {"id"} or not isinstance(body["id"], str) or not _ITEM_ID.match(body["id"]):
        raise HTTPException(status_code=400, detail='send {"id": "<item id from GET /api/plugins/installs>"} and nothing else')
    p = next((x for x in _installers() if x.id == plugin_id), None)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no plugin {plugin_id!r} that offers downloads")
    if not await registry.ensure_running(p):
        raise HTTPException(status_code=503, detail=f"{p.name} isn't running")
    try:
        r = await registry.call(p, "install", "POST", timeout=30.0, json={"id": body["id"]})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"{p.name} unreachable ({type(e).__name__})")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        raise HTTPException(status_code=r.status_code if r.status_code in (400, 404, 409, 507) else 502,
                            detail=str(detail or f"{p.name} answered {r.status_code}")[:300])
    got, _ = normalize_install({"items": [r.json()]} if r.headers.get("content-type", "").startswith("application/json") else None)
    return got[0] if got else {"id": body["id"], "state": "downloading"}


# ------------------------------------------------------------------ external model servers (Ollama, LM Studio ...)

async def _models_at(server) -> Optional[int]:
    """How many models a standard OpenAI server lists right now (None: it doesn't answer)."""
    p = registry.Plugin(server.id, server.name, server.port, server.provision, server)
    try:
        r = await registry.call(p, "openai", suffix="/models", timeout=2.0)
        data = r.json().get("data") if r.status_code == 200 else None
        return len(data) if isinstance(data, list) else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


@router.get("/api/model-servers", summary="Standard OpenAI-compatible servers added by port, and ones found running")
async def list_model_servers():
    servers, errors = external.load(external.FILE)
    out = []
    for s in servers:
        n = await _models_at(s)
        out.append({**s.to_dict(), "running": n is not None, "models": n or 0})
    added = {s.port for s in servers} | {s.id for s in servers}
    apps = {m.port for m in registry._manifests()}
    detected = []
    for k in external.KNOWN:
        if k.port in added or k.id in added or k.port in apps:
            continue
        n = await _models_at(k)
        if n is not None:
            detected.append({**k.to_dict(), "models": n})
    return {"servers": out, "detected": detected, "errors": errors}


@router.put("/api/model-servers/{server_id}", summary="Add or change a standard OpenAI-compatible server (loopback port)")
async def put_model_server(server_id: str, body: Dict[str, Any] = Body(...)):
    servers, _ = external.load(external.FILE)
    try:
        new = external.merged(servers, {**body, "id": server_id})
    except ContractError as e:
        raise HTTPException(status_code=400, detail=str(e))
    external.save(external.FILE, new)
    llm.invalidate()
    return {"server": next(s.to_dict() for s in new if s.id == server_id)}


@router.delete("/api/model-servers/{server_id}", summary="Remove a server (it keeps running; ai-serve just stops using it)")
async def delete_model_server(server_id: str):
    servers, _ = external.load(external.FILE)
    if not any(s.id == server_id for s in servers):
        raise HTTPException(status_code=404, detail="no such server")
    external.save(external.FILE, [s for s in servers if s.id != server_id])
    llm.invalidate()
    return {"removed": server_id}
