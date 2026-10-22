"""Plugins in the dashboard (docs/plugins.md): which apps provide what, what they hold, and freeing it by hand.
Release goes to the plugin's own `release` endpoint (busy -> 409 unless force). Tests: tests/unit/test_plugins_api.py."""
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.plugins import external, llm, registry
from backend.plugins.contracts import ContractError

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
