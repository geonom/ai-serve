"""Dashboard "Services" tab: overview, start, stop and open sibling apps (StoryTime,
llm-benchy, ...). Rules in backend/services/domain.py, OS access in runtime.py.

GET  /api/services                 registered services, status, running instances
POST /api/services/{id}/start      start on the manifest's port (409 if occupied)
POST /api/services/{id}/stop       {"pid": root pid} -- required when >1 instance runs
POST /api/services/{id}/exposure   {"level": "local"|"tailnet"|"owner"} (backend/services/exposure.py)
GET  /api/services/{id}/log?tail=N output of a copy started from here
GET  /api/services/{id}/screenshot the manifest's screenshot (image inside the project)
GET  /api/services/{id}/status     facts from the running app's own `status_path`
GET  /api/services/manifest-schema JSON Schema for projects integrating themselves

Discovery: every direct child of SERVICE_ROOTS with a llama-serve.service.json,
plus services.json include/exclude (optional). See docs/service-manifest.md.
"""
import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, StrictBool

from backend.config import LOGS_DIR, SERVICE_AUTOSTART_FILE, SERVICE_EXPOSURE_FILE, SERVICE_ROOTS, SERVICES_FILE
from backend import events
from backend.routers import media
from backend.services import runtime, tailscale
from backend.services.exposure import (
    Exposure,
    ExposureError,
    ExposureSettings,
    foreign_routes,
    may_change_exposure,
    parse_serve_status,
    plan_exposure,
    reconcile_serve,
    tailnet_port,
)
from backend.services.autostart import AutostartSettings, plan_autostart
from backend.services.gate import GateManager
from backend.services.domain import (
    SCREENSHOT_TYPES,
    ManifestError,
    ServiceError,
    ServiceManifest,
    ServiceRegistry,
    ServicesConfig,
    descends_from,
    discover_instances,
    find_project_dirs,
    manifest_schema,
    open_target,
    parse_status_facts,
    plan_start,
    plan_stop,
    screenshot_file,
)
from backend.state import state

logger = logging.getLogger("ai-serve-ui.services")
router = APIRouter(tags=["services"])

SERVICE_LOGS_DIR = LOGS_DIR
# Ports a service manifest may not claim: this backend.
from backend.config import GATEWAY_PORT  # noqa: E402
_RESERVED_PORTS = frozenset({8030, GATEWAY_PORT})
# Copies started from here (service id -> Popen), for "starting"/"exited" status + reaping.
_launched: Dict[str, subprocess.Popen] = {}


GATES = GateManager()


class StopRequest(BaseModel):
    pid: Optional[int] = None


class ExposureRequest(BaseModel):
    level: Exposure


def _builtins():
    """Services the backend itself runs (none since ComfyUI became the image-serve app, 2026-10-21; the mechanism
    stays for later)."""
    return []


def _registry() -> ServiceRegistry:
    """Projects under SERVICE_ROOTS that carry a manifest, plus services.json's
    include/exclude. Re-read on every request -- adding a manifest needs no restart."""
    try:
        with open(SERVICES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = None
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"{SERVICES_FILE}: {e}")
    try:
        config = ServicesConfig.from_dict(data)
        refs, root_errors = find_project_dirs(SERVICE_ROOTS, config)
        registry = ServiceRegistry.load(refs, reserved_ports=_RESERVED_PORTS, exclude_ids=config.excluded_ids,
                                        builtins=_builtins())
    except ManifestError as e:
        raise HTTPException(status_code=500, detail=str(e))
    registry.errors = root_errors + registry.errors
    return registry


def _get(registry: ServiceRegistry, service_id: str):
    try:
        return registry.get(service_id)
    except ServiceError as e:
        raise HTTPException(status_code=e.status, detail=e.message)


def _log_path(service_id: str) -> str:
    return os.path.join(SERVICE_LOGS_DIR, f"service-{service_id}.log")


# --------------------------------------------------------------------------- exposure

def _peer_address(request: Request) -> Optional[str]:
    return request.client.host if request.client else None


def _load_exposure() -> ExposureSettings:
    try:
        with open(SERVICE_EXPOSURE_FILE, encoding="utf-8") as f:
            return ExposureSettings.from_dict(json.load(f))
    except FileNotFoundError:
        return ExposureSettings()
    except (OSError, ValueError) as e:
        logger.error(f"{SERVICE_EXPOSURE_FILE} unreadable, treating every service as local: {e}")
        return ExposureSettings()


def _save_exposure(settings: ExposureSettings) -> None:
    tmp = SERVICE_EXPOSURE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)
    os.replace(tmp, SERVICE_EXPOSURE_FILE)


def _load_autostart() -> AutostartSettings:
    try:
        with open(SERVICE_AUTOSTART_FILE, encoding="utf-8") as f:
            return AutostartSettings.from_dict(json.load(f))
    except FileNotFoundError:
        return AutostartSettings()
    except (OSError, ValueError) as e:
        logger.error(f"{SERVICE_AUTOSTART_FILE} unreadable, no service autostarts: {e}")
        return AutostartSettings(errors=(str(e),))


def _save_autostart(settings: AutostartSettings) -> None:
    tmp = SERVICE_AUTOSTART_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)
    os.replace(tmp, SERVICE_AUTOSTART_FILE)


async def run_autostart() -> Dict[str, Any]:
    """Backend startup: start every service switched to autostart that isn't
    running yet -- through the same start path as the dashboard's Start button."""
    registry = _registry()
    procs = await asyncio.to_thread(runtime.snapshot)
    running = {m.id for m in registry.services if discover_instances(m, procs)}
    results: Dict[str, Any] = {}
    for sid in plan_autostart(_load_autostart(), [m.id for m in registry.services], running):
        try:
            results[sid] = await start_service(sid)
            logger.info(f"Autostarted service {sid}")
        except HTTPException as e:
            results[sid] = {"error": e.detail}
            logger.warning(f"Autostart of {sid} failed: {e.detail}")
    return results


async def apply_exposure(registry: Optional[ServiceRegistry] = None) -> Dict[str, Any]:
    """Bring gates and `tailscale serve` in line with the saved levels. Called after
    every change and at backend startup (gates live in this process). Returns
    {service id: error} for services whose level couldn't be applied."""
    registry = registry or _registry()
    settings = _load_exposure()
    info = await asyncio.to_thread(tailscale.info)
    owner = info.owner if info else None
    desired: Dict[int, str] = {}
    gates: Dict[int, Any] = {}
    managed = set()
    errors: Dict[str, str] = {}
    for m in registry.services:
        try:
            managed.add(tailnet_port(m))
        except ExposureError:
            pass
        level = settings.level(m.id)
        if level == Exposure.LOCAL:
            continue
        if info is None:
            errors[m.id] = "Tailscale is not running"
            continue
        try:
            plan = plan_exposure(m, level, owner)
        except ExposureError as e:
            errors[m.id] = str(e)
            continue
        desired[plan.route.https_port] = plan.route.target
        if plan.gate:
            gates[plan.gate.listen_port] = plan.gate
    for port in GATES.running_ports() - set(gates):
        await GATES.stop(port)
    for spec in gates.values():
        try:
            await GATES.ensure(spec)
        except OSError as e:  # gate port taken -- leave the route off rather than open it ungated
            errors.update({m.id: f"gate port {spec.listen_port}: {e}" for m in registry.services
                           if plan_exposure(m, Exposure.OWNER, owner).gate == spec})
            desired = {p: t for p, t in desired.items() if t != f"http://127.0.0.1:{spec.listen_port}"}
    if info is not None:
        actual = parse_serve_status(await asyncio.to_thread(tailscale.serve_status))
        service_ports = frozenset(m.port for m in registry.services)
        for action, port, target in reconcile_serve(desired, actual, managed, service_ports):
            try:
                if action == "set":
                    await asyncio.to_thread(tailscale.serve_set, port, target)
                else:
                    await asyncio.to_thread(tailscale.serve_off, port)
            except RuntimeError as e:
                logger.error(str(e))
                errors.update({m.id: str(e) for m in registry.services if _safe_tport(m) == port})
    return errors


def _safe_tport(m) -> Optional[int]:
    try:
        return tailnet_port(m)
    except ExposureError:
        return None


def _exposure_view(m, settings: ExposureSettings, info, actual: Dict[int, str], errors: Dict[str, str]):
    level = settings.level(m.id)
    view: Dict[str, Any] = {"level": level.value, "url": None, "warning": None, "error": errors.get(m.id),
                            "tailnet_port": _safe_tport(m)}
    try:
        plan = plan_exposure(m, level, info.owner if info else None)
    except ExposureError as e:
        view["error"] = view["error"] or str(e)
        return view
    view["warning"] = plan.warning
    foreign = foreign_routes(m, actual)
    if foreign and info:
        urls = ", ".join(f"https://{info.dns_name}{'' if p == 443 else f':{p}'}/" for p in foreign)
        note = (f"Already reachable in the tailnet via {urls} -- a `tailscale serve` route set up outside "
                f"the dashboard. Remove it (tailscale serve --https=<port> off) to control access here.")
        view["warning"] = f"{view['warning']} {note}" if view["warning"] else note
    if plan.route and info:
        if actual.get(plan.route.https_port) == plan.route.target:
            view["url"] = plan.url(info.dns_name)
        else:
            view["error"] = view["error"] or "not applied in tailscale serve"
    return view


@router.post("/api/services/{service_id}/exposure", summary="Choose who can reach a service via Tailscale")
async def set_exposure(service_id: str, req: ExposureRequest, request: Request):
    registry = _registry()
    m = _get(registry, service_id)
    info = await asyncio.to_thread(tailscale.info)
    peer = _peer_address(request)
    login = await asyncio.to_thread(tailscale.whois, peer) if peer and info else None
    if not may_change_exposure(peer, login, info.owner if info else None):
        logger.warning(f"exposure change for {m.id} refused for {peer} ({login})")
        raise HTTPException(status_code=403, detail="Exposure can only be changed on this machine "
                                                    "or from one of the owner's tailnet devices.")
    if req.level != Exposure.LOCAL:
        if info is None:
            raise HTTPException(status_code=503, detail="Tailscale is not running on this machine.")
        try:
            plan_exposure(m, req.level, info.owner)
        except ExposureError as e:
            raise HTTPException(status_code=400, detail=str(e))
    _save_exposure(_load_exposure().with_level(m.id, req.level))
    errors = await apply_exposure(registry)
    if m.id in errors:
        raise HTTPException(status_code=500, detail=errors[m.id])
    logger.info(f"exposure of {m.id} set to {req.level.value} by {peer} ({login or 'local'})")
    return {"id": m.id, "level": req.level.value}


@router.get("/api/services", summary="Registered services and their running instances")
async def list_services():
    registry = _registry()
    procs = await asyncio.to_thread(runtime.snapshot)
    settings = _load_exposure()
    autostart = _load_autostart()
    info = await asyncio.to_thread(tailscale.info)
    actual = parse_serve_status(await asyncio.to_thread(tailscale.serve_status)) if info else {}
    out = []
    for m in registry.services:
        instances = discover_instances(m, procs)
        launched = _launched.get(m.id)
        managed_pids = {launched.pid} if launched and launched.poll() is None else set()
        inst_out = []
        for i in instances:
            port = m.port if m.port in i.ports else min(i.ports)
            healthy = await runtime.health(port, m.health_path) if m.health_path else None
            inst_out.append({
                "pid": i.root_pid, "ports": sorted(i.ports), "started_at": i.started_at,
                "managed": bool(managed_pids & i.pids) or bool(m.controller)
                           or any(descends_from(i.root_pid, mp, procs) for mp in managed_pids),
                "healthy": healthy,
                "exposed": [f"{a}:{p}" for a, p in i.exposed],
                "open": open_target(m, port),
            })
        if instances:
            status = "running"
        elif launched and launched.poll() is None:
            status = "starting"
        elif launched and launched.poll() is not None:
            status = "exited"
        else:
            status = "stopped"
        out.append({
            "id": m.id, "name": m.name, "description": m.description, "port": m.port,
            "project_dir": str(m.project_dir), "source": registry.source(m.id), "status": status,
            "exit_code": launched.poll() if status == "exited" else None,
            "instances": inst_out, "has_log": os.path.isfile(_log_path(m.id)),
            "exposure": _exposure_view(m, settings, info, actual, {}),
            "autostart": autostart.enabled(m.id),
            "category": m.category, "icon": m.icon,
            "has_screenshot": m.screenshot is not None, "has_status": m.status_path is not None, "ui": m.ui,
            "identity": m.identity, "maturity": m.maturity,
        })
    return {
        "services": out, "errors": registry.errors + [{"dir": SERVICE_EXPOSURE_FILE, "error": e, "source": "exposure"}
                                                       for e in settings.errors],
        "roots": list(SERVICE_ROOTS),
        "tailscale": {"available": info is not None, "owner": info.owner if info else None,
                      "dns_name": info.dns_name if info else None},
    }


class AutostartRequest(BaseModel):
    enabled: StrictBool  # "yes"/1 must not silently switch on a process start


@router.post("/api/services/{service_id}/autostart", summary="Start a service automatically when the backend starts")
async def set_autostart(service_id: str, req: AutostartRequest):
    m = _get(_registry(), service_id)
    _save_autostart(_load_autostart().with_enabled(m.id, req.enabled))
    logger.info(f"Service {m.id}: autostart {'on' if req.enabled else 'off'}")
    return {"id": m.id, "autostart": req.enabled}


@router.get("/api/services/manifest-schema", summary="JSON Schema of llama-serve.service.json")
async def service_manifest_schema():
    return manifest_schema()


@router.post("/api/services/{service_id}/start", summary="Start a registered service")
async def start_service(service_id: str):
    m = _get(_registry(), service_id)
    procs = await asyncio.to_thread(runtime.snapshot)
    try:
        plan = plan_start(m, procs)
    except ServiceError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    previous = _launched.get(m.id)
    if previous and previous.poll() is None:
        raise HTTPException(status_code=409, detail=f"{m.name} is still starting (pid {previous.pid}).")
    log_path = _log_path(m.id)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as f:
        f.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(plan.argv)}\n".encode())
    # docs/identity.md: apps tell the owner coming through Tailscale Serve by this login
    info = await asyncio.to_thread(tailscale.info)
    if info and info.owner:
        os.environ["LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"] = info.owner
    # shared through the door: the app verifies the door's identity for its public host
    from backend.routers.door import identity_env_for
    plan.env.update(identity_env_for(m.id))
    proc = runtime.launch(plan, log_path)
    _launched[m.id] = proc
    logger.info(f"Started service {m.id} (pid {proc.pid}) on port {m.port}")
    events.record("service", "{name} started", name=m.name)
    return {"status": "starting", "pid": proc.pid, "port": m.port}


async def _reports_busy(m, inst) -> bool:
    """The app's own status says busy (`status_path`, raw `busy` field). Unreadable -> not busy."""
    if not m.status_path:
        return False
    port = m.port if m.port in inst.ports else (min(inst.ports) if inst.ports else m.port)
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            r = await client.get(f"http://127.0.0.1:{port}{m.status_path}")
        data = r.json() if r.status_code == 200 and len(r.content) <= _STATUS_MAX_BYTES else {}
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(data, dict) and data.get("busy") is True


@router.post("/api/services/restart-outdated", summary="Restart this installation's apps and plugins whose code changed")
async def restart_outdated():
    """After an update (`./ai-serve restart` calls this): apps/ and plugins/ of this ai-serve folder that still run
    code older than their files are restarted -- unless they report themselves busy. Other projects are never
    touched (backend/services/updates.py)."""
    from backend.config import _PROJECT_DIR
    from backend.services import updates
    reg = _registry()
    procs = await asyncio.to_thread(runtime.snapshot)
    rows, by_id = [], {}
    for m in reg.services:
        if m.controller or not updates.part_of_installation(str(m.project_dir), _PROJECT_DIR):   # built-ins: not ours to restart
            continue
        for inst in discover_instances(m, procs):
            rows.append((m.id, str(m.project_dir), inst.root_pid, inst.started_at))
            by_id[(m.id, inst.root_pid)] = (m, inst)
    newest = {d: await asyncio.to_thread(updates.newest_code_mtime, d) for d in {r[1] for r in rows}}
    restarted, busy = [], []
    for sid, pid in updates.outdated(rows, newest, _PROJECT_DIR):
        m, inst = by_id[(sid, pid)]
        if await _reports_busy(m, inst):
            busy.append(sid)
            continue
        await stop_service(sid, StopRequest(pid=pid))
        try:
            await start_service(sid)
            restarted.append(sid)
        except HTTPException as e:
            logger.warning(f"{sid} stopped for an update but didn't start again: {e.detail}")
            restarted.append(f"{sid} (stopped; starts again when needed)")
    return {"restarted": restarted, "busy": busy}


@router.post("/api/services/{service_id}/stop", summary="Stop a running instance of a service")
async def stop_service(service_id: str, req: Optional[StopRequest] = None):
    m = _get(_registry(), service_id)
    procs = await asyncio.to_thread(runtime.snapshot)
    try:
        pids = plan_stop(m, discover_instances(m, procs), req.pid if req else None)
    except ServiceError as e:
        raise HTTPException(status_code=e.status, detail=e.message)
    result = await asyncio.to_thread(runtime.terminate, pids)
    launched = _launched.get(m.id)
    if launched and launched.pid in pids:
        launched.poll()  # reap
        del _launched[m.id]  # stopped on purpose -> "stopped", not "exited" (= crashed)
    logger.info(f"Stopped service {m.id}: {result}")
    events.record("service", "{name} stopped", name=m.name)
    return {"status": "stopped", **result}


@router.get("/api/services/{service_id}/log", summary="Log of a copy started from the dashboard")
async def service_log(service_id: str, tail: int = 200):
    m = _get(_registry(), service_id)
    return {"log": runtime.tail(_log_path(m.id), max(1, min(tail, 2000)))}


@router.get("/api/services/{service_id}/screenshot", summary="The app's screenshot from its manifest")
async def service_screenshot(service_id: str):
    m = _get(_registry(), service_id)
    path = screenshot_file(m)  # re-checked now: inside the project, raster image, <= 5 MB
    if path is None:
        raise HTTPException(status_code=404, detail="no screenshot")
    return FileResponse(path, media_type=SCREENSHOT_TYPES[path.suffix.lower()],
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "max-age=60"})


_STATUS_MAX_BYTES = 64 * 1024


@router.get("/api/services/{service_id}/status", summary="Short facts the running app reports about itself")
async def service_status(service_id: str):
    """Fetched from the app's own loopback port (`status_path` in its manifest) and
    reduced to at most six plain {label, value} strings -- nothing else is passed on."""
    m = _get(_registry(), service_id)
    procs = await asyncio.to_thread(runtime.snapshot)
    instances = discover_instances(m, procs)
    if not instances:
        return {"id": m.id, "running": False, "facts": []}
    if not m.status_path:
        return {"id": m.id, "running": True, "facts": []}
    inst = instances[0]
    port = m.port if m.port in inst.ports else min(inst.ports)
    try:
        async with httpx.AsyncClient(timeout=2.0, follow_redirects=False) as client:
            r = await client.get(f"http://127.0.0.1:{port}{m.status_path}")
        if r.status_code != 200 or len(r.content) > _STATUS_MAX_BYTES:
            return {"id": m.id, "running": True, "facts": []}
        facts = parse_status_facts(r.json())
    except (httpx.HTTPError, ValueError):
        facts = []
    return {"id": m.id, "running": True, "facts": facts}
