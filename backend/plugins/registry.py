"""Which apps provide what, and talking to them only through their contract (backend/plugins/contracts.py,
docs/plugins.md). Plugins are found through Apps & Services (re-read per call: adding one needs no restart);
several plugins of a capability come in id order. Tests: tests/unit/test_plugin_registry.py.

The GPU manager learns from here which plugin processes hold GPU memory (`gpu_consumers`, from each GPU plugin's
`status.workers`) and frees them through `release` -- no plugin is known to it by name."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from backend.cluster.consumers import Consumer
from backend.plugins.contracts import ContractError, Provision, normalize_status, plugin_url

logger = logging.getLogger("ai-serve-ui.plugins")
_transport: Optional[httpx.AsyncBaseTransport] = None   # tests put a MockTransport here
_start_locks: Dict[str, asyncio.Lock] = {}
TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class Plugin:
    id: str
    name: str
    port: int
    provision: Provision
    manifest: Any


def _manifests() -> List[Any]:
    try:
        from backend.routers import services as svc
        return list(svc._registry().services)
    except Exception as e:   # a broken services.json must not take capabilities down with it
        logger.warning(f"plugins: services registry unreadable: {e}")
        return []


def _external() -> List[Any]:
    """Standard OpenAI servers the owner added by port (backend/plugins/external.py): llm plugins without an app."""
    from backend.plugins import external
    servers, errors = external.load(external.FILE)
    for e in errors:
        logger.warning(f"model servers: {e}")
    return servers


def providers(capability: str) -> List[Plugin]:
    found = list(_manifests())
    taken = {m.id for m in found} | {m.port for m in found}
    found += [s for s in _external() if s.id not in taken and s.port not in taken]   # an app wins over an entry
    out = [Plugin(m.id, m.name, m.port, m.provides[capability], m) for m in found
           if capability in (getattr(m, "provides", None) or {})]
    return sorted(out, key=lambda p: p.id)


def provider(capability: str) -> Optional[Plugin]:
    found = providers(capability)
    return found[0] if found else None


def url(p: Plugin, endpoint: str, suffix: str = "") -> str:
    return plugin_url(p.port, p.provision, endpoint, suffix)


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, transport=_transport) if _transport else httpx.AsyncClient(timeout=timeout)


async def call(p: Plugin, endpoint: str, method: str = "GET", suffix: str = "", timeout: float = TIMEOUT_S,
               **kw) -> httpx.Response:
    """One request to a declared endpoint (ContractError if it isn't declared). Network errors raise httpx's."""
    target = url(p, endpoint, suffix)
    async with _client(timeout) as c:
        return await c.request(method, target, **kw)


async def status(p: Plugin) -> Optional[Dict[str, Any]]:
    """The plugin's checked status, or None when it doesn't answer (or declares no status)."""
    if "status" not in p.provision.endpoints:
        return None
    try:
        r = await call(p, "status", timeout=3.0)
        return normalize_status(r.json()) if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


async def running(p: Plugin) -> bool:
    if getattr(p.manifest, "external", False):        # a standard OpenAI server: does it list its models?
        try:
            async with _client(3.0) as c:
                return (await c.get(url(p, "openai", "/models"))).status_code == 200
        except httpx.HTTPError:
            return False
    if "status" in p.provision.endpoints:
        return await status(p) is not None
    health = getattr(p.manifest, "health_path", None) or "/"
    try:
        async with _client(2.0) as c:
            return (await c.get(f"http://127.0.0.1:{p.port}{health}")).status_code < 500
    except httpx.HTTPError:
        return False


async def ensure_running(p: Plugin, timeout_s: float = 40.0) -> bool:
    """Start the plugin through Apps & Services if it doesn't answer (one start at a time per plugin)."""
    if await running(p):
        return True
    if getattr(p.manifest, "external", False):
        return False                                  # the owner runs it; ai-serve never starts it
    lock = _start_locks.setdefault(p.id, asyncio.Lock())
    async with lock:
        if await running(p):
            return True
        from fastapi import HTTPException
        from backend.routers import services as svc
        from backend.services import runtime
        from backend.services.domain import ServiceError, plan_start
        try:
            procs = await asyncio.to_thread(runtime.snapshot)
            plan = plan_start(p.manifest, procs)
        except (ServiceError, HTTPException) as e:
            logger.warning(f"plugin {p.id} not reachable and not startable: {getattr(e, 'message', e)}")
            return False
        logger.info(f"Starting plugin {p.id} on demand")
        svc._launched[p.id] = runtime.launch(plan, svc._log_path(p.id))
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if await running(p):
                return True
            await asyncio.sleep(0.5)
        logger.warning(f"plugin {p.id} did not come up in time")
        return False


# ------------------------------------------------------------------ for the GPU manager

def _all_gpu_plugins() -> List[Plugin]:
    seen, out = set(), []
    for m in _manifests():
        for cap, prov in sorted((getattr(m, "provides", None) or {}).items()):
            if prov.gpu != "none" and m.id not in seen:
                seen.add(m.id)
                out.append(Plugin(m.id, m.name, m.port, prov, m))
    return sorted(out, key=lambda p: p.id)


def _root_pid(m: Any) -> Optional[int]:
    from backend.cluster.local import _cmdline_pid
    for fragment in getattr(m, "match", ()) or ():
        pid = _cmdline_pid(fragment)
        if pid:
            return pid
    return None


async def gpu_consumers() -> List[Consumer]:
    """Every running GPU plugin's workers as GPU-manager consumers (`<plugin>:<worker>`). Exclusive plugins are
    never evicted; a plugin that doesn't answer is skipped (its memory counts as "other programs", untouched)."""
    out: List[Consumer] = []
    for p in _all_gpu_plugins():
        st = await status(p)
        if st is None:
            continue
        exclusive = p.provision.gpu == "exclusive"
        kind = "exclusive" if exclusive else p.provision.capability
        workers = st["workers"]
        if not workers and exclusive:
            continue          # an exclusive plugin with nothing loaded (e.g. its model server stopped) holds no GPU
        if not workers:
            pid = await asyncio.to_thread(_root_pid, p.manifest)
            if pid:
                out.append(Consumer(id=p.id, kind=kind, label=p.name, root_pids={pid}, plugin=p.id,
                                    busy=st["busy"] or exclusive, evictable_kind=not exclusive))
            continue
        for w in workers:
            pid = w["pid"] or await asyncio.to_thread(_root_pid, p.manifest)
            if not pid:
                continue
            out.append(Consumer(id=f"{p.id}:{w['id']}", kind=kind, label=w["label"], root_pids={pid}, plugin=p.id,
                                last_used=w["last_used"] or 0.0,
                                busy=w["busy"] or exclusive, evictable_kind=not exclusive))
    return out


async def release(consumer_id: str, force: bool = False) -> Dict[str, Any]:
    """Free one plugin worker's GPU memory (`<plugin>:<worker>`, or `<plugin>` for all its idle ones).
    KeyError: no such plugin; PermissionError: the plugin says it's busy (409)."""
    pid, _, worker = consumer_id.partition(":")
    p = next((x for x in _all_gpu_plugins() if x.id == pid), None)
    if p is None or "release" not in p.provision.endpoints:
        raise KeyError(consumer_id)
    body: Dict[str, Any] = {"force": force}
    if worker:
        body["worker"] = worker
    r = await call(p, "release", "POST", json=body, timeout=120.0)
    if r.status_code == 409:
        raise PermissionError(f"{p.name}: busy")
    if r.status_code >= 400:
        raise RuntimeError(f"{p.name}: release failed ({r.status_code})")
    try:
        return r.json()
    except ValueError:
        return {}
