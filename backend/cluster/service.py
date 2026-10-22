"""Runs routing plans against node agents (I/O). Rules: backend/cluster/routing.py;
behaviour under test: tests/unit/test_cluster_service.py.

- reuse/start: an Upstream through the agent's authenticated proxy.
- copy: in the background, with fallback to the next node when one refuses the
  download; the caller gets 503 + Retry-After right away instead of hanging.
- Idle reaper: stops instances the coordinator started once their keep-alive ran
  out (never instances it didn't start; never deletes files)."""
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from backend.cluster.client import NodeUnreachable, agent_call
from backend.cluster.routing import NodeSnapshot, plan_route

logger = logging.getLogger("ai-serve-ui.cluster")
_MB = 1024 * 1024
START_TIMEOUT_S = 600
COPY_TIMEOUT_S = 6 * 3600
RETRY_AFTER_S = 120


class CopyRefused(Exception):
    pass


def registry():
    from backend.routers.cluster import REGISTRY  # lazy: the router imports this module
    return REGISTRY


async def send_plan_to(rec, plan: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await agent_call(rec, "POST", "/v1/models/download", plan)
    except NodeUnreachable as e:
        raise CopyRefused(str(e))


async def send_model_to(rec, name: str, local_path: str) -> Dict[str, Any]:
    from backend.routers.cluster import plan_for_local
    return await send_plan_to(rec, await plan_for_local(rec, name, local_path))


async def wait_download(rec, name: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            st = (await agent_call(rec, "GET", "/v1/downloads")).get("downloads", {}).get(name) or {}
        except NodeUnreachable as e:
            return f"node unreachable: {e}"
        if st.get("state") == "done":
            return "done"
        if st.get("state") == "failed":
            return st.get("error") or "download failed"
        await asyncio.sleep(3)
    return "timed out"


def _for_node(plan: Dict[str, Any], rec) -> Dict[str, Any]:
    """Point coordinator sources at the URL this device uses to reach us."""
    out = dict(plan)
    srcs = []
    for u in plan.get("sources") or []:
        if "/api/cluster/models/" in u:
            if not rec.coordinator_url:
                continue
            u = f"{rec.coordinator_url}/api/cluster/models/{u.rsplit('/', 1)[-1]}"
        srcs.append(u)
    out["sources"] = srcs
    return out


@dataclass
class Upstream:
    kind: str                      # local / plugin (an llm plugin's OpenAI server) / node
    base: str
    headers: Dict[str, str] = field(default_factory=dict)
    label: str = ""
    port: Optional[int] = None
    session: Optional[Tuple[str, str]] = None
    model: Optional[str] = None    # the model name the upstream knows (a plugin's catalog id)

    def url(self, path: str) -> str:
        return self.base + path


class ClusterService:
    def __init__(self, snapshot_ttl_s: float = 5.0):
        self.sessions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.copies: Dict[str, Dict[str, Any]] = {}
        self._ttl = snapshot_ttl_s
        self._cache: Tuple[float, List[NodeSnapshot]] = (0.0, [])

    def has_nodes(self) -> bool:
        return bool(registry().all())

    async def snapshots(self) -> List[NodeSnapshot]:
        if self._ttl and time.monotonic() - self._cache[0] < self._ttl:
            return self._cache[1]

        async def one(rec):
            try:
                node = await agent_call(rec, "GET", "/v1/node", timeout=5.0)
                models = await agent_call(rec, "GET", "/v1/models", timeout=5.0)
            except NodeUnreachable:
                return NodeSnapshot(id=rec.id, online=False, priority=0, runtime_kinds=frozenset(),
                                    accelerators=[], models={}, instances=[])
            return NodeSnapshot(
                id=rec.id, online=True, priority=0,
                runtime_kinds=frozenset(node.get("runtime_kinds") or []),
                accelerators=node.get("accelerators") or [],
                models={m["name"]: int(m.get("size_mb") or 0) for m in models.get("models") or []},
                instances=node.get("instances") or [])
        snaps = list(await asyncio.gather(*(one(r) for r in registry().all())))
        self._cache = (time.monotonic(), snaps)
        return snaps

    def _upstream(self, rec, iid: str, model: str, keep_alive: int, owned: bool) -> Upstream:
        key = (rec.id, iid)
        if key not in self.sessions:
            from backend import events
            events.record("device", f"{os.path.basename(model)} läuft auf {rec.id} (hier war kein Platz)")
        s = self.sessions.setdefault(key, {"active": 0, "owned": owned, "model": model})
        s.update(last_used=time.time(), keep_alive=keep_alive)
        return Upstream(kind="node", base=f"{rec.url}/v1/instances/{iid}/proxy",
                        headers={"Authorization": f"Bearer {rec.token}"},
                        label=f"{os.path.basename(model)} @ {rec.id}", session=key)

    async def upstream_for(self, model_name: str, local_path: Optional[str], ctx_size: int,
                           keep_alive: int, local_size_mb: Optional[int] = None,
                           mode: str = "chat", pooling: Optional[str] = None) -> Upstream:
        """local_path/local_size_mb: the model file on this machine, if there is one
        (then it can be copied to a device)."""
        snaps = await self.snapshots()
        base = os.path.basename(model_name)
        size_mb = local_size_mb if local_path else None
        if size_mb is None:
            sizes = [sz for s in snaps if s.online for m, sz in s.models.items() if os.path.basename(m) == base]
            size_mb = max(sizes) if sizes else None
        if size_mb is None:
            raise HTTPException(status_code=507, detail=f"{base}: not on this machine and on no device")
        plan = plan_route(base, size_mb, ctx_size, snaps, mode=mode, pooling=pooling)
        reg = registry()
        if plan.action == "reuse":
            owned = (plan.node, plan.instance) in self.sessions and self.sessions[(plan.node, plan.instance)]["owned"]
            return self._upstream(reg.get(plan.node), plan.instance, plan.model_path, keep_alive, owned)
        if plan.action == "start":
            rec = reg.get(plan.node)
            try:
                body = {"model": plan.model_path, "ctx_size": ctx_size, "gpu": plan.gpu}
                if mode != "chat":  # older agents reject unknown fields -- chat keeps working with them
                    body["mode"] = mode
                if pooling:
                    body["pooling"] = pooling
                inst = await agent_call(rec, "POST", "/v1/instances", body, timeout=30.0)
                deadline = time.monotonic() + START_TIMEOUT_S
                while inst.get("state") == "loading" and time.monotonic() < deadline:
                    await asyncio.sleep(1)
                    inst = await agent_call(rec, "GET", f"/v1/instances/{inst['id']}")
            except NodeUnreachable as e:
                raise HTTPException(status_code=502, detail=f"starting {base} on {rec.id} failed: {e}")
            if inst.get("state") != "ready":
                raise HTTPException(status_code=502, detail=f"{base} on {rec.id} did not come up ({inst.get('state')})")
            self._cache = (0.0, [])
            logger.info(f"Cluster: started {base} on {rec.id} (instance {inst['id']})")
            return self._upstream(rec, inst["id"], plan.model_path, keep_alive, owned=True)
        if plan.action == "copy" and local_path:
            cur = self.copies.get(base)
            if not cur or cur["state"] == "failed":   # never a second copy of a running/finished one
                self.copies[base] = {"state": "copying", "node": None, "tried": [], "errors": []}
                self.copies[base]["task"] = asyncio.create_task(self._copy(base, local_path, plan.candidates))
            node = self.copies[base].get("node") or plan.candidates[0]
            raise HTTPException(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)},
                                detail=f"{base} doesn't fit on this machine; it is being copied to device "
                                       f"{node} -- retry in a few minutes")
        raise HTTPException(status_code=507, detail=f"{base}: {plan.reason or 'no device can take it'}")

    async def place_model(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """"Best device" for a download plan (name, size, sha256, sources): the
        first device with room that takes it; one that refuses (quota, disk) is
        skipped for the next. Sources whose URL is coordinator-specific are
        rewritten per device (each agent knows us under its own URL)."""
        name = plan["name"]
        snaps = await self.snapshots()
        route = plan_route(name, int(plan["size"]) >> 20, 8192, snaps)
        if route.action in ("reuse", "start"):
            return {"state": "present", "node": route.node}
        if route.action != "copy":
            raise HTTPException(status_code=507, detail=f"{name}: {route.reason or 'no device can take it'}")
        cur = self.copies.get(name)
        if cur and cur["state"] == "copying":
            return {"state": "copying", "node": cur.get("node")}
        self.copies[name] = {"state": "copying", "node": None, "tried": [], "errors": []}
        self.copies[name]["task"] = asyncio.create_task(self._copy(name, None, route.candidates, plan))
        return {"state": "copying", "node": route.candidates[0], "candidates": route.candidates}

    async def _copy(self, name: str, local_path: Optional[str], candidates: List[str],
                    plan: Optional[Dict[str, Any]] = None) -> None:
        st = self.copies[name]
        for node in candidates:
            rec = registry().get(node)
            if rec is None:
                continue
            st["node"] = node
            st["tried"].append(node)
            try:
                if plan is not None:
                    await send_plan_to(rec, _for_node(plan, rec))
                else:
                    await send_model_to(rec, name, local_path)
                result = await wait_download(rec, name, COPY_TIMEOUT_S)
            except CopyRefused as e:
                result = str(e)
            if result == "done":
                st["state"] = "done"
                self._cache = (0.0, [])
                logger.info(f"Cluster: copied {name} to {node}")
                return
            st["errors"].append(f"{node}: {result}")
            logger.warning(f"Cluster: {node} could not take {name} ({result}) -- trying the next device")
        st.update(state="failed", node=None)

    def begin(self, up: Upstream) -> None:
        s = self.sessions.get(up.session)
        if s:
            s["active"] += 1
            s["last_used"] = time.time()

    def end(self, up: Upstream, keep_alive: Optional[int]) -> None:
        s = self.sessions.get(up.session)
        if s:
            s["active"] = max(0, s["active"] - 1)
            s["last_used"] = time.time()
            if keep_alive:
                s["keep_alive"] = keep_alive

    async def reap_idle(self) -> None:
        now = time.time()
        for key, s in list(self.sessions.items()):
            if not s["owned"] or s["active"] or now - s["last_used"] <= s.get("keep_alive", 300):
                continue
            rec = registry().get(key[0])
            try:
                if rec:
                    await agent_call(rec, "DELETE", f"/v1/instances/{key[1]}")
                logger.info(f"Cluster: stopped idle {s['model']} on {key[0]}")
            except NodeUnreachable as e:
                logger.warning(f"Cluster: stopping idle instance on {key[0]} failed: {e}")
            self.sessions.pop(key, None)
            self._cache = (0.0, [])

    async def run_reaper(self, interval_s: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.reap_idle()
            except Exception as e:
                logger.warning(f"Cluster reaper: {e}")


SERVICE = ClusterService()
