"""Cluster: enroll other devices (node agents) as AI resources. Rules:
backend/cluster/enrollment.py. The whole /api/cluster/ surface sits behind the
dashboard access policy (this machine + the network owner's devices), so a new
device must already be one of the owner's overlay devices to even get the agent or
enroll -- and needs a one-time code from the dashboard on top."""
import hashlib
import logging
import os
import time

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict

from backend import network
import asyncio
import re

from backend.cluster.client import NodeUnreachable, agent_call, fetch_release, hf_meta
from backend.cluster.hashcache import HashCache
from backend.cluster.models import HfRef, ModelPlanError, plan_model
from backend.cluster.enrollment import EnrollError, EnrollmentBook, NodeRegistry, validate_enrollment
from backend.cluster.runtimes import NodeFacts, RuntimeError_, choose_runtime, parse_release
from backend.config import CLUSTER_NODES_FILE, MODEL_DIRS, _PROJECT_DIR

logger = logging.getLogger("ai-serve-ui.cluster")
router = APIRouter()

AGENT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "node_agent", "llama_serve_node.py")
REGISTRY = NodeRegistry(CLUSTER_NODES_FILE)
HASHES = HashCache(os.path.join(_PROJECT_DIR, "cluster_hashes.json"))
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf$")
BOOK = EnrollmentBook()


def _agent_bytes() -> bytes:
    with open(AGENT_FILE, "rb") as f:
        return f.read()


@router.post("/api/cluster/enrollments", status_code=201, summary="Create a one-time code to enroll a device")
async def create_enrollment(request: Request):
    e = BOOK.create()
    sha = hashlib.sha256(_agent_bytes()).hexdigest()
    host = request.headers.get("host", "")
    name = host.rsplit(":", 1)[0].strip("[]")
    if not host or name in ("localhost", "127.0.0.1", "::1"):
        # a loopback address would be useless on the other device
        overlay = network.active().hostname()
        from backend.config import GATEWAY_PORT
        host = f"{overlay}:{GATEWAY_PORT}" if overlay else f"<this machine's overlay address>:{GATEWAY_PORT}"
    base = f"http://{host}"
    verify = f"import hashlib; assert hashlib.sha256(open('agent.py','rb').read()).hexdigest() == '{sha}', 'checksum mismatch'"
    return {
        "code": e.code, "expires_at": e.expires_at, "agent_sha256": sha, "coordinator": base,
        "command": (f"curl -fsS -o agent.py {base}/api/cluster/agent.py && python3 -c \"{verify}\" && "
                    f"python3 agent.py --enroll {base} --code {e.code}"),
    }


@router.get("/api/cluster/agent.py", summary="The node agent (single file, stdlib only)")
async def agent_file():
    data = _agent_bytes()
    return Response(content=data, media_type="text/x-python",
                    headers={"X-Agent-SHA256": hashlib.sha256(data).hexdigest(),
                             "Content-Disposition": "attachment; filename=agent.py"})


@router.post("/api/cluster/enroll", status_code=201, summary="Called by a node agent with a one-time code")
async def enroll(request: Request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    peer = request.client.host if request.client else None
    if not BOOK.redeem(body.get("code") if isinstance(body, dict) else None):
        logger.warning(f"Enrollment with an invalid/used/expired code from {peer}")
        raise HTTPException(status_code=403, detail="invalid, used or expired enrollment code")
    net = network.active()
    try:
        rec = validate_enrollment(body, peer=peer, is_private=net.is_private, now=time.time())
    except EnrollError as e:
        raise HTTPException(status_code=400, detail=str(e))
    host = request.headers.get("host", "")
    if re.fullmatch(r"[A-Za-z0-9.\-\[\]:]{1,255}", host):
        from dataclasses import replace
        rec = replace(rec, coordinator_url=f"{request.url.scheme}://{host}")
    REGISTRY.add(rec)
    logger.info(f"Enrolled node {rec.id} at {rec.url}")
    return {"node_id": rec.id, "coordinator_login": net.owner()}


@router.get("/api/cluster/nodes", summary="Enrolled nodes (without their tokens)")
async def list_nodes():
    return {"nodes": REGISTRY.listing()}


@router.delete("/api/cluster/nodes/{node_id}", summary="Forget an enrolled node")
async def remove_node(node_id: str):
    if not REGISTRY.remove(node_id):
        raise HTTPException(status_code=404, detail="no such node")
    logger.info(f"Removed node {node_id}")
    return {"removed": node_id}


def _node(node_id: str):
    rec = REGISTRY.get(node_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="no such node")
    return rec


@router.get("/api/cluster/nodes/{node_id}", summary="Live info from a node's agent")
async def node_info(node_id: str):
    rec = _node(node_id)
    try:
        info = await agent_call(rec, "GET", "/v1/node")
        runtimes = await agent_call(rec, "GET", "/v1/runtimes")
        downloads = await agent_call(rec, "GET", "/v1/downloads")
    except NodeUnreachable as e:
        return {"id": rec.id, "url": rec.url, "online": False, "error": str(e)}
    return {"id": rec.id, "url": rec.url, "online": True, **info, "runtimes": runtimes, "downloads": downloads}


class RuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: Optional[str] = None
    tag: Optional[str] = None


@router.post("/api/cluster/nodes/{node_id}/runtime", status_code=202,
             summary="Have a node install the matching llama.cpp runtime")
async def install_runtime(node_id: str, req: RuntimeRequest):
    rec = _node(node_id)
    try:
        info = await agent_call(rec, "GET", "/v1/node")
    except NodeUnreachable as e:
        raise HTTPException(status_code=502, detail=str(e))
    f = info.get("facts") or {}
    cuda = f.get("cuda_driver")
    facts = NodeFacts(platform=str(f.get("platform") or ""), machine=str(f.get("machine") or ""),
                      gpu_vendors=tuple(f.get("gpu_vendors") or ()),
                      cuda_driver=tuple(cuda) if isinstance(cuda, list) and len(cuda) == 2 else None,
                      rocm=bool(f.get("rocm")), vulkan=bool(f.get("vulkan")))
    try:
        spec = choose_runtime(parse_release(await fetch_release(req.tag)), facts, req.variant)
    except RuntimeError_ as e:
        raise HTTPException(status_code=400, detail=str(e))
    body = {"tag": spec.tag, "variant": spec.variant,
            "files": [{"name": a.name, "url": a.url, "sha256": a.sha256, "size": a.size} for a in spec.files]}
    try:
        state = await agent_call(rec, "POST", "/v1/runtimes", body)
    except NodeUnreachable as e:
        raise HTTPException(status_code=502, detail=str(e))
    logger.info(f"Node {rec.id}: installing llama.cpp {spec.tag} {spec.variant} "
                f"({', '.join(a.name for a in spec.files)})")
    return {"node": rec.id, "tag": spec.tag, "variant": spec.variant,
            "files": [a.name for a in spec.files], "size_mb": round(sum(a.size for a in spec.files) / 1048576),
            "agent": state}


def _local_model(name: str):
    """A .gguf directly in one of the model dirs (symlinks the owner placed there are
    followed), by plain file name only."""
    if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
        return None
    for d in MODEL_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


@router.get("/api/cluster/models/{name}", summary="Download a local model (for node agents)")
async def serve_model(name: str):
    path = _local_model(name)
    if path is None:
        raise HTTPException(status_code=404, detail="no such model")
    return FileResponse(path, media_type="application/octet-stream", filename=name)


async def plan_for_local(rec, name: str, local_path: str):
    """Download plan for a model that lives on this machine (used by the background
    copy when the gateway routes to a device)."""
    sha = await asyncio.to_thread(HASHES.sha256, local_path)
    return plan_model(rec.coordinator_url or None, (name, os.path.getsize(local_path), sha), None, None)


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_file: Optional[str] = None
    revision: Optional[str] = None


async def _model_plan(req: "ModelRequest", coordinator_url: Optional[str]):
    hf = HfRef.parse(req.hf_repo, req.hf_file, req.revision) if (req.hf_repo or req.hf_file) else None
    name = req.model or (hf.name if hf else None)
    local = None
    if name:
        path = _local_model(name)
        if req.model and path is None:
            raise ModelPlanError(f"no local model {req.model!r}")
        if path:
            local = (name, os.path.getsize(path), None)
    meta = await hf_meta(hf) if hf else None
    if local and not hf:  # with Hugging Face as a source, its sha256 is authoritative
        local = (local[0], local[1], await asyncio.to_thread(HASHES.sha256, _local_model(local[0])))
    return plan_model(coordinator_url, local, hf, meta)


@router.post("/api/cluster/nodes/{node_id}/models", status_code=202,
             summary="Have a node download a model (from here and/or Hugging Face, whichever is faster)")
async def send_model(node_id: str, req: ModelRequest):
    rec = _node(node_id)
    try:
        plan = await _model_plan(req, rec.coordinator_url or None)
    except ModelPlanError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    try:
        state = await agent_call(rec, "POST", "/v1/models/download", plan)
    except NodeUnreachable as e:
        raise HTTPException(status_code=502, detail=str(e))
    logger.info(f"Node {rec.id}: downloading {plan['name']} ({plan['size'] >> 20} MB) from {plan['sources']}")
    return {"node": rec.id, **{k: plan[k] for k in ("name", "size", "sources")}, "agent": state}


@router.post("/api/cluster/models", status_code=202,
             summary="Send a model to the best device; a device that refuses the download is skipped")
async def place_model(req: ModelRequest):
    from backend.cluster.service import SERVICE
    try:
        # coordinator sources are rewritten per device by the service
        plan = await _model_plan(req, "http://coordinator")
    except ModelPlanError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"name": plan["name"], **await SERVICE.place_model(plan)}


@router.delete("/api/cluster/nodes/{node_id}/downloads/{name}",
               summary="Delete a model the node downloaded (only on the user's explicit request)")
async def delete_download(node_id: str, name: str):
    rec = _node(node_id)
    if not _MODEL_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="not a model file name")
    try:
        result = await agent_call(rec, "DELETE", f"/v1/downloads/{name}")
    except NodeUnreachable as e:
        raise HTTPException(status_code=409 if "409" in str(e) else 502, detail=str(e))
    logger.info(f"Node {rec.id}: deleted download {name} (user request)")
    return result


@router.get("/api/cluster/copies", summary="Models being copied to devices by the gateway (with fallback)")
async def copies():
    from backend.cluster.service import SERVICE
    return {"copies": {k: {kk: vv for kk, vv in v.items() if kk != "task"} for k, v in SERVICE.copies.items()},
            "sessions": [{"node": n, "instance": i, **s} for (n, i), s in SERVICE.sessions.items()]}


@router.put("/api/cluster/nodes/{node_id}/budgets",
            summary='Memory budget per accelerator: {"<index>": MB or null (default: half the RAM on shared memory)}')
async def set_budgets(node_id: str, budgets: Dict[str, Optional[int]]):
    rec = _node(node_id)
    try:
        result = await agent_call(rec, "PUT", "/v1/budgets", budgets)
    except NodeUnreachable as e:
        raise HTTPException(status_code=400 if "answered 400" in str(e) else 502, detail=str(e))
    from backend.cluster.service import SERVICE
    SERVICE._cache = (0.0, [])
    logger.info(f"Node {rec.id}: budgets set to {budgets}")
    return result
