"""Instance management, model listing, GPU status, logs, compare."""
import asyncio
import glob
import logging
import platform
import re
import shutil
import signal
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import DEFAULT_PARALLEL, LLAMA_SERVER_BIN, LLAMA_TOKENIZE_BIN, LOGS_DIR, MODEL_DIRS
from backend.model_manager import (
    get_model_size_mb,
    check_model_file_exists,
    fetch_gpu_status,
    find_active_model_port,
    find_draft_model_path,
    find_gguf_by_name,
    find_mmproj_path,
    get_bundled_flags,
    get_context_length,
    get_gpu_runtime_for_name,
    get_llamacpp_version,
    get_pid_on_port,
    get_upstream_error_detail,
    is_port_busy,
    kill_process_on_port,
    reserve_free_port,
    resolve_model_path,
    scan_all_gguf_paths,
    stop_server_by_port,
)
from backend.response_processing import (
    gguf_has_bundled_mtp,
    is_loadable_gguf,
    is_mtp_draft,
    read_gguf_header,
)
from backend.plugins import llm as llm_plugins
from backend.plugins import registry as plugin_registry
from backend.state import state
from backend.version import BACKEND_VERSION, VERSION_LOG

logger = logging.getLogger("ai-serve-ui.dashboard")
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    model_path: str = Field(..., description="Absolute path to the .gguf model file.")
    port: Optional[int] = Field(
        None,
        description="Port to run this llama-server instance on. If omitted, a free port is "
                     "picked automatically. If set to a port that's already running a managed "
                     "instance, that instance is stopped and replaced -- unless it has requests "
                     "actively in flight, in which case the call is refused with 409 (see `force`).",
    )
    force: bool = Field(
        False,
        description="If `port` names an existing managed instance with requests currently in "
                     "flight, replace it anyway instead of refusing with 409. Concurrent callers "
                     "sharing this gateway (e.g. two independent clients that both default to the "
                     "same port) can otherwise silently kill each other's in-flight work.",
    )
    ignore_strata: bool = Field(
        False,
        description="Skip the GPU manager's check entirely (escape hatch; the old name is kept for callers). "
                     "By default a load is refused with 507 while a gpu-exclusive plugin holds the GPUs.",
    )
    ctx_size: int = Field(32000, description="Context size in tokens (-c).")
    threads: int = Field(8, description="CPU thread count (-t).")
    n_gpu_layers: int = Field(-1, description="Layers to offload to GPU (-ngl). -1 offloads all layers.")
    gpu_id: Optional[int] = Field(
        None,
        description="GPU index to pin this instance to via CUDA_VISIBLE_DEVICES. "
                     "If omitted, the GPU with the most free VRAM is picked automatically.",
    )
    mmproj_path: Optional[str] = Field(
        None,
        description="Absolute path to a vision projector (mmproj) .gguf file to enable multimodal input. "
                     "Must be an absolute path — a bare filename will not resolve and the server will fail to start.",
    )
    parallel: Optional[int] = Field(
        None,
        description="Parallel slots (-np). Omitted = LLAMA_DEFAULT_PARALLEL (1): with more slots "
                     "llama.cpp splits/reserves for concurrent requests, which cost single-stream "
                     "speed on every model measured (up to ~2x for a model that nearly fills the "
                     "card). Set >1 only if you really send concurrent requests to this instance; "
                     "0 = llama.cpp's own auto choice (4).",
    )
    ubatch_size: Optional[int] = Field(
        None,
        description="Physical batch size (-ub). llama.cpp's own default (512) is a hard floor for "
                     "vision input: mtmd_helper_decode_image_chunk asserts n_ubatch >= the image's "
                     "token count, and llama-server crashes outright (not a graceful error) the moment "
                     "an image needs more tokens than that. Bump this (1024+) for any mmproj_path load "
                     "that will see real images, not just small icons/crops.",
    )
    batch_size: Optional[int] = Field(
        None,
        description="Logical batch size (-b). Raise alongside ubatch_size when the vision projector's "
                     "prompt processing pass needs it — llama.cpp requires n_batch >= n_ubatch.",
    )
    embedding: bool = Field(False, description="Enable embedding pooling mode (--embedding). Only for dedicated embedding models — leave off for normal chat/completion models.")
    pooling: Optional[str] = Field(
        None,
        description="Explicit --pooling override (none/mean/cls/last/rank). Leave unset to trust the "
                     "model's own GGUF metadata. Set explicitly for embedding models whose metadata is "
                     "unreliable — e.g. bge-m3 needs 'cls'; some quantized bge-m3 GGUFs omit pooling "
                     "metadata entirely, which silently resolves to 'none' and makes /v1/embeddings fail "
                     "outright with 'Pooling type none is not OAI compatible'.",
    )
    reranking: bool = Field(
        False,
        description="Enable the reranking endpoint (--reranking) for a dedicated cross-encoder reranker "
                     "model (e.g. bge-reranker-v2-m3) — not the same model as an embedding model like "
                     "bge-m3, which only produces vectors, not relevance scores. Implies embedding=true.",
    )
    jinja: bool = Field(True, description="Enable Jinja chat template rendering (--jinja).")
    draft_model_path: Optional[str] = Field(
        None,
        description="Absolute path to a draft model .gguf file for MTP speculative decoding. "
                     "If omitted and mtp is not false, a sidecar draft model is auto-detected next to model_path.",
    )
    mtp: Optional[bool] = Field(
        None,
        description="Set to false to disable MTP speculative decoding auto-detection "
                     "(draft sidecar lookup and bundled-nextn-tensor detection).",
    )


class CompareRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful assistant."
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7


class ModelInfo(BaseModel):
    name: str
    path: str
    mtp_draft_path: Optional[str] = None
    mtp_bundled: bool = False
    mmproj_path: Optional[str] = None
    vision_bundled: bool = False
    context_length: Optional[int] = None


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    vision_modules: List[ModelInfo]
    draft_modules: List[ModelInfo] = []


class GPUInfo(BaseModel):
    index: int
    name: str
    total: int
    used: int
    free: int
    usage_percent: float
    driver_version: Optional[str] = None
    # Apple Silicon only: where `used` comes from (ioreg_gpu | psutil_system_ram) and Metal's GPU limit
    memory_source: Optional[str] = None
    gpu_limit_mb: Optional[int] = None


class GPUStatusResponse(BaseModel):
    gpus: List[GPUInfo]
    error: Optional[str] = None


class InstanceStatus(BaseModel):
    # Optional because in-process services with no subprocess/port (OmniVoice)
    # are surfaced here too, the same way ComfyUI already is.
    port: Optional[int]
    running: bool
    ready: bool
    model: str
    pid: Optional[int]
    health: Dict[str, Any]
    n_gpu_layers: int
    gpu_id: Optional[int]
    ctx_size: Optional[int] = None
    embedding: Optional[bool] = None
    reranking: Optional[bool] = None
    pooling: Optional[str] = None
    draft_model: Optional[str] = None
    mtp_active: Optional[bool] = None
    active_requests: int = 0
    generating: bool = False
    tps: Optional[float] = None
    mmproj_path: Optional[str] = None
    # llama-server -np the instance was started with (None = llama.cpp's own auto
    # choice, or an instance adopted after a backend restart).
    parallel: Optional[int] = None
    # Set on the four port-less TTS engine entries (OmniVoice/Qwen3-TTS/CosyVoice/
    # Chatterbox) and on ComfyUI (frees VRAM, keeps the process) -- which endpoint the frontend's generic Stop & Unload button should
    # POST to, since none of them have a port to go through /api/stop with. See the
    # "TTS engines" section of CLAUDE.md for why this exists (CosyVoice/Chatterbox
    # previously had no working Stop button at all).
    unload_endpoint: Optional[str] = None
    plugin: Optional[str] = None   # the plugin (docs/plugins.md) this entry belongs to


class StatusResponse(BaseModel):
    instances: List[InstanceStatus]


class IndividualResponse(BaseModel):
    port: int
    model: str
    content: str
    time_taken: Optional[float] = None
    tokens: Optional[int] = None
    tps: Optional[float] = None
    status: str


class CompareResponse(BaseModel):
    responses: List[IndividualResponse]


class ActiveModelInfo(BaseModel):
    port: int
    model: str
    gpu_id: Optional[int]
    active_requests: int
    last_used: float
    keep_alive: int
    keep_alive_remaining: float
    ctx_size: Optional[int] = None
    draft_model: Optional[str] = None
    mtp_active: Optional[bool] = None


class ActiveModelsResponse(BaseModel):
    active_models: List[ActiveModelInfo]


class UnloadModelRequest(BaseModel):
    port: Optional[int] = None
    model_name: Optional[str] = None
    force: bool = Field(
        False,
        description="Unload even if the target instance has requests currently in flight. "
                     "Without this, a busy instance is skipped (409 for a single `port`; "
                     "silently skipped, and reported, for a `model_name` match).",
    )


class TokenizeRequest(BaseModel):
    content: str
    model_name: Optional[str] = Field(
        None,
        description="Which model's tokenizer to use, matched the same fuzzy way as "
                     "find_gguf_by_name (bare filename, stem, or substring). If a "
                     "llama-server instance for it is already running, tokenizes through "
                     "that instance's own /tokenize (exact match to what it'll actually see). "
                     "Otherwise falls back to the llama-tokenize CLI directly against the GGUF "
                     "file -- vocab-only, no VRAM/full-weight load -- so a caller can get an "
                     "exact count before deciding whether a model is worth loading at all. "
                     "Omitted entirely: uses whichever model is already running, if any.",
    )


class TokenizeResponse(BaseModel):
    success: bool
    tokens_count: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/models", summary="List Available GGUF Models", response_model=ModelsResponse)
async def list_models():
    models: list = []
    vision_modules: list = []
    draft_modules: list = []
    seen_paths: set = set()
    all_paths: list = []

    for directory in MODEL_DIRS:
        if not os.path.isdir(directory):
            continue
        files = glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True)
        for f in files:
            name = os.path.basename(f)
            abs_path = os.path.abspath(f)
            if abs_path in seen_paths:
                continue
            all_paths.append(abs_path)
            if "mmproj" in name.lower():
                vision_modules.append({"name": name, "path": abs_path})
            elif is_mtp_draft(name):
                draft_modules.append({"name": name, "path": abs_path})
            elif is_loadable_gguf(name):
                models.append({"name": name, "path": abs_path})
            seen_paths.add(abs_path)

    # Reuse the directory scan just done above instead of letting each matcher below
    # re-walk MODEL_DIRS per model — that O(models x dirs) re-globbing plus a full
    # header decode per model (including whatever the tokenizer vocab array turns out
    # to be) is what made this endpoint take 17s+, blocking the whole event loop for
    # that entire time since it's all synchronous file I/O. Cached below by (mtime,
    # size), and run off the event loop so even a cold cache doesn't stall other
    # requests (gateway chat, GPU polling, etc).
    def _enrich(m: dict) -> None:
        m["mtp_draft_path"] = find_draft_model_path(m["path"], all_paths)
        m["mmproj_path"] = find_mmproj_path(m["path"], all_paths)
        mtp_bundled = vision_bundled = False
        if m["mtp_draft_path"] is None or m["mmproj_path"] is None:
            mtp_bundled, vision_bundled = get_bundled_flags(m["path"])
        m["mtp_bundled"] = m["mtp_draft_path"] is None and mtp_bundled
        m["vision_bundled"] = m["mmproj_path"] is None and vision_bundled
        m["context_length"] = get_context_length(m["path"])

    def _enrich_all() -> None:
        for m in models:
            _enrich(m)

    await asyncio.get_running_loop().run_in_executor(None, _enrich_all)

    return {"models": models, "vision_modules": vision_modules, "draft_modules": draft_modules}


@router.get("/api/models/header", summary="Read a GGUF model's full metadata header")
async def get_model_header(path: Optional[str] = None, model_name: Optional[str] = None):
    if path:
        abs_path = os.path.abspath(path)
    elif model_name:
        abs_path = find_gguf_by_name(model_name)
        if abs_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"No GGUF file matching '{model_name}' found in configured model directories.",
            )
    else:
        raise HTTPException(status_code=400, detail="Provide either 'path' or 'model_name'.")

    allowed = any(
        os.path.isdir(directory) and abs_path.startswith(os.path.abspath(directory) + os.sep)
        for directory in MODEL_DIRS
    )
    if not allowed or not abs_path.lower().endswith(".gguf") or not os.path.isfile(abs_path):
        raise HTTPException(
            status_code=400,
            detail="Path must point to an existing .gguf file inside a configured model directory.",
        )
    try:
        return read_gguf_header(abs_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read GGUF header: {e}")


@router.get("/api/gpu/status", summary="Get NVIDIA GPU Metrics", response_model=GPUStatusResponse)
async def get_gpu_status():
    return await fetch_gpu_status()


@router.get("/api/docs/raw", summary="Get raw API documentation in Markdown format")
async def get_raw_api_docs():
    doc_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "API.md")
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load API documentation: {e}")


async def _start_plugin_via_api(req: "StartRequest", found) -> Dict[str, Any]:
    """/api/start with a model an llm plugin serves (e.g. Strata through its adapter): the plugin loads it and this
    answers once it serves -- a server's /health may answer long before the model is loaded, and a caller that polls
    right after /api/start (llm-benchy does) would start too early. The model is served on the **plugin's** port,
    not req.port: callers use the `port` of this answer. llama-server-only fields (gpu_id, mmproj, mtp, ubatch,
    parallel ...) don't apply."""
    p, entry = found
    ctx = entry.get("context_length")
    if req.ctx_size and ctx and req.ctx_size > ctx:
        raise HTTPException(status_code=400, detail=(
            f"{entry['id']} serves at most {ctx} tokens of context; requested ctx_size {req.ctx_size}."))
    st = await plugin_registry.status(p)
    was_ready = bool(st and st["ready"])
    try:
        await llm_plugins.load(p, entry["id"])
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "already running" if was_ready else "started", "port": p.port, "engine": p.id,
            "model": entry["id"], "max_context": ctx, "requested_port": req.port}


@router.post("/api/start", summary="Start a new llama-server instance")
async def start_server(req: StartRequest):
    # Validate before touching any port so a bad request never leaves a
    # reservation placeholder (below) dangling in state.managed_processes. Shared with
    # get_or_load_model()'s own preflight (see check_model_file_exists's own docstring
    # for why a dangling symlink -- e.g. a model relocated onto an external drive that
    # isn't currently mounted -- needs a distinct message from a plain missing path.
    found = await llm_plugins.find(req.model_path)
    if found is not None:
        return await _start_plugin_via_api(req, found)
    check_model_file_exists(req.model_path, "Model file")
    if req.mmproj_path:
        check_model_file_exists(req.mmproj_path, "Vision module (mmproj) file")

    # port omitted -> auto-pick and atomically reserve a free one (guards against
    # two concurrent /api/start calls both defaulting to the same port and each
    # killing the other's freshly-started instance). port explicitly given ->
    # preserve the existing "replace whatever's on this port" semantics.
    auto_picked = req.port is None
    if auto_picked:
        port = await reserve_free_port()
    else:
        port = req.port
        if port in state.managed_processes:
            old_info = state.managed_processes[port]
            await _raise_if_busy(port, old_info, req.force)
            old_proc = old_info.get("process")
            if old_proc and old_proc.poll() is None:
                old_proc.terminate()
                old_proc.wait()

        kill_process_on_port(port)

    draft_model_path = req.draft_model_path
    bundled_mtp = False
    if draft_model_path is None and req.mtp is not False:
        draft_model_path = find_draft_model_path(req.model_path)
        if draft_model_path is None:
            bundled_mtp = gguf_has_bundled_mtp(req.model_path)

    # --reranking implies --embedding on llama-server's own side (common/arg.cpp
    # forces params.embedding=true when --reranking is passed) -- mirrored here so
    # our own state bookkeeping (the "Mode: Embedding" badge, /api/status) stays
    # truthful for reranker instances too, rather than relying on the C++ side effect.
    embedding = req.embedding or req.reranking
    reranking = req.reranking
    if (draft_model_path or bundled_mtp) and embedding:
        logger.warning(
            "MTP and --embedding are incompatible (crashes llama-server's pooling graph); "
            "disabling embedding mode since MTP speculative decoding was requested."
        )
        embedding = False
        reranking = False

    target_gpu_id = req.gpu_id
    if target_gpu_id is None and sys.platform != "darwin":
        try:
            gpu_status = await fetch_gpu_status()
            if gpu_status.get("gpus"):
                target_gpu_id = sorted(gpu_status["gpus"], key=lambda g: g["free"], reverse=True)[0]["index"]
            else:
                target_gpu_id = 0
        except Exception:
            target_gpu_id = 0

    # Central GPU manager: reserve the memory (unloading idle consumers if needed; fast
    # no otherwise -- callers retry). 507 while a gpu-exclusive plugin holds the GPUs. ignore_strata
    # keeps its old meaning: skip the check entirely (escape hatch).
    reservation = None
    if not req.ignore_strata and sys.platform != "darwin" and target_gpu_id is not None:
        from backend.cluster.local import MANAGER
        from backend.response_processing import estimate_kv_overhead_mb
        need = int(get_model_size_mb(req.model_path) + estimate_kv_overhead_mb(req.ctx_size))
        try:
            reservation, _, why = await MANAGER.reserve(target_gpu_id, need,
                                                        f"llm:{os.path.basename(req.model_path)}")
        except HTTPException:
            if state.managed_processes.get(port, {}).get("reserving"):
                state.managed_processes.pop(port, None)
            raise
        if reservation is None:
            if state.managed_processes.get(port, {}).get("reserving"):
                state.managed_processes.pop(port, None)
            raise HTTPException(status_code=507, detail=f"No room for {os.path.basename(req.model_path)}: {why}")

    from backend.model_manager import llama_env
    env = llama_env()
    if target_gpu_id is not None and sys.platform != "darwin":
        # Always pin to a single GPU: leaving CUDA_VISIBLE_DEVICES unset lets llama.cpp
        # see every GPU and auto tensor-split -ngl -1 across all of them, which is a
        # known slow/hanging failure mode for some models rather than a hard OOM.
        env["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
        logger.info(f"CUDA_VISIBLE_DEVICES={target_gpu_id} for port {port}")

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", req.model_path,
        "--port", str(port),
        "-c", str(req.ctx_size),
        "-t", str(req.threads),
        "-ngl", str(req.n_gpu_layers),
        "--host", "127.0.0.1",
    ]
    if embedding:
        cmd.append("--embedding")
    if reranking:
        cmd.append("--reranking")
    if req.pooling:
        cmd.extend(["--pooling", req.pooling])
    if req.jinja:
        cmd.append("--jinja")
    if req.mmproj_path:
        cmd.extend(["--mmproj", req.mmproj_path])
        logger.info(f"Enabling vision module: {req.mmproj_path}")
    parallel = req.parallel if req.parallel is not None else DEFAULT_PARALLEL
    parallel = parallel if parallel > 0 else None
    if parallel:
        cmd.extend(["-np", str(parallel)])
    if req.ubatch_size:
        cmd.extend(["-ub", str(req.ubatch_size)])
    if req.batch_size:
        cmd.extend(["-b", str(req.batch_size)])
    if draft_model_path:
        cmd.extend(["-md", draft_model_path, "--spec-type", "draft-mtp"])
        logger.info(f"Enabling MTP speculative decoding with draft head: {draft_model_path}")
    elif bundled_mtp:
        cmd.extend(["--spec-type", "draft-mtp"])
        logger.info(f"Enabling MTP speculative decoding using bundled nextn tensors in: {req.model_path}")

    try:
        log_file_path = os.path.join(LOGS_DIR, f"llama-server-{port}.log")
        log_file = open(log_file_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
        if reservation is not None:
            from backend.cluster.local import MANAGER
            MANAGER.confirm(reservation.id, proc.pid)  # counts until it has allocated, or exits
        state.managed_processes[port] = {
            "process": proc,
            "log_file": log_file,
            "model": req.model_path,
            "port": port,
            "ctx_size": req.ctx_size,
            "threads": req.threads,
            "n_gpu_layers": req.n_gpu_layers,
            "gpu_id": target_gpu_id,
            "embedding": embedding,
            "reranking": reranking,
            "pooling": req.pooling,
            "draft_model": draft_model_path,
            "mtp_active": bool(draft_model_path) or bundled_mtp,
            "mmproj_path": req.mmproj_path,
            "parallel": parallel,
            "ubatch_size": req.ubatch_size,
            "batch_size": req.batch_size,
            "last_used": time.time(),
            "active_requests": 0,
            "keep_alive": 300,
            "keep_alive_task": None,
        }
        return {"status": "started", "pid": proc.pid, "port": port}
    except Exception as e:
        if auto_picked and state.managed_processes.get(port, {}).get("reserving"):
            state.managed_processes.pop(port, None)
        logger.error(f"Failed to start llama-server on port {port}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _raise_if_busy(port: int, info: Dict[str, Any], force: bool):
    """Refuse to kill an instance that's actually busy (see model_manager.is_port_busy).

    Mirrors the busy-guard the TTS engines' own unload endpoints already use
    (409 while generating) and the ctx_size-escalation path in
    get_or_load_model() already requires (active_requests == 0 before an
    evict-and-restart) -- stop_server_by_port() itself has no such check, so
    every caller that can reach it manually (this endpoint, /api/models/unload,
    /api/models/unload_all, /api/start's port-replace path) needs its own.
    """
    if not force and await is_port_busy(port, info):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Port {port} is running '{info.get('model')}' and is busy "
                f"({info.get('active_requests', 0)} request(s) tracked via this app's own "
                "gateway, plus a live /slots check for traffic that bypasses it); refusing "
                "to stop it. Wait for it to go idle, or pass force=true to stop it anyway."
            ),
        )


async def _release_plugin(p, force: bool) -> None:
    """Unload what an llm plugin holds (for an exclusive one like Strata: stop its server). 409 while busy."""
    try:
        await plugin_registry.release(p.id, force=force)
    except PermissionError as e:
        raise HTTPException(status_code=409, detail=f"{e} -- refusing to unload it; retry later or pass force=true")
    except (KeyError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=f"{p.name}: release failed ({e})")


def _gpu_plugin_by_port(port: int):
    """A GPU plugin listening on `port` (a model server like Strata's adapter, image-serve ...): stop/unload by port
    goes to its release (callers like llm-benchy address it by port)."""
    return next((p for p in plugin_registry._all_gpu_plugins() if p.port == port), None)


@router.post("/api/stop", summary="Stop a managed llama-server instance")
async def stop_server(port: int, force: bool = False):
    plugin = _gpu_plugin_by_port(port)
    if plugin is not None:
        await _release_plugin(plugin, force)
        return {"status": "stopped", "port": port}
    info = state.managed_processes.get(port)
    if info is not None:
        await _raise_if_busy(port, info, force)
    await stop_server_by_port(port)
    return {"status": "stopped", "port": port}

@router.get("/terminate", summary="terminate the server")
async def terminate():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "terminating"}

@router.get("/api/version", summary="Get backend performance/capability version, sub-component versions, and host system info")
async def get_version():
    # Apple Silicon's RAM doubles as GPU memory (fetch_gpu_status reports it as a "GPU"
    # below too) — a discrete-GPU box's system RAM and VRAM are separate pools with very
    # different bandwidth, which is the distinction that actually matters for inference.
    is_unified = platform.system() == "Darwin" and "arm64" in platform.machine()

    try:
        gpu_status = await fetch_gpu_status()
        gpus = gpu_status.get("gpus", [])
        # Sniffed per-GPU (not once globally) so a mixed-vendor box — e.g. one NVIDIA +
        # one AMD card — reports the correct compute stack for each card individually.
        for g in gpus:
            g["runtime"] = get_gpu_runtime_for_name(g.get("name", ""))
    except Exception:
        gpus = []

    return {
        "version": BACKEND_VERSION,
        "changed": VERSION_LOG[-1]["change"],
        "date": VERSION_LOG[-1]["date"],
        "log": list(reversed(VERSION_LOG)),
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "ram_architecture": "unified" if is_unified else "discrete",
        "gpus": gpus,
        "components": {
            "llamacpp": get_llamacpp_version(),
        },
    }

@router.get("/api/status", summary="Get status of all managed instances", response_model=StatusResponse)
async def get_status():
    results = []
    # Iterate a snapshot, not the live dict -- entries get pruned (popped) below as
    # dead ones are found, and mutating a dict while iterating over it raises.
    for port, info in list(state.managed_processes.items()):
        proc = info.get("process")
        running = proc is not None and proc.poll() is None
        if proc is not None and not running:
            # The process exited on its own (crash, OOM-kill, manually killed outside
            # this app, ...) rather than via our own stop_server_by_port() -- nothing
            # else prunes state.managed_processes for that case, so without this the
            # entry lingers here forever: it uses no VRAM anymore (the process is
            # actually gone), but keeps showing up in the Servers & GPUs tab as a
            # permanently "Starting" card (ready is always False for it) until someone
            # manually clicks Stop & Unload, which just force-pops it regardless of
            # whether the process is even still alive. Also cancel its keep_alive_task
            # here, not just pop the dict entry -- otherwise that task fires later,
            # finds no entry in managed_processes, and falls through to
            # kill_process_on_port(port), which could kill an unrelated process that's
            # since started listening on the same now-reused port number.
            if info.get("keep_alive_task"):
                info["keep_alive_task"].cancel()
            log_file = info.get("log_file")
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
            state.managed_processes.pop(port, None)
            logger.info(f"Pruned dead instance on port {port} (model {info.get('model')}) from managed_processes.")
            continue
        ready = False
        health_details: Dict[str, Any] = {}
        if running:
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        ready = True
                        health_details = resp.json()
            except Exception:
                pass

        # active_requests only counts calls this app itself mediated through the
        # gateway (/v1/chat/completions, /api/chat/completions) -- a client hitting
        # the instance's own port directly (bypassing this app's gateway entirely)
        # is invisible to that counter, so a genuinely busy instance could still
        # show "Ready". Polling the instance's own /slots gives ground truth
        # (is_processing per slot) regardless of who's actually talking to it.
        # Live TPS is derived by diffing n_decoded for the same slot/task between
        # successive polls -- /slots itself has no elapsed-time or tokens/sec field.
        generating = info.get("active_requests", 0) > 0
        tps: Optional[float] = None
        if ready:
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    slots_resp = await client.get(f"http://127.0.0.1:{port}/slots")
                    if slots_resp.status_code == 200:
                        now = time.time()
                        prev_samples = info.get("_slot_samples", {})
                        new_samples: Dict[int, Dict[str, Any]] = {}
                        tps_sum = 0.0
                        tps_found = False
                        for slot in slots_resp.json():
                            slot_id = slot.get("id")
                            is_processing = bool(slot.get("is_processing"))
                            if is_processing:
                                generating = True
                            next_token = (slot.get("next_token") or [{}])[0]
                            n_decoded = next_token.get("n_decoded", 0)
                            task_id = slot.get("id_task")
                            prev = prev_samples.get(slot_id)
                            if (
                                is_processing
                                and prev
                                and prev.get("task_id") == task_id
                                and n_decoded > prev.get("n_decoded", 0)
                            ):
                                dt = now - prev["ts"]
                                if dt > 0:
                                    tps_sum += (n_decoded - prev["n_decoded"]) / dt
                                    tps_found = True
                            new_samples[slot_id] = {"n_decoded": n_decoded, "task_id": task_id, "ts": now}
                        info["_slot_samples"] = new_samples
                        if tps_found:
                            tps = round(tps_sum, 2)
            except Exception:
                pass

        results.append({
            "port": port,
            "running": running,
            "ready": ready,
            "model": info["model"],
            "pid": proc.pid if running else None,
            "health": health_details,
            "n_gpu_layers": info.get("n_gpu_layers", -1),
            "gpu_id": info.get("gpu_id"),
            "ctx_size": info.get("ctx_size"),
            "embedding": info.get("embedding", False),
            "reranking": info.get("reranking", False),
            "pooling": info.get("pooling"),
            "draft_model": info.get("draft_model"),
            "mtp_active": info.get("mtp_active", bool(info.get("draft_model"))),
            # Same counter the OpenAI gateway increments/decrements around each chat/completions
            # call (see gateway.py) -- already used by /api/models/active for the same purpose,
            # just not previously surfaced here for the Servers & GPU tab's own instance cards.
            "active_requests": info.get("active_requests", 0),
            # Ground-truth liveness from the instance's own /slots -- true whenever
            # any slot is actually decoding, regardless of whether the request came
            # through this app's gateway or hit the instance's port directly.
            "generating": generating,
            "tps": tps,
            "mmproj_path": info.get("mmproj_path"),
            "parallel": info.get("parallel"),
        })

    # Workers of GPU plugins (speech, transcription, other model servers ...; backend/plugins/registry.py): listed
    # while loaded; an llm plugin's port is where its OpenAI server answers, others have none (the plugin owns its
    # processes). Unloaded through the plugin's release.
    from backend.plugins import registry
    for _p in registry._all_gpu_plugins():
        _st = await registry.status(_p)
        for _w in (_st or {}).get("workers", []):
            if not _w["loaded"]:
                continue
            results.append({
                "port": _p.port if "llm" in _p.manifest.provides else None, "running": True,
                "ready": bool(_st.get("ready")),
                "model": _w["label"], "plugin": _p.id, "pid": _w["pid"],
                "health": {"status": "busy" if _w["busy"] else "idle"},
                "n_gpu_layers": -1, "gpu_id": _w["gpu"], "ctx_size": None,
                "embedding": False, "reranking": False, "pooling": None, "draft_model": None, "mtp_active": False,
                "active_requests": 1 if _w["busy"] else 0, "generating": _w["busy"], "tps": None,
                "unload_endpoint": f"/api/plugins/{_p.id}/release?worker={_w['id']}",
            })

    return {"instances": results}


_TOKENIZE_COUNT_RE = re.compile(r"Total number of tokens:\s*(\d+)")


async def _tokenize_via_cli(model_path: str, content: str) -> dict:
    """Exact token count via the llama-tokenize CLI directly against a GGUF file --
    vocab-only, no VRAM/full-weight load (confirmed live: ~1s against a 27B model,
    dominated by mmap/file-open, not compute). Fallback path for /api/tokenize when no
    llama-server instance for the requested model is currently running."""
    if not os.path.isfile(LLAMA_TOKENIZE_BIN) and not shutil.which(LLAMA_TOKENIZE_BIN):
        return {"success": False, "tokens_count": 0,
                "error": f"llama-tokenize not found at '{LLAMA_TOKENIZE_BIN}' (set LLAMA_TOKENIZE_BIN)."}
    try:
        proc = await asyncio.create_subprocess_exec(
            LLAMA_TOKENIZE_BIN, "-m", model_path, "--stdin", "--show-count", "--no-parse-special",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=content.encode("utf-8")), timeout=30.0)
        match = _TOKENIZE_COUNT_RE.search(stdout.decode("utf-8", "ignore"))
        if match:
            return {"success": True, "tokens_count": int(match.group(1))}
        return {"success": False, "tokens_count": 0,
                "error": f"llama-tokenize produced no parseable count: {stderr.decode('utf-8', 'ignore')[:300]}"}
    except asyncio.TimeoutError:
        return {"success": False, "tokens_count": 0, "error": "llama-tokenize timed out after 30s"}
    except Exception as e:
        return {"success": False, "tokens_count": 0, "error": f"llama-tokenize failed: {e}"}


@router.post("/api/tokenize", summary="Tokenize text using the active model", response_model=TokenizeResponse)
async def tokenize_text(req: TokenizeRequest):
    port = find_active_model_port(req.model_name)
    if port:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"http://127.0.0.1:{port}/tokenize", json={"content": req.content})
                if resp.status_code == 200:
                    return {"success": True, "tokens_count": len(resp.json().get("tokens", []))}
                return {"success": False, "tokens_count": 0,
                        "error": f"Upstream tokenize returned {resp.status_code}"}
        except Exception as e:
            logger.error(f"Error querying tokenize on port {port}: {e}")
            return {"success": False, "tokens_count": 0, "error": str(e)}

    # No running instance for this model -- fall back to the llama-tokenize CLI directly
    # against the GGUF (tokenizer-only, doesn't load the model into VRAM). Only possible
    # if the caller told us *which* model via model_name; with nothing running and no
    # name given there's nothing to tokenize against.
    if req.model_name:
        model_path = find_gguf_by_name(req.model_name)
        if model_path:
            return await _tokenize_via_cli(model_path, req.content)
        return {"success": False, "tokens_count": 0,
                "error": f"No running instance and no GGUF file matching model_name='{req.model_name}' "
                         f"found under configured model directories."}

    available = sorted({os.path.basename(p) for p in scan_all_gguf_paths()})[:20]
    return {"success": False, "tokens_count": 0,
            "error": "No active model server found, and no model_name given to tokenize against "
                     "directly. Pass model_name (any of: " + ", ".join(available) +
                     (", ..." if len(available) == 20 else "") + ")."}


@router.get("/api/models/active", summary="Get active/loaded models", response_model=ActiveModelsResponse)
async def get_active_models():
    now = time.time()
    active_list = []
    for port, info in state.managed_processes.items():
        proc = info.get("process")
        if proc and proc.poll() is None:
            ka = info.get("keep_alive", 300)
            remaining = max(0.0, info.get("last_used", now) + ka - now)
            active_list.append({
                "port": port,
                "model": info["model"],
                "gpu_id": info.get("gpu_id"),
                "active_requests": info.get("active_requests", 0),
                "last_used": info.get("last_used", 0.0),
                "keep_alive": ka,
                "keep_alive_remaining": round(remaining, 1),
                "ctx_size": info.get("ctx_size"),
                "draft_model": info.get("draft_model"),
                "mtp_active": info.get("mtp_active", bool(info.get("draft_model"))),
            })

    return {"active_models": active_list}


@router.post("/api/models/unload", summary="Manually unload a model")
async def unload_model(req: UnloadModelRequest):
    if req.port is not None:
        plugin = _gpu_plugin_by_port(req.port)
        if plugin is not None:
            await _release_plugin(plugin, req.force)
            return {"status": "success", "message": f"{plugin.name} on port {req.port} unloaded."}
        if req.port in state.managed_processes:
            await _raise_if_busy(req.port, state.managed_processes[req.port], req.force)
            await stop_server_by_port(req.port)
            return {"status": "success", "message": f"Model on port {req.port} unloaded."}
        else:
            kill_process_on_port(req.port)
            return {"status": "success", "message": f"Killed any process on port {req.port}."}
    found = await llm_plugins.find(req.model_name) if req.model_name is not None else None
    if found is not None:
        await _release_plugin(found[0], req.force)
        return {"status": "success", "message": f"{found[1]['id']} unloaded ({found[0].name})."}
    if req.model_name is not None:
        unloaded_ports = []
        skipped_busy = []
        for port, info in list(state.managed_processes.items()):
            if req.model_name.lower() in info["model"].lower():
                if not req.force and await is_port_busy(port, info):
                    skipped_busy.append(port)
                    continue
                await stop_server_by_port(port)
                unloaded_ports.append(port)
        if not unloaded_ports and not skipped_busy:
            raise HTTPException(status_code=404, detail=f"No active model matching '{req.model_name}'.")
        msg = f"Unloaded {len(unloaded_ports)} model(s) matching '{req.model_name}'."
        if skipped_busy:
            msg += f" Skipped {len(skipped_busy)} with requests in flight (port(s) {skipped_busy}) -- pass force=true to unload anyway."
        return {"status": "success", "message": msg}
    raise HTTPException(status_code=400, detail="Must provide port or model_name.")


@router.post("/api/models/unload_all", summary="Unload all running models")
async def unload_all_models(gpu_id: Optional[int] = None, force: bool = False):
    candidates = [
        (port, info) for port, info in state.managed_processes.items()
        if gpu_id is None or info.get("gpu_id") == gpu_id
    ]
    unloaded_ports = []
    skipped_busy = []
    for port, info in candidates:
        if not force and await is_port_busy(port, info):
            skipped_busy.append(port)
            continue
        await stop_server_by_port(port)
        unloaded_ports.append(port)
    # llm plugins that hold a GPU (an exclusive one like Strata holds all of them, so it counts for any gpu_id)
    for p in plugin_registry.providers("llm"):
        if p.provision.gpu == "none":
            continue
        st = await plugin_registry.status(p)
        if not st or not st["workers"]:
            continue
        if gpu_id is not None and p.provision.gpu != "exclusive" and all(w["gpu"] not in (None, gpu_id) for w in st["workers"]):
            continue
        try:
            await _release_plugin(p, force)
            unloaded_ports.append(p.port)
        except HTTPException as e:
            if e.status_code != 409:
                raise
            skipped_busy.append(p.port)
    scope = f" on GPU {gpu_id}" if gpu_id is not None else ""
    msg = f"Unloaded {len(unloaded_ports)} model(s){scope}."
    if skipped_busy:
        msg += f" Skipped {len(skipped_busy)} with requests in flight (port(s) {skipped_busy}) -- pass force=true to unload anyway."
    return {"status": "success", "message": msg}


@router.post("/api/chat/compare", summary="Broadcast prompt to all ready instances", response_model=CompareResponse)
async def compare_chat(req: CompareRequest):
    ready_ports = []
    for port, info in state.managed_processes.items():
        proc = info.get("process")
        if proc and proc.poll() is None:
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        ready_ports.append(port)
            except Exception:
                pass

    if not ready_ports:
        raise HTTPException(status_code=400, detail="No ready llama-server instances found.")

    async def _get_response(port: int) -> Dict[str, Any]:
        user_content = []
        if req.image_base64:
            img_data = req.image_base64
            if not img_data.startswith("data:"):
                img_data = f"data:image/jpeg;base64,{img_data}"
            user_content.append({"type": "image_url", "image_url": {"url": img_data}})
        if req.audio_base64:
            audio_data = req.audio_base64
            if audio_data.startswith("data:"):
                try:
                    header, data_part = audio_data.split(",", 1)
                    audio_format = "wav"
                    if "audio/" in header:
                        raw_fmt = header.split("audio/")[1].split(";")[0].lower()
                        audio_format = "wav" if "wav" in raw_fmt else ("mp3" if "mp3" in raw_fmt or "mpeg" in raw_fmt else raw_fmt)
                    audio_data_base64 = data_part
                except Exception:
                    audio_format, audio_data_base64 = "wav", audio_data
            else:
                audio_format, audio_data_base64 = "wav", audio_data
            user_content.append({"type": "input_audio", "input_audio": {"data": audio_data_base64, "format": audio_format}})
        user_content.append({"type": "text", "text": req.prompt})

        payload = {
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                t0 = asyncio.get_event_loop().time()
                resp = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload)
                duration = asyncio.get_event_loop().time() - t0
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "") or message.get("thought", "")
                full_content = (f"<think>\n{reasoning}\n</think>\n\n{content}" if content else f"<think>\n{reasoning}\n</think>") if reasoning else content
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                tps = round(tokens / duration, 2) if duration > 0 else 0
                return {"port": port, "model": state.managed_processes[port]["model"].split("/").pop(),
                        "content": full_content, "time_taken": round(duration, 2), "tokens": tokens, "tps": tps, "status": "success"}
        except httpx.HTTPError as exc:
            return {"port": port, "model": state.managed_processes[port]["model"].split("/").pop(),
                    "content": f"Upstream error: {get_upstream_error_detail(exc)}", "status": "error"}
        except Exception as exc:
            return {"port": port, "model": state.managed_processes[port]["model"].split("/").pop(),
                    "content": str(exc), "status": "error"}

    responses = await asyncio.gather(*[_get_response(p) for p in ready_ports])
    return {"responses": responses}


@router.get("/api/system/status", summary="Get status of all integrated services")
async def get_system_status():
    # Lazy import: media.py owns the TTS engine state; importing it at module load
    # would be a needless cross-router dependency (and a latent circular one).
    from backend.routers.media import tts_status_snapshot as _tts_status_snapshot
    tts_snapshot = await _tts_status_snapshot()
    from backend import stt_client
    stt_status = await stt_client.status()

    # ComfyUI is the image plugin offering its job API (image-serve): ready = its ComfyUI runs
    comfy_ready, comfy_running, comfy_pid = False, False, None
    comfy_plugin = next((p for p in plugin_registry.providers("image") if "comfyui_api" in p.provision.endpoints), None)
    if comfy_plugin is not None:
        st = await plugin_registry.status(comfy_plugin)
        comfy_running = st is not None
        workers = (st or {}).get("workers") or []
        comfy_ready = bool(workers)
        comfy_pid = workers[0]["pid"] if workers else None

    llama_running = False
    llama_ready = False
    instances = []
    for port, info in state.managed_processes.items():
        proc = info.get("process")
        running = proc is not None and proc.poll() is None
        ready = False
        health: Dict[str, Any] = {}
        if running:
            llama_running = True
            try:
                async with httpx.AsyncClient(timeout=0.2) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        ready = True
                        llama_ready = True
                        health = resp.json()
            except Exception:
                pass
        instances.append({"port": port, "running": running, "ready": ready, "model": info["model"],
                          "pid": proc.pid if running else None, "health": health,
                          "n_gpu_layers": info.get("n_gpu_layers", -1), "gpu_id": info.get("gpu_id")})

    return {
        "comfyui": {"status": "ready" if comfy_ready else ("running" if comfy_running else "idle"),
                    "pid": comfy_pid},
        "llama": {"status": "ready" if llama_ready else ("running" if llama_running else "idle"),
                  "active_instances": len(state.managed_processes), "instances": instances},
        "whisperx": {"status": (stt_status or {}).get("whisperx", {}).get("status", "unloaded"),
                     "loaded": bool((stt_status or {}).get("whisperx", {}).get("loaded"))},
        "diarize": {"status": (stt_status or {}).get("diarize", {}).get("status", "unloaded"),
                    "loaded": bool((stt_status or {}).get("diarize", {}).get("loaded"))},
        "stt_serve": {"available": stt_status is not None},
        # Engine-agnostic aggregate of tts-serve's engines. Same payload as
        # GET /api/tts/status.
        "tts": tts_snapshot,
        "omnivoice": {"status": tts_snapshot["engines"].get("omnivoice", {}).get("status") or "unloaded"},
    }


def _tts_related_backend_lines(scan: int = 600, keep: int = 150) -> str:
    """OmniVoice logs in-process to backend.log; pull just the TTS-relevant lines
    from its tail so /api/logs/tts doesn't need the whole backend log."""
    path = os.path.join(LOGS_DIR, "backend.log")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            recent = f.readlines()[-scan:]
    except Exception as e:
        return f"(failed to read backend.log: {e})\n"
    kws = ("tts", "omnivoice", "qwen3", "cosyvoice", "chatterbox", "voxcpm")
    hits = [ln for ln in recent if any(k in ln.lower() for k in kws)]
    return "".join(hits[-keep:])


@router.get("/api/logs/{service}", summary="Get logs for a service")
async def get_service_logs(service: str, port: Optional[int] = None):
    if service == "tts":
        # Engine-agnostic: OmniVoice logs into backend.log (in-process); each
        # isolated-venv worker into its own <name>.log. Return every relevant tail so
        # a caller doesn't have to know which engine ran. Not merged chronologically
        # -- different line formats and clocks -- just concatenated under headers.
        sections = []
        backend_tts = _tts_related_backend_lines()
        if backend_tts.strip():
            sections.append("=== backend.log (TTS-related, last 600 lines scanned) ===\n" + backend_tts)
        from backend import tts_client
        serve_logs = await tts_client.logs(tail=200)
        if serve_logs.strip():
            sections.append(f"=== {getattr(tts_client.plugin(), 'name', 'tts')} workers (current run of each) ===\n" + serve_logs)
        return {"logs": "\n\n".join(sections) if sections else "No TTS logs yet."}

    if service in ("whisperx", "diarize", "nemotron", "nemotron_diarize", "stt"):
        # STT runs in the stt plugin: its own service log plus what its logs endpoint returns.
        from backend import stt_client
        parts = []
        _p = stt_client.plugin()
        svc_log = os.path.join(LOGS_DIR, f"service-{_p.id}.log") if _p else ""
        if svc_log and os.path.exists(svc_log):
            with open(svc_log, "r", encoding="utf-8", errors="ignore") as f:
                parts.append(f"=== {_p.name} ===\n" + "".join(f.readlines()[-300:]))
        worker = await stt_client.logs(tail=300)
        if worker.strip():
            parts.append("=== workers ===\n" + worker)
        return {"logs": "\n\n".join(parts) or "No STT logs yet (no transcription plugin running)."}
    if service in ("backend", "omnivoice"):
        filename = "backend.log"
    elif service in ("qwen3tts", "qwen3", "cosyvoice", "chatterbox", "voxcpm"):
        # The worker's own output, kept by tts-serve.
        from backend import tts_client
        engine = "qwen3" if service == "qwen3tts" else service
        return {"logs": await tts_client.logs(tail=500, engine=engine) or "No log yet (or tts-serve not running)."}
    elif service == "comfyui":   # the image plugin offering ComfyUI (image-serve) keeps its log
        p = next((p for p in plugin_registry.providers("image") if "comfyui_api" in p.provision.endpoints), None)
        if p is None or "logs" not in p.provision.endpoints:
            return {"logs": "No ComfyUI plugin is set up."}
        try:
            r = await plugin_registry.call(p, "logs", timeout=5.0, params={"tail": 500})
            return {"logs": r.json().get("logs", "") if r.status_code == 200 else f"{p.name}: {r.status_code}"}
        except (httpx.HTTPError, ValueError):
            return {"logs": f"{p.name} is not running."}
    elif service == "llama-server":
        if port:
            filename = f"llama-server-{port}.log"
        else:
            log_files = glob.glob(os.path.join(LOGS_DIR, "llama-server-*.log"))
            filename = os.path.basename(max(log_files, key=os.path.getmtime)) if log_files else "llama-server.log"
    else:
        return {"logs": "No log file found."}

    path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(path):
        return {"logs": f"Log file {filename} does not exist yet."}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return {"logs": "".join(lines[-200:])}
    except Exception as e:
        return {"logs": f"Failed to read logs: {e}"}
