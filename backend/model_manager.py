"""Model lifecycle: load, unload, GPU routing, keep-alive, process scanning."""
import asyncio
import glob
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import time
import struct
from typing import Any, Dict, List, Optional

import httpx
import psutil
from fastapi import HTTPException
from backend.procs import processes

from backend.config import LLAMA_SERVER_BIN, LOGS_DIR, MODEL_DIRS, DEFAULT_PARALLEL, DEFAULT_CHAT_MODEL
from backend.response_processing import (
    estimate_kv_overhead_mb,
    gguf_context_length,
    gguf_has_bundled_mtp,
    gguf_has_bundled_vision,
    is_loadable_gguf,
    is_mtp_draft,
)
from backend.state import state

logger = logging.getLogger("ai-serve-ui.model_manager")

# Per-model asyncio locks to prevent concurrent load races.
_model_load_locks: Dict[str, asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_upstream_error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            err_json = exc.response.json()
            if isinstance(err_json, dict):
                error_obj = err_json.get("error")
                if isinstance(error_obj, dict):
                    msg = error_obj.get("message")
                    if msg:
                        return msg
                elif error_obj:
                    return str(error_obj)
                detail = err_json.get("detail")
                if detail:
                    if isinstance(detail, list):
                        return "; ".join(
                            str(d.get("msg", d)) if isinstance(d, dict) else str(d)
                            for d in detail
                        )
                    return str(detail)
            elif isinstance(err_json, str):
                return err_json
        except Exception:
            pass
        if exc.response.text:
            return exc.response.text.strip()
        return f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"
    return str(exc)


def _tail_log_file(path: str, n: int = 20) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except Exception:
        return ""


def check_model_file_exists(path: str, label: str) -> None:
    """Raise a clear, actionable HTTPException if `path` doesn't resolve to a real
    file -- rather than letting a missing file fall through silently (get_model_size_mb()
    already swallows this into a 0.0 that skips the VRAM check below entirely) and
    surface only much later as an opaque "llama-server crashed during startup" plus a
    raw llama.cpp stderr line once actually spawned.

    Specifically distinguishes a *dangling symlink* from an outright missing path: this
    project's own models_dir can (and does, per a relocation reported by another local
    Claude session working on model storage) contain symlinks pointing at an external/
    removable drive (e.g. large GGUFs moved off a nearly-full disk onto a USB SSD)
    -- os.path.exists() on the *symlink itself* returns False whenever that drive isn't
    currently mounted, identically to a genuinely deleted file, so the two need to be
    told apart in the message or "is the drive unplugged?" would never occur to someone
    debugging a plain typo'd path, and vice versa.
    """
    from fastapi import HTTPException

    if os.path.exists(path):
        return
    if os.path.islink(path):
        try:
            target = os.readlink(path)
        except OSError:
            target = None
        detail = (
            f"{label} not found: {path} -- this is a symlink"
            + (f" pointing at '{target}'" if target else "")
            + " that doesn't currently resolve. If it points at an external/removable "
              "drive (e.g. a model relocated off a full local disk), check that the "
              "drive is mounted and try again."
        )
    else:
        detail = f"{label} not found: {path}"
        if not os.path.isabs(path):
            detail += " (this is not an absolute path -- check it's correct and absolute)."
    raise HTTPException(status_code=400, detail=detail)


def get_model_size_mb(model_path: str) -> float:
    try:
        if os.path.exists(model_path):
            return os.path.getsize(model_path) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error checking model size for {model_path}: {e}")
    return 0.0


_llamacpp_version_cache: Optional[Dict[str, Any]] = None


def get_llamacpp_version() -> Dict[str, Any]:
    """Query the configured llama-server binary for its build version.

    Cached in-process since the binary on disk doesn't change without a backend
    restart (only a full rebuild of llama.cpp itself does).
    """
    global _llamacpp_version_cache
    if _llamacpp_version_cache is not None:
        return _llamacpp_version_cache

    result = {"binary": LLAMA_SERVER_BIN, "build_number": None, "commit": None, "raw": None, "error": None}
    try:
        proc = subprocess.run(
            [LLAMA_SERVER_BIN, "--version"], capture_output=True, text=True, timeout=5,
        )
        raw = (proc.stdout + proc.stderr).strip()
        result["raw"] = raw
        # llama.cpp's own --version output format has changed across builds:
        # older builds print "version: 9986 (91c631b21)"; newer ones print
        # "version: 0.3.0-dev (build 10665, commit ca3d5a3e1)". Try the new
        # form first, fall back to the old one so either build is parseable.
        match = re.search(r"\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)", raw)
        if not match:
            match = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", raw)
        if match:
            result["build_number"] = int(match.group(1))
            result["commit"] = match.group(2)
    except Exception as e:
        result["error"] = str(e)

    _llamacpp_version_cache = result
    return result


def resolve_model_path(model_name_or_path: Optional[str]) -> str:
    models = []
    seen_paths: set = set()
    for directory in MODEL_DIRS:
        if not os.path.isdir(directory):
            continue
        files = glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True)
        for f in files:
            name = os.path.basename(f)
            abs_path = os.path.abspath(f)
            if (
                abs_path not in seen_paths
                and "mmproj" not in name.lower()
                and not is_mtp_draft(name)
                and is_loadable_gguf(name)
            ):
                models.append({"name": name, "path": abs_path})
                seen_paths.add(abs_path)

    if not models:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No GGUF models found in configured directories.")

    if not model_name_or_path or model_name_or_path.lower() in ["local-model", "default"]:
        # "Auto": a running instance -- but only one that can chat (2026-10-20: with bge-m3 and a reranker
        # running, Auto chats went to the reranker: "the current context does not logits computation")
        def can_chat(path: str) -> bool:
            # reading models (NuExtract, *OCR*) are chat models by kind but answer documents, not chats
            if re.search(r"nuextract|ocr", os.path.basename(path), re.IGNORECASE):
                return False
            t = get_model_traits(path)
            return t is None or t.get("kind") == "chat"
        for port, info in list(state.managed_processes.items()):
            proc = info.get("process")
            if (proc and proc.poll() is None and not info.get("embedding") and not info.get("reranking")
                    and can_chat(info["model"])):
                return info["model"]
        # a symlink onto a drive that isn't mounted (2026-10-24: the external SSD after a reboot) can't be the
        # default -- a model asked for by name still resolves to it, for the "is the drive mounted?" message
        models = [m for m in models if not (os.path.islink(m["path"]) and not os.path.exists(m["path"]))] or models
        for m in models:
            if m["name"].lower() == DEFAULT_CHAT_MODEL.lower():
                return m["path"]
        tiny = re.compile(r"(^|[-_.])(i?q1|i?q2|i?q3)(_|\.|$)", re.IGNORECASE)   # too small to answer reliably
        for m in models:
            name_lower = m["name"].lower()
            if ("gemma" in name_lower and ("12b" in name_lower or "4" in name_lower) and not tiny.search(m["name"])
                    and can_chat(m["path"])):
                return m["path"]
        return next((m["path"] for m in models if can_chat(m["path"])), models[0]["path"])

    if os.path.isabs(model_name_or_path) and os.path.exists(model_name_or_path):
        return model_name_or_path

    for m in models:
        if (
            m["name"].lower() == model_name_or_path.lower()
            or os.path.basename(m["path"]).lower() == model_name_or_path.lower()
        ):
            return m["path"]

    for directory in MODEL_DIRS:
        full_path = os.path.join(directory, model_name_or_path)
        if os.path.exists(full_path):
            return os.path.abspath(full_path)

    return model_name_or_path


def scan_all_gguf_paths() -> List[str]:
    """Recursively list every .gguf file under MODEL_DIRS, once. Reused by the
    per-model MTP/mmproj matchers below so listing N models doesn't re-walk the
    directory tree O(N) times each — that quadratic blowup is what made /api/models
    take 17s+ with a few dozen models, and it blocks the whole event loop while it runs
    since these are synchronous filesystem calls.
    """
    paths: List[str] = []
    for directory in MODEL_DIRS:
        if not os.path.isdir(directory):
            continue
        paths.extend(
            os.path.abspath(f) for f in glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True)
        )
    return paths


def find_draft_model_path(model_path: str, all_gguf_paths: Optional[List[str]] = None) -> Optional[str]:
    """Find an MTP draft head paired with a main model, e.g. mtp-gemma-4-12b-it.gguf
    pairs with gemma-4-12b-it-UD-Q4_K_XL.gguf (main model name starts with the draft's stem).
    Pass a pre-scanned all_gguf_paths (scan_all_gguf_paths()) when calling this for many
    models in a loop to avoid re-walking the directory tree each time.
    """
    main_stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    paths = all_gguf_paths if all_gguf_paths is not None else scan_all_gguf_paths()

    for f in paths:
        name = os.path.basename(f)
        if not is_mtp_draft(name):
            continue
        draft_stem = os.path.splitext(name)[0][len("mtp-"):].lower()
        if draft_stem and main_stem.startswith(draft_stem):
            return f
    return None


def find_gguf_by_name(model_name: str) -> Optional[str]:
    """Resolve a bare filename, stem, or fuzzy substring to an absolute .gguf path,
    searching every GGUF under MODEL_DIRS — main models, mmproj sidecars, and MTP draft
    heads alike. Unlike resolve_model_path() this has no "default model" fallback and
    doesn't exclude mmproj/draft files, since it's meant for ad hoc lookups (e.g. the
    header inspector) rather than picking a model to load.
    """
    name_lower = model_name.lower()
    candidates = scan_all_gguf_paths()

    for c in candidates:
        base = os.path.basename(c)
        if base.lower() == name_lower or os.path.splitext(base)[0].lower() == name_lower:
            return c
    for c in candidates:
        if name_lower in os.path.basename(c).lower():
            return c
    return None


_MMPROJ_STEM_RE = re.compile(r'^(?:mmproj-(.+)|(.+)-mmproj(?:-[^.]+)?)$', re.IGNORECASE)


def find_mmproj_path(model_path: str, all_gguf_paths: Optional[List[str]] = None) -> Optional[str]:
    """Find a vision projector (mmproj) sidecar paired with a main model. Handles both
    naming conventions seen in the wild: '<stem>-mmproj-<quant>.gguf' (e.g.
    gemma-4-12b-it-mmproj-F32.gguf) and 'mmproj-<stem>.gguf' (e.g.
    mmproj-gemma-4-12b-it-qat-q4_0.gguf), matching whichever candidate stem is the
    longest prefix of the main model's filename to prefer the most specific pairing.
    Pass a pre-scanned all_gguf_paths (scan_all_gguf_paths()) when calling this for many
    models in a loop to avoid re-walking the directory tree each time.
    """
    main_stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    paths = all_gguf_paths if all_gguf_paths is not None else scan_all_gguf_paths()

    best_match = None
    best_len = -1
    for f in paths:
        name = os.path.basename(f)
        if "mmproj" not in name.lower():
            continue
        stem = os.path.splitext(name)[0]
        m = _MMPROJ_STEM_RE.match(stem)
        if not m:
            continue
        candidate_stem = (m.group(1) or m.group(2)).lower()
        if candidate_stem and main_stem.startswith(candidate_stem) and len(candidate_stem) > best_len:
            best_match = f
            best_len = len(candidate_stem)
    return best_match


_bundle_flag_cache: Dict[str, tuple] = {}  # path -> ((mtime, size), (mtp_bundled, vision_bundled))


def get_bundled_flags(path: str) -> tuple:
    """Cached (mtp_bundled, vision_bundled) for a model — both require a full header
    scan (skipping past whatever the tokenizer vocab array turns out to be) which,
    across a few dozen models on every /api/models call, was the entire reason that
    endpoint took 15s+. Cache key is (mtime, size) so edited/replaced files are
    re-scanned automatically but unchanged ones are free on every subsequent call.
    """
    try:
        st = os.stat(path)
        cache_key = (st.st_mtime, st.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _bundle_flag_cache.get(path)
        if cached and cached[0] == cache_key:
            return cached[1]

    result = (gguf_has_bundled_mtp(path), gguf_has_bundled_vision(path))
    if cache_key is not None:
        _bundle_flag_cache[path] = (cache_key, result)
    return result


_traits_cache: Dict[str, tuple] = {}  # path -> ((mtime, size), traits)


def get_model_traits(path: str) -> Optional[Dict[str, Any]]:
    """Cached kind (chat/embedding/rerank/base), architecture and thinking support from the GGUF header
    (backend/llm_catalog.py), same (mtime, size) cache key as get_bundled_flags. None if unreadable."""
    from backend.llm_catalog import traits
    from backend.response_processing import read_gguf_header
    try:
        st = os.stat(path)
        key = (st.st_mtime, st.st_size)
    except OSError:
        return None
    cached = _traits_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    try:
        result = traits(read_gguf_header(path, full_keys=("tokenizer.chat_template",))["metadata"],
                        os.path.basename(path))
    except (OSError, ValueError, struct.error):
        result = None
    _traits_cache[path] = (key, result)
    return result


_context_length_cache: Dict[str, tuple] = {}  # path -> ((mtime, size), context_length)


def get_context_length(path: str) -> Optional[int]:
    """Cached trained context length for a model (see gguf_context_length) -- same
    (mtime, size) cache-key convention as get_bundled_flags above, so a few dozen models
    on every /api/models call stays cheap after the first scan."""
    try:
        st = os.stat(path)
        cache_key = (st.st_mtime, st.st_size)
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _context_length_cache.get(path)
        if cached and cached[0] == cache_key:
            return cached[1]

    result = gguf_context_length(path)
    if cache_key is not None:
        _context_length_cache[path] = (cache_key, result)
    return result


def find_active_model_port(model_name: Optional[str] = None) -> Optional[int]:
    if model_name:
        for port, info in list(state.managed_processes.items()):
            proc = info.get("process")
            if proc and proc.poll() is None:
                active_name = os.path.basename(info["model"]).lower()
                if model_name.lower() in active_name or active_name in model_name.lower():
                    return port
    for port, info in list(state.managed_processes.items()):
        proc = info.get("process")
        if proc and proc.poll() is None:
            return port
    return None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _detect_apple_gpu_used_mb() -> Optional[int]:
    """Best-effort GPU-attributable memory usage on Apple Silicon, sourced from
    ioreg's IOAccelerator "PerformanceStatistics" dict ("In use system memory",
    bytes) -- the same source community GPU monitors (asitop, mactop, stats.app)
    read for a per-process-free equivalent of nvidia-smi's memory.used, without
    needing sudo (unlike `powermetrics --samplers gpu_power`).

    NOT verified live -- this machine is confirmed NVIDIA-only (see CLAUDE.md),
    so there is no Apple hardware here to test against. Written from documented/
    community-confirmed ioreg key names, not reproduced firsthand; the exact key
    has reportedly shifted across macOS versions. Must be confirmed live on real
    Apple Silicon before this is trusted over the psutil fallback it silently
    degrades to on any parse failure -- returns None (never raises) so the caller
    always has a working value either way.
    """
    try:
        output = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            encoding="utf-8", stderr=subprocess.DEVNULL,
        )
        match = re.search(r'"In use system memory"\s*=\s*(\d+)', output)
        if match:
            return int(match.group(1)) // (1024 * 1024)
    except Exception:
        pass
    return None


def apple_gpu_limit_mb(total_mb: int, wired_limit_mb: int = 0) -> int:
    """What Metal lets the GPU use on Apple Silicon (unified memory): `sysctl iogpu.wired_limit_mb` when the owner
    set one, else Apple's default recommendedMaxWorkingSetSize -- about 2/3 of the RAM up to 36 GB, 3/4 above.
    Ollama and LM Studio size models against this too, not against the RAM that happens to be unused."""
    if wired_limit_mb > 0:
        return min(wired_limit_mb, total_mb)
    return int(total_mb * (0.75 if total_mb > 36 * 1024 else 2 / 3))


def apple_gpu_free_mb(limit_mb: int, gpu_used_mb: Optional[int], available_mb: int) -> int:
    """Free GPU memory on unified memory: the Metal limit minus what the GPU already holds (ioreg). macOS
    compresses or pages out other apps' memory when the GPU needs it, so their RAM doesn't count against this.
    Without the ioreg figure, only the RAM that's unused right now (psutil `available`) -- the cautious old way."""
    if gpu_used_mb is None:
        return available_mb
    return max(0, limit_mb - gpu_used_mb)


def _apple_wired_limit_mb() -> int:
    try:
        out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0) if out.returncode == 0 else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def llama_env() -> Dict[str, str]:
    """The environment for a llama-server: a prebuilt llama.cpp (installer, runtimes/) keeps its libraries next to
    the binary, so that folder goes first on the library path (harmless for a system install)."""
    env = os.environ.copy()
    lib_dir = os.path.dirname(LLAMA_SERVER_BIN)
    if lib_dir:
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            env[var] = os.pathsep.join(p for p in (lib_dir, env.get(var, "")) if p)
    return env


async def fetch_gpu_status() -> Dict[str, Any]:
    """Return GPU metrics from nvidia-smi (or psutil fallback).
    Called by the /api/gpu/status route handler AND internally by model loading.
    """
    if sys.platform == "darwin":
        try:
            mem = psutil.virtual_memory()
            total_mb = int(mem.total / (1024 * 1024))
            # Prefer GPU-attributable usage (ioreg) over whole-system RAM used --
            # unified memory means every unrelated process (browser, OS, ...) also
            # counts toward psutil's mem.used, which is a much noisier proxy for
            # "how much memory does the running model actually need" than nvidia-
            # smi's memory.used is on a discrete GPU. Falls back to the old
            # whole-system figure when ioreg is unavailable or unparseable (e.g. a
            # future macOS version renaming the key) -- see _detect_apple_gpu_used_mb
            # docstring for why this path isn't yet trusted as "confirmed live".
            gpu_used_mb = _detect_apple_gpu_used_mb()
            used_mb = gpu_used_mb if gpu_used_mb is not None else int(mem.used / (1024 * 1024))
            limit_mb = apple_gpu_limit_mb(total_mb, _apple_wired_limit_mb())
            return {
                "gpus": [{
                    "index": 0,
                    "name": "Apple Silicon (Unified Memory)",
                    "total": total_mb,
                    "gpu_limit_mb": limit_mb,
                    "used": used_mb,
                    "free": apple_gpu_free_mb(limit_mb, gpu_used_mb, int(mem.available / (1024 * 1024))),
                    "usage_percent": round((used_mb / total_mb) * 100, 1) if total_mb else 0.0,
                    "driver_version": None,
                    "memory_source": "ioreg_gpu" if gpu_used_mb is not None else "psutil_system_ram",
                }]
            }
        except Exception as e:
            return {"gpus": [], "error": f"Failed to get unified memory status: {e}"}

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader",
        ]
        output = subprocess.check_output(cmd, encoding="utf-8")
        gpus = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "total": int(parts[3]),
                "used": int(parts[4]),
                "free": int(parts[5]),
                "usage_percent": round((int(parts[4]) / int(parts[3])) * 100, 1),
            })
        return {"gpus": gpus}
    except Exception as e:
        try:
            mem = psutil.virtual_memory()
            return {
                "gpus": [{
                    "index": 0,
                    "name": "System Memory (CPU Fallback)",
                    "total": int(mem.total / (1024 * 1024)),
                    "used": int(mem.used / (1024 * 1024)),
                    "free": int(mem.available / (1024 * 1024)),
                    "usage_percent": round(mem.percent, 1),
                    "driver_version": None,
                }]
            }
        except Exception:
            logger.error(f"Failed to get GPU status: {e}")
            return {"gpus": [], "error": str(e)}


_gpu_runtime_cache_by_vendor: Dict[str, Dict[str, Any]] = {}


def _detect_gpu_vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n or "arc" in n or "iris" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    if "system memory" in n or "cpu fallback" in n:
        return "cpu"
    return "unknown"


def _detect_nvidia_runtime() -> Dict[str, Any]:
    # The CUDA version a driver supports is only in nvidia-smi's plain banner output
    # ("CUDA Version: 12.2"), not exposed as a --query-gpu field.
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding="utf-8", stderr=subprocess.DEVNULL)
        match = re.search(r"CUDA Version:\s*([\d.]+)", output)
        if match:
            return {"vendor": "nvidia", "kind": "cuda", "version": match.group(1)}
    except Exception:
        pass
    return {"vendor": "nvidia", "kind": "cuda", "version": None}


def _detect_amd_runtime() -> Dict[str, Any]:
    # The installed ROCm stack version is a plain text file ROCm itself writes.
    try:
        with open("/opt/rocm/.info/version", "r", encoding="utf-8") as f:
            version = f.read().strip()
        if version:
            return {"vendor": "amd", "kind": "rocm", "version": version}
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showdriverversion"], encoding="utf-8", stderr=subprocess.DEVNULL
        )
        match = re.search(r"([\d.]+)", output)
        if match:
            return {"vendor": "amd", "kind": "rocm", "version": match.group(1)}
    except Exception:
        pass
    return {"vendor": "amd", "kind": "rocm", "version": None}


def _detect_intel_runtime() -> Dict[str, Any]:
    # No single ubiquitous CLI like nvidia-smi/rocm-smi for Intel; best-effort via
    # clinfo's OpenCL driver version reported under an Intel platform entry.
    try:
        output = subprocess.check_output(["clinfo"], encoding="utf-8", stderr=subprocess.DEVNULL)
        if "Intel" in output:
            match = re.search(r"Driver Version\s+([\d.]+)", output)
            if match:
                return {"vendor": "intel", "kind": "level_zero", "version": match.group(1)}
    except Exception:
        pass
    return {"vendor": "intel", "kind": "level_zero", "version": None}


def _detect_apple_runtime() -> Dict[str, Any]:
    return {"vendor": "apple", "kind": "metal", "version": platform.mac_ver()[0] or None}


_GPU_RUNTIME_DETECTORS = {
    "nvidia": _detect_nvidia_runtime,
    "amd": _detect_amd_runtime,
    "intel": _detect_intel_runtime,
    "apple": _detect_apple_runtime,
}


def get_gpu_runtime_for_name(gpu_name: str) -> Dict[str, Any]:
    """Look up the GPU compute runtime (CUDA/ROCm/Level Zero/Metal) for one GPU by its
    display name — sniffed per-GPU rather than once globally, so a mixed-vendor box
    (e.g. one NVIDIA + one AMD card) reports the correct stack for each card instead of
    one blanket guess. Each vendor's detector is cached in-process (doesn't change
    without a driver reinstall/reboot); "cpu"/"unknown" vendors short-circuit to a
    static null result without touching the cache.

    This is separate from the per-GPU `driver_version` in fetch_gpu_status() — that's
    the display driver, this is the compute runtime/toolkit version it exposes, which
    is what actually gates which CUDA/ROCm-built llama.cpp binary will run.
    """
    vendor = _detect_gpu_vendor(gpu_name)
    if vendor not in _GPU_RUNTIME_DETECTORS:
        return {"vendor": vendor, "kind": None, "version": None}

    if vendor not in _gpu_runtime_cache_by_vendor:
        _gpu_runtime_cache_by_vendor[vendor] = _GPU_RUNTIME_DETECTORS[vendor]()
    return _gpu_runtime_cache_by_vendor[vendor]


async def get_free_vram_for_gpu(gpu_id: int) -> int:
    try:
        status = await fetch_gpu_status()
        for g in status.get("gpus", []):
            if g["index"] == gpu_id:
                return g["free"]
    except Exception as e:
        logger.error(f"Error checking free VRAM for GPU {gpu_id}: {e}")
    return 0


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# find_free_port() hands a port back to the OS the instant it closes the probe
# socket, before the caller has actually bound anything to it -- two callers
# asking within that gap can be handed the identical "free" port. The actual
# collision that matters in practice is between our own concurrent callers
# (two /api/start requests, or a dashboard start racing a gateway auto-load),
# not some unrelated third-party process; serializing the pick-and-register
# step under one lock and immediately checking/reserving the result in
# state.managed_processes (written before anything here awaits) closes that
# case, since every other caller checks the same dict before returning a port.
_port_alloc_lock = asyncio.Lock()


async def reserve_free_port() -> int:
    """Atomically pick a port no other in-flight start has already claimed.

    Registers a placeholder entry in state.managed_processes for the returned
    port -- the caller MUST overwrite it with the real process info (on
    success) or pop it (on failure) once done, or the port will look
    permanently occupied.
    """
    async with _port_alloc_lock:
        for _ in range(50):
            port = find_free_port()
            if port not in state.managed_processes:
                state.managed_processes[port] = {"process": None, "reserving": True}
                return port
        raise RuntimeError("Could not find a free port for a new llama-server instance")


def get_pid_on_port(port: int) -> Optional[int]:
    """Find PID of process listening on port."""
    for proc, pinfo in processes(["pid"]):
        try:
            for conn in proc.connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    return pinfo["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def kill_process_on_port(port: int):
    """Kill the llama-server process on this port. Skips non-llama-server processes."""
    for proc, pinfo in processes(["pid", "name", "cmdline"]):
        try:
            for conn in proc.connections(kind="inet"):
                if conn.laddr.port == port:
                    cmdline = pinfo.get("cmdline") or []
                    is_llama = any("llama-server" in os.path.basename(part) for part in cmdline if part)
                    if is_llama:
                        logger.info(f"Killing llama-server PID {pinfo['pid']} on port {port}")
                        proc.terminate()
                        proc.wait(timeout=2)
                    else:
                        logger.warning(
                            f"Port {port} occupied by non-llama process "
                            f"PID {pinfo['pid']} ({pinfo['name']}); skipping kill"
                        )
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass


async def _reserve_for_load(gpu_id: Optional[int], preferred: int, need_mb: int, label: str):
    """Reserve `need_mb` with the GPU manager -- on the requested GPU, or the
    preferred (most free) one first, then the others. Returns (reservation, reason);
    (None, "") when the manager isn't usable, so the old capacity check decides."""
    from backend.cluster.local import MANAGER
    try:
        if gpu_id is not None:
            candidates = [gpu_id]
        else:
            gpus = (await fetch_gpu_status()).get("gpus", [])
            candidates = [preferred] + [g["index"] for g in sorted(gpus, key=lambda g: g["free"], reverse=True)
                                        if g["index"] != preferred]
        reasons = []
        for g in candidates:
            res, _, reason = await MANAGER.reserve(g, need_mb, f"llm:{label}")
            if res is not None:
                return res, ""
            reasons.append(reason)
        return None, "; ".join(r for r in reasons if r)
    except HTTPException:
        raise  # e.g. 507: Strata holds the GPUs exclusively
    except Exception as e:  # the manager must never make a load fail on its own
        logger.warning(f"GPU manager unavailable: {e}")
        return None, ""


async def stop_server_by_port(port: int):
    info = state.managed_processes.get(port)
    if info is not None:
        proc = info.get("process")
        log_file = info.get("log_file")
        if info.get("keep_alive_task"):
            info["keep_alive_task"].cancel()
        if proc and proc.poll() is None:
            proc.terminate()
            for _ in range(20):
                if proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
            else:
                proc.kill()
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
        # Concurrent callers can race here (e.g. two unload requests for the same
        # port); pop with a default instead of del so the loser is a no-op, not a KeyError.
        state.managed_processes.pop(port, None)
        logger.info(f"Stopped server on port {port}")
    else:
        kill_process_on_port(port)


async def is_port_busy(port: int, info: Dict[str, Any]) -> bool:
    """Ground-truth busy check -- active_requests alone is NOT enough here.

    active_requests only counts calls this app itself mediated through the
    gateway (/v1/chat/completions, /api/chat/completions). A caller that
    started an instance via /api/start and then talks to its port directly
    (bypassing this app's gateway entirely -- confirmed live to be a real,
    common pattern: a client with its own model-lifecycle management calls
    /api/start once, then sends the actual inference traffic straight to the
    instance's own port for the rest of a long-running session) leaves
    active_requests permanently at 0 regardless of how busy the instance
    really is. Poll the instance's own /slots for ground truth instead --
    the same source get_status()'s own generating field already uses for
    this identical reason.

    Shared by reset_keep_alive() below (confirmed live: a keep-alive timer
    reset once at session start, then never touched again by a client whose
    actual request traffic bypasses the gateway, fired and killed a model out
    from under 14+ minutes of still-active direct-port traffic -- silent
    countdown, no real-activity check, exactly this gap) and dashboard.py's
    manual-stop endpoints (/api/stop, /api/models/unload*, /api/start's
    port-replace path), which need the identical check before killing
    anything on demand.
    """
    if info.get("active_requests", 0) > 0:
        return True
    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/slots")
            if resp.status_code == 200:
                return any(slot.get("is_processing") for slot in resp.json())
    except Exception:
        pass
    return False


def reset_keep_alive(port: int, keep_alive_seconds: int):
    info = state.managed_processes.get(port)
    if not info:
        return
    if info.get("keep_alive_task"):
        info["keep_alive_task"].cancel()
    info["keep_alive"] = keep_alive_seconds

    async def _keep_alive_check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            while await is_port_busy(port, info):
                await asyncio.sleep(5)
            logger.info(f"Keep-alive expired for model {info['model']} on port {port}. Unloading...")
            await stop_server_by_port(port)
        except asyncio.CancelledError:
            pass

    info["keep_alive_task"] = asyncio.create_task(_keep_alive_check())


class _AdoptedProcess:
    """Wraps a psutil.Process so an adopted instance satisfies the subset of
    subprocess.Popen's interface (.poll/.pid/.terminate/.kill/.wait) that
    managed_processes entries are read through everywhere else.

    psutil.Process has no .poll() at all -- every managed_processes entry
    started via Popen (get_or_load_model/_load_locked, /api/start) stores the
    real Popen object, but scan_existing_llama_servers() (run on every backend
    startup, adopting llama-server instances that survived a restart thanks to
    start_new_session=True) used to store the raw psutil.Process instead.
    Calling .poll() on that -- which /api/status, stop_server_by_port(), and
    the keep-alive checker all do unconditionally -- raised an uncaught
    AttributeError instead of reporting the instance's real liveness.
    """

    def __init__(self, proc: psutil.Process):
        self._proc = proc

    @property
    def pid(self):
        return self._proc.pid

    def poll(self) -> Optional[int]:
        try:
            if self._proc.is_running() and self._proc.status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.NoSuchProcess:
            return -1
        return 0

    def terminate(self):
        try:
            self._proc.terminate()
        except psutil.NoSuchProcess:
            pass

    def kill(self):
        try:
            self._proc.kill()
        except psutil.NoSuchProcess:
            pass

    def wait(self, timeout=None):
        try:
            self._proc.wait(timeout=timeout)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass


def _foreign_ports() -> set:
    """Ports that belong to someone else's server: model servers the owner added (an Ollama, LM Studio or their own
    llama-server -- backend/plugins/external.py) and apps' ports. A llama-server there is theirs: never adopted, so
    never stopped by a keep-alive timer (found 2026-10-21: a hand-started llama-server on 1234, added as a model
    server, was adopted at a gateway restart and unloaded 5 min later)."""
    ports = set()
    try:
        from backend.plugins import external, registry
        ports |= {s.port for s in external.load(external.FILE)[0]}
        ports |= {m.port for m in registry._manifests()}
    except Exception as e:
        logger.warning(f"adoption: could not read model servers/apps ({e}) -- adopting by port only")
    return ports


def scan_existing_llama_servers():
    logger.info("Scanning for existing llama-server processes...")
    foreign = _foreign_ports()
    for proc, pinfo in processes(["pid", "name", "cmdline"]):
        try:
            cmd = pinfo.get("cmdline")
            if not cmd:
                continue
            if not any("llama-server" in os.path.basename(part) for part in cmd if part):
                continue

            port = 8080
            model = None
            ctx_size = 32000
            n_gpu_layers = -1
            draft_model = None
            spec_type_value = ""
            pooling = None
            mmproj_path = None

            # GPU pinning is never a CLI flag here -- get_or_load_model()/dashboard.py's
            # /api/start both pin a GPU purely via a CUDA_VISIBLE_DEVICES env var on the
            # subprocess (model_manager.py's own "CUDA_VISIBLE_DEVICES={id} for port
            # {port}" log line is the only place this is recorded elsewhere). A prior
            # version of this loop searched cmd for "--gpu-id"/"--gpu_id", a flag
            # llama-server is never actually launched with -- confirmed live: it never
            # matched anything, so every adopted instance's gpu_id silently came back
            # None regardless of which GPU it was actually pinned to. Confirmed live
            # (2026-08-27) that psutil.Process.environ() correctly recovers it instead
            # (same-user process, readable via /proc/<pid>/environ). This mattered well
            # beyond a cosmetic /api/status field: _load_locked()'s existing-instance
            # reuse check matches on gpu_id when a caller supplies one, so a None here
            # made every already-adopted instance permanently unmatchable by any request
            # that specified gpu_id explicitly -- spawning a brand new, redundant
            # instance on the same physical GPU instead of reusing the adopted one.
            try:
                gpu_id_env = proc.environ().get("CUDA_VISIBLE_DEVICES")
                gpu_id = int(gpu_id_env) if gpu_id_env not in (None, "") else None
            except Exception:
                gpu_id = None

            it = iter(cmd)
            for part in it:
                if part in ("--port", "-p"):
                    try:
                        port = int(next(it))
                    except Exception:
                        pass
                elif part in ("-m", "--model"):
                    try:
                        model = next(it)
                    except Exception:
                        pass
                elif part in ("-c", "--ctx-size", "--ctx_size"):
                    try:
                        ctx_size = int(next(it))
                    except Exception:
                        pass
                elif part in ("-ngl", "--n-gpu-layers", "--n_gpu_layers"):
                    try:
                        n_gpu_layers = int(next(it))
                    except Exception:
                        pass
                elif part in ("-md", "--spec-draft-model", "--model-draft"):
                    try:
                        draft_model = next(it)
                    except Exception:
                        pass
                elif part == "--spec-type":
                    try:
                        spec_type_value = next(it)
                    except Exception:
                        spec_type_value = ""
                elif part == "--pooling":
                    try:
                        pooling = next(it)
                    except Exception:
                        pooling = None
                elif part == "--mmproj":
                    try:
                        mmproj_path = next(it)
                    except Exception:
                        mmproj_path = None

            embedding = "--embedding" in cmd or "--embeddings" in cmd
            reranking = "--reranking" in cmd or "--rerank" in cmd
            bundled_mtp = draft_model is None and "draft-mtp" in spec_type_value.split(",")

            if port in foreign:
                logger.info(f"Not adopting the llama-server on port {port}: it belongs to an added model server or an app")
                continue
            if model and port and port not in state.managed_processes:
                logger.info(
                    f"Adopting existing llama-server on port {port} "
                    f"(PID {proc.pid}, model {model}, ctx {ctx_size})"
                )
                state.managed_processes[port] = {
                    "process": _AdoptedProcess(proc),
                    "model": os.path.abspath(model),
                    "port": port,
                    "ctx_size": ctx_size,
                    "n_gpu_layers": n_gpu_layers,
                    "gpu_id": gpu_id,
                    "keep_alive": 300,
                    "keep_alive_task": None,
                    "last_used": time.time(),
                    "active_requests": 0,
                    "embedding": embedding,
                    "reranking": reranking,
                    "pooling": pooling,
                    "draft_model": os.path.abspath(draft_model) if draft_model else None,
                    "mtp_active": bool(draft_model) or bundled_mtp,
                    "mmproj_path": mmproj_path,
                }
                # Confirmed live (2026-08-27): without this, an adopted instance never
                # gets a keep_alive_task at all -- the "keep_alive": 300 line above is
                # just a stored number, reset_keep_alive() is what actually schedules the
                # asyncio task that enforces it. A get_or_load_model() reuse hit later
                # calls reset_keep_alive() itself, but an adopted instance that's never
                # matched again (e.g. because a caller only ever requests a gpu_id the
                # adoption's own gpu_id parsing bug -- see above -- happened to make
                # unmatchable) would otherwise run forever with no expiry mechanism at
                # all, regardless of how long it actually sits idle.
                reset_keep_alive(port, 300)
        except Exception as e:
            logger.error(f"Error parsing process {proc.pid}: {e}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

async def get_or_load_model(
    model_path: str,
    gpu_id: Optional[int] = None,
    ctx_size: int = 32000,
    threads: int = 8,
    n_gpu_layers: int = -1,
    keep_alive: int = 300,
    mmproj_path: Optional[str] = None,
    embedding: bool = False,
    draft_model_path: Optional[str] = None,
    use_mtp: bool = True,
    pooling: Optional[str] = None,
    reranking: bool = False,
    parallel: Optional[int] = None,
) -> int:
    abs_model = os.path.abspath(model_path)
    if abs_model not in _model_load_locks:
        _model_load_locks[abs_model] = asyncio.Lock()
    async with _model_load_locks[abs_model]:
        if mmproj_path is None and not (embedding or reranking):
            # a vision module next to the model (<stem>-mmproj-*.gguf) comes along, so image input works through the
            # gateway too -- before 2026-10-21 only OCR and a manual /api/start attached it, although the model
            # catalog already reported such models as vision-capable
            mmproj_path = find_mmproj_path(model_path)
        bundled_mtp = False
        if draft_model_path is None and use_mtp:
            draft_model_path = find_draft_model_path(model_path)
            if draft_model_path is None:
                bundled_mtp = gguf_has_bundled_mtp(model_path)
        return await _load_locked(
            model_path, gpu_id, ctx_size, threads, n_gpu_layers, keep_alive, mmproj_path, embedding,
            draft_model_path, bundled_mtp, pooling, reranking, parallel,
        )


async def _load_locked(
    model_path: str,
    gpu_id: Optional[int],
    ctx_size: int,
    threads: int,
    n_gpu_layers: int,
    keep_alive: int,
    mmproj_path: Optional[str],
    embedding: bool,
    draft_model_path: Optional[str] = None,
    bundled_mtp: bool = False,
    pooling: Optional[str] = None,
    reranking: bool = False,
    parallel: Optional[int] = None,
) -> int:
    from fastapi import HTTPException

    # None -> DEFAULT_PARALLEL (1); 0 (either way) -> llama.cpp's own auto choice
    # (4 slots), recorded as 4 for the reuse comparison below.
    if parallel is None:
        parallel = DEFAULT_PARALLEL
    parallel = parallel if parallel and parallel > 0 else None

    # --reranking implies --embedding on llama-server's own side (common/arg.cpp
    # forces params.embedding=true when --reranking is passed) -- mirrored here so
    # our own state bookkeeping (the "Mode: Embedding" badge, /api/status) stays
    # truthful for reranker instances too, rather than relying on the C++ side effect.
    if reranking:
        embedding = True

    for port, info in list(state.managed_processes.items()):
        if os.path.abspath(info["model"]) == os.path.abspath(model_path):
            if gpu_id is None or info.get("gpu_id") == gpu_id:
                proc = info.get("process")
                if proc and proc.poll() is None:
                    existing_ctx = info.get("ctx_size") or 0
                    existing_parallel = info.get("parallel") or 4
                    # Confirmed live: stopping an instance out from under a caller
                    # that's still using it is not hypothetical -- a multi-page
                    # OCR document holds active_requests >= 1 across many
                    # sequential per-page completions (see ocr.py's _run_ocr()),
                    # and a second, concurrent caller requesting a bigger ctx_size
                    # for the same model mid-document used to evict that instance
                    # immediately, crashing the first caller's in-flight request
                    # with a KeyError once its loop's next page tried to touch the
                    # now-gone managed_processes[port] entry. Only evict an idle
                    # instance; a busy one is served at its current (smaller)
                    # ctx_size for this one request rather than interrupted --
                    # the requester can retry once it frees up if it truly needs
                    # the larger context.
                    wants_more_slots = (parallel or 4) > existing_parallel
                    if (ctx_size > existing_ctx or wants_more_slots) and info.get("active_requests", 0) == 0:
                        # Reusing as-is would silently cap this request to a smaller
                        # context than it asked for -- restart with the larger one
                        # instead. The reverse (requested <= running) still reuses:
                        # a smaller/equal ask doesn't justify evicting a bigger,
                        # possibly-warm context that another caller may still want.
                        logger.info(
                            f"Requested ctx_size {ctx_size}/parallel {parallel or 4} exceeds running "
                            f"instance's {existing_ctx}/{existing_parallel} on port {port} for "
                            f"{model_path} -- restarting with the larger setting."
                        )
                        await stop_server_by_port(port)
                        break
                    info["last_used"] = time.time()
                    reset_keep_alive(port, keep_alive)
                    return port

    check_model_file_exists(model_path, "Model file")
    # Strata holding the GPUs is the GPU manager's call now (exclusive reservation): the
    # reservation below answers 507 while it runs or loads.
    if mmproj_path:
        check_model_file_exists(mmproj_path, "Vision module (mmproj) file")
    if draft_model_path:
        check_model_file_exists(draft_model_path, "MTP draft model file")

    target_gpu_id = gpu_id
    if target_gpu_id is None:
        try:
            status = await fetch_gpu_status()
            if status.get("gpus"):
                target_gpu_id = sorted(status["gpus"], key=lambda g: g["free"], reverse=True)[0]["index"]
            else:
                target_gpu_id = 0
        except Exception:
            target_gpu_id = 0

    model_size_mb = get_model_size_mb(model_path)
    free_vram = await get_free_vram_for_gpu(target_gpu_id)

    reservation = None
    if sys.platform != "darwin" and model_size_mb > 0:
        required_free = model_size_mb + estimate_kv_overhead_mb(ctx_size)
        # Central GPU manager (backend/cluster/local.py): every load reserves its
        # memory -- unloading idle consumers of any kind (other LLMs, ComfyUI, TTS
        # engines, STT; least recently used first; never busy or pinned) if needed --
        # so two loads can't both count on the same free VRAM before either has
        # allocated it. Fast yes/no, no waiting (callers retry).
        # purpose per model *and* context size: the GPU manager learns needs per purpose (KV grows with ctx)
        reservation, why = await _reserve_for_load(gpu_id, target_gpu_id, int(required_free),
                                                   f"{os.path.basename(model_path)}@{ctx_size}")
        if reservation is not None:
            target_gpu_id = reservation.gpu
        # refused by the manager (even if measured room exists: it may be promised to
        # a load in progress), or the manager is unusable and measured room is short
        if reservation is None and (why or free_vram < required_free):
            better_gpu = None
            better_gpu_free = 0
            try:
                status = await fetch_gpu_status()
                for g in status.get("gpus", []):
                    if g["index"] != target_gpu_id and g["free"] >= required_free:
                        if g["free"] > better_gpu_free:
                            better_gpu = g["index"]
                            better_gpu_free = g["free"]
            except Exception:
                pass

            if better_gpu is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        (f"No room on GPU {target_gpu_id}: {why}. " if why else
                         f"Not enough VRAM on GPU {target_gpu_id} ({free_vram}MB free, "
                         f"need {required_free:.0f}MB). ")
                        + f"Model could be loaded on GPU {better_gpu} ({better_gpu_free}MB free)."
                    ),
                )

            loaded_list = []
            for port, info in state.managed_processes.items():
                proc = info.get("process")
                if proc and proc.poll() is None:
                    loaded_list.append(
                        f"- {os.path.basename(info['model'])} on GPU {info.get('gpu_id')} "
                        f"(Port {port}, Active requests: {info.get('active_requests', 0)}, "
                        f"Last used: {round(time.time() - info.get('last_used', 0))}s ago)"
                    )
            raise HTTPException(
                status_code=507,
                detail=(
                    f"Insufficient VRAM on all GPUs (need {required_free:.0f}MB). "
                    + (f"GPU manager: {why}. " if why else "")
                    + f"Currently loaded:\n" + ("\n".join(loaded_list) or "None")
                ),
            )

    if (draft_model_path or bundled_mtp) and embedding:
        logger.warning(
            "MTP and --embedding are incompatible (crashes llama-server's pooling graph); "
            "disabling embedding mode since MTP speculative decoding was requested."
        )
        embedding = False
        reranking = False

    port = await reserve_free_port()
    env = llama_env()
    if target_gpu_id is not None and sys.platform != "darwin":
        env["CUDA_VISIBLE_DEVICES"] = str(target_gpu_id)
        logger.info(f"CUDA_VISIBLE_DEVICES={target_gpu_id} for port {port}")

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", model_path,
        "--port", str(port),
        "-c", str(ctx_size),
        "-t", str(threads),
        "-ngl", str(n_gpu_layers),
        "--host", "127.0.0.1",
        "--jinja",
    ]
    if embedding:
        cmd.append("--embedding")
    if reranking:
        cmd.append("--reranking")
    if pooling:
        cmd.extend(["--pooling", pooling])
    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])
        logger.info(f"Enabling vision module: {mmproj_path}")
    if draft_model_path:
        cmd.extend(["-md", draft_model_path, "--spec-type", "draft-mtp"])
        logger.info(f"Enabling MTP speculative decoding with draft head: {draft_model_path}")
    elif bundled_mtp:
        cmd.extend(["--spec-type", "draft-mtp"])
        logger.info(f"Enabling MTP speculative decoding using bundled nextn tensors in: {model_path}")
    # 1 slot unless the caller asked for more: llama.cpp otherwise auto-picks 4, which
    # costs single-stream speed on every model measured, most when the model nearly
    # fills the card (gemma-4-31B + MTP on the 3090: 17.7 -> 39.8 t/s decode;
    # gemma-4-12B on the 3090: +7% with MTP, +11% without; also ~0.6-1 GB less VRAM).
    # With 1 slot, concurrent requests queue inside llama-server.
    if parallel:
        cmd.extend(["-np", str(parallel)])

    try:
        log_file_path = os.path.join(LOGS_DIR, f"llama-server-{port}.log")
        log_file = open(log_file_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
        if reservation is not None:
            from backend.cluster.local import MANAGER
            MANAGER.confirm(reservation.id, proc.pid)  # now bound to this process
        state.managed_processes[port] = {
            "process": proc,
            "log_file": log_file,
            "model": model_path,
            "port": port,
            "ctx_size": ctx_size,
            "threads": threads,
            "n_gpu_layers": n_gpu_layers,
            "gpu_id": target_gpu_id,
            "last_used": time.time(),
            "active_requests": 0,
            "keep_alive": keep_alive,
            "keep_alive_task": None,
            "embedding": embedding,
            "reranking": reranking,
            "pooling": pooling,
            "draft_model": draft_model_path,
            "mtp_active": bool(draft_model_path) or bundled_mtp,
            "mmproj_path": mmproj_path,
            "parallel": parallel,
            "loading": True,  # the GPU manager treats a loading instance as busy
        }

        ready = False
        crashed = False
        elapsed = 0
        for i in range(60):
            if proc.poll() is not None:
                crashed = True
                break
            await asyncio.sleep(1)
            elapsed = i + 1
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        ready = True
                        break
            except Exception:
                pass

        if not ready:
            reason = "crashed during startup" if crashed else f"did not become ready within {elapsed}s"
            log_tail = _tail_log_file(log_file_path)
            logger.error(f"Server on port {port} {reason}.")
            await stop_server_by_port(port)
            detail = f"llama-server {reason}."
            if log_tail:
                detail += f" Last log lines:\n{log_tail}"
            raise HTTPException(status_code=500, detail=detail)

        logger.info(f"Server ready on port {port} after {elapsed}s.")
        from backend import events
        if target_gpu_id is not None:
            events.record("model", "{model} loaded on GPU {gpu} ({s} s)", model=os.path.basename(model_path),
                          gpu=target_gpu_id, s=elapsed)
        else:
            events.record("model", "{model} loaded ({s} s)", model=os.path.basename(model_path), s=elapsed)
        state.managed_processes[port]["loading"] = False
        if reservation is not None:
            # loaded: from now on its real, measured usage counts -- not the estimate
            # (the KV estimate is generous; a lingering reservation would block room)
            from backend.cluster.local import MANAGER
            MANAGER.release(reservation.id)

        reset_keep_alive(port, keep_alive)
        return port

    except Exception as e:
        if reservation is not None and reservation.pid is None:
            from backend.cluster.local import MANAGER
            MANAGER.release(reservation.id)  # Popen never ran -- give the room back now
        # If we never got past reserving the port (e.g. Popen itself raised),
        # state.managed_processes still holds the reservation placeholder --
        # drop it so the port doesn't look permanently occupied.
        if state.managed_processes.get(port, {}).get("reserving"):
            state.managed_processes.pop(port, None)
        logger.error(f"Failed to start llama-server: {e}")
        raise HTTPException(status_code=500, detail=str(e))
