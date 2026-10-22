"""Plugin contracts -- pure rules (no I/O). Spec for plugin authors: docs/plugins.md. Tests:
tests/unit/test_plugin_contracts.py.

Optional capabilities come from apps that declare `provides` in their llama-serve.service.json instead of being
built into the gateway (owner, 2026-10-21: "es sollte tatsächlich nichts davon hardcoded sein, sondern alles
pluginable"). The gateway's core keeps llama.cpp, the GPU manager, the catalogs, access rules and Apps & Services.
The owner chose the capability list the same day (incl. video, audio, documents, translation, moderation, 3D,
agent tools via MCP and realtime speech) so later plugins don't each invent their own interface.

Threat model: a plugin is an app under SERVICE_ROOTS -- trusted like any app there (whoever can put a manifest
there could already edit that app). What a plugin *returns* is not trusted: catalog entries and job status are
checked field by field, unknown fields dropped, ids can't clash with another plugin's (first owner keeps them), a
broken plugin yields errors, not a broken catalog, and result files are only ever paths on the plugin itself. The
gateway only calls `http://127.0.0.1:<manifest port><declared path>`: no hosts, schemes, queries, `..` or control
characters in paths -- a manifest can't point the gateway elsewhere. Agent tools carry a `risk`; without one they
count as the riskiest kind (execute).

Vocabulary:
- capability  llm | image | video | audio | tts | stt | documents | translate | moderation | model3d | tools |
              realtime | access
- endpoint    a named path of the plugin for one part of a capability (catalog, generate, jobs, ...)
- jobs        the one pattern for long work: POST {jobs} -> 202 {id}; GET {jobs}/{id} -> status; POST
              {jobs}/{id}/cancel; result files under the plugin's own paths
- gpu         none | shared (asks the GPU manager before loading) | exclusive (owns all GPUs while it runs)
- catalog     the plugin's answer to "what can you do?" -- entries with an `id` the gateway routes requests by,
              each with `license` and `commercial_use`
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# capability -> required / optional endpoints, the catalog's list key, and whether it may use a GPU
CAPABILITIES: Dict[str, Dict[str, Any]] = {
    # an OpenAI-compatible server (MLX, vLLM, Ollama, Strata ...): {openai}/models, /chat/completions,
    # /embeddings, /rerank; optional sparse embeddings (lexical weights, e.g. bge-m3's) and `load` (POST {"model"}:
    # load now and answer once it serves -- for a server that takes minutes to load)
    "llm": {"required": ("openai",), "optional": ("sparse", "load"), "list": "data"},
    # comfyui_api: a path prefix the gateway forwards /api/comfyui/* to (compatibility with apps that use ComfyUI's
    # job API from before image plugins existed)
    "image": {"required": ("catalog", "generate"), "optional": ("edit", "cancel", "jobs", "comfyui_api"), "list": "data"},
    "video": {"required": ("catalog", "jobs"), "optional": (), "list": "data"},
    "audio": {"required": ("catalog", "jobs"), "optional": (), "list": "data"},          # music, sound effects
    "tts": {"required": ("catalog", "generate"), "optional": ("status", "jobs"), "list": "engines"},
    "stt": {"required": ("catalog", "transcribe"), "optional": ("status", "live", "jobs"), "list": "engines"},
    "documents": {"required": ("catalog", "jobs"), "optional": (), "list": "data"},      # OCR / extraction
    "translate": {"required": ("catalog", "translate"), "optional": (), "list": "data"},
    "moderation": {"required": ("catalog", "moderate"), "optional": (), "list": "data"},  # OpenAI moderations shape
    "model3d": {"required": ("catalog", "jobs"), "optional": (), "list": "data"},
    # agent tools: an MCP server (streamable HTTP); its tools/list result is the catalog
    "tools": {"required": ("mcp",), "optional": (), "list": "tools", "no_gpu": True},
    # realtime speech conversation over a WebSocket (OpenAI-Realtime-like)
    "realtime": {"required": ("catalog", "realtime"), "optional": (), "list": "data"},
    # access for other people (the door): the gateway pushes the owner's shares, reads status
    "access": {"required": ("shares",), "optional": ("status", "security", "people"), "list": None, "no_gpu": True},
}
GPU_MODES = ("none", "shared", "exclusive")
# every capability may have these: status (what it holds, which processes), release (free idle GPU memory), logs,
# install (GET: what it can download -- models, engines; POST {"id"}: start one; the plugin picks the files and
# checks them -- normalize_install). GPU plugins must: shared -> status + release, exclusive -> status.
COMMON_ENDPOINTS = ("status", "release", "logs", "install")
INSTALL_STATES = ("missing", "downloading", "installed", "failed")
MAX_ENTRIES = 500
JOB_STATUSES = ("queued", "running", "done", "failed", "cancelled")

_PATH = re.compile(r"^/[A-Za-z0-9._~/-]{0,200}$")
_ENTRY_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# language model names as servers really use them: Ollama `name:tag`, LM Studio `publisher/model`, hf.co/... paths
_LLM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_TOOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ASPECT = re.compile(r"^\d{1,2}:\d{1,2}$")
_SIZE = re.compile(r"^\d{2,5}x\d{2,5}$")
_LANG = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")
_MIME = re.compile(r"^[a-z]+/[a-z0-9.+-]{1,60}$")
_FILE = re.compile(r"^[A-Za-z0-9._ -]{1,128}$")


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Provision:
    capability: str
    endpoints: Mapping[str, str]
    gpu: str = "none"


def _path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _PATH.match(value) or "//" in value or "/../" in value + "/" \
            or value.endswith("/..") or "/./" in value + "/":
        raise ContractError(f"{where}: a plain path starting with / (no host, query, '..' or '//')")
    return value


def parse_provides(raw: Any) -> Dict[str, Provision]:
    """The `provides` block of a service manifest."""
    if not isinstance(raw, dict) or not raw:
        raise ContractError("provides is an object: {capability: {endpoint: path, gpu?}}")
    out = {}
    for cap, decl in raw.items():
        spec = CAPABILITIES.get(cap)
        if spec is None:
            hint = " (renamed to llm)" if cap == "chat" else ""
            raise ContractError(f"unknown capability {cap!r}{hint} (known: {', '.join(CAPABILITIES)})")
        if not isinstance(decl, dict):
            raise ContractError(f"provides.{cap} must be an object")
        known = set(spec["required"]) | set(spec["optional"]) | set(COMMON_ENDPOINTS) | {"gpu"}
        unknown = sorted(set(decl) - known)
        if unknown:
            raise ContractError(f"provides.{cap}: unknown {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        missing = [e for e in spec["required"] if e not in decl]
        if missing:
            raise ContractError(f"provides.{cap}: missing {', '.join(missing)}")
        gpu = decl.get("gpu", "none")
        if gpu not in GPU_MODES:
            raise ContractError(f"provides.{cap}.gpu: one of {', '.join(GPU_MODES)}")
        if spec.get("no_gpu") and gpu != "none":
            raise ContractError(f"provides.{cap}: never takes a GPU of its own")
        needed = {"shared": ("status", "release"), "exclusive": ("status",)}.get(gpu, ())
        lacking = [e for e in needed if e not in decl]
        if lacking:
            raise ContractError(f"provides.{cap}: gpu {gpu} needs {', '.join(lacking)} (the GPU manager asks it "
                                f"what it holds and how to free it)")
        endpoints = {k: _path(v, f"provides.{cap}.{k}") for k, v in decl.items() if k != "gpu"}
        if cap == "llm" and not endpoints["openai"].endswith("/v1"):
            raise ContractError("provides.llm.openai: the path of the OpenAI-compatible API, ending in /v1")
        out[cap] = Provision(cap, endpoints, gpu)
    return out


def plugin_url(port: int, provision: Provision, endpoint: str, suffix: str = "") -> str:
    """The only URL shape the gateway calls a plugin at (`suffix`: e.g. "/<job id>" under {jobs})."""
    if endpoint not in provision.endpoints:
        raise ContractError(f"{provision.capability}: no {endpoint!r} endpoint declared")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ContractError("plugin port must be 1024-65535")
    if suffix:
        _path(suffix, "suffix")
    return f"http://127.0.0.1:{port}{provision.endpoints[endpoint]}{suffix}"


# ------------------------------------------------------------------ untrusted answers: small helpers

def _text(v: Any, limit: int, default: str = "") -> str:
    if not isinstance(v, str):
        return default
    return re.sub(r"[\x00-\x1f\x7f]", " ", v).strip()[:limit]


def _strings(v: Any, allowed: Optional[Sequence[str]] = None, pattern: Optional[re.Pattern] = None,
             limit: int = 50) -> List[str]:
    if not isinstance(v, list):
        return []
    out = [x for x in v if isinstance(x, str) and (allowed is None or x in allowed) and (pattern is None or pattern.match(x))]
    return list(dict.fromkeys(out))[:limit]


def _int(v: Any, lo: int, hi: int) -> Optional[int]:
    return v if type(v) is int and lo <= v <= hi else None


def _tasks(raw: Dict[str, Any], allowed: Sequence[str], default: Sequence[str]) -> List[str]:
    tasks = raw.get("tasks", list(default))
    if not isinstance(tasks, list) or not tasks or any(t not in allowed for t in tasks):
        raise ContractError(f"tasks: {' | '.join(allowed)}")
    return list(dict.fromkeys(tasks))


# ------------------------------------------------------------------ per-capability entry shapes

IMAGE_TASKS = ("generate", "edit", "style", "inpaint", "upscale", "remove_background", "segment")
VIDEO_TASKS = ("text_to_video", "image_to_video", "video_to_video")
AUDIO_TASKS = ("music", "sound")
MODEL3D_TASKS = ("text_to_3d", "image_to_3d")
LLM_KINDS = ("chat", "embedding", "rerank", "sparse")
TOOL_RISKS = ("read", "write", "external", "execute")      # from harmless to "runs code / changes things"


def _image(raw):
    steps = raw.get("steps") if isinstance(raw.get("steps"), dict) else {}
    return {"tasks": _tasks(raw, IMAGE_TASKS, ["generate"]),
            "aspect_ratios": _strings(raw.get("aspect_ratios"), pattern=_ASPECT) or ["1:1"],
            "text_in_image": raw.get("text_in_image") is True, "layout": raw.get("layout") is True,
            "steps": {"default": _int(steps.get("default"), 1, 200), "adjustable": steps.get("adjustable") is not False,
                      "max": _int(steps.get("max"), 1, 200)}}


def _video(raw):
    return {"tasks": _tasks(raw, VIDEO_TASKS, ["text_to_video"]), "max_seconds": _int(raw.get("max_seconds"), 1, 600),
            "resolutions": _strings(raw.get("resolutions"), pattern=_SIZE), "fps": _int(raw.get("fps"), 1, 120)}


def _audio(raw):
    return {"tasks": _tasks(raw, AUDIO_TASKS, ["music"]), "max_seconds": _int(raw.get("max_seconds"), 1, 3600)}


def _tts(raw):
    modes = raw.get("modes") if isinstance(raw.get("modes"), list) else []
    return {"languages": _strings(raw.get("languages"), pattern=_LANG, limit=200),
            "params": _strings(raw.get("params"), pattern=re.compile(r"^[a-z][a-z0-9_]{0,31}$"), limit=16),
            "modes": [m for m in modes if isinstance(m, dict)][:20], "loaded": raw.get("loaded") is True,
            "default_reference": raw.get("default_reference") is True}


def _stt(raw):
    fields = raw.get("fields") if isinstance(raw.get("fields"), list) else []
    return {"loaded": raw.get("loaded") is True, "outputs": _strings(raw.get("outputs"), limit=10),
            "fields": [f for f in fields if isinstance(f, dict) and isinstance(f.get("name"), str)][:30]}


def _documents(raw):
    return {"modes": _strings(raw.get("modes"), pattern=re.compile(r"^[a-z][a-z_-]{0,40}$")),
            "inputs": _strings(raw.get("inputs"), allowed=("image", "pdf", "epub", "docx", "html", "text"))}


def _translate(raw):
    return {"languages": _strings(raw.get("languages"), pattern=_LANG, limit=500)}


def _moderation(raw):
    return {"categories": _strings(raw.get("categories"), pattern=re.compile(r"^[a-z][a-z0-9_/-]{0,40}$"))}


def _model3d(raw):
    return {"tasks": _tasks(raw, MODEL3D_TASKS, ["image_to_3d"]),
            "formats": _strings(raw.get("formats"), allowed=("glb", "gltf", "obj", "ply", "stl", "usdz"))}


def _realtime(raw):
    return {"voices": _strings(raw.get("voices"), pattern=_TOKEN), "languages": _strings(raw.get("languages"), pattern=_LANG)}


def _kind_from_name(name: Any) -> str:
    """A standard OpenAI server's /models says nothing about the kind (Ollama, LM Studio): guess from the name."""
    n = name.lower() if isinstance(name, str) else ""
    return "rerank" if "rerank" in n else "embedding" if "embed" in n else "chat"


def _llm(raw):
    kind = raw.get("kind") or _kind_from_name(raw.get("id"))
    if kind not in LLM_KINDS:
        raise ContractError(f"kind: {' | '.join(LLM_KINDS)}")
    ctx = raw.get("context_length")
    return {"kind": kind, "context_length": ctx if type(ctx) is int and 0 < ctx <= 10_000_000 else None,
            "thinking": raw.get("thinking") is True, "vision": raw.get("vision") is True}


_SHAPES: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "llm": _llm, "image": _image, "video": _video, "audio": _audio, "tts": _tts, "stt": _stt,
    "documents": _documents, "translate": _translate, "moderation": _moderation, "model3d": _model3d,
    "realtime": _realtime,
}


def _tool(item: Dict[str, Any]) -> Dict[str, Any]:
    """One MCP tool (tools/list): name -> id; without a known risk it counts as the riskiest kind."""
    name = item.get("name")
    if not isinstance(name, str) or not _TOOL_ID.match(name):
        raise ContractError(f"invalid tool name {name!r}")
    risk = item.get("risk") if item.get("risk") in TOOL_RISKS else "execute"
    schema = item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {"type": "object"}
    return {"id": name, "label": _text(item.get("title"), 120) or name, "description": _text(item.get("description"), 1000),
            "risk": risk, "input_schema": schema, "available": True}


def _common(item: Dict[str, Any], capability: str) -> Dict[str, Any]:
    eid = item.get("id")
    pattern = _LLM_ID if capability == "llm" else _ENTRY_ID
    if not isinstance(eid, str) or not pattern.match(eid) or "//" in eid or ".." in eid.split("/") \
            or any(seg in ("", ".", "..") for seg in eid.split("/")):
        raise ContractError(f"invalid id {eid!r}")
    cu = item.get("commercial_use")
    return {"id": eid, "label": _text(item.get("label") or item.get("name"), 120) or eid,
            "description": _text(item.get("description"), 500),
            # an OpenAI /models list only names what the server can serve right now
            "available": True if capability == "llm" else item.get("available") is True,
            "license": _text(item.get("license"), 60), "commercial_use": cu if isinstance(cu, bool) else None,
            "default": item.get("default") is True,
            "aliases": _strings(item.get("aliases"), pattern=re.compile(r"^[A-Za-z0-9._-]{1,128}$"), limit=10)}


def normalize_catalog(capability: str, plugin_id: str, raw: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A plugin's catalog answer -> (checked entries tagged with the plugin, errors). Never raises."""
    spec = CAPABILITIES.get(capability)
    if spec is None or spec["list"] is None:
        return [], [f"{capability} has no catalog"]
    key = spec["list"]
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return [], [f"{plugin_id}: catalog must be an object with a {key!r} list"]
    entries, errors = [], []
    if len(items) > MAX_ENTRIES:
        errors.append(f"{plugin_id}: only the first {MAX_ENTRIES} of {len(items)} entries")
        items = items[:MAX_ENTRIES]
    for i, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ContractError("not an object")
            if capability == "tools":
                entry = _tool(item)
            else:
                entry = {**_common(item, capability), **_SHAPES[capability](item)}
            entries.append({**entry, "plugin": plugin_id})
        except ContractError as e:
            errors.append(f"{plugin_id}: entry {i}: {e}")
    return entries, errors


def merge_catalogs(parts: Sequence[Tuple[str, List[Dict[str, Any]]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Several plugins' entries -> one list. Plugins in id order (stable, not "whoever answered first"); an id
    belongs to the first plugin that lists it -- a later plugin can't take it over."""
    merged, owner, clashes = [], {}, []
    for plugin_id, entries in sorted(parts, key=lambda p: p[0]):
        for e in entries:
            if e["id"] in owner:
                clashes.append({"id": e["id"], "kept": owner[e["id"]], "ignored": plugin_id})
                continue
            owner[e["id"]] = plugin_id
            merged.append(e)
    return merged, clashes


# ------------------------------------------------------------------ status (what a plugin holds)

_HOST = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_HTTPS = re.compile(r"^https://[a-z0-9.-]{1,253}(/[A-Za-z0-9._~/-]{0,200})?$")


def normalize_status(raw: Any) -> Dict[str, Any]:
    """A plugin's status answer -> checked. Unreadable = busy (the GPU manager then touches nothing of it).
    `workers` are the plugin's processes holding models (the GPU manager attributes VRAM by them); `domain` and
    `overview` are for access plugins (the public names the door answers on, shown to the owner)."""
    if not isinstance(raw, dict):
        return {"busy": True, "ready": False, "facts": [], "workers": [], "domain": None, "overview": None}
    facts = []
    for f in raw.get("facts") if isinstance(raw.get("facts"), list) else []:
        if isinstance(f, dict):
            facts.append({"label": _text(f.get("label"), 60), "value": _text(f.get("value"), 200)})
        elif isinstance(f, str):
            facts.append({"label": "", "value": _text(f, 200)})
    workers = []
    for w in raw.get("workers") if isinstance(raw.get("workers"), list) else []:
        if not isinstance(w, dict) or not isinstance(w.get("id"), str) or not _ENTRY_ID.match(w["id"]):
            continue
        last = w.get("last_used")
        workers.append({"id": w["id"], "label": _text(w.get("label"), 120) or w["id"],
                        "pid": w["pid"] if type(w.get("pid")) is int and w["pid"] > 1 else None,
                        "busy": w.get("busy") is True, "loaded": w.get("loaded") is True,
                        "gpu": w["gpu"] if type(w.get("gpu")) is int and 0 <= w["gpu"] < 64 else None,
                        "last_used": float(last) if isinstance(last, (int, float)) and not isinstance(last, bool) and last > 0 else None})
    domain = raw.get("domain") if isinstance(raw.get("domain"), str) and _HOST.match(raw["domain"]) else None
    overview = raw.get("overview") if isinstance(raw.get("overview"), str) and _HTTPS.match(raw["overview"]) else None
    # ready: serving right now without loading first (missing = ready) -- a model server that is stopped or still
    # loading says false
    return {"busy": raw.get("busy") is True, "ready": raw.get("ready") is not False, "facts": facts[:6],
            "workers": workers[:50], "domain": domain,
            "overview": overview}


# ------------------------------------------------------------------ downloads a plugin offers

def normalize_install(raw: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A plugin's install list (GET {install}) -> (checked items, errors). Never raises. Only what's shown and the
    item id the owner may ask for -- no URLs or paths: which files, from where and with which checksum is the
    plugin's own business (it ships that list)."""
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return [], ["install list must be an object with an 'items' list"]
    out, errors = [], []
    for i, item in enumerate(items[:MAX_ENTRIES]):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not _ENTRY_ID.match(item["id"]):
            errors.append(f"item {i}: invalid id {item.get('id') if isinstance(item, dict) else item!r}")
            continue
        state = item.get("state") if item.get("state") in INSTALL_STATES else "missing"
        out.append({"id": item["id"], "label": _text(item.get("label"), 120) or item["id"],
                    "description": _text(item.get("description"), 300), "size_mb": _int(item.get("size_mb"), 0, 10_000_000),
                    "done_mb": _int(item.get("done_mb"), 0, 10_000_000), "state": state,
                    "error": _text(item.get("error"), 300), "license": _text(item.get("license"), 60)})
    return out, errors


# ------------------------------------------------------------------ long jobs (one pattern for all)

def normalize_job(raw: Any) -> Dict[str, Any]:
    """A plugin's job status (GET {jobs}/{id}) -> checked. Result files are only paths on the plugin itself; the
    gateway fetches them from there and hands them on -- a plugin can't make anyone download from elsewhere."""
    if not isinstance(raw, dict):
        raise ContractError("job status must be an object")
    jid = raw.get("id")
    if not isinstance(jid, str) or not _JOB_ID.match(jid):
        raise ContractError("job id: letters, digits, _ and -")
    status = raw.get("status")
    if status not in JOB_STATUSES:
        raise ContractError(f"job status: {' | '.join(JOB_STATUSES)}")
    progress = raw.get("progress")
    if progress is not None and (isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0 <= progress <= 1):
        raise ContractError("job progress: 0..1")
    result = None
    if isinstance(raw.get("result"), dict):
        res = raw["result"]
        files = []
        for f in res.get("files") or []:
            if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not _FILE.match(f["name"]):
                raise ContractError("result file: {name, type, url}")
            ftype = f.get("type") if isinstance(f.get("type"), str) and _MIME.match(f["type"]) else "application/octet-stream"
            files.append({"name": f["name"], "type": ftype, "url": _path(f.get("url"), "result file url")})
        result = {"files": files[:100]}
        if isinstance(res.get("text"), str):
            result["text"] = res["text"][:1_000_000]
        if isinstance(res.get("data"), dict):
            result["data"] = res["data"]
    return {"id": jid, "status": status, "progress": float(progress) if progress is not None else None,
            "message": _text(raw.get("message"), 300), "result": result,
            "error": _text(raw.get("error"), 500) or None}
