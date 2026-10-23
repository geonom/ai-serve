"""ComfyUI in image-serve: the process (start / adopt / stop / idle-free), workflows (generate, edit, style transfer,
batch, LoRAs), the GPU questions before models load, and ComfyUI's job API under /api/comfyui/* (apps from before
image plugins use it; ai-serve forwards /api/comfyui/* here). Moved verbatim from ai-serve-ui's
backend/routers/media.py on 2026-10-21 -- the history of every rule in here is in ai-serve-ui's CLAUDE.md
("ComfyUI integration"); only the connections changed: the GPU manager is reached over ai-serve's coordination API
(image_serve/gpu.py), state lives in image_serve/state.py, settings in image_serve/config.py."""
import asyncio
import base64
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import httpx
import psutil
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticUndefined

from image_serve.config import (COMFYUI_DEFAULT_GPU_ID, COMFYUI_DIR, COMFYUI_MODELS_ROOT, COMFYUI_PYTHON,
                                GENERATED_IMAGES_DIR, LOGS_DIR, WORKFLOWS_DIR)
from image_serve.gpu import Reservation  # noqa: F401  (the type reserve() answers with)
from image_serve.state import state

logger = logging.getLogger("image-serve.comfy")
router = APIRouter()


# ComfyUI's folder name -> sub dirs of a models root (the layout ComfyUI's own models/ dir has)
_MODEL_PATHS = {"checkpoints": ["checkpoints/"], "diffusion_models": ["diffusion_models/", "unet/"],
                "text_encoders": ["text_encoders/", "clip/"], "vae": ["vae/"], "loras": ["loras/"],
                "clip_vision": ["clip_vision/"], "controlnet": ["controlnet/"], "upscale_models": ["upscale_models/"],
                "embeddings": ["embeddings/"], "model_patches": ["model_patches/"]}


def extra_model_paths_args(comfy_dir: str, models_root: str, state_dir: str) -> List[str]:
    """A models root outside ComfyUI's own models/ (COMFYUI_MODELS_ROOT -- where downloads land) is handed to ComfyUI
    as an extra model paths file; ComfyUI otherwise never looks there. Nothing extra when they're the same dir."""
    if os.path.realpath(models_root) == os.path.realpath(os.path.join(comfy_dir, "models")):
        return []
    lines = ["image_serve:", f"    base_path: {json.dumps(models_root)}"]   # a JSON string is a valid YAML scalar
    for name, subs in _MODEL_PATHS.items():
        lines.append(f"    {name}: |")
        lines += [f"        {s}" for s in subs]
    path = os.path.join(state_dir, "comfyui-model-paths.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# written by image-serve at each ComfyUI start (COMFYUI_MODELS_ROOT)\n" + "\n".join(lines) + "\n")
    return ["--extra-model-paths-config", path]


def find_comfyui_process() -> Optional[psutil.Process]:
    """ComfyUI on our port, whoever started it (cwd = COMFYUI_DIR, main.py --port <ours>)."""
    root = os.path.realpath(COMFYUI_DIR)
    for p in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            cmd = p.info["cmdline"] or []
            if "main.py" in cmd and "--port" in cmd and cmd[cmd.index("--port") + 1] == str(state.comfyui_port) \
                    and os.path.realpath(p.info["cwd"] or "") == root:
                return p
        except (psutil.Error, TypeError, IndexError, ValueError):
            continue
    return None


def get_pid_on_port(port: int) -> Optional[int]:
    p = find_comfyui_process()
    return p.pid if p else None


def adopt_running() -> None:
    """A ComfyUI that survived an image-serve restart (own session) is picked up again: pid and --cuda-device."""
    p = find_comfyui_process()
    if p is None or state.comfyui_pid:
        return
    state.comfyui_pid = p.pid
    try:
        cmd = p.cmdline()
        state.comfyui_gpu_id = int(cmd[cmd.index("--cuda-device") + 1]) if "--cuda-device" in cmd else None
    except (psutil.Error, ValueError, IndexError):
        pass

    class _Adopted:   # enough of Popen for the code below (poll/terminate/wait/pid)
        pid = p.pid

        def poll(self):
            return None if p.is_running() and p.status() != psutil.STATUS_ZOMBIE else 0

        def terminate(self):
            p.terminate()

        def wait(self, timeout=None):
            return p.wait(timeout=timeout or 30)

    state.comfyui_process = _Adopted()
    logger.info(f"Adopted running ComfyUI (pid {p.pid}, GPU {state.comfyui_gpu_id})")


# ComfyUI's own external model path (configured via its extra_model_paths.yaml),
# separate from llama.cpp's MODEL_DIRS.
_COMFYUI_LORAS_DIR = os.path.join(COMFYUI_MODELS_ROOT, "loras")

_SAFE_IMAGE_FILENAME_RE = re.compile(r'^[A-Za-z0-9 _.\-]+$')


def _sanitize_image_filename(name: str) -> str:
    """Turn a user-supplied filename into a safe basename under GENERATED_IMAGES_DIR.

    Same path-traversal class of bug the voices/ upload path already guards
    against — os.path.basename() drops any directory component a caller might try
    to smuggle in, and the charset check rejects anything else that isn't a plain
    name. Always forces a .png extension since that's what ComfyUI's SaveImage
    node actually produces here, regardless of what extension (if any) was given.
    """
    name = os.path.basename(name.strip())
    if not name or name in (".", ".."):
        raise ValueError("filename must be a non-empty file name, not a path.")
    if not _SAFE_IMAGE_FILENAME_RE.match(name):
        raise ValueError(
            f"filename '{name}' contains characters outside letters, digits, spaces, '.', '_', '-'."
        )
    stem = os.path.splitext(name)[0] or name
    return f"{stem}.png"


class ComfyUIStartRequest(BaseModel):
    # kept for callers from before image-serve; ignored -- ComfyUI always runs on the internal COMFY_PORT behind this
    # app (8189 is this app's own port now; a caller's port must never move ComfyUI onto it)
    port: int = 8189
    # Which GPU ComfyUI is allowed to see, passed straight through as ComfyUI's own
    # native `--cuda-device` CLI flag (comfy/cli_args.py) -- not a
    # CUDA_VISIBLE_DEVICES env var we set ourselves, unlike llama-server's gpu_id
    # (get_or_load_model()/StartRequest.gpu_id), which has no equivalent CLI flag.
    # None (default) leaves every GPU visible to the process; on Linux that means
    # PyTorch's own default of cuda:0 among whatever's visible -- there is no
    # most-free-VRAM auto-pick the way llama-server's own gpu_id-omitted path has.
    # Only takes effect at process launch: ComfyUI is one persistent process shared
    # by every /api/comfyui/generate|edit|style-transfer call, so changing GPUs means
    # stopping and restarting it (POST /api/comfyui/stop then /api/comfyui/start with
    # a new gpu_id), not something settable per-request on an already-running instance.
    gpu_id: Optional[int] = None


class LoraSpec(BaseModel):
    name: str
    strength: float = 1.0


class ComfyUIGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_name: str
    aspect_ratio: str = "1:1"
    base_resolution: int = 1024
    steps: Optional[int] = None
    seed: Optional[int] = None
    # Optional — Ernie Turbo / Z-Image-Turbo workflows bypass their LoraLoaderModelOnly
    # node entirely (model wired straight from the base checkpoint) unless lora_name is
    # set. A LoRA is architecture-specific (trained against one base checkpoint's
    # weights); mismatching one against the wrong workflow will fail to load or produce
    # garbage — there's no compatibility check here, same as model_name/CLIP/VAE pairing.
    lora_name: Optional[str] = None
    lora_strength: Optional[float] = None
    # Stack more than one LoRA. If set, takes precedence over lora_name/lora_strength
    # (which stay as the single-LoRA shorthand for existing callers, e.g. the batch
    # endpoint). loras[0] activates the workflow's own LoraLoaderModelOnly node exactly
    # like lora_name always has; loras[1:] get chained in after it via
    # _extend_lora_chain -- see that function for why chaining instead of one node.
    loras: Optional[List[LoraSpec]] = None
    # Optional -- when set, the completed image is also written to disk under
    # GENERATED_IMAGES_DIR/<filename>.png (sanitized; see _sanitize_image_filename).
    # Omitted entirely -> unchanged behavior, image only comes back as base64.
    filename: Optional[str] = None
    # Which GPU ComfyUI should run on, forwarded to _ensure_comfyui_running() ->
    # ComfyUIStartRequest.gpu_id. Only takes effect if ComfyUI isn't already running --
    # see ComfyUIStartRequest.gpu_id and _ensure_comfyui_running() for why an
    # already-running instance can't be silently moved to a different GPU per-request.
    # Explicitly POST /api/comfyui/stop first if you need to force a GPU switch.
    gpu_id: Optional[int] = None


class ComfyBatchItem(BaseModel):
    """One image in a batch -- only prompt is required, everything else falls back
    to the batch's top-level defaults so a batch JSON can be as short as a flat list
    of prompts or as detailed as per-item overrides."""
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    aspect_ratio: Optional[str] = None
    base_resolution: Optional[int] = None
    # Per-item, not inherited from defaults -- a shared filename across every item
    # in a batch would just mean each one overwrites the last. Leave unset to skip
    # saving that item to disk (base64 via the batch's task_id lookup still works).
    filename: Optional[str] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    lora_name: Optional[str] = None
    lora_strength: Optional[float] = None


class ComfyBatchDefaults(BaseModel):
    negative_prompt: str = "ugly, deformed, bad anatomy, blurry"
    model_name: str = "z_image_turbo-Q6_K.gguf"
    aspect_ratio: str = "1:1"
    base_resolution: int = 1024
    steps: Optional[int] = None
    seed: Optional[int] = None
    lora_name: Optional[str] = None
    lora_strength: Optional[float] = None
    # See ComfyUIGenerateRequest.gpu_id -- applies once, to the whole batch's
    # auto-start, not per item (all items in a batch share the one ComfyUI process).
    gpu_id: Optional[int] = None


class ComfyBatchRequest(BaseModel):
    defaults: ComfyBatchDefaults = ComfyBatchDefaults()
    items: List[ComfyBatchItem]


class ComfyUIEditRequest(BaseModel):
    image_base64: str
    prompt: str
    # Fastest + most reliable instruction-following of the three quants tested
    # (Q2_K/Q3_K_S/Q4_K_S all ~40-55s with the Lightning LoRA; Q3_K_S under-applied
    # edits in testing despite being the mid-size quant — quant quality here doesn't
    # scale monotonically with bit-depth). All three remain selectable via model_name.
    model_name: str = "qwen-image-edit-2511-Q2_K.gguf"   # the older pre-2511 files were removed 2026-10-21
    steps: Optional[int] = None
    seed: Optional[int] = None
    denoise: Optional[float] = None
    # Stacked on top of the workflow's own load-bearing Lightning speed-LoRA (never
    # replaces it -- that LoRA is what makes the 4-step KSampler config work at all).
    # See _extend_lora_chain.
    loras: Optional[List[LoraSpec]] = None
    filename: Optional[str] = None
    # See ComfyUIGenerateRequest.gpu_id.
    gpu_id: Optional[int] = None


class ComfyUIStyleTransferRequest(BaseModel):
    content_image_base64: str
    style_image_base64: str
    prompt: str = "Apply the artistic style of the second image to the first image, keeping the composition and content of the first image."
    # Multi-image-capable 2511 checkpoints (TextEncodeQwenImageEditPlus).
    # qwen-image-edit-2511-Q2_K.gguf offers faster generation with multi-image support.
    model_name: str = "qwen-image-edit-2511-Q2_K.gguf"
    steps: Optional[int] = None
    seed: Optional[int] = None
    denoise: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Denoise strength for KSampler (0.0 = retain content, 1.0 = full style transfer)")
    # Stacked on top of the workflow's own load-bearing Lightning speed-LoRA (never
    # replaces it). See _extend_lora_chain.
    loras: Optional[List[LoraSpec]] = None
    filename: Optional[str] = None
    # See ComfyUIGenerateRequest.gpu_id.
    gpu_id: Optional[int] = None


class ComfyUITaskResponse(BaseModel):
    task_id: str
    status: str
    progress: Dict[str, Any]
    model_used: str
    runtime_seconds: float
    image_base64: Optional[str] = None
    error: Optional[str] = None
    # Basename actually written under GENERATED_IMAGES_DIR, if a filename was
    # requested and the save succeeded -- null otherwise (no filename requested,
    # still pending, or the save itself failed after a successful generation).
    filename: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    # extensions: layout = elements with a relative box [left, top, right, bottom] in 0..1 (models with `layout`,
    # e.g. MOSS-Image, image_serve/moss.py);
    # other models come from image plugins (backend/plugins/images.py; e.g. Iris in sd-serve)
    model: Optional[str] = None
    layout: Optional[List[Dict[str, Any]]] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    # extension: an aspect ratio from GET /v1/images/models instead of `size`; each provider turns it into a
    # size it was trained for (backend/images/catalog.py)
    aspect_ratio: Optional[str] = None


class ImageGenerationResponseDataItem(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageGenerationResponseDataItem]


# ---------------------------------------------------------------------------
# Diffusion model / LoRA listing
# ---------------------------------------------------------------------------


async def _query_live_comfyui_models() -> set:
    await _ensure_comfyui_running()
    models: set = set()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for node_type in ("UnetLoaderGGUF", "UNETLoader"):
                resp = await client.get(f"http://127.0.0.1:{state.comfyui_port}/object_info/{node_type}")
                resp.raise_for_status()
                info = resp.json()
                options = info.get(node_type, {}).get("input", {}).get("required", {}).get("unet_name")
                if options:
                    models.update(options[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query ComfyUI for available models: {e}")
    return models


@router.get("/api/comfyui/models", summary="List available diffusion model checkpoints for image generation")
async def list_comfyui_models():
    # Queries ComfyUI's own node schemas rather than scanning a directory ourselves —
    # this is the actual, currently-valid set of model_name values for
    # /api/comfyui/generate (UnetLoaderGGUF for Ernie Turbo/Z-Image-Turbo's GGUF
    # checkpoints, UNETLoader for Ideogram-4's safetensors ones), so it can't drift out
    # of sync with what ComfyUI would actually accept the way a hardcoded or
    # independently-scanned list could.
    models = await _query_live_comfyui_models()
    return {"models": sorted(models)}


def _request_field_summary(model_cls) -> Dict[str, Any]:
    """Introspect a request model's actual Pydantic fields (name/type/default/required)
    instead of hand-maintaining a parallel description of its parameters — this is the
    literal set of fields FastAPI validates against, so it can never drift out of sync
    with what the endpoint really accepts the way a hand-written param list could.
    """
    fields = {}
    for name, field in model_cls.model_fields.items():
        default = field.default
        if default is PydanticUndefined:
            default = None
        type_str = str(field.annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")
        fields[name] = {"type": type_str, "required": field.is_required(), "default": default}
    return fields


@router.get("/api/comfyui/workflows", summary="List image workflows, their endpoint, routing rule, and parameters")
async def list_comfyui_workflows():
    live_models = await _query_live_comfyui_models()

    def _model_entries(names):
        return [{"name": n, "available": n in live_models} for n in names]

    workflows = [
        {
            "name": "Ernie Turbo",
            "workflow_file": "Ernie Turbo.json",
            "endpoint": "/api/comfyui/generate",
            "request_model": _request_field_summary(ComfyUIGenerateRequest),
            "routing_rule": 'Selected when model_name contains "ernie" (case-insensitive).',
            "known_models": _model_entries(["ernie-image-turbo-Q6_K.gguf", "ernie-image-turbo-Q4_1.gguf", "ernie-image-turbo-UD-Q4_K_M.gguf"]),
        },
        {
            "name": "Z-Image-Turbo",
            "workflow_file": "Z-Image-Turbo.json",
            "endpoint": "/api/comfyui/generate",
            "request_model": _request_field_summary(ComfyUIGenerateRequest),
            "routing_rule": 'Default — selected when model_name contains neither "ernie" nor "ideogram". Passing an unrelated checkpoint (e.g. a Flux/Chroma/qwen-image-* filename) still routes here and will fail with a CLIP/VAE tensor-shape mismatch, since this rule is a substring match, not an allowlist.',
            "known_models": _model_entries(["z_image_turbo-Q6_K.gguf"]),
        },
        {
            "name": "Qwen-Image-2.1",
            "workflow_file": "Qwen-Image-2.1.json",
            "endpoint": "/api/comfyui/generate",
            "request_model": _request_field_summary(ComfyUIGenerateRequest),
            "routing_rule": 'Selected when model_name contains "qwen-image-2.1", "qwen_image_2.1", or "qwen-image-2-1".',
            "known_models": _model_entries(["qwen-image-2.1-UC-Q4_K_M.gguf", "qwen-image-2.1-Q4_K_M.gguf"]),
            "notes": "Text-to-image only (no reference image support here — see Qwen-Image-Edit for that). "
                     "aspect_ratio is ignored: TextEncodeQwenImage21's own resolution input only supports a "
                     "square output without a reference image, so base_resolution is used as-is. "
                     "cfg MUST be 1 (verified live) — cfg=4 produced a severely blown-out, mirrored/duplicated "
                     "image; cfg=1 fixed it, even with no Lightning-style LoRA in this template, suggesting "
                     "this 'UC' checkpoint is itself guidance-distilled. Text encoder must be the full VL "
                     "checkpoint including its vision tower (qwen3vl_8b_int8_convrot.safetensors) — confirmed "
                     "live that a text-only GGUF of the *same* real Qwen3-VL-8B-Instruct model (unsloth's) "
                     "cannot work here at all: ComfyUI's own te-family auto-detection (comfy/sd.py's "
                     "detect_te_model()) only recognizes the qwen3vl family via a vision-tower tensor key "
                     "(model.visual.deepstack_merger_list...), so a vision-stripped checkpoint silently falls "
                     "through to the wrong (smaller, ~2560-dim) text-model branch regardless of the real "
                     "language tower's size or quant — not a quality/VRAM tradeoff, a hard mechanism "
                     "requirement. Needed a ComfyUI core upgrade (v0.34.2 -> v0.37.0, first landed the "
                     "qwen_image21 model class + TextEncodeQwenImage21 node) plus two small patches to the "
                     "installed ComfyUI-GGUF custom node: qwen_image21 wasn't in loader.py's IMG_ARCH_LIST "
                     "(the primary path, keyed off the GGUF's own general.architecture metadata) or in "
                     "tools/convert.py's arch_list (a tensor-key-based fallback loader.py uses when that "
                     "metadata is absent or unrecognized -- needed separately because unsloth's own "
                     "Qwen-Image-2.1 GGUF conversion doesn't set that metadata field the same way "
                     "abenzerps' does, so it takes the fallback path the other quant never exercised); "
                     "loader.py's 1D-tensor dequant also only special-cased BF16, not the other quant types "
                     "stable-diffusion.cpp -- abenzerps' own conversion tool -- uses for norm weights. "
                     "Two quants verified live: 'UC' (abenzerps, an uncensored community fine-tune) and the "
                     "plain unsloth quant of the official Qwen/Qwen-Image-2.1 base model -- both need cfg=1, "
                     "both work with the identical text encoder/VAE pairing, both produce clean high-quality "
                     "output for an asymmetric subject (tested: a cat on a table) and a mirrored/doubled "
                     "wheel artifact on a symmetric one (tested: a bicycle) -- reproducing on the *base* "
                     "model too rules out 'community fine-tune quality issue' as the explanation; more "
                     "likely a real characteristic of this pipeline/architecture on symmetric subjects, "
                     "still not root-caused.",
        },
        {
            "name": "Ideogram-4",
            "workflow_file": "Ideogram-4.json",
            "endpoint": "/api/comfyui/generate",
            "request_model": _request_field_summary(ComfyUIGenerateRequest),
            "routing_rule": 'Selected when model_name contains "ideogram".',
            "known_models": _model_entries(["ideogram4_nvfp4_mixed.safetensors"]),
            "notes": "Dual-model guidance — also always loads ideogram4_unconditional_nvfp4_mixed.safetensors "
                     "(fixed pairing, not selectable via model_name). lora_name is a no-op here (no LoRA node in "
                     "this template). Has a safety filter baked into the model weights with a much higher "
                     "false-positive rate on plain-text prompts than on structured JSON captions "
                     "(verified: 2/2 plain-text blocked, 4/4 structured-JSON clean in testing) — prefer JSON "
                     "captions, ideally built with the installed Ideogram4PromptBuilderKJ node in ComfyUI's own UI.",
        },
        {
            "name": "Qwen-Image-Edit",
            "workflow_file": "Qwen-Image-Edit.json",
            "endpoint": "/api/comfyui/edit",
            "request_model": _request_field_summary(ComfyUIEditRequest),
            "routing_rule": None,
            "known_models": _model_entries(["qwen-image-edit-2511-Q2_K.gguf", "qwen-image-edit-2511-Q3_K_S.gguf"]),
            "notes": "Single reference image. Qwen-Image-Edit 2511 through TextEncodeQwenImageEditPlus with lightx2v's 2511 "
                     "Lightning 4-step LoRA (2026-10-21: with the older node and the older model's LoRA the 2511 "
                     "checkpoints ignored instructions). Keeps a character the same across many images. The older "
                     "pre-2511 checkpoints were removed; their names now mean the 2511 Q2_K.",
        },
        {
            "name": "Qwen-Image-Style-Transfer",
            "workflow_file": "Qwen-Image-Style-Transfer.json",
            "endpoint": "/api/comfyui/style-transfer",
            "request_model": _request_field_summary(ComfyUIStyleTransferRequest),
            "routing_rule": None,
            "known_models": _model_entries(["qwen-image-edit-2511-Q3_K_S.gguf", "qwen-image-edit-2511-Q2_K.gguf"]),
            "notes": "Needs a multi-image-capable checkpoint (TextEncodeQwenImageEditPlus) — both '2511' quants "
                     "tested and confirmed working (same fox+style-reference pair produced a correctly blended "
                     "result on both Q3_K_S and Q2_K). Uses the 2511 Lightning LoRA since 2026-10-21.",
        },
    ]
    return {"workflows": workflows}


_LORA_METADATA_FIELDS = ("trigger_words", "description", "base_model", "recommended_strength")


def _read_lora_metadata(lora_abs_path: str) -> Dict[str, Any]:
    """Optional JSON sidecar with the same stem as a LoRA .safetensors file --
    e.g. z-image-pen.safetensors + z-image-pen.json -- same convention voices/*.wav
    + voices/*.txt already uses for transcripts, just JSON here since a LoRA has
    several distinct, genuinely optional fields (trigger words to put in the
    prompt, a free-text description, which base checkpoint it's compatible with,
    a recommended strength) rather than one blob of free text.

    All fields optional, and a missing/malformed sidecar just means no metadata --
    never an error, and never required for the LoRA to work exactly as it did
    before this existed. Only recognized fields are surfaced (not an arbitrary
    passthrough of whatever the JSON contains) so the API response shape stays
    predictable regardless of what someone puts in the file.
    """
    json_path = os.path.splitext(lora_abs_path)[0] + ".json"
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
    except Exception as e:
        logger.warning(f"Failed to read LoRA metadata sidecar {json_path}: {e}")
        return {}
    return {k: data[k] for k in _LORA_METADATA_FIELDS if data.get(k) is not None}


@router.get("/api/comfyui/loras", summary="List available LoRA files for image generation")
async def list_comfyui_loras():
    """Recursively lists LoRA .safetensors files, each optionally enriched with
    trigger_words/description/base_model/recommended_strength from a same-name
    .json sidecar (see _read_lora_metadata) -- present only when set, so a LoRA
    with no sidecar is just {"name": "..."}, unchanged from before this existed.
    """
    if not os.path.isdir(_COMFYUI_LORAS_DIR):
        return {"loras": []}
    # LoRAs can live in per-model subfolders (e.g. "z-image-turbo/foo.safetensors") —
    # walk recursively and return paths exactly as ComfyUI's own LoraLoaderModelOnly
    # combo reports them (forward-slash-joined, relative to the loras root), since
    # that's the literal string comfyui_generate() needs to pass back as lora_name.
    loras = []
    for root, _dirs, files in os.walk(_COMFYUI_LORAS_DIR):
        rel_root = os.path.relpath(root, _COMFYUI_LORAS_DIR)
        for f in files:
            if not f.lower().endswith(".safetensors") or f.startswith("put_"):
                continue
            rel_path = f if rel_root == "." else f"{rel_root}/{f}"
            entry = {"name": rel_path}
            entry.update(_read_lora_metadata(os.path.join(root, f)))
            loras.append(entry)
    loras.sort(key=lambda x: x["name"])
    return {"loras": loras}


@router.get("/api/comfyui/images", summary="List images saved via the filename option (generate/edit/style-transfer/batch)")
async def list_comfyui_images():
    if not os.path.isdir(GENERATED_IMAGES_DIR):
        return {"images": []}
    files = [f for f in os.listdir(GENERATED_IMAGES_DIR) if f.lower().endswith(".png")]
    return {"images": sorted(files)}


@router.get("/api/comfyui/images/{filename}", summary="Fetch a saved image by filename")
async def get_comfyui_image(filename: str):
    try:
        safe_name = _sanitize_image_filename(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    path = os.path.join(GENERATED_IMAGES_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No saved image named '{safe_name}'.")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------


@router.post("/api/comfyui/start", summary="Start ComfyUI instance")
async def start_comfyui(req: ComfyUIStartRequest):
    if state.comfyui_process and state.comfyui_process.poll() is None:
        return {
            "status": "already running", "port": state.comfyui_port, "pid": state.comfyui_pid,
            "gpu_id": state.comfyui_gpu_id,
        }
    # Unnamed GPU -> COMFYUI_DEFAULT_GPU_ID (config/env); covers explicit starts and
    # every auto-start via _ensure_comfyui_running(), which funnels through here.
    gpu_id = req.gpu_id if req.gpu_id is not None else COMFYUI_DEFAULT_GPU_ID
    if gpu_id == "auto":   # the GPU with the most room right now (it may move later, see _comfyui_gpu_preflight)
        from image_serve.gpu import MANAGER
        room = await MANAGER.room()
        gpu_id = max(room, key=lambda g: (room[g], -g)) if room else None
    from image_serve.gpu import MANAGER
    await MANAGER.raise_if_exclusive(gpu_id, "ComfyUI")  # 507 while an exclusive plugin holds the GPUs
    # Loopback only: who else reaches ComfyUI's UI is chosen in the dashboard's
    # Services tab (Tailscale levels), not by listening on every interface.
    cmd = [COMFYUI_PYTHON, "main.py", "--port", str(state.comfyui_port), "--listen", "127.0.0.1"]
    cmd += extra_model_paths_args(COMFYUI_DIR, COMFYUI_MODELS_ROOT, LOGS_DIR)
    if gpu_id is not None:
        cmd += ["--cuda-device", str(gpu_id)]
    try:
        log_path = os.path.join(LOGS_DIR, "comfyui.log")
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=COMFYUI_DIR, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        state.comfyui_process = proc
        state.comfyui_pid = proc.pid
        state.comfyui_gpu_id = gpu_id
        state.comfyui_log_file = log_file
        return {"status": "started", "pid": proc.pid, "port": state.comfyui_port, "gpu_id": gpu_id}
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _comfyui_busy_reason() -> Optional[str]:
    """Why ComfyUI must not be stopped/unloaded right now, or None if idle. Checks
    both our own tracked tasks (submitted via this app, image maybe still being
    retrieved) and ComfyUI's own /queue (also catches jobs queued straight into its
    native UI, which never touch comfyui_tasks)."""
    tracked = len([t for t in state.comfyui_tasks.values() if t.get("status") in ("pending", "processing")])
    if tracked:
        return f"{tracked} image task(s) still running via this app"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"http://127.0.0.1:{state.comfyui_port}/queue")
            if resp.status_code == 200:
                data = resp.json()
                n = len(data.get("queue_running") or []) + len(data.get("queue_pending") or [])
                if n:
                    return f"{n} job(s) in ComfyUI's own queue"
    except Exception:
        pass  # not reachable -> nothing running in it either
    return None


async def _raise_if_comfyui_busy(force: bool, action: str):
    if force:
        return
    reason = await _comfyui_busy_reason()
    if reason:
        raise HTTPException(
            status_code=409,
            detail=(f"ComfyUI is busy ({reason}); refusing to {action} it. Wait for it to "
                    "go idle, or pass force=true to do it anyway."),
        )


@router.post("/api/comfyui/unload", summary="Free ComfyUI's VRAM (unload models) but keep the process running")
async def unload_comfyui(force: bool = False):
    """Same /free call the idle timer makes. The process stays up, so the next
    generate skips the ~13s ComfyUI startup and only pays the model load (~11s for
    Z-Image-Turbo). 409 while a job is running unless force=true -- note that with
    force, ComfyUI itself still finishes (and then frees after) a running job."""
    await _raise_if_comfyui_busy(force, "unload")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            _comfyui_models_freed()
            await client.post(f"http://127.0.0.1:{state.comfyui_port}/free",
                              json={"unload_models": True, "free_memory": True})
    except httpx.HTTPError:
        return {"status": "not running"}
    task = state.comfyui_idle_task
    if task and not task.done():
        task.cancel()
    state.comfyui_active_model = None
    logger.info("ComfyUI VRAM freed on request.")
    return {"status": "unloaded"}


@router.post("/api/comfyui/stop", summary="Stop ComfyUI instance")
async def stop_comfyui(force: bool = False):
    await _raise_if_comfyui_busy(force, "stop")
    task = state.comfyui_idle_task
    if task and not task.done():
        task.cancel()
    if state.comfyui_process and state.comfyui_process.poll() is None:
        state.comfyui_process.terminate()
        state.comfyui_process.wait()
        state.comfyui_process = None
        state.comfyui_pid = None
        state.comfyui_gpu_id = None
        if hasattr(state, "comfyui_log_file") and state.comfyui_log_file:
            try:
                state.comfyui_log_file.close()
            except Exception:
                pass
            state.comfyui_log_file = None
        return {"status": "stopped"}
    # state.comfyui_process is empty (e.g. adoption hasn't run, or this is a stale
    # request racing a restart) -- find the real process directly by cwd/cmdline
    # rather than kill_process_on_port(), which only ever matches "llama-server"
    # in the cmdline and would silently no-op here, leaving ComfyUI running.
    proc = find_comfyui_process()
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if hasattr(state, "comfyui_log_file") and state.comfyui_log_file:
        try:
            state.comfyui_log_file.close()
        except Exception:
            pass
        state.comfyui_log_file = None
    state.comfyui_gpu_id = None
    return {"status": "stopped (no active managed process)"}


@router.get("/api/comfyui/status", summary="Get status of ComfyUI instance")
async def get_comfyui_status():
    running = state.comfyui_process is not None and state.comfyui_process.poll() is None
    ready = False
    pid = state.comfyui_pid if running else None
    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            resp = await client.get(f"http://127.0.0.1:{state.comfyui_port}/system_stats")
            if resp.status_code == 200:
                ready = True
                running = True
                if not pid:
                    pid = get_pid_on_port(state.comfyui_port)
    except Exception:
        pass
    return {
        "running": running, "ready": ready, "port": state.comfyui_port, "pid": pid,
        "gpu_id": state.comfyui_gpu_id if running else None,
    }


@router.get("/api/comfyui/progress", summary="Get current image generation progress")
async def get_comfyui_progress():
    return state.comfyui_progress


@router.get("/api/comfyui/task/{task_id}", summary="Get ComfyUI task status and output", response_model=ComfyUITaskResponse)
async def get_comfyui_task(task_id: str):
    if task_id not in state.comfyui_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = state.comfyui_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", {}),
        "model_used": task.get("model_used", ""),
        "runtime_seconds": task.get("runtime_seconds", 0.0),
        "image_base64": task.get("image_base64"),
        "error": task.get("error"),
        "filename": task.get("filename"),
    }


# ---------------------------------------------------------------------------
# ComfyUI image generation internals
# ---------------------------------------------------------------------------

COMFYUI_START_TIMEOUT_S = int(os.environ.get("COMFYUI_START_TIMEOUT_S") or "120")


async def _ensure_comfyui_running(gpu_id: Optional[int] = None):
    """Auto-starts ComfyUI if it isn't already up -- called from every generate/edit/
    style-transfer/models/workflows entry point so a caller never has to remember to
    call /api/comfyui/start explicitly first.

    `gpu_id` only has an effect on a *fresh* auto-start: ComfyUI is one persistent
    process shared by every caller, so once it's running, its GPU is already fixed
    (see ComfyUIStartRequest.gpu_id) -- a later request's gpu_id can't retroactively
    move it without killing whatever else might be mid-generation on it. A mismatch
    is logged, not silently ignored, so an operator can tell "wrong GPU" from "no GPU
    request went through at all" -- restarting it on the right GPU is a deliberate
    POST /api/comfyui/stop + /api/comfyui/start(gpu_id=...), not automatic here.
    """
    status = await get_comfyui_status()
    if not status["ready"]:
        logger.info(f"ComfyUI not ready — starting automatically (gpu_id={gpu_id})...")
        await start_comfyui(ComfyUIStartRequest(port=state.comfyui_port, gpu_id=gpu_id))
        # a first start sets up its database and scans the models (found 2026-10-22: >15 s on a fresh ComfyUI;
        # slower machines need more) -- wait up to COMFYUI_START_TIMEOUT_S, but not for a process that already died
        for _ in range(COMFYUI_START_TIMEOUT_S):
            await asyncio.sleep(1)
            status = await get_comfyui_status()
            if status["ready"]:
                logger.info("ComfyUI became ready.")
                break
            proc = state.comfyui_process
            if proc is not None and proc.poll() is not None:
                raise HTTPException(status_code=500, detail="ComfyUI exited while starting -- see its log (logs/comfyui.log).")
        else:
            raise HTTPException(status_code=500, detail=f"ComfyUI didn't start within {COMFYUI_START_TIMEOUT_S} s -- see logs/comfyui.log.")
    elif gpu_id is not None and state.comfyui_gpu_id != gpu_id:
        logger.warning(
            f"ComfyUI is already running on gpu_id={state.comfyui_gpu_id}; "
            f"ignoring requested gpu_id={gpu_id} for this call. Stop and restart "
            f"ComfyUI (POST /api/comfyui/stop, then /api/comfyui/start with the "
            f"desired gpu_id) to actually move it to a different GPU."
        )


COMFYUI_IDLE_FREE_S = 180


def _reset_comfyui_idle_timer(port: int):
    """Cancel-and-recreate ComfyUI's single idle-unload timer (same convention as the
    TTS/STT keep-alives). Called on every job submit AND completion, so the countdown
    only ever runs from the most recent activity -- previously each finished job
    spawned its own independent 300s timer that freed the model if the queue happened
    to be empty at that instant, so a steady stream of jobs ~2-3 min apart got its
    model unloaded (and reloaded) between almost every pair of jobs."""
    task = state.comfyui_idle_task
    if task and not task.done():
        task.cancel()
    state.comfyui_idle_task = asyncio.create_task(_comfyui_idle_free(port))


async def _comfyui_idle_free(port: int):
    while True:
        await asyncio.sleep(COMFYUI_IDLE_FREE_S)
        try:
            async with httpx.AsyncClient() as client:
                queue_resp = await client.get(f"http://127.0.0.1:{port}/queue")
                if queue_resp.status_code != 200:
                    return
                data = queue_resp.json()
                # Jobs queued straight into ComfyUI (its native UI) don't go through
                # our submit/complete hooks -- if the queue is busy, keep waiting
                # rather than giving up and leaving the model pinned forever.
                if data.get("queue_running") or data.get("queue_pending"):
                    continue
                _comfyui_models_freed()
                await client.post(f"http://127.0.0.1:{port}/free", json={"unload_models": True, "free_memory": True})
                state.comfyui_active_model = None
                logger.info(f"ComfyUI VRAM freed after {COMFYUI_IDLE_FREE_S}s idle.")
                return
        except Exception as e:
            logger.error(f"Failed VRAM cleanup: {e}")
            return


async def _track_and_retrieve_comfyui(
    client_id: str, prompt_id: str, port: int, start_time: float, model_name: str, filename: Optional[str] = None,
):
    state.comfyui_tasks[prompt_id] = {
        "status": "pending",
        "progress": {"step": 0, "max_steps": 0, "node": "", "value": 0},
        "model_used": model_name,
        "runtime_seconds": 0.0,
        "image_base64": None,
        "error": None,
        "filename": None,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{port}/ws?clientId={client_id}") as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        msg_data = data.get("data", {})
                        if msg_type == "progress":
                            step = msg_data.get("value", 0)
                            max_steps = msg_data.get("max", 0)
                            percent = round((step / max_steps) * 100, 1) if max_steps > 0 else 0.0
                            progress_info = {"step": step, "max_steps": max_steps, "node": state.comfyui_progress.get("node", ""), "value": percent}
                            state.comfyui_progress.update(progress_info)
                            if prompt_id in state.comfyui_tasks:
                                state.comfyui_tasks[prompt_id]["progress"] = progress_info
                                state.comfyui_tasks[prompt_id]["status"] = "processing"
                        elif msg_type == "executing":
                            node = msg_data.get("node")
                            if node is None and msg_data.get("prompt_id") == prompt_id:
                                break
                            elif node:
                                state.comfyui_progress["node"] = node
                                if prompt_id in state.comfyui_tasks:
                                    state.comfyui_tasks[prompt_id]["progress"]["node"] = node
                                    state.comfyui_tasks[prompt_id]["status"] = "processing"
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
    except Exception as e:
        logger.error(f"WebSocket tracking failed: {e}")

    image_base64 = None
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for _ in range(60):
                await asyncio.sleep(1)
                hist_resp = await client.get(f"http://127.0.0.1:{port}/history/{prompt_id}")
                if hist_resp.status_code == 200:
                    hist_data = hist_resp.json()
                    if prompt_id in hist_data:
                        failed = history_error(hist_data[prompt_id])
                        if failed:      # ComfyUI gave up on it: say why now instead of waiting out the timeout
                            if prompt_id in state.comfyui_tasks:
                                state.comfyui_tasks[prompt_id].update({"status": "failed", "error": failed})
                            break
                        outputs = hist_data[prompt_id].get("outputs", {})
                        for _, node_output in outputs.items():
                            if "images" in node_output:
                                image_info = node_output["images"][0]
                                img_url = (f"http://127.0.0.1:{port}/view"
                                           f"?filename={image_info['filename']}"
                                           f"&subfolder={image_info['subfolder']}"
                                           f"&type={image_info['type']}")
                                img_resp = await client.get(img_url)
                                if img_resp.status_code == 200:
                                    image_base64 = base64.b64encode(img_resp.content).decode("utf-8")
                                    break
                        if image_base64:
                            break
    except Exception as e:
        logger.error(f"Error polling history: {e}")
        if prompt_id in state.comfyui_tasks:
            state.comfyui_tasks[prompt_id]["status"] = "failed"
            state.comfyui_tasks[prompt_id]["error"] = str(e)

    state.comfyui_last_used = time.time()
    _reset_comfyui_idle_timer(port)
    _release_comfyui_reservation()

    saved_filename = None
    if image_base64 and filename:
        # Already validated (sanitized) by the caller at request time -- this
        # re-sanitizes defensively rather than trusting that, and only fails the
        # save (logged), not the whole generation: the image is still returned via
        # image_base64 either way, saving to disk is strictly additive.
        try:
            safe_name = _sanitize_image_filename(filename)
            save_path = os.path.join(GENERATED_IMAGES_DIR, safe_name)
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
            saved_filename = safe_name
            logger.info(f"Saved generated image to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save generated image as '{filename}': {e}")

    if prompt_id in state.comfyui_tasks:
        if image_base64:
            state.comfyui_tasks[prompt_id].update({
                "status": "completed", "image_base64": image_base64,
                "runtime_seconds": round(time.time() - start_time, 2), "filename": saved_filename,
            })
        elif state.comfyui_tasks[prompt_id]["status"] != "failed":
            state.comfyui_tasks[prompt_id].update({"status": "failed", "error": "Failed to retrieve generated image within timeout."})


def history_error(entry: Any) -> Optional[str]:
    """A ComfyUI /history entry that ended in an error -> a short reason (node type + ComfyUI's message, no
    traceback -- it carries paths), else None. Before this a failed prompt looked like a missing image and the caller
    waited 60 s for "Failed to retrieve generated image within timeout" (found 2026-10-23)."""
    status = entry.get("status") if isinstance(entry, dict) else None
    if not isinstance(status, dict) or status.get("status_str") != "error":
        return None
    for m in status.get("messages") or []:
        if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == "execution_error" and isinstance(m[1], dict):
            d = m[1]
            msg = " ".join(str(d.get("exception_message") or "").split())[:240]
            return f"ComfyUI failed in {d.get('node_type') or 'a node'}: {msg or d.get('exception_type') or 'error'}"
    return "ComfyUI failed on this image (see the plugin's ComfyUI log)"


def _extend_lora_chain(workflow: dict, anchor_lora_node_id: str, extra_loras: List[LoraSpec]) -> None:
    """Stacks additional LoraLoaderModelOnly nodes after one that's already wired
    into a workflow (anchor_lora_node_id), rewiring whichever node currently
    consumes the anchor's "model" output to read from the end of the new chain
    instead. Used for both:
      - Z-Image-Turbo/Ernie generate: anchor is the template's own optional LoRA
        node, already activated (or not) by the existing lora_name handling above.
      - Qwen-Image-Edit / Qwen-Image-Style-Transfer: anchor is a load-bearing
        Lightning speed-LoRA baked into the template (required for correct
        4-step sampling) — extra_loras get chained AFTER it, never replacing it.

    No-op if extra_loras is empty, so zero/single-LoRA behavior is byte-for-byte
    unchanged for every existing caller (including the batch endpoint, which only
    ever sets lora_name/lora_strength).
    """
    if not extra_loras:
        return
    consumers = [
        (nid, key)
        for nid, node in workflow.items()
        for key, value in node.get("inputs", {}).items()
        if isinstance(value, list) and value[0] == anchor_lora_node_id
    ]
    current_id = anchor_lora_node_id
    for i, lora in enumerate(extra_loras):
        new_id = f"{anchor_lora_node_id}_extra{i}"
        workflow[new_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [current_id, 0],
                "lora_name": lora.name,
                "strength_model": lora.strength,
            },
            "_meta": {"title": f"Extra LoRA {i + 1}"},
        }
        current_id = new_id
    for nid, key in consumers:
        workflow[nid]["inputs"][key] = [current_id, 0]


def _build_patched_generate_workflow(req: ComfyUIGenerateRequest) -> dict:
    """Load the right workflow template for req.model_name and patch in prompt/size/
    seed/LoRA -- the same node-patching logic /api/comfyui/generate has always used,
    extracted so the batch runner below can reuse it verbatim per item."""
    _comfyui_dir = WORKFLOWS_DIR
    model_lower = req.model_name.lower()
    if "ernie" in model_lower:
        workflow_file = "Ernie Turbo.json"
    elif "ideogram" in model_lower:
        workflow_file = "Ideogram-4.json"
    elif "qwen-image-2.1" in model_lower or "qwen_image_2.1" in model_lower or "qwen-image-2-1" in model_lower:
        workflow_file = "Qwen-Image-2.1.json"
    else:
        workflow_file = "Z-Image-Turbo.json"
    workflow_path = os.path.join(_comfyui_dir, workflow_file)
    try:
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load workflow template: {e}")

    base_res = req.base_resolution
    width, height = base_res, base_res
    if req.aspect_ratio == "3:4":
        width, height = int(base_res * 0.75), base_res
    elif req.aspect_ratio == "4:3":
        width, height = base_res, int(base_res * 0.75)
    elif req.aspect_ratio == "16:9":
        width, height = base_res, int(base_res * (9 / 16))
    elif req.aspect_ratio == "9:16":
        width, height = int(base_res * (9 / 16)), base_res
    width = (width // 64) * 64
    height = (height // 64) * 64

    # Ernie Turbo / Z-Image-Turbo each carry a LoraLoaderModelOnly node wired between
    # their base checkpoint and KSampler, but KSampler's "model" input points straight
    # at the base checkpoint by default in the template — the LoRA node sits present
    # but disconnected from the execution path unless a LoRA is requested, at which
    # point we rewire KSampler to route through it instead.
    lora_node_id = next((nid for nid, n in workflow.items() if n.get("class_type") == "LoraLoaderModelOnly"), None)
    # loras (list) takes precedence over the older singular lora_name/lora_strength,
    # which stay around as shorthand for existing single-LoRA callers (e.g. batch).
    # effective_loras[0] activates the template's own LoRA node exactly like
    # lora_name always has; effective_loras[1:] get stacked on via _extend_lora_chain
    # after the loop below.
    if req.loras:
        effective_loras = list(req.loras)
    elif req.lora_name:
        effective_loras = [LoraSpec(name=req.lora_name, strength=req.lora_strength if req.lora_strength is not None else 1.0)]
    else:
        effective_loras = []

    for node_id, node in list(workflow.items()):
        class_type = node.get("class_type")
        title = node.get("_meta", {}).get("title", "")
        if class_type == "FluxResolutionNode":
            del workflow[node_id]
        elif class_type == "KSampler":
            node["inputs"]["seed"] = req.seed if req.seed is not None else random.randint(1, 10**14 - 1)
            if req.steps is not None:
                node["inputs"]["steps"] = req.steps
            if effective_loras and lora_node_id:
                node["inputs"]["model"] = [lora_node_id, 0]
        elif class_type == "CLIPTextEncode":
            node["inputs"]["text"] = req.negative_prompt if "Negative" in title else req.prompt
        elif class_type == "UnetLoaderGGUF":
            node["inputs"]["unet_name"] = req.model_name
            if req.model_name.lower().endswith(".safetensors"):
                # e.g. Qwen-Image 2.1's official int8 checkpoint: the core loader, not ComfyUI-GGUF's
                node["class_type"] = "UNETLoader"
                node["inputs"]["weight_dtype"] = "default"
        elif class_type == "UNETLoader" and "Unconditional" not in title:
            # Ideogram-4's dual-model guidance pairs one fixed "unconditional" checkpoint
            # (DualModelGuider) with the user-selectable conditional one — only the
            # conditional loader should ever be overridden by model_name.
            node["inputs"]["unet_name"] = req.model_name
        elif class_type == "LoraLoaderModelOnly" and effective_loras:
            node["inputs"]["lora_name"] = effective_loras[0].name
            node["inputs"]["strength_model"] = effective_loras[0].strength
        elif class_type in ["EmptySD3LatentImage", "EmptyFlux2LatentImage", "EmptyLatentImage"]:
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
        elif class_type == "TextEncodeQwenImage21":
            # This node's own "resolution" input is a single int -- it produces a
            # SQUARE latent (resolution x resolution) with no reference image
            # supplied, unlike every other workflow's separate Empty*LatentImage
            # width/height. A caller-requested aspect_ratio is therefore not
            # achievable through plain /api/comfyui/generate for this model (the
            # node's own width/height computation only activates when a reference
            # image is given, which is an edit-style input this endpoint doesn't
            # take) -- base_resolution is used as-is and aspect_ratio is ignored,
            # rather than silently squashing/cropping to a wrong aspect.
            node["inputs"]["prompt"] = req.prompt
            node["inputs"]["negative_prompt"] = req.negative_prompt
            node["inputs"]["resolution"] = req.base_resolution
        elif class_type == "Ideogram4Scheduler":
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            if req.steps is not None:
                node["inputs"]["steps"] = req.steps
        elif class_type == "RandomNoise":
            node["inputs"]["noise_seed"] = req.seed if req.seed is not None else random.randint(1, 10**14 - 1)

    if lora_node_id and len(effective_loras) > 1:
        _extend_lora_chain(workflow, lora_node_id, effective_loras[1:])

    return workflow


# --------------------------------------------------------------------------- GPU manager
# ComfyUI asks the central GPU manager before a job -- only when the job needs other
# model files than the ones already loaded (rules: tests/unit/test_comfyui_gpu.py).


COMFY_SAMPLING_MARGIN_MB = 1500   # activations / latents while sampling, on top of the weights
_COMFY_LOADERS = {
    "UNETLoader": ("unet_name",), "UnetLoaderGGUF": ("unet_name",), "CheckpointLoaderSimple": ("ckpt_name",),
    "CLIPLoader": ("clip_name",), "CLIPLoaderGGUF": ("clip_name",), "DualCLIPLoader": ("clip_name1", "clip_name2"),
    "VAELoader": ("vae_name",), "LoraLoaderModelOnly": ("lora_name",), "LoraLoader": ("lora_name",),
}
_COMFY_MODEL_SUBDIRS = ("diffusion_models", "unet", "text_encoders", "clip", "vae", "loras", "checkpoints")


def comfy_model_files(workflow: Dict[str, Any]) -> frozenset:
    """Model files a workflow makes ComfyUI load. A LoRA node only counts when some
    node uses its output (the templates carry a bypassed LoRA node)."""
    used = {str(v[0]) for n in workflow.values() if isinstance(n, dict)
            for v in (n.get("inputs") or {}).values() if isinstance(v, list) and v}
    files = set()
    for nid, node in workflow.items():
        keys = _COMFY_LOADERS.get((node or {}).get("class_type"))
        if not keys or (node["class_type"].startswith("Lora") and str(nid) not in used):
            continue
        for k in keys:
            v = (node.get("inputs") or {}).get(k)
            if isinstance(v, str) and v:
                files.add(v)
    return frozenset(files)


def _comfy_file_mb(name: str) -> int:
    roots = [COMFYUI_MODELS_ROOT, os.path.join(COMFYUI_DIR, "models")]
    for root in roots:
        for sub in _COMFY_MODEL_SUBDIRS:
            p = os.path.join(root, sub, name)
            if os.path.isfile(p):
                return int(os.path.getsize(p) >> 20)
    return 0


async def _comfy_held_mb(gpu: int) -> int:
    from image_serve.gpu import MANAGER
    return await MANAGER.held_mb(gpu)


def _release_comfyui_reservation() -> None:
    """A finished (or failed) job means the models are allocated -- from now on ComfyUI's measured VRAM
    counts. Held open instead, the reservation kept ComfyUI 'loading' (busy, not unloadable) for up to
    2 min after every image and refused other image models in that time (found live 2026-10-18)."""
    rid, state.comfyui_reservation_id = state.comfyui_reservation_id, None
    if rid:
        from image_serve.gpu import MANAGER
        MANAGER.release(rid)


def _comfyui_models_freed() -> None:
    state.comfyui_loaded_models = None
    _release_comfyui_reservation()


async def _restart_comfyui_on(gpu: int) -> None:
    """Move an idle ComfyUI to another GPU -- a restart, since ComfyUI picks its GPU at process start."""
    logger.info(f"Moving ComfyUI to GPU {gpu}")
    await stop_comfyui(force=False)
    await _ensure_comfyui_running(gpu_id=gpu)


async def _comfyui_gpu_preflight(workflow: Dict[str, Any], label: str, pinned: bool = False) -> None:
    files = comfy_model_files(workflow)
    if files and files == state.comfyui_loaded_models:
        return  # already loaded -- nothing to ask
    gpu = state.comfyui_gpu_id if state.comfyui_gpu_id is not None else 0
    # ComfyUI computes with one model at a time (text encoder, then UNet, then VAE) and offloads the others
    # itself when VRAM is short -- the largest one plus the sampling margin is what it can't do without;
    # more room only lets it keep the rest cached. (The sum made ERNIE impossible on the 16 GB card.)
    full = max((_comfy_file_mb(f) for f in files), default=0) + COMFY_SAMPLING_MARGIN_MB
    held = await _comfy_held_mb(gpu)  # ComfyUI frees its old models itself when switching
    need = full - held
    if not pinned and state.comfyui_process is not None:
        # models load anyway (they changed): if another GPU has clearly more room and nothing of ComfyUI's is
        # running, move it there instead of crowding/unloading others (2026-10-19/20: it sat on GPU 1, GPU 0 free)
        from image_serve.gpu import choose_gpu
        from image_serve.gpu import MANAGER
        room = await MANAGER.room()
        room[gpu] = room.get(gpu, 0) + held
        target, mode = choose_gpu(room, full, gpu, movable=await _comfyui_busy_reason() is None)
        if mode == "move":
            await _restart_comfyui_on(target)
            gpu, need = target, full
    if need > 0:
        from image_serve.gpu import MANAGER
        res, _, why = await MANAGER.reserve(gpu, need, f"image:{label}")  # 507 while an exclusive plugin holds it
        if res is None:
            raise HTTPException(status_code=507, detail=f"No room for the image model on GPU {gpu}: {why}")
        if state.comfyui_process is not None and getattr(state.comfyui_process, "pid", None):
            MANAGER.confirm(res.id, state.comfyui_process.pid)
        _release_comfyui_reservation()   # an older one still open is superseded
        state.comfyui_reservation_id = res.id
    state.comfyui_loaded_models = files


async def _queue_comfyui_generate(req: ComfyUIGenerateRequest) -> tuple:
    """Build+queue a generate workflow on ComfyUI. Returns (client_id, prompt_id,
    start_time) -- does NOT wait for completion; caller decides whether to track it
    as a fire-and-forget task (single /generate) or await it inline (batch runner)."""
    start_time = time.time()
    state.comfyui_active_model = req.model_name
    state.comfyui_last_used = start_time
    await _ensure_comfyui_running(gpu_id=req.gpu_id)
    _reset_comfyui_idle_timer(state.comfyui_port)
    workflow = _build_patched_generate_workflow(req)
    await _comfyui_gpu_preflight(workflow, req.model_name or "image", pinned=req.gpu_id is not None)

    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    state.comfyui_progress = {"step": 0, "max_steps": 0, "node": "", "value": 0}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"http://127.0.0.1:{state.comfyui_port}/prompt", json=payload)
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {e}")
    return client_id, prompt_id, start_time


async def queue_workflow(workflow: Dict[str, Any], label: str) -> Dict[str, str]:
    """Queue a workflow built elsewhere (MOSS-Image builds its own, image_serve/moss.py): ComfyUI started if needed,
    GPU room asked for its models, then tracked like any job -- the caller awaits it with _await_comfyui_task."""
    start_time = time.time()
    state.comfyui_active_model = label
    state.comfyui_last_used = start_time
    await _ensure_comfyui_running()
    _reset_comfyui_idle_timer(state.comfyui_port)
    await _comfyui_gpu_preflight(workflow, label)
    client_id = str(uuid.uuid4())
    state.comfyui_progress = {"step": 0, "max_steps": 0, "node": "", "value": 0}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"http://127.0.0.1:{state.comfyui_port}/prompt",
                                     json={"prompt": workflow, "client_id": client_id})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI unreachable: {type(e).__name__}")
    if resp.status_code != 200:   # ComfyUI's validation message names the node/input -- worth passing on
        raise HTTPException(status_code=500, detail=f"ComfyUI refused the {label} workflow: {resp.text[:300]}")
    prompt_id = resp.json().get("prompt_id")
    asyncio.create_task(_track_and_retrieve_comfyui(client_id, prompt_id, state.comfyui_port, start_time, label))
    return {"status": "queued", "task_id": prompt_id}


@router.post("/api/comfyui/generate", summary="Generate image via ComfyUI")
async def comfyui_generate(req: ComfyUIGenerateRequest):
    if req.filename:
        try:
            _sanitize_image_filename(req.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    client_id, prompt_id, start_time = await _queue_comfyui_generate(req)
    asyncio.create_task(
        _track_and_retrieve_comfyui(client_id, prompt_id, state.comfyui_port, start_time, req.model_name, req.filename)
    )
    return {"status": "queued", "task_id": prompt_id}


_COMFY_BATCH_MAX_ITEMS = 500


def _merge_batch_item(defaults: ComfyBatchDefaults, item: ComfyBatchItem) -> dict:
    """Resolve one batch item against the batch's defaults into a full, validated
    ComfyUIGenerateRequest dict -- done once at submit time so a malformed item
    (including a bad filename) fails the /batch call immediately (400) instead of
    surfacing mid-run."""
    merged = defaults.dict()
    merged.update({k: v for k, v in item.dict(exclude={"prompt"}).items() if v is not None})
    merged["prompt"] = item.prompt
    if merged.get("filename"):
        merged["filename"] = _sanitize_image_filename(merged["filename"])
    return ComfyUIGenerateRequest(**merged).dict()


async def _run_comfyui_batch(batch_id: str):
    """Sequential runner for an image-generation batch: one item at a time (ComfyUI
    itself has no concept of "my" queue vs. someone else's, so running items
    concurrently would just contend for the same GPU anyway). Pause is checked
    between items only -- the in-flight item is always let finish rather than
    interrupted, since Z-Image-Turbo generations are seconds-long and a half-
    finished ComfyUI job is more state to reconcile than it's worth. Cancel is the
    one case that reaches into the in-flight item too, via /interrupt (see
    cancel_comfyui_batch), so an abort doesn't sit and wait for the current item.
    """
    batch = state.comfyui_batch
    if not batch or batch["id"] != batch_id:
        return
    items = batch["items"]

    for idx, item in enumerate(items):
        batch = state.comfyui_batch
        if not batch or batch["id"] != batch_id or batch["status"] == "cancelled":
            return
        while batch["status"] == "paused":
            await asyncio.sleep(0.5)
            batch = state.comfyui_batch
            if not batch or batch["id"] != batch_id or batch["status"] == "cancelled":
                return

        batch["current_index"] = idx
        item["status"] = "processing"
        try:
            gen_req = ComfyUIGenerateRequest(**item["request"])
            client_id, prompt_id, start_time = await _queue_comfyui_generate(gen_req)
            item["task_id"] = prompt_id

            # Don't just await _track_and_retrieve_comfyui() directly here: when an
            # in-flight item gets interrupted (see cancel_comfyui_batch), its prompt_id
            # never lands in ComfyUI's /history, so that function's own history-poll
            # loop only gives up after its full ~60s timeout -- verified live, a
            # cancel took ~40s to actually reflect in this item's status with a plain
            # await. Racing it against a short cancellation-poll loop instead means an
            # abort reflects within ~0.2s of the /interrupt call actually landing.
            # The tracker task isn't killed if we stop waiting on it -- it keeps
            # running and will update state.comfyui_tasks[prompt_id] whenever it does
            # finish, same as any single-generate task a caller stops polling.
            track_task = asyncio.create_task(
                _track_and_retrieve_comfyui(
                    client_id, prompt_id, state.comfyui_port, start_time, gen_req.model_name, gen_req.filename,
                )
            )
            was_cancelled = False
            while not track_task.done():
                await asyncio.sleep(0.2)
                current_batch = state.comfyui_batch
                if not current_batch or current_batch["id"] != batch_id or current_batch["status"] == "cancelled":
                    was_cancelled = True
                    break

            if was_cancelled:
                item["status"] = "cancelled"
            elif track_task.done():
                task = state.comfyui_tasks.get(prompt_id, {})
                if task.get("status") == "completed":
                    item["status"] = "completed"
                    item["runtime_seconds"] = task.get("runtime_seconds")
                else:
                    item["status"] = "failed"
                    item["error"] = task.get("error") or "Generation did not complete."
        except Exception as e:
            item["status"] = "failed"
            item["error"] = str(e)

        batch = state.comfyui_batch
        if not batch or batch["id"] != batch_id:
            return
        batch["completed_count"] = idx + 1
        if batch["status"] == "cancelled":
            for later in items[idx + 1:]:
                if later["status"] == "pending":
                    later["status"] = "cancelled"
            return

    batch = state.comfyui_batch
    if batch and batch["id"] == batch_id and batch["status"] != "cancelled":
        batch["status"] = "completed"


@router.post("/api/comfyui/batch", summary="Start a batch of Z-Image-Turbo (or other) generations from a JSON list of prompts")
async def start_comfyui_batch(req: ComfyBatchRequest):
    """Accepts the batch JSON directly as the request body -- the same endpoint
    serves both a curl/API caller and the Image Gen tab's file-upload flow (the
    frontend just reads the uploaded file client-side and POSTs its parsed
    contents here), so there's no separate multipart upload path to maintain."""
    existing = state.comfyui_batch
    if existing and existing["status"] in ("running", "paused"):
        raise HTTPException(
            status_code=409,
            detail="A batch is already running or paused. Cancel it, or wait for it to finish, before starting another.",
        )
    if not req.items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list.")
    if len(req.items) > _COMFY_BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large ({len(req.items)} items) -- max {_COMFY_BATCH_MAX_ITEMS} per batch.",
        )

    try:
        resolved_items = [_merge_batch_item(req.defaults, item) for item in req.items]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch item: {e}")

    batch_id = str(uuid.uuid4())
    state.comfyui_batch = {
        "id": batch_id,
        "status": "running",
        "total": len(resolved_items),
        "completed_count": 0,
        "current_index": None,
        "created_at": time.time(),
        "items": [
            {
                "index": i,
                "request": resolved,
                "status": "pending",
                "task_id": None,
                "error": None,
                "runtime_seconds": None,
            }
            for i, resolved in enumerate(resolved_items)
        ],
    }
    asyncio.create_task(_run_comfyui_batch(batch_id))
    return {"batch_id": batch_id, "total": len(resolved_items)}


@router.get("/api/comfyui/batch/status", summary="Get the current/last image-generation batch's progress")
async def get_comfyui_batch_status():
    batch = state.comfyui_batch
    if not batch:
        return {"status": "none"}
    return {
        "id": batch["id"],
        "status": batch["status"],
        "total": batch["total"],
        "completed_count": batch["completed_count"],
        "current_index": batch["current_index"],
        "items": [_comfyui_batch_item_view(it) for it in batch["items"]],
    }


def _comfyui_batch_item_view(it: dict) -> dict:
    requested_filename = it["request"].get("filename")
    # Prefer the *actual* outcome recorded on the underlying task (None if the
    # write itself failed, e.g. a disk error) over just echoing the requested name
    # -- they only diverge in that rare failure case, but echoing the request
    # unconditionally would otherwise claim a save that didn't really happen.
    actual_filename = requested_filename
    if it["task_id"] and requested_filename:
        task = state.comfyui_tasks.get(it["task_id"])
        if task is not None and task.get("status") == "completed":
            actual_filename = task.get("filename")
    return {
        "index": it["index"],
        "prompt": it["request"]["prompt"],
        "model_name": it["request"]["model_name"],
        "status": it["status"],
        # Full image is deliberately not inlined here -- with many items this
        # response is polled every couple seconds, and base64 images would
        # bloat it fast. Completed items already have a task_id; fetch the
        # actual image via the existing GET /api/comfyui/task/{task_id}.
        "task_id": it["task_id"],
        "error": it["error"],
        "runtime_seconds": it["runtime_seconds"],
        # Requested filename until the item finishes, then the confirmed saved
        # name (or null if the save failed) -- retrievable via
        # GET /api/comfyui/images/{filename} once non-null.
        "filename": actual_filename,
    }


@router.post("/api/comfyui/batch/pause", summary="Pause a running batch after its current item finishes")
async def pause_comfyui_batch():
    batch = state.comfyui_batch
    if not batch or batch["status"] != "running":
        raise HTTPException(status_code=409, detail="No running batch to pause.")
    batch["status"] = "paused"
    return {"status": "paused"}


@router.post("/api/comfyui/batch/resume", summary="Resume a paused batch")
async def resume_comfyui_batch():
    # No new task to spawn here: the original runner from start_comfyui_batch() is
    # still alive, parked in its "while status == paused: sleep" loop -- flipping
    # the flag is all that's needed for it to wake up and continue from the same
    # index. Spawning a second _run_comfyui_batch() here would race the still-alive
    # first one over the same items once both saw "running".
    batch = state.comfyui_batch
    if not batch or batch["status"] != "paused":
        raise HTTPException(status_code=409, detail="No paused batch to resume.")
    batch["status"] = "running"
    return {"status": "running"}


@router.post("/api/comfyui/batch/cancel", summary="Abort a running or paused batch")
async def cancel_comfyui_batch():
    batch = state.comfyui_batch
    if not batch or batch["status"] not in ("running", "paused"):
        raise HTTPException(status_code=409, detail="No active batch to cancel.")
    batch["status"] = "cancelled"
    # Interrupts whatever ComfyUI is currently rendering so an abort doesn't sit and
    # wait for the in-flight item -- the runner's own pause check only happens
    # between items, so without this, cancelling would still block on the current one.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"http://127.0.0.1:{state.comfyui_port}/interrupt")
    except Exception:
        pass
    return {"status": "cancelled"}


def _decode_image_base64(image_b64: str) -> bytes:
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        return base64.b64decode(image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}")


async def _upload_image_to_comfyui(image_bytes: bytes) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upload_resp = await client.post(
                f"http://127.0.0.1:{state.comfyui_port}/upload/image",
                files={"image": (f"{uuid.uuid4()}.png", image_bytes, "image/png")},
            )
            upload_resp.raise_for_status()
            return upload_resp.json()["name"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image to ComfyUI: {e}")


@router.post("/api/comfyui/edit", summary="Edit an image via ComfyUI (Qwen-Image-Edit)")
async def comfyui_edit(req: ComfyUIEditRequest):
    req.model_name = _catalog_file(req.model_name, "edit")   # catalog ids work here too
    if req.filename:
        try:
            _sanitize_image_filename(req.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    start_time = time.time()
    state.comfyui_active_model = req.model_name
    state.comfyui_last_used = start_time
    await _ensure_comfyui_running(gpu_id=req.gpu_id)
    _reset_comfyui_idle_timer(state.comfyui_port)
    _comfyui_dir = WORKFLOWS_DIR
    workflow_path = os.path.join(_comfyui_dir, "Qwen-Image-Edit.json")
    try:
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load workflow template: {e}")

    image_bytes = _decode_image_base64(req.image_base64)
    uploaded_name = await _upload_image_to_comfyui(image_bytes)

    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        if class_type == "LoadImage":
            node["inputs"]["image"] = uploaded_name
        elif class_type == "UnetLoaderGGUF":
            node["inputs"]["unet_name"] = req.model_name
        elif class_type == "TextEncodeQwenImageEditPlus":   # 2511 checkpoints (older ones removed 2026-10-21)
            title = node.get("_meta", {}).get("title", "")
            if "Positive" in title:
                node["inputs"]["prompt"] = req.prompt
        elif class_type == "KSampler":
            node["inputs"]["seed"] = req.seed if req.seed is not None else random.randint(1, 10**14 - 1)
            if req.steps is not None:
                node["inputs"]["steps"] = req.steps
            if req.denoise is not None:
                node["inputs"]["denoise"] = req.denoise

    if req.loras:
        lora_node_id = next((nid for nid, n in workflow.items() if n.get("class_type") == "LoraLoaderModelOnly"), None)
        if lora_node_id:
            _extend_lora_chain(workflow, lora_node_id, req.loras)

    client_id = str(uuid.uuid4())
    await _comfyui_gpu_preflight(workflow, getattr(req, "model", None) or "qwen-image-edit",
                                 pinned=getattr(req, "gpu_id", None) is not None)
    payload = {"prompt": workflow, "client_id": client_id}
    state.comfyui_progress = {"step": 0, "max_steps": 0, "node": "", "value": 0}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"http://127.0.0.1:{state.comfyui_port}/prompt", json=payload)
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id")
            asyncio.create_task(
                _track_and_retrieve_comfyui(client_id, prompt_id, state.comfyui_port, start_time, req.model_name, req.filename)
            )
            return {"status": "queued", "task_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {e}")


@router.post("/api/comfyui/style-transfer", summary="Apply a reference image's style to a content image via ComfyUI")
async def comfyui_style_transfer(req: ComfyUIStyleTransferRequest):
    req.model_name = _catalog_file(req.model_name, "style")   # catalog ids work here too
    if req.filename:
        try:
            _sanitize_image_filename(req.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    start_time = time.time()
    state.comfyui_active_model = req.model_name
    state.comfyui_last_used = start_time
    await _ensure_comfyui_running(gpu_id=req.gpu_id)
    _reset_comfyui_idle_timer(state.comfyui_port)
    _comfyui_dir = WORKFLOWS_DIR
    workflow_path = os.path.join(_comfyui_dir, "Qwen-Image-Style-Transfer.json")
    try:
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load workflow template: {e}")

    content_bytes = _decode_image_base64(req.content_image_base64)
    style_bytes = _decode_image_base64(req.style_image_base64)
    content_name = await _upload_image_to_comfyui(content_bytes)
    style_name = await _upload_image_to_comfyui(style_bytes)

    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        title = node.get("_meta", {}).get("title", "")
        if class_type == "LoadImage":
            node["inputs"]["image"] = style_name if "Style" in title else content_name
        elif class_type == "UnetLoaderGGUF":
            node["inputs"]["unet_name"] = req.model_name
        elif class_type == "TextEncodeQwenImageEditPlus":
            if "Positive" in title:
                node["inputs"]["prompt"] = req.prompt
        elif class_type == "KSampler":
            node["inputs"]["seed"] = req.seed if req.seed is not None else random.randint(1, 10**14 - 1)
            if req.steps is not None:
                node["inputs"]["steps"] = req.steps
            if req.denoise is not None:
                node["inputs"]["denoise"] = req.denoise

    if req.loras:
        lora_node_id = next((nid for nid, n in workflow.items() if n.get("class_type") == "LoraLoaderModelOnly"), None)
        if lora_node_id:
            _extend_lora_chain(workflow, lora_node_id, req.loras)

    client_id = str(uuid.uuid4())
    await _comfyui_gpu_preflight(workflow, getattr(req, "model", None) or "qwen-image-style-transfer",
                                 pinned=getattr(req, "gpu_id", None) is not None)
    payload = {"prompt": workflow, "client_id": client_id}
    state.comfyui_progress = {"step": 0, "max_steps": 0, "node": "", "value": 0}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"http://127.0.0.1:{state.comfyui_port}/prompt", json=payload)
            resp.raise_for_status()
            prompt_id = resp.json().get("prompt_id")
            asyncio.create_task(
                _track_and_retrieve_comfyui(client_id, prompt_id, state.comfyui_port, start_time, req.model_name, req.filename)
            )
            return {"status": "queued", "task_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue prompt: {e}")


# ---------------------------------------------------------------------------
# OpenAI image generation compatibility
# ---------------------------------------------------------------------------

# A synchronous /v1 image call waits for ComfyUI as long as a job can sensibly take: it used to give up after
# 120 s while ComfyUI kept computing (Qwen-Image 2.1 int8, 40 steps ~150 s) -- a "Timeout" for the caller and
# a GPU that stayed busy for the next request. The job itself is tracked over ComfyUI's websocket.
COMFYUI_SYNC_TIMEOUT_S = 1800


async def _await_comfyui_task(prompt_id: str, timeout_s: int = COMFYUI_SYNC_TIMEOUT_S) -> str:
    for _ in range(timeout_s):
        await asyncio.sleep(1)
        task = state.comfyui_tasks.get(prompt_id)
        if task:
            if task["status"] == "completed":
                return task["image_base64"]
            elif task["status"] == "failed":
                if task.get("error") == "cancelled":
                    raise HTTPException(status_code=409, detail="Image generation cancelled")
                raise HTTPException(status_code=500, detail=f"Image generation failed: {task.get('error')}")
    raise HTTPException(status_code=504, detail=f"Image generation took longer than {timeout_s // 60} minutes.")


def openai_size_to_comfy(size: Optional[str]) -> tuple:
    """OpenAI `size` -> (base_resolution, aspect_ratio) for ComfyUIGenerateRequest."""
    if size == "512x512":
        return 512, "1:1"
    if size in ("1024x768", "4:3"):
        return 1024, "4:3"
    if size in ("768x1024", "3:4"):
        return 1024, "3:4"
    if size in ("16:9", "1280x720", "1792x1024", "1536x1024"):
        return 1024, "16:9"
    if size in ("9:16", "720x1280", "1024x1792", "1024x1536"):
        return 1024, "9:16"
    return 1024, "1:1"


def _catalog_file(name: Optional[str], task: str) -> Optional[str]:
    """A catalog id (or old filename) for an edit/style request -> its checkpoint file; 400 if that catalog
    model can't do the task or isn't installed; unknown names pass through unchanged."""
    from image_serve import catalog
    try:
        model = catalog.for_task(name, task)
    except catalog.CatalogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if model is None:
        return name
    if not catalog.listing_row(model)["available"]:
        raise HTTPException(status_code=400, detail=f"{model.label} is not installed on this machine")
    return model.file
