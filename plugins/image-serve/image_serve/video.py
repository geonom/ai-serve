"""Video through ComfyUI -- the `video` capability (ai-serve-ui docs/plugins.md "video" and "Long work: jobs"),
added 2026-11-13 (owner: LTX-2.5 and Wan 2.2 tested live on the 3090, "bitte ins Backend einbauen", with optional
LoRAs for the API and the Playground).

Models (all distilled / few-step variants; files in COMFYUI_MODELS_ROOT):
- `ltx-2.5`       LTX-2.5 22B distilled, Lightricks' int8 transformer (21.5 GB; int8 kernels: 1.8 s per step vs 4.8 s
                  for GGUF Q5 on the 3090) + their int8 Gemma 4 12B text encoder: text -> video WITH sound, 24 fps,
                  8 steps. TWO_STAGE sizes render at half size and double with the latent upscaler (Lightricks'
                  two-stage graph). LTX-2 Community License (free below $10M revenue).
- `ltx-2.5-q3`    the same as unsloth's dynamic UD-Q3_K_XL (12 GB) -- meant for 16 GB cards.
- `wan-2.2`       Wan 2.2 I2V A14B with the lightx2v 4-step merge (high + low noise experts), GGUF Q4_K_M: picture ->
                  video, 16 fps, no sound. Apache 2.0.
- `wan-2.2-q3`    the same as Q3_K_M (both experts fit the 3090 at once -- measured to see whether that saves the swap).
- `wan-2.2-5b`    Wan 2.2 TI2V 5B Turbo GGUF Q8_0: smaller, faster, weaker; 24 fps.
- `minimax-h3`    MiniMax H3 FL2VA (33B, pruned) as unsloth's GGUF Q4_K + Comfy-Org's int8 Qwen3-VL-32B encoder (unsloth's
                  GGUF encoder has no architecture tag -- ComfyUI-GGUF refuses it, 2026-11-14), with lightx2v's
                  8-step turbo LoRA (6 steps, as Comfy-Org's template); `minimax-h3-4step` with the 4-step 768p LoRA.
                  Text or first frame -> video with stereo sound, 24 fps, frames on a 17k+5 grid. MiniMax Community
                  License. Added 2026-11-14 (owner: "download minimax h3 now"); speed LoRAs in loras/minimax-h3-speed/.
- `sulphur-3`     SulphurAI's uncensored MiniMax H3 fine-tune (Abiray's GGUF Q4_K_M), otherwise as minimax-h3 (2026-11-26).
Graphs follow Comfy-Org's templates (video_ltx2_5_t2v.json, video_wan2_2_14B_i2v.json, video_wan2_2_5B_ti2v.json)
in API form, measured 2026-11-13 (CLAUDE.md "Video").

LoRAs: `.safetensors` files in loras/<family>/ (ltx-2.5, wan-2.2) are offered per model (`loras` in the catalog) and
applied with `lora` (+ `lora_strength`, default 1.0). A LoRA of the other family is refused (wrong architecture).

Jobs like audio.py: POST -> 202 {id}; the job uploads the start picture, queues the workflow (ComfyUI started and the
GPU asked like for every image) and waits; results are MP4 files in GENERATED_VIDEO_DIR (never deleted here).
Pure parts (catalog, request checks, workflows) are tested in tests/test_video.py."""
from __future__ import annotations

import asyncio
import base64
import binascii
import os
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

TASKS = ("text_to_video", "image_to_video")
JOB_TTL_S = 6 * 3600
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# LTX: these sizes render at half size (8 steps), then the latent upscaler doubles them (3 steps) -- Lightricks'
# two-stage graph; faster than one stage at full size (1024x576: 20 s vs 960x544 one stage 25 s, int8, 3090)
TWO_STAGE = frozenset({"1024x576", "576x1024", "1280x704", "704x1280"})
NEGATIVE = "cartoon, video game, blurry, distorted faces, extra limbs, static image, text, watermark, low quality"
LTX_STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
# steps per job (`steps`): more steps can add detail and steadier motion (owner, 2026-11-14) at a linear cost in time
MIN_STEPS, MAX_STEPS = 4, 20


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoModel:
    id: str
    label: str
    family: str                              # LoRA folder: loras/<family>/
    tasks: Tuple[str, ...]
    files: Tuple[Tuple[str, str], ...]       # (models-root subdir, file); order is what the workflow uses
    description: str
    license: str
    fps: int
    min_seconds: int
    max_seconds: int
    default_seconds: int
    resolutions: Tuple[str, ...]             # first = default
    audio: bool = False
    sage_default: bool = False               # SageAttention on unless the request says otherwise (Wan: on)
    default_steps: int = 4                   # the distillation's own count (LTX 8, the 4-step Wan merges 4)


_WAN_TE = ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
_WAN_RES = ("832x480", "480x832", "640x640")
_WAN_DESC = "Picture -> video: animates a start picture following the prompt; no sound."

CATALOG: Tuple[VideoModel, ...] = (
    VideoModel("ltx-2.5", "LTX-2.5 (distilled, int8)", "ltx-2.5", ("text_to_video", "image_to_video"),
               (("diffusion_models", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
                ("text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
                ("vae", "ltx-2.5-video-vae-conv-bf16.safetensors"), ("vae", "ltx-2.5-audio-vae-bf16.safetensors")),
               "Text or picture -> video with matching sound (voices, ambience). Fast: ~20 s for 4 s at 1024x576 on a 3090 "
               "(Lightricks' int8 file, 21.5 GB -- needs a 24 GB card).",
               "LTX-2 Community License (free below $10M annual revenue)", 24, 2, 8, 4,
               ("1024x576", "576x1024", "768x768", "960x544", "544x960", "1280x704", "704x1280"), audio=True,
               default_steps=8),
    VideoModel("ltx-2.5-q3", "LTX-2.5 (distilled, Q3)", "ltx-2.5", ("text_to_video", "image_to_video"),
               (("unet", "ltx-2.5-22b-distilled-UD-Q3_K_XL.gguf"),
                ("text_encoders", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
                ("vae", "ltx-2.5-video-vae-conv-bf16.safetensors"), ("vae", "ltx-2.5-audio-vae-bf16.safetensors")),
               "LTX-2.5 as unsloth's dynamic Q3 (12 GB): for 16 GB cards (~65 s for 4 s on a 4060 Ti).",
               "LTX-2 Community License (free below $10M annual revenue)", 24, 2, 8, 4,
               ("1024x576", "576x1024", "768x768", "960x544", "544x960", "1280x704", "704x1280"), audio=True,
               default_steps=8),
    VideoModel("wan-2.2", "Wan 2.2 14B (4 steps, Q4)", "wan-2.2", ("image_to_video",),
               (("unet", "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q4_K_M.gguf"),
                ("unet", "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q4_K_M.gguf"),
                _WAN_TE, ("vae", "wan_2.1_vae.safetensors")),
               _WAN_DESC, "Apache 2.0", 16, 2, 6, 5, _WAN_RES, sage_default=True),
    VideoModel("wan-2.2-q3", "Wan 2.2 14B (4 steps, Q3)", "wan-2.2", ("image_to_video",),
               (("unet", "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412-Q3_K_M.gguf"),
                ("unet", "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412-Q3_K_M.gguf"),
                _WAN_TE, ("vae", "wan_2.1_vae.safetensors")),
               _WAN_DESC + " Smaller files: both halves fit a 24 GB card at once.", "Apache 2.0", 16, 2, 6, 5, _WAN_RES,
               sage_default=True),
    VideoModel("wan-2.2-5b", "Wan 2.2 5B Turbo (Q8)", "wan-2.2-5b", ("image_to_video",),
               (("unet", "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf"), _WAN_TE, ("vae", "wan2.2_vae.safetensors")),
               "Picture -> video with the small Wan model: faster, less detail; no sound.", "Apache 2.0", 24, 2, 5, 5,
               ("1280x704", "704x1280", "832x480", "480x832"), sage_default=True),
)
_H3_FILES = (("diffusion_models", "minimax_h3_fl2va_pruned-Q4_K.gguf"),
             ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
             ("vae", "minimax_h3_video_vae_fp16.safetensors"), ("vae", "minimax_h3_audio_vae_fp32.safetensors"))
_H3_RES = ("864x480", "480x864", "1056x608", "608x1056", "1344x768", "768x1344")
_H3_LICENSE = "MiniMax Community License"
CATALOG = CATALOG + (
    VideoModel("minimax-h3", "MiniMax H3 (turbo 8-step LoRA, Q4)", "minimax-h3", ("text_to_video", "image_to_video"),
               _H3_FILES + (("loras", "minimax-h3-speed/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),),
               "Text or picture -> video with stereo sound, 33B (unsloth GGUF Q4, int8 Qwen3-VL encoder, lightx2v's turbo LoRA).",
               _H3_LICENSE, 24, 2, 10, 5, _H3_RES, audio=True, default_steps=6),
    VideoModel("minimax-h3-4step", "MiniMax H3 (turbo 4-step LoRA, Q4)", "minimax-h3",
               ("text_to_video", "image_to_video"),
               _H3_FILES + (("loras", "minimax-h3-speed/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),),
               "MiniMax H3 with the 4-step turbo LoRA (trained for 768p): faster, possibly less detail.",
               _H3_LICENSE, 24, 2, 10, 5, _H3_RES, audio=True, default_steps=4),
    # 2026-11-26 (owner: "runterladen und testen"): SulphurAI's uncensored MiniMax H3 fine-tune (s3, RL-baked), as
    # Abiray's GGUF Q4_K_M; a non-turbo base, run with the same 8-step turbo LoRA as minimax-h3
    VideoModel("sulphur-3", "Sulphur-3 (MiniMax H3 fine-tune, uncensored, Q4)", "minimax-h3",
               ("text_to_video", "image_to_video"),
               (("diffusion_models", "Sulphur-3-Q4_K_M.gguf"),) + _H3_FILES[1:]
               + (("loras", "minimax-h3-speed/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),),
               "SulphurAI's uncensored fine-tune of MiniMax H3: text or picture -> video with stereo sound "
               "(Abiray's GGUF Q4_K_M, the same encoder and turbo LoRA as minimax-h3).",
               _H3_LICENSE, 24, 2, 10, 5, _H3_RES, audio=True, default_steps=6),
)
DEFAULTS = {"text_to_video": "ltx-2.5", "image_to_video": "ltx-2.5"}   # LTX: faster and with sound (owner, 2026-11-13)


def resolve(name: Optional[str]) -> Optional[VideoModel]:
    key = (name or "").strip().lower()
    return next((m for m in CATALOG if m.id == key), None)


def available(m: VideoModel, models_root: str) -> bool:
    return all(os.path.isfile(os.path.join(models_root, d, f)) for d, f in m.files)


def loras(m: VideoModel, models_root: str) -> List[str]:
    """LoRA names (file stems) in loras/<family>/ -- sorted, plain names only."""
    folder = os.path.join(models_root, "loras", m.family)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n[:-len(".safetensors")] for n in names
                  if n.endswith(".safetensors") and _LORA.match(n[:-len(".safetensors")])
                  and os.path.isfile(os.path.join(folder, n)))


def upscale_available(models_root: str) -> bool:
    return os.path.isfile(os.path.join(models_root, "latent_upscale_models", UPSCALER))


# whose prompt rules the gateway's prompt writer (POST /v1/video/prompt) follows: LTX-2's own (same Gemma 4 encoder),
# Wan 2.2's picture -> video prompt extension (for the 5B too)
PROMPT_GUIDES = {"ltx-2.5": "ltx-2", "wan-2.2": "wan-2.2", "wan-2.2-5b": "wan-2.2", "minimax-h3": "minimax-h3"}


def listing(models_root: str, sage: bool = False) -> List[Dict[str, Any]]:
    """`sage`: SageAttention can be used here -- then each row says its default (true/false), else null."""
    return [{"id": m.id, "label": m.label, "description": m.description, "tasks": list(m.tasks),
             "max_seconds": m.max_seconds, "min_seconds": m.min_seconds, "default_seconds": m.default_seconds,
             "resolutions": list(m.resolutions), "fps": m.fps, "audio": m.audio, "license": m.license,
             "loras": loras(m, models_root), "available": available(m, models_root),
             "sage_attention": m.sage_default if sage else None,
             "prompt_guide": PROMPT_GUIDES.get(m.family),
             "steps": {"default": m.default_steps, "min": MIN_STEPS, "max": MAX_STEPS, "adjustable": True},
             "default": m.id in DEFAULTS.values()} for m in CATALOG]


# ------------------------------------------------------------------ requests

_LORA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,120}$")
_RES = re.compile(r"^(\d{3,4})x(\d{3,4})$")


@dataclass
class VideoRequest:
    model: VideoModel
    prompt: str
    seconds: float
    width: int
    height: int
    seed: int
    negative: str = NEGATIVE
    image: Optional[bytes] = None
    lora: Optional[str] = None
    lora_strength: float = 1.0
    sage: bool = False
    steps: int = 4

    @property
    def frames(self) -> int:
        return frame_count(self.model, self.seconds)


def frame_count(m: VideoModel, seconds: float) -> int:
    """LTX wants 8n+1 frames, Wan 4n+1, MiniMax H3 17k+5 (rounded up, as Comfy-Org's template does)."""
    if m.family == "minimax-h3":
        n = max(5, round(seconds * m.fps))
        return n + (5 - n % 17) % 17
    step = 8 if m.family == "ltx-2.5" else 4
    n = max(1, round(seconds * m.fps / step))
    return n * step + 1


def _num(v: Any, name: str, lo: float, hi: float, integer: bool = False) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise VideoError(f"{name} must be a number")
    if not lo <= v <= hi:
        raise VideoError(f"{name}: {lo:g}..{hi:g}")
    return int(v) if integer else float(v)


def _image(raw: Any) -> bytes:
    if not isinstance(raw, str) or not raw:
        raise VideoError("image_base64: the start picture (PNG/JPEG/WebP, base64 or a data: URL)")
    if raw.startswith("data:"):
        head, _, raw = raw.partition(",")
        if not re.match(r"^data:image/(png|jpeg|jpg|webp);base64$", head):
            raise VideoError("image_base64: only data:image/png|jpeg|webp;base64 URLs")
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise VideoError("image_base64: not valid base64")
    if len(data) > MAX_IMAGE_BYTES:
        raise VideoError("image_base64: at most 20 MB")
    if not (data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8") or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")):
        raise VideoError("image_base64: a PNG, JPEG or WebP picture")
    return data


def parse_request(body: Any, models_root: str) -> VideoRequest:
    """The jobs body ({model?, task?, prompt, image_base64?, seconds?, resolution?, seed?, negative_prompt?, lora?,
    lora_strength?, sage_attention?, steps?}) -> a checked request. A missing model takes the default of `task` (with a picture:
    image_to_video, else text_to_video)."""
    if not isinstance(body, dict):
        raise VideoError("the body must be a JSON object")
    task = body.get("task")
    if task is not None and task not in TASKS:
        raise VideoError(f"task: {' | '.join(TASKS)}")
    if body.get("model"):
        model = resolve(body["model"]) if isinstance(body["model"], str) else None
        if model is None:
            raise VideoError(f"unknown video model {body.get('model')!r} -- {', '.join(m.id for m in CATALOG)}")
    else:
        model = resolve(DEFAULTS[task or ("image_to_video" if body.get("image_base64") else "text_to_video")])
    if task is not None and task not in model.tasks:
        raise VideoError(f"{model.label} does {' / '.join(model.tasks)}, not {task}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VideoError("prompt: describe what happens in the video")
    if len(prompt) > 4000:
        raise VideoError("prompt: at most 4000 characters")
    negative = body.get("negative_prompt")
    if negative is not None and (not isinstance(negative, str) or len(negative) > 2000):
        raise VideoError("negative_prompt: text, at most 2000 characters")
    image = None
    picture_only = "text_to_video" not in model.tasks
    if body.get("image_base64") or picture_only or task == "image_to_video":
        if "image_to_video" not in model.tasks:
            raise VideoError(f"{model.label} makes video from text only -- leave out image_base64")
        image = _image(body.get("image_base64"))
    seconds = _num(body.get("seconds"), "seconds", model.min_seconds, model.max_seconds)
    res = body.get("resolution") or model.resolutions[0]
    if res not in model.resolutions:
        raise VideoError(f"resolution for {model.label}: {', '.join(model.resolutions)}")
    w, h = (int(x) for x in _RES.match(res).groups())
    if model.family == "ltx-2.5" and res in TWO_STAGE and not upscale_available(models_root):
        raise VideoError(f"{res} needs the LTX latent upscaler ({UPSCALER}), which is not installed")
    seed = _num(body.get("seed"), "seed", 0, 2 ** 53, integer=True)
    lora = body.get("lora")
    if lora is not None and lora != "":
        if not isinstance(lora, str) or lora not in loras(model, models_root):
            known = ", ".join(loras(model, models_root)) or "none installed"
            raise VideoError(f"lora for {model.label}: {known}")
    else:
        lora = None
    strength = _num(body.get("lora_strength"), "lora_strength", 0, 2)
    steps = body.get("steps")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                              or not MIN_STEPS <= steps <= MAX_STEPS):
        raise VideoError(f"steps: a whole number {MIN_STEPS}..{MAX_STEPS} (default {model.default_steps})")
    sage = body.get("sage_attention", model.sage_default)
    if not isinstance(sage, bool):
        raise VideoError("sage_attention: true or false")
    return VideoRequest(model, prompt.strip(), float(seconds or model.default_seconds), w, h,
                        int(seed) if seed is not None else random.randint(1, 2 ** 31),
                        negative.strip() if isinstance(negative, str) and negative.strip() else NEGATIVE,
                        image, lora, 1.0 if strength is None else strength, sage,
                        model.default_steps if steps is None else steps)


# ------------------------------------------------------------------ workflows (ComfyUI API form)

def ltx_sigmas(steps: int) -> str:
    """LTX distilled has a fixed 8-step schedule; other counts sample the same curve (piecewise linear) at
    `steps` + 1 points, so the slow start and the steep end keep their shape."""
    base = [float(x) for x in LTX_STAGE1_SIGMAS.split(",")]
    if steps == len(base) - 1:
        return LTX_STAGE1_SIGMAS
    seg = len(base) - 1
    out = []
    for j in range(steps + 1):
        t = j * seg / steps
        i = min(int(t), seg - 1)
        out.append(base[i] + (base[i + 1] - base[i]) * (t - i))
    return ", ".join(f"{v:.6f}".rstrip("0").rstrip(".") if v not in (0.0, 1.0) else f"{v:.1f}" for v in out)


def _loader(name: str) -> Dict[str, Any]:
    if name.endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


def _lora_node(model_ref: List[Any], r: VideoRequest) -> Dict[str, Any]:
    return {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": model_ref, "lora_name": f"{r.model.family}/{r.lora}.safetensors", "strength_model": r.lora_strength}}


def _sage_node(model_ref: List[Any]) -> Dict[str, Any]:
    """KJNodes' per-model SageAttention patch (needs the `sageattention` package in ComfyUI's Python)."""
    return {"class_type": "PathchSageAttentionKJ", "inputs": {"model": model_ref, "sage_attention": "auto"}}


def _save(images: List[Any], fps: int, prefix: str, audio: Optional[List[Any]] = None) -> Dict[str, Any]:
    video = {"class_type": "CreateVideo", "inputs": {"images": images, "fps": fps}}
    if audio:
        video["inputs"]["audio"] = audio
    return {"vid": video, "save": {"class_type": "SaveVideo", "inputs": {
        "video": ["vid", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}}


def ltx_workflow(r: VideoRequest, prefix: str, sage: bool = False, image_name: Optional[str] = None) -> Dict[str, Any]:
    unet, te, vvae, avae = (f for _, f in r.model.files)
    upscale = f"{r.width}x{r.height}" in TWO_STAGE
    w, h = (r.width // 2, r.height // 2) if upscale else (r.width, r.height)
    frames, fps = r.frames, r.model.fps
    model = ["unet", 0]
    g: Dict[str, Any] = {
        "unet": _loader(unet),
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "ltxv", "device": "default"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": vvae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": avae}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": r.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": r.negative, "clip": ["clip", 0]}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0],
                                                               "frame_rate": float(fps)}},
        "vlat": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": frames,
                                                                   "batch_size": 1}},
        "alat": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": fps,
                                                                   "batch_size": 1, "audio_vae": ["avae", 0]}},
        "av": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["vlat", 0], "audio_latent": ["alat", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": r.seed}},
        "samp": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "sig": {"class_type": "ManualSigmas", "inputs": {"sigmas": ltx_sigmas(r.steps)}},
    }
    if image_name:   # picture -> video (Lightricks' i2v graph): the picture, cropped to the output's shape, pins frame 1
        g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        g["img_fit"] = {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos",
                                                               "width": r.width, "height": r.height, "crop": "center"}}
        g["img_prep"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["img_fit", 0], "img_compression": 18}}
        g["i2v"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["vvae", 0], "image": ["img_prep", 0], "latent": ["vlat", 0],
            "strength": 0.7 if upscale else 1.0, "bypass": False}}
        g["av"]["inputs"]["video_latent"] = ["i2v", 0]
    if r.lora:
        g["lora"] = _lora_node(model, r)
        model = ["lora", 0]
    if sage:
        g["sage"] = _sage_node(model)
        model = ["sage", 0]
    g["guide"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": model, "positive": ["cond", 0],
                                                                "negative": ["cond", 1], "video_cfg": 1.0, "audio_cfg": 1.0}}
    g["run"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guide", 0],
                                                                  "sampler": ["samp", 0], "sigmas": ["sig", 0],
                                                                  "latent_image": ["av", 0]}}
    g["sep"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["run", 0]}}
    final = "sep"
    if upscale:   # Lightricks' second stage: double the video latent, keep the audio latent, refine 3 steps
        g["upm"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
        g["up"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["sep", 0], "upscale_model": ["upm", 0],
                                                                   "vae": ["vvae", 0]}}
        g["av2"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["up", 0], "audio_latent": ["sep", 1]}}
        if image_name:
            g["i2v2"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["vvae", 0], "image": ["img_prep", 0], "latent": ["up", 0], "strength": 1.0, "bypass": False}}
            g["av2"]["inputs"]["video_latent"] = ["i2v2", 0]
        g["noise2"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": r.seed + 1}}
        g["sig2"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_STAGE2_SIGMAS}}
        g["guide2"] = {"class_type": "LTXVDualCFGGuider", "inputs": {"model": model, "positive": ["cond", 0],
                                                                     "negative": ["cond", 1], "video_cfg": 1.0,
                                                                     "audio_cfg": 1.0}}
        g["run2"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise2", 0], "guider": ["guide2", 0],
                                                                       "sampler": ["samp", 0], "sigmas": ["sig2", 0],
                                                                       "latent_image": ["av2", 0]}}
        g["sep2"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["run2", 0]}}
        final = "sep2"
    g["dec"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": [final, 0], "vae": ["vvae", 0], "tile_size": 512,
                                                           "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}}
    g["adec"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": [final, 1], "audio_vae": ["avae", 0]}}
    g.update(_save(["dec", 0], fps, prefix, audio=["adec", 0]))
    return g


def wan_workflow(r: VideoRequest, image_name: str, prefix: str, sage: bool = False) -> Dict[str, Any]:
    """Wan 2.2 A14B: the high-noise expert does the first half of the steps, the low-noise expert the rest
    (lightx2v merge: cfg 1; 4 steps by default)."""
    high, low, te, vae = (f for _, f in r.model.files)
    g: Dict[str, Any] = {
        "hi": _loader(high), "lo": _loader(low),
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": r.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": r.negative, "clip": ["clip", 0]}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0],
                                                            "vae": ["vae", 0], "width": r.width, "height": r.height,
                                                            "length": r.frames, "batch_size": 1,
                                                            "start_image": ["img", 0]}},
    }
    hi, lo = ["hi", 0], ["lo", 0]
    if r.lora:   # Wan LoRAs come per expert in practice; one file is applied to both
        g["lora_hi"], g["lora_lo"] = _lora_node(hi, r), _lora_node(lo, r)
        hi, lo = ["lora_hi", 0], ["lora_lo", 0]
    if sage:     # SageAttention for Wan only (owner, 2026-11-13: ~26 -> ~20 s per step on the 3090)
        g["sage_hi"], g["sage_lo"] = _sage_node(hi), _sage_node(lo)
        hi, lo = ["sage_hi", 0], ["sage_lo", 0]
    g["hs"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": hi, "shift": 5.0}}
    g["ls"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": lo, "shift": 5.0}}
    split = (r.steps + 1) // 2              # the high-noise expert takes the first half (rounded up)
    common = {"steps": r.steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
              "positive": ["i2v", 0], "negative": ["i2v", 1]}
    g["k1"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["hs", 0], "add_noise": "enable",
                                                            "noise_seed": r.seed, "latent_image": ["i2v", 2],
                                                            "start_at_step": 0, "end_at_step": split,
                                                            "return_with_leftover_noise": "enable"}}
    g["k2"] = {"class_type": "KSamplerAdvanced", "inputs": {**common, "model": ["ls", 0], "add_noise": "disable",
                                                            "noise_seed": 0, "latent_image": ["k1", 0],
                                                            "start_at_step": split, "end_at_step": 10000,
                                                            "return_with_leftover_noise": "disable"}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["k2", 0], "vae": ["vae", 0]}}
    g.update(_save(["dec", 0], r.model.fps, prefix))
    return g


def wan5b_workflow(r: VideoRequest, image_name: str, prefix: str, sage: bool = False) -> Dict[str, Any]:
    """Wan 2.2 TI2V 5B Turbo: one model, 4 steps by default, cfg 1 (the template's 20 steps / cfg 5 are for the base model)."""
    unet, te, vae = (f for _, f in r.model.files)
    model = ["unet", 0]
    g: Dict[str, Any] = {
        "unet": _loader(unet),
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": r.prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": r.negative, "clip": ["clip", 0]}},
        "lat": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"vae": ["vae", 0], "width": r.width,
                                                                    "height": r.height, "length": r.frames,
                                                                    "batch_size": 1, "start_image": ["img", 0]}},
    }
    if r.lora:
        g["lora"] = _lora_node(model, r)
        model = ["lora", 0]
    if sage:
        g["sage"] = _sage_node(model)
        model = ["sage", 0]
    g["ms"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": 8.0}}
    g["k"] = {"class_type": "KSampler", "inputs": {"model": ["ms", 0], "seed": r.seed, "steps": r.steps, "cfg": 1.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
                                                   "positive": ["pos", 0], "negative": ["neg", 0],
                                                   "latent_image": ["lat", 0]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["vae", 0]}}
    g.update(_save(["dec", 0], r.model.fps, prefix))
    return g


def h3_workflow(r: VideoRequest, prefix: str, sage: bool = False, image_name: Optional[str] = None) -> Dict[str, Any]:
    """MiniMax H3 (Comfy-Org's video_minimax_h3_t2v / _i2v template): Qwen3-VL text encoder, the turbo LoRA,
    res_multistep with a plain guider (no CFG), one joint video+audio latent decoded by both VAEs."""
    unet, te, vvae, avae, speed = (f for _, f in r.model.files)
    g: Dict[str, Any] = {
        "unet": _loader(unet),
        "clip": {"class_type": "CLIPLoaderGGUF" if te.endswith(".gguf") else "CLIPLoader",
                 "inputs": {"clip_name": te, "type": "minimax"}},
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": vvae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": avae}},
        "speed": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": speed,
                                                                  "strength_model": 1.0}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vvae", 0], "prompt": r.prompt, "width": r.width, "height": r.height,
            "length": r.frames}},
    }
    if image_name:
        g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        g["cond"]["inputs"]["first_frame"] = ["img", 0]
    model = ["speed", 0]
    if r.lora:
        g["lora"] = _lora_node(model, r)
        model = ["lora", 0]
    if sage:
        g["sage"] = _sage_node(model)
        model = ["sage", 0]
    g["sched"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": r.steps,
                                                             "denoise": 1.0}}
    g["samp"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": r.seed}}
    g["guide"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["cond", 0]}}
    g["run"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guide", 0],
                                                                  "sampler": ["samp", 0], "sigmas": ["sched", 0],
                                                                  "latent_image": ["cond", 1]}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["run", 0], "vae": ["vvae", 0]}}
    g["adec"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["run", 0], "vae": ["avae", 0]}}
    g.update(_save(["dec", 0], r.model.fps, prefix, audio=["adec", 0]))
    return g


def workflow(r: VideoRequest, image_name: Optional[str], prefix: str, sage: bool = False) -> Dict[str, Any]:
    """`sage`: SageAttention is installed in ComfyUI's Python. Used when the request wants it (`r.sage`; default on
    for Wan, off for LTX -- LTX gained only 48 -> 45 s, measured 2026-11-13)."""
    sage = sage and r.sage
    if r.model.family == "ltx-2.5":
        return ltx_workflow(r, prefix, sage, image_name if r.image is not None else None)
    if r.model.family == "minimax-h3":
        return h3_workflow(r, prefix, sage, image_name if r.image is not None else None)
    if r.model.family == "wan-2.2-5b":
        return wan5b_workflow(r, image_name or "", prefix, sage)
    return wan_workflow(r, image_name or "", prefix, sage)


# ------------------------------------------------------------------ jobs

@dataclass
class Job:
    id: str
    model: str
    created: float = field(default_factory=time.time)
    status: str = "queued"           # queued | running | done | failed | cancelled
    message: str = ""
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    file: Optional[str] = None
    started: Optional[float] = None
    finished: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None


_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Jobs:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def new(self, model: str) -> Job:
        now = time.time()
        for k in [k for k, j in self._jobs.items() if now - j.created > JOB_TTL_S and j.status != "running"]:
            del self._jobs[k]
        job = Job(id="vid-" + secrets.token_hex(12), model=model)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id) if isinstance(job_id, str) and _ID.match(job_id) else None


JOBS = Jobs()


def view(job: Job, progress: Optional[float], jobs_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": job.id, "status": job.status, "progress": progress, "message": job.message,
                           "error": job.error}
    if job.status == "done" and job.file:
        name = os.path.basename(job.file)
        data = dict(job.info, model=job.model)
        if job.started and job.finished:
            data["runtime_seconds"] = round(job.finished - job.started, 1)
        out["result"] = {"files": [{"name": name, "type": "video/mp4", "url": f"{jobs_path}/{job.id}/files/{name}"}],
                         "data": data}
    return out
