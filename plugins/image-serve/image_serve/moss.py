"""Pure rules for MOSS-Image in image-serve (no I/O; tests tests/test_moss_rules.py, test_moss_app.py): the MOSS variants (base model x
turbo LoRA -- every combination is its own entry so the owner can compare them, 2026-10-19), the catalog, how an
untrusted generate request becomes a workflow. (Moved here from ai-serve-ui's backend/images/moss.py and the gateway's
image catalog, 2026-10-21; from the comfyui-moss plugin into image-serve, 2026-10-22.)"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from image_serve.moss_layout import Element, LayoutError, moss_prompt, parse_layout, parse_size

WORKFLOW = Path(__file__).resolve().parent.parent / "workflows" / "MOSS-Image.json"   # (moved from comfyui-moss, 2026-10-22)
# Turbo = distilled step LoRAs on top of MOSS-Image-SFT, no CFG:
# - 4-step: akhaliq/MOSS-Image-SFT-4-steps v2.0, "Euler-native", made for a plain LoRA loader.
# - 8-step: lodestones/MOSS-Image-SFT-8-steps-dmd lora.safetensors at strength 1 (its card).
# Both authors train against the bf16 base; their deltas are partly below int8's rounding, so the bf16 base
# (10 GB, needs more VRAM) is the reference, int8 the faster/smaller one.
INT8 = "moss_image_sft_int8_convrot.safetensors"
BF16 = "moss_image_sft_bf16.safetensors"
# Q5_K GGUF of the SFT model by leejet (stable-diffusion.cpp's author; QKV fused like ComfyUI's own MOSS model, so it
# loads through ComfyUI-GGUF with patches/comfyui-gguf-moss-image.patch). For machines without an NVIDIA GPU: int8's
# fast kernels are CUDA-only -- without CUDA, int8 didn't finish one step at 256 px in 20 min on the CPU, this file a
# whole image in 229 s (2026-10-23). With CUDA int8 stays faster (27 s vs 70 s for 30 steps on a 3090).
GGUF = "MOSS-Image-SFT-Q5_K.gguf"
_V2 = "MOSS/moss_image_sft_4step_v2.0.safetensors"
_L8 = "MOSS/moss-sft-8step-dmd-lodestones.safetensors"
VARIANTS: Dict[str, Dict[str, Any]] = {
    "moss": {"unet": INT8, "steps": 30, "cfg": 3.0, "lora": None, "label": "MOSS-Image",
             "description": "Text in images, long prompts, positions (layout); 30 steps."},
    "moss-bf16": {"unet": BF16, "steps": 30, "cfg": 3.0, "lora": None, "label": "MOSS-Image bf16",
                  "description": "Like MOSS-Image, but the full bf16 main model (10 GB) – the reference for comparisons."},
    "moss-turbo-8": {"unet": INT8, "steps": 8, "cfg": 1.0, "lora": _L8, "strength": 1.0, "label": "MOSS Turbo 8",
                     "description": "lodestones' 8-step LoRA on the compact main model."},
    "moss-turbo-8-bf16": {"unet": BF16, "steps": 8, "cfg": 1.0, "lora": _L8, "strength": 1.0, "label": "MOSS Turbo 8 (bf16)",
                          "description": "lodestones' 8-step LoRA on the bf16 main model (as trained)."},
    "moss-turbo-4": {"unet": INT8, "steps": 4, "cfg": 1.0, "lora": _V2, "strength": 1.0, "label": "MOSS Turbo 4 v2.0",
                     "description": "akhaliq's 4-step LoRA v2.0 on the compact main model."},
    "moss-turbo-4-bf16": {"unet": BF16, "steps": 4, "cfg": 1.0, "lora": _V2, "strength": 1.0, "label": "MOSS Turbo 4 v2.0 (bf16)",
                          "description": "akhaliq's 4-step LoRA v2.0 on the bf16 main model – as its author recommends."},
}
ASPECT_SIZES = {"1:1": "1024x1024", "4:3": "1024x768", "3:4": "768x1024", "16:9": "1280x720", "9:16": "720x1280"}
MAX_N = 4
MAX_STEPS = 60


class InputError(ValueError):
    pass


def main_model(variant: str, have=lambda name: False, cuda: bool = True) -> str:
    """The main model file a variant runs on. bf16 variants: the bf16 file. The others use the compact main model:
    int8 with an NVIDIA GPU, the Q5 GGUF without one -- or whichever of the two is there when only one is.
    `have(name)` -> the file is in diffusion_models/."""
    if VARIANTS[variant]["unet"] == BF16:
        return BF16
    order = [INT8, GGUF] if cuda else [GGUF, INT8]
    return next((f for f in order if have(f)), order[0])


def label(variant: str, unet: str) -> str:
    """bf16 variants say so in their label; the others name the compact file they run on."""
    v = VARIANTS[variant]
    return v["label"] if unet == BF16 else f"{v['label']} ({'GGUF Q5' if unet == GGUF else 'int8'})"


def files_of(variant: str, unet: Optional[str] = None) -> List[str]:
    """Model files a variant needs, relative to the models root."""
    v = VARIANTS[variant]
    return [f"diffusion_models/{unet or v['unet']}"] + ([f"loras/{v['lora']}"] if v["lora"] else [])


TEXT_ENCODER = "qwen3.5_9b_w4a8_gptq.safetensors"
VAE = "flux2-vae.safetensors"


def needed_files(variant: str, unet: Optional[str] = None) -> List[str]:
    """Everything a variant loads, relative to the models root: its main model and LoRA, the text encoder, the VAE."""
    return files_of(variant, unet) + [f"text_encoders/{TEXT_ENCODER}", f"vae/{VAE}"]


def catalog(installed, unet_of=lambda variant: VARIANTS[variant]["unet"]) -> Dict[str, Any]:
    """`installed(variant)` -> bool, `unet_of(variant)` -> its main model file here. The contract's image catalog."""
    return {"object": "list", "data": [{
        "id": vid, "label": label(vid, unet_of(vid)), "description": v["description"],
        "available": bool(installed(vid)),
        "tasks": ["generate"], "aspect_ratios": list(ASPECT_SIZES), "text_in_image": True, "layout": True,
        "steps": {"default": v["steps"], "adjustable": True, "max": MAX_STEPS},
        # MOSS-Image-SFT: Apache-2.0 (OpenMOSS); both turbo LoRAs' repos state Apache-2.0 too (checked 2026-10-22)
        "license": "Apache-2.0", "commercial_use": True,
    } for vid, v in VARIANTS.items()]}


def from_openai(body: Any) -> Tuple[str, str, List[Element], int, int, int, Optional[int], Optional[int]]:
    """Untrusted generate body -> (variant, prompt, elements, width, height, n, seed, steps), or InputError."""
    if not isinstance(body, dict):
        raise InputError("send a JSON object")
    variant = str(body.get("model") or "moss").strip().lower()
    if variant not in VARIANTS:
        raise InputError(f"unknown model {body.get('model')!r} -- MOSS variants: {', '.join(VARIANTS)}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("prompt is required")
    aspect, size = body.get("aspect_ratio"), body.get("size")
    if aspect is not None:
        if size is not None:
            raise InputError("give either size or aspect_ratio, not both")
        if aspect not in ASPECT_SIZES:
            raise InputError(f"MOSS-Image supports the aspect ratios {', '.join(ASPECT_SIZES)}")
        size = ASPECT_SIZES[aspect]
    try:
        elements = parse_layout(body.get("layout"))
        width, height = parse_size(size)
    except LayoutError as e:
        raise InputError(str(e))
    n = body.get("n", 1)
    n = 1 if n is None else n
    if type(n) is not int or not 1 <= n <= MAX_N:
        raise InputError(f"n must be 1..{MAX_N}")
    seed, steps = body.get("seed"), body.get("steps")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 10**14):
        raise InputError("seed must be an integer 0..10^14")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= MAX_STEPS):
        raise InputError(f"steps must be 1..{MAX_STEPS}")
    return variant, prompt.strip(), elements, width, height, n, seed, steps


def build_workflow(prompt: str, elements: List[Element], width: int, height: int, *, seed: Optional[int] = None,
                   steps: Optional[int] = None, variant: str = "moss", unet: Optional[str] = None) -> dict:
    v = VARIANTS[variant]
    wf = copy.deepcopy(json.loads(WORKFLOW.read_text()))
    unet = unet or v["unet"]
    if unet.endswith(".gguf"):      # ComfyUI-GGUF's loader (with the moss_image patch); no weight_dtype input
        wf["1"] = {"class_type": "UnetLoaderGGUF", "_meta": {"title": "MOSS-Image GGUF"}, "inputs": {"unet_name": unet}}
    else:
        wf["1"]["inputs"]["unet_name"] = unet
    wf["4"]["inputs"]["text"] = moss_prompt(prompt, elements, width, height)
    wf["6"]["inputs"].update(width=width, height=height)
    wf["7"]["inputs"]["seed"] = seed if seed is not None else random.randint(1, 10**14 - 1)
    wf["7"]["inputs"]["steps"] = steps or v["steps"]
    wf["7"]["inputs"]["cfg"] = v["cfg"]
    if v["lora"]:
        wf["10"] = {"class_type": "LoraLoaderModelOnly", "_meta": {"title": "Turbo LoRA"},
                    "inputs": {"model": ["1", 0], "lora_name": v["lora"], "strength_model": v["strength"]}}
        wf["7"]["inputs"]["model"] = ["10", 0]
    return wf
