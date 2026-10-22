"""Pure rules of sd-serve's ai-serve plugin (no I/O; tests tests/test_rules.py).

Threat model: requests come from ai-serve, which forwards callers it doesn't fully trust (Spielwiese guests through
the door, other apps). sd-server reads generation parameters out of the prompt itself -- a
`<sd_cpp_extra_args>{json}</sd_cpp_extra_args>` block can set any field, including file paths it then loads
(`ad_model`, upscalers). `build_request` is therefore the only way to a body: such a block in a prompt is refused,
sizes are the trained buckets only, and the one block sd-server gets is ours, with nothing but the seed.
(Moved here from ai-serve-ui's backend/images/iris.py, 2026-10-21.)"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

MODEL_ID = "iris-turbo"
ALIASES = ("iris", "iris-3b-turbo")
# Turbo was distilled at about one megapixel in these 7 aspect buckets (model card)
SIZES = ("1024x1024", "752x1344", "1344x752", "896x1344", "1344x896", "896x1200", "1200x896")
ASPECT_SIZES = {"1:1": "1024x1024", "4:3": "1200x896", "3:4": "896x1200", "16:9": "1344x752", "9:16": "752x1344"}
MAX_PROMPT_CHARS = 1200   # sd-server caps captions at 300 tokens
MAX_N = 4


class InputError(ValueError):
    pass


def build_request(prompt: Any, size: Optional[str], n: Any, seed: Any) -> Dict[str, Any]:
    """Untrusted request -> the body for sd-server's /v1/images/generations (or InputError)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise InputError("prompt is required")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise InputError(f"prompt too long (max {MAX_PROMPT_CHARS} characters)")
    if "sd_cpp_extra_args" in prompt.casefold():
        raise InputError("prompt must not contain sd_cpp_extra_args")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in prompt):
        raise InputError("prompt contains control characters")
    size = size or SIZES[0]
    if size not in SIZES:
        raise InputError(f"size must be one of {', '.join(SIZES)} (the sizes Iris Turbo was trained on)")
    n = 1 if n is None else n
    if type(n) is not int or not 1 <= n <= MAX_N:
        raise InputError(f"n must be 1..{MAX_N}")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise InputError("seed must be an integer 0..2^31-1")
    extra = json.dumps({"seed": seed})
    return {"prompt": f"{prompt} <sd_cpp_extra_args>{extra}</sd_cpp_extra_args>", "size": size, "n": n,
            "output_format": "png"}


def from_openai(body: Any) -> Dict[str, Any]:
    """The plugin contract's generate body (OpenAI images + aspect_ratio/steps/seed/layout) -> build_request."""
    if not isinstance(body, dict):
        raise InputError("send a JSON object")
    model = body.get("model")
    if model is not None and str(model).strip().lower() not in (MODEL_ID, *ALIASES):
        raise InputError(f"this plugin serves {MODEL_ID!r} only")
    if body.get("layout"):
        raise InputError("Iris can't place things by layout -- use a model with layout")
    if body.get("steps") is not None:
        raise InputError("Iris Turbo has a fixed 4-step schedule -- steps can't be set")
    aspect, size = body.get("aspect_ratio"), body.get("size")
    if aspect is not None:
        if size is not None:
            raise InputError("give either size or aspect_ratio, not both")
        if aspect not in ASPECT_SIZES:
            raise InputError(f"Iris supports the aspect ratios {', '.join(ASPECT_SIZES)}")
        size = ASPECT_SIZES[aspect]
    return build_request(body.get("prompt"), size, body.get("n"), body.get("seed"))


def catalog(installed: bool) -> Dict[str, Any]:
    return {"object": "list", "data": [{
        "id": MODEL_ID, "label": "Iris-3B Turbo", "aliases": list(ALIASES), "available": installed,
        "description": "Rechnet direkt in Pixeln, 4 Schritte, ~6 s; englische Schrift im Bild sauber, deutsche nicht.",
        "tasks": ["generate"], "aspect_ratios": list(ASPECT_SIZES), "text_in_image": True, "layout": False,
        "steps": {"default": 4, "adjustable": False, "max": 4},
        "license": "Apache-2.0", "commercial_use": True}]}


def pick_device(list_devices: str, gpu_name: Optional[str]) -> str:
    """sd-server's `--list-devices` output ("name<TAB>description" lines) -> the device to put Iris on: the one whose
    description is the GPU ai-serve picked (by name: Vulkan numbers GPUs differently from nvidia-smi), else the first
    non-CPU device (a Mac's Metal GPU), else CPU."""
    devices = []
    for line in list_devices.splitlines():
        name, _, desc = line.partition("\t")
        if name.strip():
            devices.append((name.strip(), desc.strip()))
    if gpu_name:
        for name, desc in devices:
            if desc == gpu_name.strip():
                return name
    for name, desc in devices:
        if name.upper() != "CPU":
            return name
    return "CPU"


def release_asset(system: str, machine: str, has_vulkan: bool, assets: List[str]) -> Optional[str]:
    """Which prebuilt stable-diffusion.cpp archive fits this machine (None: build it yourself, see CLAUDE.md)."""
    def find(*parts):
        return next((a for a in assets if all(p in a for p in parts) and a.endswith(".zip")), None)
    if system == "Darwin" and machine == "arm64":
        return find("Darwin", "arm64")
    if system == "Linux" and machine == "x86_64":
        if has_vulkan:
            return find("Linux", "x86_64", "vulkan")
        return next((a for a in assets if "Linux" in a and "x86_64" in a and a.endswith("x86_64.zip")), None)
    return None
