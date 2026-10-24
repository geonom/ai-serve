"""Every image model the gateway offers, across providers -- one list for apps to pick from.

Before this, apps chose an image model by typing a checkpoint filename (StoryTime's settings), which only
reached the main ComfyUI and broke silently on a typo; MOSS-Image and Iris weren't reachable that way at all.
`GET /v1/images/models` lists this catalog (with live availability: is the model file on disk), and
`/v1/images/generations` accepts its ids, plus an `aspect_ratio` that each provider turns into a size it
was trained for. Old filenames still resolve, so stored settings keep working. Pure apart from
`listing()`'s file check. Tests: tests/unit/test_image_catalog.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ASPECTS = ("1:1", "4:3", "3:4", "16:9", "9:16")
DEFAULT = "z-image-turbo"
TASKS = ("generate", "edit", "style", "remove_background")   # text-to-image, edit one image, content + style image,
                                                             # cut out the foreground (alpha channel)
DEFAULTS = {"generate": DEFAULT, "edit": "qwen-edit-2511", "style": "qwen-edit-2511", "remove_background": "birefnet"}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class ImageModel:
    id: str
    label: str
    provider: str              # comfyui  (Iris, MOSS: image plugins since 2026-10-21)
    file: str                  # checkpoint, relative to the models root's unet/ or diffusion_models/
    description: str
    aspects: Tuple[str, ...] = ASPECTS
    text_in_image: bool = False
    layout: bool = False
    aliases: Tuple[str, ...] = ()
    steps_default: int = 8         # what the workflow/variant uses when the caller gives no steps
    steps_adjustable: bool = True  # Iris' schedule is part of the model
    steps_max: int = 60
    tasks: Tuple[str, ...] = ("generate",)
    max_images: int = 0            # input images an edit takes (0 = none: text to image)


CATALOG: Tuple[ImageModel, ...] = (
    ImageModel("z-image-turbo", "Z-Image Turbo", "comfyui", "z_image_turbo-Q6_K.gguf",
               "Fast and photorealistic, the default (ComfyUI, ~9 s).", steps_default=8),
    ImageModel("ernie-turbo", "ERNIE-Image Turbo", "comfyui", "ernie-image-turbo-Q6_K.gguf",
               "Fast, a bit more illustrative (ComfyUI).", steps_default=4),
    # 40 steps: the official pipeline uses 40-50; at 20 (our old default) images came out washed out and
    # blotchy (user 2026-10-19, confirmed side by side on a fixed seed)
    ImageModel("qwen-image-2.1", "Qwen-Image 2.1", "comfyui", "qwen-image-2.1-Q4_K_M.gguf",
               "Rich in detail, needs ~40 steps (~100 s); symmetric subjects sometimes come out mirrored (ComfyUI, GGUF Q4).",
               text_in_image=True, steps_default=40),
    ImageModel("qwen-image-2.1-int8", "Qwen-Image 2.1 (offiziell, int8)", "comfyui", "qwen_image_2.1_int8_convrot.safetensors",
               "The official Comfy-Org file instead of the GGUF – slightly finer details, but slower here (~4 s per step).",
               text_in_image=True, steps_default=40),
    ImageModel("ideogram-4", "Ideogram 4", "comfyui", "ideogram4_nvfp4_mixed.safetensors",
               "Slow; its built-in safety filter also blocks harmless prompts (ComfyUI).", text_in_image=True,
               steps_default=20),
    # Editing: Qwen-Image-Edit 2511 with lightx2v's 2511 Lightning 4-step LoRA baked into the workflows
    # (TextEncodeQwenImageEditPlus; takes a second image, so it also does style transfer).
    # the older pre-2511 Qwen-Image-Edit files were removed (owner, 2026-10-21) -- their names, still stored in
    # language_learning and Munin, now mean this one
    ImageModel("qwen-edit-2511", "Qwen-Image-Edit 2511 (Q2_K)", "comfyui", "qwen-image-edit-2511-Q2_K.gguf",
               "The default for editing and style transfer; keeps characters consistent over many images, takes two images (~50 s).",
               tasks=("edit", "style"), steps_default=4, steps_max=20, max_images=5,
               aliases=("qwen-edit", "qwen-edit-q4", "qwen_image_edit-q2_k.gguf", "qwen_image_edit-q4_k_s.gguf")),
    ImageModel("qwen-edit-2511-q3", "Qwen-Image-Edit 2511 (Q3_K_S)", "comfyui", "qwen-image-edit-2511-Q3_K_S.gguf",
               "Fastest (~40 s), but followed instructions less well in tests.", tasks=("edit", "style"),
               steps_default=4, steps_max=20, max_images=5),
    # ComfyUI's own BiRefNet node (comfy_extras/nodes_bg_removal.py, since 2026-05); Comfy-Org's repackaged file
    ImageModel("birefnet", "BiRefNet (remove background)", "comfyui", "birefnet.safetensors",
               "Cuts out the foreground: the same image back as a PNG with transparency (~1 GB VRAM, seconds).",
               aspects=(), tasks=("remove_background",), steps_default=1, steps_adjustable=False, steps_max=1,
               max_images=1),
)

def resolve(name: Optional[str]) -> Optional[ImageModel]:
    """A catalog entry by id, alias or (for settings stored before the catalog) checkpoint filename."""
    if not name:
        return None
    key = name.strip().lower()
    for m in CATALOG:
        if key == m.id or key in m.aliases or key == m.file.lower() or key == os.path.basename(m.file).lower():
            return m
    return None


def _on_disk(models_root: str, rel: str) -> bool:
    return any(os.path.isfile(os.path.join(models_root, d, rel)) for d in ("unet", "diffusion_models"))


from image_serve.config import COMFYUI_MODELS_ROOT as MODELS_ROOT   # where image checkpoints live


def available(m: ImageModel, models_root: Optional[str] = None) -> bool:
    """A model from a download bundle needs all its files (encoder, VAE, LoRA too); others: the checkpoint on disk."""
    from image_serve import bundles, config
    if not config.comfy_ready():                     # no engine yet: nothing can run
        return False
    root = models_root or MODELS_ROOT
    bundle = BUNDLE_OF.get(m.id)
    return bundles.present(bundle, root) if bundle else _on_disk(root, m.file)


# catalog model -> the download bundle that brings it (image_serve/bundles.py)
BUNDLE_OF = {"z-image-turbo": "z-image-turbo", "qwen-edit-2511": "qwen-edit-2511", "ernie-turbo": "ernie-turbo",
             "birefnet": "birefnet"}


def listing_row(m: ImageModel, models_root: Optional[str] = None) -> dict:
    """What apps see of one model: no file paths, availability = its files are on disk."""
    return {"id": m.id, "label": m.label, "provider": m.provider, "description": m.description,
            "aspect_ratios": list(m.aspects), "text_in_image": m.text_in_image, "layout": m.layout,
            "available": available(m, models_root), "default": m.id == DEFAULTS[m.tasks[0]],
            # other names /v1/images/generations accepts for it (e.g. a checkpoint name saved in an app's
            # settings before the catalog) -- the bare filename, never a path
            "aliases": [*m.aliases, os.path.basename(m.file)],
            "steps": {"default": m.steps_default, "adjustable": m.steps_adjustable, "max": m.steps_max},
            "tasks": list(m.tasks), "max_images": m.max_images}


def check_steps(model: Optional[ImageModel], steps: Optional[int]) -> None:
    """None = the model's default. Iris only runs its own fixed schedule."""
    if steps is None:
        return
    top = model.steps_max if model else 60
    if model and not model.steps_adjustable and steps != model.steps_default:
        raise CatalogError(f"{model.label} has a fixed {model.steps_default}-step schedule")
    if not 1 <= steps <= top:
        raise CatalogError(f"steps must be 1..{top}")


def listing(models_root: Optional[str] = None, task: str = "generate") -> List[dict]:
    """task: generate (default -- what apps listed before edit models were in here) | edit | style |
    remove_background | all.
    `default` marks the default for the task asked for."""
    if task != "all" and task not in TASKS:
        raise CatalogError(f"task must be one of {', '.join(TASKS)}, all")
    rows = []
    for m in CATALOG:
        if task == "all" or task in m.tasks:
            row = listing_row(m, models_root)
            # neither here nor downloadable (e.g. Ideogram 4 on a fresh machine): not offered at all
            if not row["available"] and m.id not in BUNDLE_OF:
                continue
            if task != "all":
                row["default"] = m.id == DEFAULTS[task]
            rows.append(row)
    return rows


def for_task(name: Optional[str], task: str) -> Optional[ImageModel]:
    """The catalog model for an edit/style/generate request: None name -> the task's default; a catalog
    model that can't do the task -> CatalogError; an unknown name -> None (passed on unchanged, as before)."""
    model = resolve(name) if name else resolve(DEFAULTS[task])
    if model is None:
        return None
    if task not in model.tasks:
        raise CatalogError(f"{model.label} can't {task}; models that can: "
                           + ", ".join(m.id for m in CATALOG if task in m.tasks))
    return model
