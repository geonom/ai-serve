"""image-serve settings: .env next to this project (gitignored), then the environment. Defaults are portable; this
machine's values live in .env."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    f = HERE / ".env"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
COMFYUI_DIR = os.environ.get("COMFYUI_DIR") or str(HERE / "ComfyUI")
# the Python that runs ComfyUI (torch etc.); default: ComfyUI's own venv next to it
COMFYUI_PYTHON = os.environ.get("COMFYUI_PYTHON") or os.path.join(COMFYUI_DIR, "venv", "bin", "python")
COMFYUI_MODELS_ROOT = os.environ.get("COMFYUI_MODELS_ROOT") or os.path.join(COMFYUI_DIR, "models")
# ComfyUI itself, loopback only; this app proxies it. Not 8188 (ComfyUI's usual port): a ComfyUI the person already
# runs keeps its port.
COMFY_PORT = int(os.environ.get("COMFY_PORT") or "8192")
_gpu = os.environ.get("COMFYUI_DEFAULT_GPU_ID", "auto").strip()
COMFYUI_DEFAULT_GPU_ID = "auto" if _gpu == "auto" else (int(_gpu) if _gpu else None)
GATEWAY = os.environ.get("AI_SERVE_URL", "http://127.0.0.1:8030").rstrip("/")
LOGS_DIR = os.environ.get("LOGS_DIR", str(HERE / "logs"))
GENERATED_IMAGES_DIR = os.environ.get("GENERATED_IMAGES_DIR", str(HERE / "generated_images"))
WORKFLOWS_DIR = str(HERE / "workflows")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)


MOSS_MIN_COMFYUI = (0, 42, 0)     # MOSS-Image support landed in ComfyUI v0.42.0


def comfy_version():
    """(major, minor, patch) of the ComfyUI in COMFYUI_DIR (its comfyui_version.py), None if unreadable."""
    import re
    try:
        with open(os.path.join(COMFYUI_DIR, "comfyui_version.py"), encoding="utf-8") as f:
            m = re.search(r'__version__\s*=\s*["\'](\d+)\.(\d+)\.(\d+)', f.read())
    except OSError:
        return None
    return tuple(int(x) for x in m.groups()) if m else None


def moss_possible() -> bool:
    """MOSS-Image can run here: the ComfyUI here is new enough -- or there is none yet, and setup.sh will clone its
    pinned release (new enough). An older ComfyUI configured in .env (e.g. one the owner runs) hides MOSS."""
    if not os.path.isfile(os.path.join(COMFYUI_DIR, "main.py")):
        return True
    v = comfy_version()
    return v is not None and v >= MOSS_MIN_COMFYUI


def comfy_ready() -> bool:
    """ComfyUI and its Python are there (setup.sh ran, or an existing ComfyUI is configured in .env)."""
    return os.path.isfile(os.path.join(COMFYUI_DIR, "main.py")) and os.access(COMFYUI_PYTHON, os.X_OK)
