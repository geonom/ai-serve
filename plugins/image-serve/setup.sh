#!/usr/bin/env bash
# image-serve on a fresh machine: its own small venv, ComfyUI (pinned release) with ComfyUI-GGUF (pinned, plus the
# patch in patches/ for Qwen-Image 2.1), ComfyUI's own Python environment with the PyTorch build for this machine
# (CUDA with an NVIDIA GPU, else PyPI's build -- Metal on Apple Silicon -- or CPU), and optionally model bundles:
#   ./setup.sh [--models z-image-turbo,qwen-edit-2511,ernie-turbo,moss,moss-bf16] [--dry-run]
# --dry-run makes every decision (which ComfyUI, which PyTorch build) and prints what it would run -- installs nothing
# (the CI runs it on Linux and macOS: a missing nvidia-smi once ended the setup on a Mac).
# An existing ComfyUI is used as it is when COMFYUI_DIR points at it (.env); then nothing is cloned there.
# Needs: git, uv (https://docs.astral.sh/uv/). Re-running is fine.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd -P)"
COMFYUI_TAG="v0.42.4"                           # >= v0.42: MOSS-Image
GGUF_COMMIT="6ea2651e7"                         # city96/ComfyUI-GGUF, 2026-01-12
MODELS=""; DRY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="${2:-}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
command -v uv >/dev/null || [ -n "$DRY" ] || { echo "uv is needed: https://docs.astral.sh/uv/" >&2; exit 1; }
run() { if [ -n "$DRY" ]; then echo "  would run: $*"; else "$@"; fi; }

echo "== image-serve's own environment"
run uv venv -q --allow-existing .venv
run uv pip install -q --python .venv/bin/python -r requirements.txt
[ -f .env ] || run cp .env.example .env
if [ -f .env ]; then set -a; . ./.env; set +a; fi
COMFYUI_DIR="${COMFYUI_DIR:-$HERE/ComfyUI}"

UPGRADED=""
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "== ComfyUI $COMFYUI_TAG"
    run git clone -q --depth 1 --branch "$COMFYUI_TAG" https://github.com/Comfy-Org/ComfyUI.git "$COMFYUI_DIR"
elif [ "$(cd "$COMFYUI_DIR" && pwd -P)" = "$HERE/ComfyUI" ] && [ -d "$COMFYUI_DIR/.git" ]; then
    # our own ComfyUI (cloned here before): bring it to the pinned release -- e.g. one set up before v0.42 (MOSS).
    # A ComfyUI configured elsewhere (COMFYUI_DIR in .env) is never touched.
    HAVE="$(git -C "$COMFYUI_DIR" describe --tags --exact-match 2>/dev/null || echo unknown)"
    if [ "$HAVE" != "$COMFYUI_TAG" ]; then
        echo "== ComfyUI $HAVE -> $COMFYUI_TAG"
        run git -C "$COMFYUI_DIR" fetch -q --depth 1 origin tag "$COMFYUI_TAG"
        run git -C "$COMFYUI_DIR" checkout -q "$COMFYUI_TAG"
        UPGRADED=1
    fi
fi
GGUF="$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF" ]; then
    echo "== ComfyUI-GGUF (GGUF models)"
    run git clone -q https://github.com/city96/ComfyUI-GGUF.git "$GGUF"
    run git -C "$GGUF" checkout -q "$GGUF_COMMIT"
    run git -C "$GGUF" apply "$HERE/patches/comfyui-gguf-qwen-image21.patch"
fi

PY="${COMFYUI_PYTHON:-$COMFYUI_DIR/venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "== ComfyUI's Python environment (PyTorch for this machine -- a few GB)"
    # CUDA 13 build when the driver can run it: ComfyUI's optimized int8 kernels (MOSS-Image, the int8 Qwen-Image)
    # need it -- with cu128 a MOSS image took 153 s instead of 29 s on a 4060 Ti
    CUDA_MAJOR=0                      # no nvidia-smi (a Mac, AMD, CPU): set -e + pipefail must not end the setup here
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1 || true)"
    fi
    if nvidia-smi -L >/dev/null 2>&1 && [ "${CUDA_MAJOR:-0}" -ge 13 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    elif nvidia-smi -L >/dev/null 2>&1; then TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$(uname -s)" = "Darwin" ]; then TORCH_INDEX=""
    else TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi
    echo "  PyTorch build: ${TORCH_INDEX:-PyPI (Metal on Apple Silicon)}"
    run env COMFYUI_DIR="$COMFYUI_DIR" TORCH_INDEX="$TORCH_INDEX" scripts/setup-comfyui-venv.sh "$(dirname "$(dirname "$PY")")"
fi

if [ -n "$UPGRADED" ] && [ -x "$PY" ]; then
    echo "== ComfyUI's requirements for $COMFYUI_TAG"
    run uv pip install -q --python "$PY" -r "$COMFYUI_DIR/requirements.txt"
fi

if [ -n "$MODELS" ]; then
    echo "== models: $MODELS"
    run .venv/bin/python -m image_serve.bundles ${MODELS//,/ }
fi
echo "ok -- ComfyUI in $COMFYUI_DIR; more models: Spielwiese, dashboard (Downloads) or ./setup.sh --models <ids>"
