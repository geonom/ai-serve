#!/usr/bin/env bash
# image-serve on a fresh machine: its own small venv, ComfyUI (pinned release) with ComfyUI-GGUF (pinned, plus the
# patch in patches/ for Qwen-Image 2.1), ComfyUI's own Python environment with the PyTorch build for this machine
# (CUDA with an NVIDIA GPU, else PyPI's build -- Metal on Apple Silicon -- or CPU), and optionally model bundles:
#   ./setup.sh [--models z-image-turbo,qwen-edit-2511,ernie-turbo,moss,moss-bf16]
# An existing ComfyUI is used as it is when COMFYUI_DIR points at it (.env); then nothing is cloned there.
# Needs: git, uv (https://docs.astral.sh/uv/). Re-running is fine.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd -P)"
COMFYUI_TAG="v0.42.4"                           # >= v0.42: MOSS-Image
GGUF_COMMIT="6ea2651e7"                         # city96/ComfyUI-GGUF, 2026-01-12
MODELS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="${2:-}"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
command -v uv >/dev/null || { echo "uv is needed: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "== image-serve's own environment"
uv venv -q --allow-existing .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
[ -f .env ] || cp .env.example .env
set -a; . ./.env; set +a
COMFYUI_DIR="${COMFYUI_DIR:-$HERE/ComfyUI}"

if [ ! -d "$COMFYUI_DIR" ]; then
    echo "== ComfyUI $COMFYUI_TAG"
    git clone -q --depth 1 --branch "$COMFYUI_TAG" https://github.com/Comfy-Org/ComfyUI.git "$COMFYUI_DIR"
fi
GGUF="$COMFYUI_DIR/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF" ]; then
    echo "== ComfyUI-GGUF (GGUF models)"
    git clone -q https://github.com/city96/ComfyUI-GGUF.git "$GGUF"
    git -C "$GGUF" checkout -q "$GGUF_COMMIT"
    git -C "$GGUF" apply "$HERE/patches/comfyui-gguf-qwen-image21.patch"
fi

PY="${COMFYUI_PYTHON:-$COMFYUI_DIR/venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "== ComfyUI's Python environment (PyTorch for this machine -- a few GB)"
    # CUDA 13 build when the driver can run it: ComfyUI's optimized int8 kernels (MOSS-Image, the int8 Qwen-Image)
    # need it -- with cu128 a MOSS image took 153 s instead of 29 s on a 4060 Ti
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)"
    if nvidia-smi -L >/dev/null 2>&1 && [ "${CUDA_MAJOR:-0}" -ge 13 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    elif nvidia-smi -L >/dev/null 2>&1; then TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$(uname -s)" = "Darwin" ]; then TORCH_INDEX=""
    else TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi
    COMFYUI_DIR="$COMFYUI_DIR" TORCH_INDEX="$TORCH_INDEX" scripts/setup-comfyui-venv.sh "$(dirname "$(dirname "$PY")")"
fi

if [ -n "$MODELS" ]; then
    echo "== models: $MODELS"
    .venv/bin/python -m image_serve.bundles ${MODELS//,/ }
fi
echo "ok -- ComfyUI in $COMFYUI_DIR; more models: Spielwiese, dashboard (Downloads) or ./setup.sh --models <ids>"
