#!/usr/bin/env bash
# Build ComfyUI's own Python environment (torch etc.) -- separate from image-serve's small .venv and from ai-serve.
#   scripts/setup-comfyui-venv.sh [target-dir]        default: ./comfyui-venv
# Reads COMFYUI_DIR from .env. TORCH_INDEX picks the PyTorch build (default CUDA 12.8; Apple Silicon / CPU: leave
# TORCH_INDEX empty to take PyPI's build). Then installs ComfyUI's requirements.txt and every custom node's.
# Exact versions this was verified with: comfyui-requirements.lock (`uv pip install -r` it instead, same indexes).
# Tip: put the venv on the same disk as uv's cache (uv cache dir) -- packages are then hardlinked, not copied.
set -euo pipefail
cd "$(dirname "$0")/.."
_dir="${COMFYUI_DIR:-}"; _idx="${TORCH_INDEX-unset}"
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$_dir" ] && COMFYUI_DIR="$_dir"                # the caller's (setup.sh) wins over .env
[ "$_idx" != "unset" ] && TORCH_INDEX="$_idx"
COMFYUI_DIR="${COMFYUI_DIR:-$(pwd -P)/ComfyUI}"
TARGET="${1:-comfyui-venv}"
TORCH_INDEX="${TORCH_INDEX-https://download.pytorch.org/whl/cu128}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
uv venv -q --python 3.12 "$TARGET"
PY="$TARGET/bin/python"
if [ -n "$TORCH_INDEX" ]; then
  uv pip install -q --python "$PY" torch torchvision torchaudio --index-url "$TORCH_INDEX"
else
  uv pip install -q --python "$PY" torch torchvision torchaudio
fi
uv pip install -q --python "$PY" -r "$COMFYUI_DIR/requirements.txt"
for r in "$COMFYUI_DIR"/custom_nodes/*/requirements.txt; do
  [ -f "$r" ] && uv pip install -q --python "$PY" -r "$r"
done
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'no cuda')"
T="$(cd "$(dirname "$TARGET")" && pwd -P)/$(basename "$TARGET")"
[ "$T" = "$(cd "$COMFYUI_DIR" && pwd -P)/venv" ] || echo "ok -- set COMFYUI_PYTHON=$T/bin/python in .env"
