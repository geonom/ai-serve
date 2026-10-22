#!/usr/bin/env bash
# venv for the Chatterbox (MLX) worker -- Apple Silicon only (MLX uses the Mac's GPU through Metal).
# ~0.3 GB here; the model (~2.5 GB) downloads on the first request (or: ./setup.sh --download).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    if [ "${FORCE_MLX:-}" != "1" ]; then
        echo "chatterbox-mlx is for Apple Silicon Macs; elsewhere use ../chatterbox (PyTorch)." >&2
        exit 1
    fi
fi
UV="${UV:-uv}"
"$UV" venv -q --python 3.12 "$DIR/venv"
"$UV" pip install -q --python "$DIR/venv/bin/python" mlx-audio soundfile numpy
if [ "${1:-}" = "--download" ]; then
    "$DIR/venv/bin/python" -c "from huggingface_hub import snapshot_download as s; s('${CHATTERBOX_MLX_MODEL:-mlx-community/chatterbox-multilingual-v3}')"
fi
echo "Done: $DIR/venv"
