#!/usr/bin/env bash
# Build venv for the Chatterbox-Multilingual TTS worker
# (providers/chatterbox/worker.py). Isolated from this project's own venv/ for the
# same reason as venv-qwen3tts: unrelated torch/transformers pins.
#
# Idempotent-ish: re-running recreates the venv. Needs `uv` (for the pinned Python)
# and network access. ~5 GB (torch) + ~1-2 GB (model, downloaded lazily on first
# generate, not here).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this provider directory
VENV="$DIR/venv"
UV="${UV:-uv}"

if ! command -v "$UV" >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

echo "=== Creating $VENV (Python 3.11) ==="
"$UV" venv --python 3.11 "$VENV"

echo "=== Installing chatterbox-tts ==="
# setuptools<80 for pkg_resources: chatterbox's watermarker dep (resemble-perth)
# imports pkg_resources, which setuptools>=81 no longer ships. A bare `uv venv` has
# no setuptools at all, so resemble-perth silently sets PerthImplicitWatermarker=None
# and Chatterbox then crashes calling None() in __init__.
"$UV" pip install --python "$VENV/bin/python" "setuptools<80" pip chatterbox-tts soundfile

if [ "${1:-}" = "--download" ]; then   # the model now instead of on the first request (ResembleAI/chatterbox)
    "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('ResembleAI/chatterbox')"
fi

echo
echo "Done. venv ready at $VENV"
echo "The backend picks it up automatically (CHATTERBOX_PYTHON_BIN default)."
echo "First 'engine=chatterbox' request downloads the model (~1-2 GB) and is slow;"
echo "warm calls take seconds."
