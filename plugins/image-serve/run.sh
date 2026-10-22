#!/usr/bin/env bash
# image-serve (127.0.0.1:$PORT): the ai-serve image plugin; it starts ComfyUI itself when an image is asked for.
# It runs even before ComfyUI is set up -- then it offers ComfyUI and the models for download (plugin `install`).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then                 # its own small environment: seconds with uv
    command -v uv >/dev/null || { echo "uv is needed (https://docs.astral.sh/uv/), or run ./setup.sh" >&2; exit 1; }
    uv venv -q .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
exec .venv/bin/python -m image_serve.app --port "${PORT:-8189}"
