#!/usr/bin/env bash
# Start tts-serve on 127.0.0.1 only -- who else may reach it is chosen in
# ai-serve-ui's Services tab (Tailscale levels), never here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT="${TTS_SERVE_PORT:-8050}"
if [ ! -x .venv/bin/python ]; then
    uv venv -q --python 3.12 .venv
    uv pip install -q --python .venv/bin/python fastapi uvicorn httpx pydantic
fi
exec .venv/bin/python -m uvicorn tts_serve.app:app --host 127.0.0.1 --port "$PORT"
