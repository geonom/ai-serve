#!/usr/bin/env bash
# Start stt-serve on 127.0.0.1 only -- who else may reach it is chosen in
# ai-serve-ui's Services tab (Tailscale levels), never here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT="${STT_SERVE_PORT:-8060}"
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python -m uvicorn stt_serve.app:app --host 127.0.0.1 --port "$PORT"
