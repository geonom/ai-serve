#!/usr/bin/env bash
# The ai-serve plugin (127.0.0.1:$PORT). Its worker (the face models) starts with the first image.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "run ./setup.sh first" >&2; exit 1; }
exec .venv/bin/python -m vision_plugin.app --port "${PORT:-8068}"
