#!/usr/bin/env bash
# The ai-serve plugin (127.0.0.1:$PORT). It starts sd-server (serve.sh) itself when an image is asked for.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "run ./setup.sh first" >&2; exit 1; }
exec .venv/bin/python -m sd_plugin.app --port "${PORT:-8065}"
