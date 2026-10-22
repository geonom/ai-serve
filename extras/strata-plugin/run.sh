#!/usr/bin/env bash
# Started by ai-serve (Apps & Services) on demand. First time: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "run ./setup.sh first" >&2; exit 1; }
exec .venv/bin/python -m strata_plugin.app --host 127.0.0.1 --port "${PORT:-8075}"
