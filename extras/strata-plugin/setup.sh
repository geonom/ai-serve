#!/usr/bin/env bash
# A small venv for the adapter (FastAPI, httpx, psutil) -- Strata itself is installed separately.
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null; then
  uv venv -q .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
echo "ok -- set STRATA_DIR in .env"
