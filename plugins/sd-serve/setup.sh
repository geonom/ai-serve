#!/usr/bin/env bash
# sd-serve on a fresh machine: the plugin's small venv, then the prebuilt stable-diffusion.cpp for this platform
# (sha256-checked). --models also downloads Iris-3B Turbo + its text encoder (~5.4 GB, into SD_MODELS_DIR or ./models).
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null; then uv venv -q .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
else python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; fi
[ -f .env ] || touch .env
.venv/bin/python -m sd_plugin.install "$@"
