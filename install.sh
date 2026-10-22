#!/usr/bin/env bash
# ai-serve installer (Linux, macOS). Creates ai-serve's own Python environment, then asks which basic services to set
# up (./ai-serve setup). Needs: git, uv (https://docs.astral.sh/uv/). Re-running it is fine.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$DIR"
case "$(uname -s)" in Linux|Darwin) ;; *) echo "ai-serve runs on Linux and macOS for now." >&2; exit 1 ;; esac
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is missing. Install it (it brings its own Python):"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "then open a new terminal and run ./install.sh again."
    exit 1
fi
echo "== ai-serve: Python environment"
[ -x venv/bin/python ] || uv venv -q --python 3.12 venv
uv pip install -q --python venv/bin/python -r backend/requirements.txt
[ -f ai-serve.local.env ] || cp ai-serve.example.env ai-serve.local.env
chmod +x ai-serve launcher.sh
exec ./ai-serve setup "$@"
