#!/usr/bin/env bash
# Build ./venv for the Qwen3-TTS worker (worker.py). Isolated because qwen-tts pins
# transformers==4.57.3, which conflicts with other engines' requirements.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this provider directory
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install qwen-tts soundfile
echo "Done. venv ready at $DIR/venv (checkpoints download on first use)."
