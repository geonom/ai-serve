#!/usr/bin/env bash
# Build the OmniVoice worker's venv (k2-fsa OmniVoice, transformers>=5.3 -- conflicts
# with qwen-tts' transformers==4.57.3, hence its own venv like every engine here).
#
# Disk: torch + CUDA libs are ~8 GB. uv hardlinks from its cache when cache and venv
# share a filesystem, so by default the venv is created next to uv's cache
# (TTS_SERVE_VENV_ROOT, default ~/.local/share/tts-serve/venvs) and ./venv is a
# symlink to it. Set TTS_SERVE_VENV_ROOT="" to build ./venv in place instead.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this provider directory
ROOT="${TTS_SERVE_VENV_ROOT-$HOME/.local/share/tts-serve/venvs}"
UV="${UV:-uv}"
if [ -n "$ROOT" ]; then
    VENV="$ROOT/omnivoice"
    mkdir -p "$ROOT"
else
    VENV="$DIR/venv"
fi
"$UV" venv --python 3.12 "$VENV"
export UV_HTTP_TIMEOUT=600
"$UV" pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==2.9.1" "torchaudio==2.9.1" "omnivoice==0.1.5" soundfile
if [ -n "$ROOT" ] && [ ! -e "$DIR/venv" ]; then
    ln -s "$VENV" "$DIR/venv"
fi
echo "Done. OmniVoice venv at $VENV (model k2-fsa/OmniVoice downloads on first use)."
