#!/usr/bin/env bash
# Build stt-serve's venv (WhisperX + pyannote.audio + the API server). WhisperX 3.8 pins torch~=2.8.
#   ./setup.sh [--download]     --download: fetch the default Whisper model (small) now, not on the first request
# PyTorch build by platform: an NVIDIA GPU (nvidia-smi) -> CUDA 12.8; a Mac -> PyPI's (CPU; WhisperX's
# ctranslate2 has no Metal backend); other Linux -> the CPU build (much smaller than the CUDA one).
# STT_SERVE_VENV puts the venv elsewhere (e.g. next to uv's cache, so wheels are hardlinked); .venv then links to it.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${STT_SERVE_VENV:-$DIR/.venv}"
UV="${UV:-uv}"
if [ "$(uname -s)" = "Darwin" ]; then
    INDEX=(--index-url https://pypi.org/simple)
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    INDEX=(--index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple)
else
    INDEX=(--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple)
fi
mkdir -p "$(dirname "$VENV")"
"$UV" venv --python 3.12 "$VENV"
export UV_HTTP_TIMEOUT=600
"$UV" pip install --python "$VENV/bin/python" "${INDEX[@]}" --index-strategy unsafe-best-match \
    "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0" \
    "whisperx==3.8.6" "pyannote.audio==4.0.4" soundfile pandas numpy \
    fastapi uvicorn httpx pydantic pytest
if [ "$VENV" != "$DIR/.venv" ] && [ ! -e "$DIR/.venv" ]; then ln -s "$VENV" "$DIR/.venv"; fi
if [ "${1:-}" = "--download" ]; then
    "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('Systran/faster-whisper-small')"
fi
echo "Done. stt-serve venv at $VENV"
