#!/usr/bin/env bash
# Builds the MOSS-Transcribe-Diarize worker's venv: Python 3.12, PyTorch for CUDA 13, the authors' package.
# The venv lives next to uv's cache on / (hardlinks instead of copies); moss/venv is a symlink to it.
set -euo pipefail
cd "$(dirname "$0")"
[ -d repo ] || git clone --depth 20 https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git repo
V="$HOME/.local/share/stt-serve/moss-venv"
uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" torch torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$V/bin/python" -e ./repo
ln -sfn "$V" venv
echo "MOSS-Transcribe-Diarize venv ready; the model (~1.8 GB) downloads on first use."
