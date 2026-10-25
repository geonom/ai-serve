#!/usr/bin/env bash
# vision-serve on a fresh machine: its venv (ONNX Runtime with CUDA where there is an NVIDIA GPU, else the plain
# one), then the model files (~250 MB, sha256-pinned; VISION_MODELS_DIR or ./models).
set -euo pipefail
cd "$(dirname "$0")"
REQ=(-r requirements.txt)
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then REQ+=("onnxruntime-gpu[cuda,cudnn]>=1.20")
else REQ+=("onnxruntime>=1.20"); fi
if command -v uv >/dev/null; then uv venv -q .venv && uv pip install -q --python .venv/bin/python "${REQ[@]}"
else python3 -m venv .venv && .venv/bin/pip install -q "${REQ[@]}"; fi
[ -f .env ] || touch .env
.venv/bin/python -m vision_plugin.install
