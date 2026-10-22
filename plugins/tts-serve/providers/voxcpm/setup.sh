#!/usr/bin/env bash
# Build venv for the VoxCPM2 TTS worker on the Nano-vLLM inference engine
# (providers/voxcpm/worker.py). Isolated for the usual torch/transformers reasons, plus
# a hard one: nano-vllm-voxcpm needs flash-attn, and flash-attn only ships prebuilt
# wheels for specific torch versions. Left alone, nano-vllm-voxcpm resolves the newest
# torch (2.14 at integration time) for which no flash-attn wheel exists -> building
# from source against a mismatching system nvcc. So torch is pinned to 2.9 (cu128)
# and flash-attn installed from its matching release wheel FIRST.
#
# Re-running recreates the venv. Needs `uv` and network. ~6 GB (torch/CUDA libs)
# + ~5 GB model (downloaded here, so the first request isn't a 10-minute cold start).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this provider directory
VENV="$DIR/venv"
UV="${UV:-uv}"
PY="$VENV/bin/python"
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

if ! command -v "$UV" >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

# The cuDNN wheel alone is ~700 MB; uv's default HTTP timeout cut it off.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

echo "=== Creating $VENV (Python 3.12) ==="
"$UV" venv --python 3.12 "$VENV"

echo "=== Installing torch 2.9 (cu128) ==="
"$UV" pip install --python "$PY" "torch==2.9.*" "torchaudio==2.9.*" \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing flash-attn (prebuilt wheel for torch 2.9 / cp312) ==="
"$UV" pip install --python "$PY" "$FLASH_WHL"

echo "=== Installing nano-vllm-voxcpm (torch stays pinned) ==="
"$UV" pip install --python "$PY" nano-vllm-voxcpm soundfile \
    "torch==2.9.*" "torchaudio==2.9.*" "torchcodec<0.9" "triton<3.6" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

echo "=== Sanity check ==="
"$PY" -c "import torch, flash_attn, nanovllm_voxcpm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'flash-attn', flash_attn.__version__)"

echo "=== Downloading openbmb/VoxCPM2 (~5 GB) ==="
# Xet transfer stalled indefinitely at integration time; plain HTTP is reliable.
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 "$PY" -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))"

echo
echo "Done. venv ready at $VENV"
echo "The backend picks it up automatically (VOXCPM_PYTHON_BIN default)."
echo "First 'engine=voxcpm' request still takes ~75 s (CUDA-graph capture); warm calls"
echo "run at RTF ~0.17, and concurrent requests are batched (RTF ~0.04 at 12 in flight)."
