#!/usr/bin/env bash
# Build venv + check out the CosyVoice repo for the CosyVoice 3 worker
# (providers/cosyvoice/worker.py). Isolated from this project's own venv/ like
# venv-qwen3tts.
#
# CosyVoice is NOT a pip package -- it's a git repo you put on sys.path, plus its
# bundled third_party/Matcha-TTS. This script clones it under ./repo
# (override with COSYVOICE_REPO_DIR), builds the venv from its requirements.txt, and
# downloads Fun-CosyVoice3-0.5B into {repo}/pretrained_models/Fun-CosyVoice3-0.5B
# (override with COSYVOICE_MODEL_DIR).
#
# Needs `uv` (pinned Python 3.10), git, network, ~6 GB venv + ~2 GB model + repo.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this provider directory
REPO="$DIR/repo"
VENV="$DIR/venv"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-$REPO/pretrained_models/Fun-CosyVoice3-0.5B}"
UV="${UV:-uv}"

if ! command -v "$UV" >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

echo "=== CosyVoice repo -> $REPO ==="
mkdir -p "$(dirname "$REPO")"
if [ ! -d "$REPO/.git" ]; then
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO"
else
    git -C "$REPO" pull --ff-only || true
    git -C "$REPO" submodule update --init --recursive
fi

echo "=== Creating $VENV (Python 3.10) ==="
"$UV" venv --python 3.10 "$VENV"
# pkg_resources (setuptools<81) -- several CosyVoice / matcha deps still import it,
# and a bare `uv venv` ships no setuptools.
"$UV" pip install --python "$VENV/bin/python" "setuptools<80" pip

echo "=== Installing CosyVoice requirements (this takes a while) ==="
# CosyVoice's requirements.txt is inference + training + TRT + gRPC + Gradio all in
# one. For a pure-inference worker we drop the heaviest bits that aren't on the
# generate() path: tensorrt* (only used with load_trt=True), deepspeed (CUDA-op
# build, lazy-imported), gradio/grpcio*/tensorboard/fastapi-cli/gdown/wget.
# --index-strategy unsafe-best-match: the file's extra Azure index doesn't carry
# protobuf==4.25, and uv's default first-index-wins then fails to resolve it.
REQ_TRIMMED="$(mktemp)"
trap 'rm -f "$REQ_TRIMMED"' EXIT
# Drop training / TensorRT / serving-frontend deps not on the inference path.
# NOT openai-whisper -- cosyvoice/cli/frontend.py does a hard `import whisper`.
grep -vE '^(tensorrt-cu12|deepspeed|gradio|grpcio|grpcio-tools|tensorboard|fastapi-cli|gdown|wget)([=<>; ]|$)' \
    "$REPO/requirements.txt" > "$REQ_TRIMMED"
"$UV" pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.org/simple --no-build-isolation -r "$REQ_TRIMMED"
"$UV" pip install --python "$VENV/bin/python" huggingface_hub soundfile

echo "=== Downloading Fun-CosyVoice3-0.5B -> $MODEL_DIR ==="
# HF_HUB_ENABLE_HF_TRANSFER=1 is set globally in this machine's ~/.bashrc and
# inherited here, but hf_transfer isn't in this venv -> hf_hub raises instead of
# falling back. Force it off for the download (same workaround the qwen3tts worker
# launcher uses).
export HF_HUB_ENABLE_HF_TRANSFER=0
"$VENV/bin/python" - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=sys.argv[1])
print("model at", dest)
PY

echo
echo "Done."
echo "  repo                = $REPO"
echo "  COSYVOICE_MODEL_DIR = $MODEL_DIR"
echo "tts-serve starts it via llama-serve.provider.json."
