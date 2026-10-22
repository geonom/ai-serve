#!/usr/bin/env bash
# Build nemotron/venv for the nvidia/Nemotron-3-Diarization worker
# (nemotron/worker.py). Isolated from this project's own venv/ for a
# harder version of the reason venv-qwen3tts/venv-cosyvoice/venv-chatterbox are:
# this model's transformers support (AutoModelForAudioFrameClassification) only
# exists on transformers' git main branch as of integration time, not in any tagged
# release -- an unpinned, moving-target transformers install that must never share
# an environment with OmniVoice's own transformers>=5.3.0 pin in venv/.
#
# Idempotent-ish: re-running recreates the venv. Needs `uv` (for the pinned Python)
# and network access. ~5 GB (torch) + a few hundred MB (the 100M-param model,
# downloaded lazily on first /api/stt/transcribe call with diarize_backend=nemotron,
# not here).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # stt-serve/nemotron
VENV="$DIR/venv"
UV="${UV:-uv}"

if ! command -v "$UV" >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it (https://docs.astral.sh/uv/) or set UV=/path/to/uv." >&2
    exit 1
fi

echo "=== Creating $VENV (Python 3.12) ==="
"$UV" venv --python 3.12 "$VENV"

echo "=== Installing torch + transformers (git main) ==="
# librosa + accelerate: confirmed live to be real, undeclared runtime dependencies of
# nvidia/Nemotron-3-Diarization's transformers integration -- neither is a hard
# dependency of transformers or torch themselves, and both surface as a plain
# ImportError/RuntimeError at inference time (feature extraction and device_map=
# model loading respectively), not at model/processor load time, if missing.
"$UV" pip install --python "$VENV/bin/python" torch soundfile librosa accelerate
# Pinned releases don't yet carry AutoModelForAudioFrameClassification/
# extract_speaker_dict for this model -- confirmed against the model card, which
# itself says "pip install git+https://github.com/huggingface/transformers". This
# is a moving target by nature; re-run this script to pick up upstream changes.
"$UV" pip install --python "$VENV/bin/python" "git+https://github.com/huggingface/transformers"

echo
echo "Done. nemotron/venv ready at $VENV"
echo "The backend picks it up automatically (NEMOTRON_DIARIZE_PYTHON_BIN default)."
echo "First diarize_backend=nemotron request downloads the model (~a few hundred MB,"
echo "100M params) and is slow; warm calls take seconds."
echo "libsndfile1/ffmpeg are used via the main venv's own audio decoding -- no need"
echo "to install them again here if WhisperX already works."
