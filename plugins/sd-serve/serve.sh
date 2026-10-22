#!/usr/bin/env bash
# sd-server with Iris-3B Turbo (pixel-space text-to-image, Apache-2.0), started by the sd-serve plugin (sd_plugin/app.py)
# on 127.0.0.1:$PORT. The 4-step schedule is the authors' (turbo/distill.yaml: sigma 1, 0.938, 0.833, 0.625 -> 0,
# flow-Euler, no CFG); callers can't change it. Placement is fixed to one device (no auto-fit): auto-fit decides once
# at process start from the memory free *then* and silently keeps the weights in RAM if it's short (20 s instead of
# 5 s per image, found live 2026-10-18) -- the plugin reserves the room before it starts this.
#   SD_DEVICE  the device from `sd-server --list-devices` (Vulkan1, MTL0, CPU ...; the plugin picks it by GPU name)
#   SD_GPU_ID  with a local CUDA build in sdcpp/build/bin: the GPU index (nvidia-smi numbering)
set -euo pipefail
cd "$(dirname "$0")"
MODELS="${SD_MODELS_DIR:-./models}"
if [ -x sdcpp/build/bin/sd-server ]; then          # this machine's own CUDA build
    BIN=sdcpp/build/bin/sd-server
    export CUDA_VISIBLE_DEVICES="${SD_GPU_ID:-0}"
    DEV="${SD_DEVICE:-CUDA0}"
else                                               # the prebuilt release (sd_plugin/install.py)
    BIN=bin/sd-server
    export LD_LIBRARY_PATH="$PWD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH="$PWD/bin${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    DEV="${SD_DEVICE:-CPU}"
fi
exec "$BIN" \
    --listen-ip 127.0.0.1 --listen-port "${PORT:-8064}" \
    --diffusion-model "$MODELS/diffusion_models/iris/iris-3b-turbo-Q4_K.gguf" \
    --llm "$MODELS/text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf" \
    --sampling-method euler --sigmas "1,0.938,0.833,0.625,0" --cfg-scale 1 \
    -W 1024 -H 1024 --diffusion-fa \
    --auto-fit off --backend "diffusion=$DEV,te=$DEV" --params-backend "diffusion=$DEV,te=$DEV"
