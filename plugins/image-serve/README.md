# image-serve -- ComfyUI as an ai-serve plugin

Image generation, image editing and style transfer for [ai-serve](https://github.com/geonom/ai-serve), run by
[ComfyUI](https://github.com/Comfy-Org/ComfyUI). ai-serve reaches it through its standard image API
(`/v1/images/generations`, `/v1/images/edits`); ComfyUI starts only when an image is asked for and gives its memory
back when idle. ComfyUI's own web interface is passed through as well.

| Model | For | Download |
|---|---|---|
| Z-Image Turbo (GGUF Q6_K) | fast, photorealistic images -- the default | ~9.1 GB |
| Qwen-Image-Edit 2511 (GGUF Q2_K + Lightning 4-step LoRA) | editing by instruction, style transfer | ~16.7 GB |
| ERNIE-Image Turbo (GGUF Q6_K) | fast, a bit more illustrative | ~13.7 GB |

All three are Apache-2.0. Each file is pinned by Hugging Face repo, size and sha256 (`image_serve/bundles.py`) and
checked while downloading.

## Setup

It comes with ai-serve (`plugins/image-serve`). Nothing has to be done by hand: the plugin starts even before ComfyUI
is set up, and the Spielwiese or the dashboard (*Compute & models → Downloads*) offer ComfyUI and the models -- a
model download sets up ComfyUI first. By hand:

```bash
./setup.sh                                   # ComfyUI (pinned release) + ComfyUI-GGUF + ComfyUI's Python
./setup.sh --models z-image-turbo,qwen-edit-2511
```

`setup.sh` clones ComfyUI `v0.37.0` and city96's ComfyUI-GGUF (pinned, with `patches/` for Qwen-Image 2.1) into
`./ComfyUI` and builds ComfyUI's Python with the PyTorch for this machine: CUDA with an NVIDIA GPU, PyPI's build on
a Mac (Metal), else CPU. To use a ComfyUI you already have, set `COMFYUI_DIR` (and `COMFYUI_PYTHON`,
`COMFYUI_MODELS_ROOT`) in `.env` -- see `.env.example`. ComfyUI runs on 127.0.0.1:8192 behind this plugin (not 8188,
so a ComfyUI you already run keeps its port).

**Status:** tested on Linux with NVIDIA (fresh setup, generating, editing). On Apple Silicon ComfyUI runs on Metal
but much slower; whether Qwen-Image-Edit's fp8 text encoder works there is not verified yet.

## License

image-serve: AGPL-3.0, like ai-serve. ComfyUI (GPL-3.0) and ComfyUI-GGUF (Apache-2.0) are downloaded by `setup.sh`,
not shipped here.
