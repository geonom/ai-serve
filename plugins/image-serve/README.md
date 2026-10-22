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
| MOSS-Image (int8 + 4- and 8-step turbo LoRAs) | text in images, long prompts, placing things where you draw boxes (`layout`) | ~14.4 GB |
| MOSS-Image bf16 main model | the full-precision variants (on top of MOSS-Image) | ~9.5 GB |

All are Apache-2.0. Each file is pinned by Hugging Face repo, size and sha256 (`image_serve/bundles.py`) and
checked while downloading.

## Setup

It comes with ai-serve (`plugins/image-serve`). Nothing has to be done by hand (tried on Linux): the plugin starts even
before ComfyUI is set up, and the Spielwiese or the dashboard (*Compute & models → Downloads*) offer ComfyUI and the models -- a
model download sets up ComfyUI first. By hand:

```bash
./setup.sh                                   # ComfyUI (pinned release) + ComfyUI-GGUF + ComfyUI's Python
./setup.sh --models z-image-turbo,qwen-edit-2511,moss
```

`setup.sh` clones ComfyUI `v0.42.4` and city96's ComfyUI-GGUF (pinned, with `patches/` for Qwen-Image 2.1) into
`./ComfyUI` and builds ComfyUI's Python with the PyTorch for this machine: CUDA 13 with an NVIDIA driver that supports it (ComfyUI's
fast int8 kernels need it -- MOSS-Image is ~5x faster), else CUDA 12.8; PyPI's build on a Mac (Metal), else CPU. To
use a ComfyUI you already have, set `COMFYUI_DIR` (and `COMFYUI_PYTHON`, `COMFYUI_MODELS_ROOT`) in `.env` -- see
`.env.example`; MOSS-Image is then offered only if that ComfyUI is v0.42 or newer.

**MOSS-Image with `layout`:** `/v1/images/generations` with `"model": "moss-turbo-4"` (or `moss`, `moss-turbo-8`, the
`-bf16` variants) takes an extra `layout`: up to 20 elements `{"box": [left, top, right, bottom] (0..1), "kind":
"object"|"character"|"text", "caption": "...", "text": "..." (kind text)}` -- the Spielwiese's *With positions* page
draws them. ComfyUI runs on 127.0.0.1:8192 behind this plugin (not 8188,
so a ComfyUI you already run keeps its port).

**Getting MOSS-Image on an older setup:** image-serve's own ComfyUI (`./ComfyUI`, e.g. set up before setup.sh pinned
v0.42) is updated to the pinned release when you download MOSS-Image (Spielwiese → *With positions*, or the
dashboard's *Downloads*) or run `./setup.sh` again; a running ComfyUI is restarted once it's idle. A ComfyUI of your
own (`COMFYUI_DIR` in `.env`) is never changed: MOSS is then listed with the reason, and you update that ComfyUI
yourself (`git fetch --depth 1 origin tag v0.42.4 && git checkout v0.42.4`, then its `requirements.txt` into its
Python) or remove `COMFYUI_DIR` to let image-serve use its own.

**Status:** tested on Linux with NVIDIA (fresh setup with ComfyUI v0.42.4 and CUDA 13: Z-Image, Qwen-Image 2.1,
editing, style transfer, MOSS with and without layout, int8 and bf16; a real bundle file downloaded and checked).
ERNIE runs on the developer's machine but wasn't part of the fresh-setup test. **Apple Silicon: not tried.** setup.sh
installs PyTorch with Metal there, but expect it to be much slower, and the int8 / fp8 / w4a8 weights (MOSS,
Qwen-Image-Edit's text encoder) are made for CUDA – they may not work at all. AMD and CPU-only: not tried.

## License

image-serve: AGPL-3.0, like ai-serve. ComfyUI (GPL-3.0) and ComfyUI-GGUF (Apache-2.0) are downloaded by `setup.sh`,
not shipped here.
