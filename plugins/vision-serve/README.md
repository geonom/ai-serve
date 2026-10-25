# vision-serve -- faces in photos, as an ai-serve plugin

Finds faces in photos and turns each into a vector for recognising the same person in other photos, for
[ai-serve](https://github.com/geonom/ai-serve). ai-serve reaches it through its `vision` API: `GET /v1/vision/models`
and `POST /v1/vision/analyze` (multipart `image`, repeated for several; `tasks` = `faces` | `objects` | `both`;
`embed=false` for boxes only). The models start with the first image and give their memory back when idle.

| Model | For | Licence | Download |
|---|---|---|---|
| YuNet 2023mar (OpenCV Zoo) | face boxes + 5 landmarks, on the CPU | MIT | 0.2 MB |
| AuraFace v1 (`glintr100.onnx`, fal) | 512-dim face vectors (cosine), on the GPU where there is one | Apache-2.0 | 249 MB |

Both may be used commercially; each file is pinned by size and sha256 (`vision_plugin/install.py`). Only
`glintr100.onnx` is taken from fal's AuraFace repo -- its other files are InsightFace's (non-commercial). Objects:
the API already has the switch, no object model is bundled yet.

Each face comes as `{box, landmarks, score, quality: {sharpness, size}, embedding}`. Vectors of the same person lie
closer together; the catalog row carries starting thresholds (`maybe` 0.30, `likely` 0.45), measured on generated
pictures only -- calibrate them on your own photos. Vectors of different models are never comparable.

**Privacy:** photos and face vectors are biometric data. vision-serve stores and caches nothing; its log carries
counts and times only. Keeping vectors is up to the app that asks.

## Setup

It comes with ai-serve (`plugins/vision-serve`). By hand: `./setup.sh` (ONNX Runtime with CUDA when an NVIDIA GPU is
present, else the CPU build; then the models into `./models` or `VISION_MODELS_DIR`). Without a GPU, or when ai-serve's
GPU manager has no room, it runs on the CPU (slower, never a failure).

Limits per request: 16 images, 40 MB and 120 megapixels each. Scanned black-and-white or sepia prints and 16-bit scans
work; EXIF rotation is applied.

Tests: `.venv/bin/python -m pytest -q` (the fixtures are generated pictures, no real people).
