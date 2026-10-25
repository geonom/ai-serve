"""Model files for vision-serve, from Hugging Face, each checked against a pinned sha256 (a changed upstream file is
refused, never used). `python -m vision_plugin.install [model ...]` (default: every model).

- yunet: OpenCV Zoo's YuNet face detector (2023mar, MIT) -- boxes + 5 landmarks.
- auraface: fal's AuraFace v1 recognizer (ResNet100/ArcFace loss, Apache-2.0, trained on commercially usable data)
  -- only its own `glintr100.onnx`; the other files in that repo (scrfd, genderage, 2d106det, 1k3d68) are
  InsightFace's and not taken."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("VISION_MODELS_DIR") or HERE / "models")

FILES: Dict[str, Dict[str, object]] = {
    "yunet": {"repo": "opencv/opencv_zoo", "path": "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
              "file": "face_detection_yunet_2023mar.onnx", "size": 232589,
              "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"},
    "auraface": {"repo": "fal/AuraFace-v1", "path": "glintr100.onnx", "file": "auraface-v1-glintr100.onnx",
                 "size": 260694151, "sha256": "a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"},
}


def path_of(name: str) -> Path:
    return MODELS_DIR / str(FILES[name]["file"])


def present(name: str) -> bool:
    p = path_of(name)
    return p.is_file() and p.stat().st_size == FILES[name]["size"]


def fetch(name: str, progress: Optional[Callable[[int], None]] = None) -> Path:
    spec = FILES[name]
    target = path_of(name)
    if present(name):
        return target
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{spec['repo']}/resolve/main/{spec['path']}"
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=MODELS_DIR, prefix=".part-")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as r:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
                if progress:
                    progress(len(chunk))
        if h.hexdigest() != spec["sha256"]:
            raise SystemExit(f"{name}: sha256 mismatch ({h.hexdigest()}) -- not used")
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
        return target
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(FILES):
        print(n, fetch(n))
