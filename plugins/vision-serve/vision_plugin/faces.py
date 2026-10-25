"""Faces in photos: find them (YuNet: box + 5 landmarks + score), straighten each to the 112x112 ArcFace layout
(similarity transform from the 5 landmarks), and describe it as a 512-number vector (AuraFace). Two vectors of the
same person point the same way (cosine similarity); thresholds in app.py's catalog.

Built for old material too (Tapestry: scanned b/w and sepia prints): every image is turned into 3-channel colour,
EXIF rotation applied first, large scans searched at up to DETECT_MAX_SIDE pixels (small faces in group photos).
Pure image work, no HTTP; runs inside the worker process (worker.py). Tests: tests/test_faces.py."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 120_000_000           # a 600 dpi scan of a large print; more is refused (decompression bombs)
DETECT_MAX_SIDE = 2048
MIN_FACE_PX = 16
SCORE_THRESHOLD = 0.5               # lower than YuNet's 0.9 default: faded prints score low; apps can filter by score
# where the 5 points (eyes, nose tip, mouth corners -- as they appear in the image) sit in a 112x112 ArcFace crop
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    """Image bytes -> BGR uint8 array, EXIF rotation applied, grey/sepia/alpha -> 3 channels."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ImageError(f"image too large ({im.width}x{im.height})")
            im = ImageOps.exif_transpose(im)
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):          # 16-bit grey scans
                arr = np.asarray(im, dtype=np.float32)
                arr = (255 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0)).astype(np.uint8)
                im = Image.fromarray(arr)
            rgb = np.asarray(im.convert("RGB"))
    except ImageError:
        raise
    except (Image.DecompressionBombError, OSError, ValueError, SyntaxError) as e:
        raise ImageError(f"not a readable image ({type(e).__name__})")
    return np.ascontiguousarray(rgb[:, :, ::-1])


@dataclass
class Detection:
    box: List[float]          # x, y, w, h in pixels of the decoded image
    landmarks: np.ndarray     # 5 x 2: eye, eye, nose, mouth corner, mouth corner (left to right in the image)
    score: float


def order_points(lmk: np.ndarray) -> np.ndarray:
    """Eyes and mouth corners left-to-right in the image, so the alignment never mirrors a face."""
    lmk = lmk.astype(np.float32).copy()
    if lmk[0, 0] > lmk[1, 0]:
        lmk[[0, 1]] = lmk[[1, 0]]
    if lmk[3, 0] > lmk[4, 0]:
        lmk[[3, 4]] = lmk[[4, 3]]
    return lmk


def align(img: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(order_points(lmk), ARCFACE_DST, method=cv2.LMEDS)
    if m is None:
        raise ImageError("face could not be aligned")
    return cv2.warpAffine(img, m, (112, 112), borderMode=cv2.BORDER_REPLICATE)


def sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian of the aligned face: < ~50 blurry, > ~150 sharp (same scale for every face)."""
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


class FaceModel:
    def __init__(self, detector_path: str, recognizer_path: str, providers: Sequence[Any]):
        import onnxruntime as ort
        self.detector_path = detector_path
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.rec = ort.InferenceSession(recognizer_path, sess_options=opts, providers=list(providers))
        self.rec_input = self.rec.get_inputs()[0].name
        self.device = self.rec.get_providers()[0]

    def detect(self, img: np.ndarray) -> List[Detection]:
        h, w = img.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else img
        det = cv2.FaceDetectorYN.create(self.detector_path, "", (small.shape[1], small.shape[0]),
                                        SCORE_THRESHOLD, 0.3, 5000)
        _, found = det.detect(small)
        out = []
        for row in found if found is not None else []:
            box = (row[:4] / scale).tolist()
            if min(box[2], box[3]) < MIN_FACE_PX:
                continue
            out.append(Detection(box=box, landmarks=row[4:14].reshape(5, 2) / scale, score=float(row[14])))
        return sorted(out, key=lambda d: (d.box[0], d.box[1]))

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        vecs = self.rec.run(None, {self.rec_input: blob})[0].astype(np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    def analyze(self, img: np.ndarray, embed: bool = True) -> Dict[str, Any]:
        faces, crops = [], []
        for d in self.detect(img):
            crop = align(img, d.landmarks)
            crops.append(crop)
            faces.append({"box": [round(v, 1) for v in d.box],
                          "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in order_points(d.landmarks)],
                          "score": round(d.score, 3),
                          "quality": {"sharpness": round(sharpness(crop), 1), "size": round(min(d.box[2], d.box[3]))}})
        if embed and crops:
            for f, v in zip(faces, self.embed(crops)):
                f["embedding"] = [round(float(x), 6) for x in v]
        return {"width": int(img.shape[1]), "height": int(img.shape[0]), "faces": faces}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))
