"""The process that holds the face models (spawned by app.py). Its own process so that `release` gives every byte of
GPU memory back (ending it ends its CUDA context) and so that ai-serve's GPU manager can attribute and measure it by
pid. It sees only the GPU it was given (CUDA_VISIBLE_DEVICES, set before onnxruntime loads); gpu None = CPU.
Requests and answers go over a multiprocessing pipe; nothing is written to disk or logged about the images."""
from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional


def _providers(gpu: Optional[int]) -> List[Any]:
    import onnxruntime as ort
    if gpu is None or "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    if hasattr(ort, "preload_dlls"):       # CUDA/cuDNN from the nvidia-* wheels (onnxruntime-gpu[cuda,cudnn])
        try:
            ort.preload_dlls()
        except Exception:
            pass
    return [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT",
                                       "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]


def is_oom(msg: str) -> bool:
    m = msg.lower()
    return "out of memory" in m or "failed to allocate" in m or "cudnn_status_alloc_failed" in m


def main(conn, gpu: Optional[int], detector: str, recognizer: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)     # its per-layer notes would fill the log; errors still come back here
        from vision_plugin.faces import FaceModel, ImageError, decode
        model = FaceModel(detector, recognizer, _providers(gpu))
        conn.send({"ready": True, "device": model.device})
    except Exception as e:
        conn.send({"ready": False, "error": f"{type(e).__name__}: {e}"[:500], "oom": is_oom(str(e))})
        return
    while True:
        try:
            req: Dict[str, Any] = conn.recv()
        except (EOFError, OSError):
            return
        t0 = time.monotonic()
        try:
            out = []
            for data in req["images"]:
                try:
                    out.append(model.analyze(decode(data), embed=req.get("embed", True)))
                except ImageError as e:
                    out.append({"error": str(e), "faces": []})
            conn.send({"ok": out, "seconds": round(time.monotonic() - t0, 3)})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            conn.send({"error": msg[:500], "oom": is_oom(msg), "trace": traceback.format_exc()[-2000:]})
