#!/usr/bin/env python3
"""Chatterbox-Multilingual on Apple Silicon through MLX (mlx-audio) -- the Mac counterpart of ../chatterbox (PyTorch,
CUDA). Same tiny stdlib HTTP/JSON protocol as every tts-serve worker: GET /health, POST /generate {text, language_id,
audio_prompt_path?, exaggeration?, cfg_weight?, temperature?} -> {audio_base64 (WAV)}. The model
(CHATTERBOX_MLX_MODEL, default mlx-community/chatterbox-multilingual-v3, ~2.5 GB) loads on the first request.
Written 2026-10-21 against mlx-audio 0.6's chatterbox model (generate(text, lang_code, ref_audio, exaggeration,
cfg_weight, temperature) yielding results with .audio; model.sample_rate), checked on Linux CPU; first run on a Mac
is the real test."""
import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("CHATTERBOX_MLX_MODEL", "mlx-community/chatterbox-multilingual-v3")
_lock = threading.Lock()
_state = {"model": None, "sr": 24000, "device": None}


def _load():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    model = load_model(MODEL)
    _state["model"] = model
    _state["sr"] = int(getattr(model, "sample_rate", 24000))
    _state["device"] = f"mlx-{mx.default_device()}"


def _synthesize(req):
    import numpy as np
    kw = {"lang_code": (req.get("language_id") or "en").lower(), "verbose": False}
    if req.get("audio_prompt_path"):
        kw["ref_audio"] = req["audio_prompt_path"]
    for k in ("exaggeration", "cfg_weight", "temperature"):
        if req.get(k) is not None:
            kw[k] = float(req[k])
    parts = [np.array(r.audio) for r in _state["model"].generate(text=req["text"], **kw)]
    if not parts:
        raise RuntimeError("no audio generated")
    return np.concatenate(parts).squeeze(), _state["sr"]


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "variant": None, "device": _state["device"],
                                  "loaded": _state["model"] is not None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return
        if not req.get("text"):
            self._send_json(400, {"error": "missing required parameter 'text'"})
            return
        with _lock:
            try:
                import soundfile as sf
                if _state["model"] is None:
                    _load()
                audio, sr = _synthesize(req)
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
                self._send_json(200, {"status": "success", "device": _state["device"],
                                      "audio_base64": base64.b64encode(buf.getvalue()).decode("utf-8")})
            except Exception as e:
                self._send_json(500, {"error": str(e) or type(e).__name__})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Chatterbox (MLX) worker listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
