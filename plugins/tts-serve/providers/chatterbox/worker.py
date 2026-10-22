#!/usr/bin/env python3
"""Standalone Chatterbox-Multilingual worker.

Same design as backend/qwen3tts_server.py / backend/cosyvoice_server.py: its own
isolated venv (scripts/setup_chatterbox.sh / backend/config.py's
CHATTERBOX_PYTHON_BIN), spawned as a subprocess, driven over a tiny stdlib
HTTP/JSON protocol on 127.0.0.1. No FastAPI/uvicorn here.

`chatterbox-tts` IS a pip package, but its torch/transformers pins aren't trusted
to coexist with OmniVoice/qwen-tts, so it gets the same isolation.

Single model, no variants. `generate()` takes an optional reference clip
(audio_prompt_path) for zero-shot cloning plus expressiveness knobs
(exaggeration / cfg_weight / temperature). Output carries Resemble's Perth
neural watermark -- inherent to the model, not something added here.
"""
import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_state = {"model": None, "sr": None, "device": None}


def _load():
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # CUDA, else Apple's GPU (MPS), else CPU -- on a Mac the chatterbox-mlx provider is usually the better choice
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    except TypeError:
        # Older chatterbox-tts without the t3_model selector.
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    _state["model"] = model
    _state["sr"] = int(getattr(model, "sr", 24000))
    _state["device"] = device


def _synthesize(req):
    model = _state["model"]
    kw = {}
    if req.get("audio_prompt_path"):
        kw["audio_prompt_path"] = req["audio_prompt_path"]
    if req.get("language_id"):
        kw["language_id"] = req["language_id"]
    for k in ("exaggeration", "cfg_weight", "temperature"):
        if req.get(k) is not None:
            kw[k] = float(req[k])

    out = model.generate(req["text"], **kw)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        wav, sr = out
    else:
        wav, sr = out, _state["sr"]

    arr = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else wav
    arr = arr.squeeze()
    return arr, int(sr)


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
            self._send_json(200, {
                "status": "ok",
                "variant": None,
                "device": _state["device"],
                "loaded": _state["model"] is not None,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
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
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
                self._send_json(200, {
                    "status": "success",
                    "audio_base64": base64.b64encode(wav_io.getvalue()).decode("utf-8"),
                    "device": _state["device"],
                })
            except Exception as e:
                self._send_json(500, {"error": str(e) or type(e).__name__})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Chatterbox worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
