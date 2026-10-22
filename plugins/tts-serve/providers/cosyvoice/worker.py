#!/usr/bin/env python3
"""Standalone CosyVoice 3 worker.

Same design as backend/qwen3tts_server.py: its own isolated venv (see
scripts/setup_cosyvoice.sh / backend/config.py's COSYVOICE_PYTHON_BIN), spawned as a
subprocess and driven over a tiny stdlib HTTP/JSON protocol on 127.0.0.1. No
FastAPI/uvicorn here on purpose, to keep that venv's footprint minimal.

Unlike Qwen3-TTS, CosyVoice isn't a pip package -- it needs its git checkout on
sys.path plus its bundled third_party/Matcha-TTS. The launcher passes the checkout
location as COSYVOICE_REPO_DIR; the model dir defaults to
{repo}/pretrained_models/Fun-CosyVoice3-0.5B unless COSYVOICE_MODEL_DIR is set.

One checkpoint serves all three modes -- zero_shot / instruct / cross_lingual are
just different inference calls on the same loaded model, so there's no
swap-on-demand like the Qwen3-TTS worker has.
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socketserver


class QuickHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the socket.getfqdn() in server_bind: on a Mac whose name resolution hangs it took
    ~35 s before the server accepted connections (found in CI on macOS, 2026-10-22). server_name isn't used here."""
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

_REPO_DIR = os.environ.get("COSYVOICE_REPO_DIR", "")
_MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "") or (
    os.path.join(_REPO_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B") if _REPO_DIR else ""
)

if _REPO_DIR:
    sys.path.insert(0, _REPO_DIR)
    sys.path.insert(0, os.path.join(_REPO_DIR, "third_party", "Matcha-TTS"))

_lock = threading.Lock()
_state = {"model": None, "sr": None, "device": None, "load_wav": None}


def _load():
    import torch

    try:
        from cosyvoice.cli.cosyvoice import AutoModel as _Model
    except Exception:
        # Older checkouts don't have the AutoModel dispatcher -- fall back to the
        # explicit class for the 0.5B v3 checkpoint.
        from cosyvoice.cli.cosyvoice import CosyVoice3 as _Model  # type: ignore
    from cosyvoice.utils.file_utils import load_wav

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fp16 = device.startswith("cuda")
    try:
        model = _Model(model_dir=_MODEL_DIR, fp16=fp16)
    except TypeError:
        model = _Model(_MODEL_DIR)
    _state["model"] = model
    _state["sr"] = int(getattr(model, "sample_rate", 24000))
    _state["device"] = device
    _state["load_wav"] = load_wav


def _synthesize(req):
    import torch

    model = _state["model"]
    variant = req.get("variant", "zero_shot")
    text = req["text"]
    # CosyVoice 3's inference_* take the reference wav PATH -- the frontend loads and
    # resamples it internally (a pre-loaded tensor raises "Invalid file").
    prompt_wav = req["prompt_wav_path"]

    kw = {}
    if req.get("speed") is not None:   # the only per-call knob inference_* take (no temperature)
        kw["speed"] = float(req["speed"])

    # CosyVoice 3 wants an LLM-style prompt with a system line + <|endofprompt|>.
    # zero_shot : prompt_text  = "You are a helpful assistant.<|endofprompt|>{ref transcript}"
    # instruct2 : instruct_text = "You are a helpful assistant. {instruction}<|endofprompt|>"
    _SYS = "You are a helpful assistant."

    def _wrap_zs(ref_text):
        ref_text = ref_text or ""
        return ref_text if "<|endofprompt|>" in ref_text else _SYS + "<|endofprompt|>" + ref_text

    def _wrap_instruct(instr):
        instr = instr or ""
        return instr if "<|endofprompt|>" in instr else _SYS + " " + instr + "<|endofprompt|>"

    def _xlingual():
        return model.inference_cross_lingual(_wrap_zs("") + text, prompt_wav, stream=False, **kw)

    if variant == "instruct":
        gen = model.inference_instruct2(text, _wrap_instruct(req.get("instruct")), prompt_wav, stream=False, **kw)
    elif variant == "cross_lingual":
        gen = _xlingual()
    else:  # zero_shot
        gen = model.inference_zero_shot(text, _wrap_zs(req.get("prompt_text", "")), prompt_wav, stream=False, **kw)

    try:
        chunks = [out["tts_speech"] for out in gen]
    except RuntimeError as e:
        # inference_zero_shot underflows an internal conv (kernel 4) when both the
        # target text and the reference transcript are short. cross_lingual clones
        # from the reference audio alone and has no such minimum -- retry once.
        if variant == "zero_shot" and "Kernel size" in str(e) and "input size" in str(e):
            print(f"zero_shot conv underflow ({e}); retrying via cross_lingual", file=sys.stderr, flush=True)
            chunks = [out["tts_speech"] for out in _xlingual()]
        else:
            raise
    if not chunks:
        raise RuntimeError("CosyVoice produced no audio")
    speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    return speech.squeeze(0).float().cpu().numpy(), _state["sr"]


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

        variant = req.get("variant", "zero_shot")
        if variant not in ("zero_shot", "instruct", "cross_lingual"):
            self._send_json(400, {"error": "variant must be zero_shot | instruct | cross_lingual"})
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
    if not _MODEL_DIR or not os.path.isdir(_MODEL_DIR):
        print(f"CosyVoice model dir not found: {_MODEL_DIR!r} "
              f"(set COSYVOICE_MODEL_DIR or run scripts/setup_cosyvoice.sh)", file=sys.stderr, flush=True)
    server = QuickHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"CosyVoice worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
