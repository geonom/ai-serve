#!/usr/bin/env python3
"""Standalone Qwen3-TTS worker.

Run under its own isolated venv (see backend/config.py's QWEN3TTS_PYTHON_BIN), NOT
this project's own venv/ -- qwen-tts pins transformers==4.57.3 exactly, which
conflicts with OmniVoice's own transformers>=5.3.0 requirement (confirmed live:
installing qwen-tts into the main venv broke OmniVoice's import outright). Spawned
as a subprocess and talked to over a small stdlib HTTP/JSON protocol on
127.0.0.1, the same shape backend/routers/media.py's generate_qwen3_tts() already
uses for llama-server/ComfyUI, not the in-process import OmniVoice/WhisperX use.

Deliberately no framework dependency (no FastAPI/uvicorn) so the isolated venv's
footprint stays limited to what qwen-tts itself needs, minimizing the surface
for a second version conflict down the line.

Only one Qwen3-TTS checkpoint (Base/VoiceDesign/CustomVoice) is ever loaded at
once -- POST /generate swaps checkpoints on demand when a request's "variant"
differs from whatever is currently loaded, same policy the earlier in-process
design used, just moved into this worker since it's the one holding the model now.
"""
import argparse
import base64
import io
import json
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

_REPOS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}

# Serializes both model swap/load and generation -- one GPU, no benefit to
# concurrent access, and it keeps a variant swap from racing a generate call
# against the checkpoint that was loaded a moment ago.
_lock = threading.Lock()
_state = {"model": None, "variant": None, "device": None}


def _load(variant):
    import torch
    from qwen_tts import Qwen3TTSModel

    device_map = "cuda:0"
    dtype = torch.bfloat16
    if sys.platform == "darwin" or not torch.cuda.is_available():
        device_map = "cpu"
        dtype = torch.float32
    repo_id = _REPOS[variant]
    try:
        model = Qwen3TTSModel.from_pretrained(repo_id, device_map=device_map, dtype=dtype)
    except Exception as cuda_err:
        print(f"Qwen3-TTS CUDA load failed, falling back to CPU: {cuda_err}", file=sys.stderr, flush=True)
        model = Qwen3TTSModel.from_pretrained(repo_id, device_map="cpu", dtype=torch.float32)
        device_map = "cpu"
    _state["model"] = model
    _state["variant"] = variant
    _state["device"] = device_map


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
            self._send_json(200, {"status": "ok", "variant": _state["variant"], "device": _state["device"]})
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

        variant = req.get("variant", "base")
        if variant not in _REPOS:
            self._send_json(400, {"error": f"variant must be one of {sorted(_REPOS)}"})
            return

        with _lock:
            try:
                import soundfile as sf

                if _state["model"] is None or _state["variant"] != variant:
                    _load(variant)

                text = req["text"]
                language = req.get("language", "English")
                if variant == "base":
                    clone_kw = dict(
                        text=text, language=language,
                        ref_audio=req["ref_audio_path"], ref_text=req["ref_text"],
                    )
                    if req.get("instruct"):
                        # qwen-tts 0.1.1's generate_voice_clone() has no instruct
                        # parameter -- it flows through **kwargs into model.generate()
                        # and is dropped (never tokenised to instruct_ids). Forward it
                        # regardless so a future qwen-tts that honours it works with no
                        # code change here; warn so the current no-op is visible.
                        import inspect
                        if "instruct" in inspect.signature(_state["model"].generate_voice_clone).parameters:
                            clone_kw["instruct"] = req["instruct"]
                        else:
                            print(
                                f"WARN: variant='base' got instruct={req['instruct']!r} but this "
                                f"qwen-tts's generate_voice_clone() has no instruct parameter -- ignored.",
                                file=sys.stderr, flush=True,
                            )
                    wavs, sr = _state["model"].generate_voice_clone(**clone_kw)
                elif variant == "voice_design":
                    wavs, sr = _state["model"].generate_voice_design(
                        text=text, language=language, instruct=req["instruct"],
                    )
                else:
                    wavs, sr = _state["model"].generate_custom_voice(
                        text=text, language=language, speaker=req["speaker"], instruct=req.get("instruct"),
                    )

                audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, sr, format="WAV", subtype="PCM_16")
                self._send_json(200, {
                    "status": "success",
                    "audio_base64": base64.b64encode(wav_io.getvalue()).decode("utf-8"),
                    "device": _state["device"],
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = QuickHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Qwen3-TTS worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
