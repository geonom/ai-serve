"""OmniVoice TTS worker for tts-serve -- stdlib HTTP on 127.0.0.1 like the other
providers: GET /health, POST /generate {text, language, ref_audio_path?, ref_text?,
instruct?} -> {audio_base64 (24 kHz PCM16 WAV), device}.

Moved out of ai-serve-ui's backend process (it used to run in-process there).
As its own process, unloading = process exit, which returns every byte of VRAM --
the in-process version needed two workarounds for that (an lru_cache on a
transformers tokenizer method keeping the model alive, and torch's cuBLAS
workspace); see tts-serve's CLAUDE.md history."""
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

_lock = threading.Lock()
_state = {"model": None, "device": None}


def _load():
    import torch
    from omnivoice import OmniVoice

    device, dtype = "cuda:0", torch.float16
    if sys.platform == "darwin" or not torch.cuda.is_available():
        device, dtype = "cpu", torch.float32
    try:
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    except Exception as e:  # CUDA OOM etc. -> still serve, slower
        print(f"OmniVoice CUDA load failed, falling back to CPU: {e}", file=sys.stderr, flush=True)
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
        device = "cpu"
    _state["model"], _state["device"] = model, device
    print(f"OmniVoice loaded on {device}", file=sys.stderr, flush=True)


def _synthesize(req):
    kwargs = {"text": req["text"], "language": req.get("language") or "en"}
    if req.get("ref_audio_path"):
        kwargs["ref_audio"] = req["ref_audio_path"]
        kwargs["ref_text"] = req.get("ref_text") or None
    elif req.get("instruct"):
        kwargs["instruct"] = req["instruct"]
    audio = _state["model"].generate(**kwargs)
    return audio[0]


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
            self._send_json(200, {"status": "ok", "device": _state["device"], "loaded": _state["model"] is not None})
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
        if not isinstance(req.get("text"), str) or not req["text"].strip():
            self._send_json(400, {"error": "text is required"})
            return
        with _lock:
            try:
                import soundfile as sf
                if _state["model"] is None:
                    _load()
                audio = _synthesize(req)
                wav_io = io.BytesIO()
                sf.write(wav_io, audio, 24000, format="WAV", subtype="PCM_16")
                self._send_json(200, {"status": "success", "device": _state["device"],
                                      "audio_base64": base64.b64encode(wav_io.getvalue()).decode("utf-8")})
            except Exception as e:
                # OmniVoice rejects instruct tags outside its vocabulary with a ValueError.
                code = 400 if isinstance(e, ValueError) else 500
                self._send_json(code, {"error": str(e) or type(e).__name__})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    print(f"OmniVoice worker listening on 127.0.0.1:{args.port}", file=sys.stderr, flush=True)
    QuickHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
