"""MOSS-Transcribe-Diarize worker (OpenMOSS, Apache-2.0): transcription + speaker diarization +
timestamps in one pass, up to 90 minutes of audio, 50+ languages. Runs in its own venv
(moss/venv: Python 3.12, transformers 5, the authors' package from moss/repo) like the Nemotron
worker; stt-serve starts it on demand (engine "moss") and talks to it on 127.0.0.1 only.

GET  /health      -> {status, device, loaded}
POST /transcribe  {audio_path, prompt, max_new_tokens} -> {text, segments: [{start, end, speaker, text}], ...}
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
_lock = threading.Lock()
_state = {"model": None, "processor": None, "device": None}


def _load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, trust_remote_code=True, dtype="auto",
                                                 attn_implementation="sdpa").to(dtype=dtype).to(device).eval()
    _state.update(model=model, processor=AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True),
                  device=str(device), dtype=dtype, torch_device=device)


def _transcribe(audio_path: str, prompt: str, max_new_tokens: int) -> dict:
    t0 = time.time()
    messages = build_transcription_messages(audio_path, prompt)
    result = generate_transcription(_state["model"], _state["processor"], messages, max_new_tokens=max_new_tokens,
                                    do_sample=False, device=_state["torch_device"], dtype=_state["dtype"])
    segments = [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
                for s in parse_transcript(result["text"])]
    return {"status": "success", "text": result["text"], "segments": segments,
            "generated_tokens": result["generated_tokens"], "max_new_tokens": max_new_tokens,
            "seconds": round(time.time() - t0, 2), "device": _state["device"]}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        if self.path != "/transcribe":
            return self._send_json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception as e:
            return self._send_json(400, {"error": f"invalid JSON body: {e}"})
        if not req.get("audio_path") or not req.get("prompt"):
            return self._send_json(400, {"error": "audio_path and prompt are required"})
        with _lock:
            try:
                if _state["model"] is None:
                    _load()
                self._send_json(200, _transcribe(req["audio_path"], req["prompt"], int(req.get("max_new_tokens") or 2048)))
            except Exception as e:
                self._send_json(500, {"error": str(e) or type(e).__name__})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MOSS-Transcribe-Diarize worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
