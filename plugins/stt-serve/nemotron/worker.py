#!/usr/bin/env python3
"""Standalone nvidia/Nemotron-3-Diarization worker.

Same design as tts-serve's provider workers: its own isolated venv (nemotron/venv,
built by nemotron/setup.sh), spawned by stt-serve as a subprocess and driven over a tiny stdlib
HTTP/JSON protocol on 127.0.0.1. No FastAPI/uvicorn here on purpose, to keep that
venv's footprint minimal -- same reasoning as the other workers, just a harder
version of it here: as of integration time, this model's transformers support
(AutoModelForAudioFrameClassification) only exists on transformers' git main branch,
not in any tagged release, so this venv's transformers install is a moving target
that must never be allowed near stt-serve's own venv.

Unlike the TTS workers, this one does speaker diarization, not audio generation:
input is a path to a local 16kHz-ish audio file (the caller writes audio_data to a
temp WAV and hands over the path -- everything runs on one machine, so passing a
filesystem path is simpler and cheaper than base64-round-tripping potentially
several minutes of PCM), output is a list of {start, end, speaker} segments, not
audio. Model produces per-speaker (up to 8 channels) frame-level activity
probabilities; processor.extract_speaker_dict() (shipped by transformers alongside
the model) turns that into segments -- reimplementing that frame->segment logic
here would just be re-deriving what the model's own processor already does.

Nemotron has no request-time "expect exactly N speakers" knob (confirmed against
the model card: it always evaluates all 8 channels, "speaker counting" is a metric
the model is *evaluated* on, not a parameter a caller can constrain it with) --
num_speakers/min_speakers/max_speakers are approximated here, after the fact, by
ranking the channels extract_speaker_dict() actually returned by total speaking
duration and keeping only the top N. This is a strictly weaker guarantee than
pyannote's own num_speakers (which conditions the clustering step itself), and is
documented as such in CLAUDE.md / docs/API.md -- not hidden as if it were native
support.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MODEL_ID = "nvidia/Nemotron-3-Diarization"

_lock = threading.Lock()
_state = {"model": None, "processor": None, "device": None}


def _load():
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(_MODEL_ID, device_map=device)
    model.eval()
    _state["model"] = model
    _state["processor"] = processor
    _state["device"] = device


def _diarize(audio_path: str, num_speakers, min_speakers, max_speakers):
    import torch
    from transformers.audio_utils import load_audio

    processor = _state["processor"]
    model = _state["model"]
    sampling_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(audio_path, sampling_rate=sampling_rate)
    inputs = processor(audio, sampling_rate=sampling_rate)
    inputs = {k: v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device)
              for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits

    raw_segments = processor.extract_speaker_dict(logits, inputs["attention_mask"])[0]
    # raw_segments: list of {"Speaker": int 0-7, "Start": float seconds, "End": float seconds}
    segments = [
        {"start": float(s["Start"]), "end": float(s["End"]), "speaker": f"SPEAKER_{int(s['Speaker']):02d}"}
        for s in raw_segments
    ]

    # See module docstring: this is an after-the-fact approximation, not a native
    # speaker-count constraint. Rank speakers by total speaking duration and keep
    # only the top N when the caller asked for a specific count/bound.
    target_n = None
    if num_speakers is not None:
        target_n = int(num_speakers)
    elif min_speakers is not None or max_speakers is not None:
        detected_n = len({s["speaker"] for s in segments})
        lo = int(min_speakers) if min_speakers is not None else detected_n
        hi = int(max_speakers) if max_speakers is not None else detected_n
        target_n = max(lo, min(detected_n, hi))

    if target_n is not None:
        totals = {}
        for s in segments:
            totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + (s["end"] - s["start"])
        keep = {spk for spk, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:target_n]}
        segments = [s for s in segments if s["speaker"] in keep]

    segments.sort(key=lambda s: s["start"])
    return segments


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
                "device": _state["device"],
                "loaded": _state["model"] is not None,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/diarize":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
        except Exception as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return

        audio_path = req.get("audio_path")
        if not audio_path:
            self._send_json(400, {"error": "audio_path is required"})
            return

        with _lock:
            try:
                if _state["model"] is None:
                    _load()
                segments = _diarize(
                    audio_path, req.get("num_speakers"), req.get("min_speakers"), req.get("max_speakers"),
                )
                self._send_json(200, {
                    "status": "success",
                    "segments": segments,
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
    print(f"Nemotron-3-Diarization worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
