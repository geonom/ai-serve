#!/usr/bin/env python3
"""Standalone VoxCPM2 worker (Nano-vLLM inference engine).

Same outer shape as backend/chatterbox_server.py / backend/cosyvoice_server.py: its
own isolated venv (scripts/setup_voxcpm.sh / backend/config.py's VOXCPM_PYTHON_BIN),
spawned as a subprocess, stdlib HTTP/JSON on 127.0.0.1 (GET /health, POST /generate).

One deliberate difference: NO global generate lock. `nano-vllm-voxcpm` runs a
continuous-batching scheduler, so concurrent /generate calls are batched on the GPU
instead of queued -- measured on an RTX 3090: RTF ~0.17 for one request, ~0.04
aggregate with 12 in flight. ThreadingHTTPServer gives one thread per request; each
submits its coroutine to the single asyncio loop that owns the async server pool.

Reference audio is encoded to latents once and cached by content hash (the gateway
loudness-normalises the clip into a fresh temp file per request, so paths differ).

Env: VOXCPM_MODEL (default openbmb/VoxCPM2), VOXCPM_GPU (cuda index, default 0),
VOXCPM_GPU_MEM (fraction of that GPU nano-vllm may reserve, default 0.5),
VOXCPM_TIMESTEPS (diffusion steps, fixed at load, default 10).
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import signal
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

_load_lock = threading.Lock()
_latents_lock = threading.Lock()
_inflight_lock = threading.Lock()
_state = {"pool": None, "sr": None, "device": None, "inflight": 0}
_latents: dict = {}            # sha1(ref bytes) -> latents bytes
_loop = asyncio.new_event_loop()


def _run_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop, daemon=True).start()


def _call(coro, timeout=None):
    """Run a coroutine on the pool's loop from a handler thread."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout)


def _load():
    with _load_lock:
        if _state["pool"] is not None:
            return
        gpu = int(os.getenv("VOXCPM_GPU", "0"))

        async def _init():
            from nanovllm_voxcpm import VoxCPM
            # Constructed inside the running loop -> nano-vllm returns its async pool.
            pool = VoxCPM.from_pretrained(
                os.getenv("VOXCPM_MODEL", "openbmb/VoxCPM2"),
                inference_timesteps=int(os.getenv("VOXCPM_TIMESTEPS", "10")),
                gpu_memory_utilization=float(os.getenv("VOXCPM_GPU_MEM", "0.5")),
                devices=[gpu],
            )
            await pool.wait_for_ready()
            info = await pool.get_model_info()
            return pool, int(info["output_sample_rate"])

        pool, sr = _call(_init())
        _state.update(pool=pool, sr=sr, device=f"cuda:{gpu}")


def _ref_latents(path: str) -> bytes:
    data = open(path, "rb").read()
    key = hashlib.sha1(data).hexdigest()
    with _latents_lock:
        hit = _latents.get(key)
    if hit is not None:
        return hit
    lat = _call(_state["pool"].encode_latents(data, os.path.splitext(path)[1].lstrip(".") or "wav"))
    with _latents_lock:
        _latents[key] = lat
    return lat


def _synthesize(req):
    import numpy as np

    pool = _state["pool"]
    kw = {"cfg_value": float(req.get("cfg_value") or 2.0)}
    if req.get("temperature") is not None:
        kw["temperature"] = float(req["temperature"])
    if req.get("max_generate_length"):
        kw["max_generate_length"] = int(req["max_generate_length"])
    if req.get("ref_audio_path"):
        lat = _ref_latents(req["ref_audio_path"])
        kw["ref_audio_latents"] = lat
        # "prompt" variant: additionally continue from the reference + its transcript
        # (VoxCPM2's highest-fidelity "ultimate cloning" mode).
        if req.get("prompt_text"):
            kw["prompt_latents"] = lat
            kw["prompt_text"] = req["prompt_text"]

    async def _gen():
        return [c async for c in pool.generate(req["text"], **kw)]

    chunks = _call(_gen())
    if not chunks:
        raise RuntimeError("VoxCPM2 returned no audio")
    return np.concatenate(chunks), _state["sr"]


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
                "loaded": _state["pool"] is not None,
                "inflight": _state["inflight"],
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

        with _inflight_lock:
            _state["inflight"] += 1
        try:
            import soundfile as sf

            if _state["pool"] is None:
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
        finally:
            with _inflight_lock:
                _state["inflight"] -= 1

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _shutdown(*_):
    # nano-vllm runs its model in child processes -- stop them explicitly, or a
    # terminate() from the gateway would orphan them holding VRAM.
    pool = _state.get("pool")
    if pool is not None:
        try:
            _call(pool.stop(), timeout=30)
        except Exception:
            pass
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server = QuickHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"VoxCPM2 (nano-vllm) worker listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
