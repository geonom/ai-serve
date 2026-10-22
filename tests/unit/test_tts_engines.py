import os
"""The gateway's TTS surface: every engine (OmniVoice included) runs in tts-serve
(~/projects/tts-serve). /api/tts/generate and /v1/audio/speech forward to
it; the gateway keeps voices/ (presets, DEFAULT_TTS_VOICE). Engine-specific
validation lives in tts-serve's adapters and is tested there; tts-serve itself is
faked here."""
import base64

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.routers import media
from backend.routers.media import TTSGenerateRequest

client = TestClient(app, client=("127.0.0.1", 50000))  # this machine (access policy)
_B64 = base64.b64encode(b"RIFF----WAVEfmt ").decode()
WAV = b"RIFF\x00\x00\x00\x00WAVEfake"

ENGINES = [
    {"id": "omnivoice", "aliases": ["omni"], "default_reference": False, "loaded": True,
     "status": "idle", "pid": 12, "device": "cuda:0", "variant": None, "gpu": 0, "name": "OmniVoice"},
    {"id": "cosyvoice", "aliases": ["cosy", "cosyvoice3"], "default_reference": True, "loaded": True,
     "status": "idle", "pid": 11, "device": "cuda:0", "variant": "zero_shot", "gpu": 0, "name": "CosyVoice 3"},
    {"id": "chatterbox", "aliases": ["chatter"], "default_reference": False, "loaded": False,
     "status": "unloaded", "pid": None, "device": None, "variant": None, "gpu": 0, "name": "Chatterbox"},
    {"id": "voxcpm", "aliases": ["voxcpm2"], "default_reference": True, "loaded": False,
     "status": "unloaded", "pid": None, "device": None, "variant": None, "gpu": 0, "name": "VoxCPM2"},
    {"id": "qwen3", "aliases": ["qwen3tts", "qwen3-tts", "qwen"], "default_reference": False, "loaded": False,
     "status": "unloaded", "pid": None, "device": None, "variant": None, "gpu": 0, "name": "Qwen3-TTS"},
]


class FakeTtsServe:
    def __init__(self):
        self.up = True
        self.generated, self.unloaded, self.unload_idle_calls = [], [], []
        self.error = None

    async def ensure_running(self):
        return self.up

    async def engines(self):
        return {"engines": ENGINES, "default_engine": "cosyvoice", "errors": []} if self.up else None

    async def generate(self, payload):
        if self.error:
            raise self.error
        self.generated.append(payload)
        return WAV

    async def unload(self, engine, force=False):
        self.unloaded.append((engine, force))
        return {"id": engine, "status": "unloaded"}

    async def unload_idle(self, gpu=None):
        self.unload_idle_calls.append(gpu)
        return ["cosyvoice"]

    async def logs(self, tail=200):
        return "==> cosyvoice <==\nworker log line"


@pytest.fixture(autouse=True)
def tts(monkeypatch):
    from backend import tts_client
    fake = FakeTtsServe()
    for name in ("ensure_running", "engines", "generate", "unload", "unload_idle", "logs"):
        monkeypatch.setattr(tts_client, name, getattr(fake, name))
    return fake


def _gen(**body):
    body.setdefault("text", "Hallo Welt")
    return client.post("/api/tts/generate", json=body)


# --------------------------------------------------------------------------- forwarding

def test_unknown_engine_rejected(tts):
    r = _gen(engine="bogus")
    assert r.status_code == 400 and "Unknown TTS engine" in r.json()["detail"]
    assert tts.generated == []


def test_cosyvoice_without_reference_gets_the_default_voice(tts):
    r = _gen(engine="cosyvoice", language="de")
    assert r.status_code == 200, r.text
    p = tts.generated[0]
    assert p["engine"] == "cosyvoice" and p["language"] == "de" and p["response_format"] == "wav"
    assert base64.b64decode(p["ref_audio_base64"])[:4] == b"RIFF"
    assert p["ref_text"]  # from the default voice's .txt sidecar
    assert base64.b64decode(r.json()["audio_base64"]) == WAV


def test_alias_resolves_to_engine_id(tts):
    _gen(engine="cosy")
    assert tts.generated[0]["engine"] == "cosyvoice"


def test_engine_without_default_reference_gets_none(tts):
    _gen(engine="chatterbox")
    assert "ref_audio_base64" not in tts.generated[0]


def test_engine_specific_fields_forwarded(tts):
    _gen(engine="chatter", exaggeration=0.7, cfg_weight=0.3, temperature=0.8, instruct="x", variant="v",
         speaker="Vivian")
    p = tts.generated[0]
    assert p["params"] == {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8}
    assert (p["instruct"], p["variant"], p["speaker"]) == ("x", "v", "Vivian")


def test_keep_alive_only_forwarded_when_given(tts):
    _gen(engine="voxcpm")
    _gen(engine="voxcpm", keep_alive=30)
    assert "keep_alive" not in tts.generated[0] and tts.generated[1]["keep_alive"] == 30


def test_explicit_reference_is_sent(tts):
    _gen(engine="cosyvoice", ref_audio_base64=_B64, ref_text="my words")
    p = tts.generated[0]
    assert p["ref_text"] == "my words" and base64.b64decode(p["ref_audio_base64"])[:4] == b"RIFF"


def test_reference_path_outside_voices_rejected(tts):
    r = _gen(engine="cosyvoice", ref_voice_path="/etc/passwd")
    assert r.status_code == 400 and tts.generated == []


def test_binary_response(tts):
    r = _gen(engine="chatterbox", response_format="binary")
    assert r.status_code == 200 and r.content == WAV and r.headers["content-type"] == "audio/wav"


def test_tts_serve_errors_pass_through(tts):
    tts.error = HTTPException(status_code=400, detail="engine='qwen3' variant='base' requires ref_text")
    r = _gen(engine="qwen3")
    assert r.status_code == 400 and "requires ref_text" in r.json()["detail"]


# --------------------------------------------------------------------------- availability / fallback

@pytest.mark.parametrize("engine", [None, "cosyvoice", "omnivoice"])
def test_tts_serve_down_is_503(tts, engine):
    tts.up = False
    r = _gen(**({"engine": engine} if engine else {}))
    assert r.status_code == 503 and "tts-serve" in r.json()["detail"]


def test_omnivoice_forwarded_without_default_voice(tts):
    assert _gen(engine="omni", instruct="female, low pitch").status_code == 200
    p = tts.generated[0]
    assert p["engine"] == "omnivoice" and p["instruct"] == "female, low pitch" and "ref_audio_base64" not in p


def test_request_model_defaults():
    assert TTSGenerateRequest(text="x").engine is None
    # no engine named in the settings -> the TTS plugin's own default (this machine's local env may name one)
    assert config.DEFAULT_TTS_ENGINE == os.environ.get("DEFAULT_TTS_ENGINE", "")


# --------------------------------------------------------------------------- OpenAI shim

@pytest.mark.parametrize("body_extra, expected", [({}, "en"), ({"language": "de"}, "de")])
def test_openai_speech_shim_forwards_language(tts, body_extra, expected):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hallo", "voice": "alloy", **body_extra})
    assert r.status_code == 200 and tts.generated[0]["language"] == expected


def test_openai_speech_shim_maps_models(tts):
    for model, engine in (("voxcpm", "voxcpm"), ("qwen3-tts", "qwen3"), ("chatterbox", "chatterbox")):
        client.post("/v1/audio/speech", json={"model": model, "input": "x", "voice": "alloy"})
        assert tts.generated[-1]["engine"] == engine


def test_openai_speech_shim_preset_voice_sends_transcript(tts):
    client.post("/v1/audio/speech", json={"model": "qwen3-tts", "input": "x", "voice": "synth-default"})
    p = tts.generated[-1]
    assert p["ref_text"] and base64.b64decode(p["ref_audio_base64"])[:4] == b"RIFF"


def test_deprecated_qwen3_endpoint_forwards(tts):
    r = client.post("/api/tts/qwen3/generate", json={"text": "x", "variant": "voice_design", "instruct": "warm"})
    assert r.status_code == 200 and tts.generated[0]["engine"] == "qwen3"
    assert tts.generated[0]["variant"] == "voice_design"


# --------------------------------------------------------------------------- unload / status / logs

def test_unload_endpoints_proxy(tts):
    assert client.post("/api/tts/cosyvoice/unload").json()["status"] == "unloaded"
    assert client.post("/api/tts/qwen3/unload").status_code == 200
    assert client.post("/api/tts/unload").status_code == 200  # legacy OmniVoice path
    assert tts.unloaded == [("cosyvoice", False), ("qwen3", False), ("omnivoice", False)]


def test_status_snapshot_combines_omnivoice_and_tts_serve(tts):
    snap = client.get("/api/tts/status").json()
    assert {"omnivoice", "cosyvoice", "chatterbox", "voxcpm", "qwen3"} <= set(snap["engines"])
    assert snap["default_engine"] == "cosyvoice" and snap["tts_serve"]["available"] is True


def test_status_when_tts_serve_down(tts):
    tts.up = False
    snap = client.get("/api/tts/status").json()
    assert snap["engines"] == {} and snap["status"] == "unloaded" and snap["tts_serve"]["available"] is False


# Loaded engines in /api/status come from the plugin's status.workers like every GPU plugin's
# (tests/unit/test_stt.py::test_dashboard_lists_loaded_stt_models covers that listing).


def test_tts_logs_include_tts_serve(tts):
    assert "worker log line" in client.get("/api/logs/tts").json()["logs"]
