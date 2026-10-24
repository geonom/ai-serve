"""GET /v1/audio/models: every speech engine the gateway can drive, as tts-serve and stt-serve describe
them (owner, 2026-10-20: the Playground should offer whatever runs through the gateway, open for future
models). The gateway adds only what it knows itself: a clone engine with `default_reference` doesn't need
a clip here (the gateway brings DEFAULT_TTS_VOICE), and a service that's down is reported, not hidden."""
import pytest
from fastapi.testclient import TestClient

from backend.audio_catalog import audio_catalog
from backend.main import app

client = TestClient(app, client=("127.0.0.1", 50000))

TTS = {"default_engine": "cosyvoice", "errors": [], "engines": [
    {"id": "cosyvoice", "name": "CosyVoice 3", "description": "d", "license": "Apache-2.0", "aliases": ["cosy"],
     "default_reference": True, "loaded": False, "languages": ["de", "en"], "params": ["temperature"],
     "modes": [{"variant": "zero_shot", "label": "Klonen", "description": "", "speakers": [], "params": [],
                "inputs": {"reference": "required", "ref_text": "optional"}}]},
    {"id": "qwen3", "name": "Qwen3-TTS", "description": "d", "license": "Apache-2.0", "aliases": [],
     "default_reference": False, "loaded": True, "languages": [], "params": [],
     "modes": [{"variant": "base", "label": "Klonen", "description": "", "speakers": [], "params": [],
                "inputs": {"reference": "required", "ref_text": "required"}}]},
    {"id": "old", "name": "Old engine", "description": "", "license": "", "aliases": [],
     "default_reference": False, "loaded": False, "variants": ["x"], "params": ["temperature"]},  # no modes yet
]}
STT = {"default_engine": "whisperx", "engines": [
    {"id": "whisperx", "label": "WhisperX", "description": "d", "available": True, "loaded": False,
     "outputs": ["text"], "fields": [{"name": "diarize", "type": "bool", "label": "Sprecher trennen"}]}]}


def test_both_services():
    cat = audio_catalog(TTS, STT, default_voice="synth-default.wav")
    tts = {e["id"]: e for e in cat["tts"]["engines"]}
    assert cat["tts"]["available"] and cat["tts"]["default_engine"] == "cosyvoice"
    # the gateway brings a voice for cosyvoice, so the clip is optional here -- not for qwen3
    assert tts["cosyvoice"]["modes"][0]["inputs"]["reference"] == "optional"
    assert tts["cosyvoice"]["modes"][0]["default_voice"] == "synth-default.wav"
    assert tts["qwen3"]["modes"][0]["inputs"]["reference"] == "required"
    assert tts["qwen3"]["label"] == "Qwen3-TTS" and tts["qwen3"]["license"] == "Apache-2.0"
    # an engine that doesn't describe itself yet still shows up, with one mode per variant
    assert [m["variant"] for m in tts["old"]["modes"]] == ["x"] and tts["old"]["modes"][0]["described"] is False
    assert cat["stt"]["available"] and cat["stt"]["engines"][0]["fields"][0]["name"] == "diarize"


SYNTH = [{"id": "synthetic:warm-elder-male", "label": "Warm elder male", "description": "d", "languages": ["de"],
          "synthetic": True}]


def test_every_mode_that_takes_a_reference_offers_the_synthetic_voices():
    """2026-10-24 (owner: six voices for every engine where possible): the cloning modes list the synthetic voices;
    a mode without a reference input (Qwen3 presets, Piper) doesn't."""
    tts = dict(TTS, engines=TTS["engines"] + [
        {"id": "piper", "name": "Piper", "default_reference": False, "languages": ["de"], "params": [],
         "modes": [{"variant": "voice", "label": "Voice", "speakers": ["de_DE-thorsten-medium"],
                    "inputs": {"speaker": "optional"}}]}])
    cat = audio_catalog(tts, STT, default_voice="synth-default.wav", synthetic_voices=SYNTH)
    by = {e["id"]: e for e in cat["tts"]["engines"]}
    assert by["cosyvoice"]["modes"][0]["voices"] == SYNTH and by["qwen3"]["modes"][0]["voices"] == SYNTH
    assert "voices" not in by["piper"]["modes"][0] and "voices" not in by["old"]["modes"][0]
    assert "voices" not in audio_catalog(TTS, STT, default_voice="x")["tts"]["engines"][0]["modes"][0]


def test_a_service_that_is_down_is_reported():
    cat = audio_catalog(None, None, default_voice="v.wav")
    assert cat["tts"] == {"available": False, "engines": [], "default_engine": None,
                          "reason": cat["tts"]["reason"]} and "tts-serve" in cat["tts"]["reason"]
    assert cat["stt"]["available"] is False and cat["stt"]["engines"] == []


def test_endpoint(monkeypatch):
    from backend import stt_client, tts_client

    async def up():
        return True

    async def tts_engines():
        return TTS

    async def stt_call(endpoint, method="GET", **kw):
        assert (endpoint, method) == ("catalog", "GET")
        return STT

    monkeypatch.setattr(tts_client, "ensure_running", up)
    monkeypatch.setattr(tts_client, "engines", tts_engines)
    monkeypatch.setattr(stt_client, "ensure_running", up)
    monkeypatch.setattr(stt_client, "call", stt_call)
    r = client.get("/v1/audio/models")
    assert r.status_code == 200, r.text
    assert [e["id"] for e in r.json()["tts"]["engines"]] == ["cosyvoice", "qwen3", "old"]
    assert r.json()["stt"]["engines"][0]["id"] == "whisperx"


def test_endpoint_with_both_down():
    r = client.get("/v1/audio/models")   # conftest: tts-serve and stt-serve unreachable
    assert r.status_code == 200 and r.json()["tts"]["available"] is False and r.json()["stt"]["available"] is False


def test_generate_takes_params_the_engine_declares(monkeypatch):
    from backend.routers.media import TTSGenerateRequest
    req = TTSGenerateRequest(text="x", params={"top_p": 0.9}, temperature=0.5)
    assert req.params == {"top_p": 0.9}
    with pytest.raises(Exception):
        TTSGenerateRequest(text="x", params={"Bad Name": 1})


def test_stt_engine_is_not_a_fixed_list_any_more():
    from backend.routers.media import STTTranscribeRequest
    assert STTTranscribeRequest(engine="future-engine").engine == "future-engine"
    with pytest.raises(Exception):
        STTTranscribeRequest(engine="a b; rm")
