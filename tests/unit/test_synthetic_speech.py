"""Speaking with a synthetic voice (backend/synthetic_voices.py) through /v1/audio/speech and /api/tts/generate:
`voice: "synthetic:<id>"` -> that voice's clip in the request's language + its transcript go to tts-serve as the
reference. Unknown ids and engines that take no reference are refused (400) -- never a fallback to voices/' other
clips (some are real people)."""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import media

client = TestClient(app, client=("127.0.0.1", 50000))
CLONE = {"inputs": {"reference": "optional", "ref_text": "optional"}, "speakers": []}
ENGINES = [
    {"id": "cosyvoice", "aliases": [], "default_reference": True, "modes": [dict(CLONE, variant="zero_shot")]},
    {"id": "chatterbox", "aliases": [], "default_reference": False, "modes": [dict(CLONE, variant=None)]},
    {"id": "qwen3", "aliases": [], "default_reference": False, "modes": [
        {"variant": "base", "inputs": {"reference": "required", "ref_text": "required"}, "speakers": []},
        {"variant": "custom_voice", "inputs": {"speaker": "required"}, "speakers": ["Vivian"]}]},
    {"id": "piper", "aliases": [], "default_reference": False, "modes": [
        {"variant": "voice", "inputs": {"speaker": "optional"}, "speakers": ["de_DE-thorsten-medium"]}]},
]


class Fake:
    def __init__(self):
        self.generated = []

    async def ensure_running(self):
        return True

    async def engines(self):
        return {"engines": ENGINES, "default_engine": "cosyvoice"}

    async def generate(self, payload):
        self.generated.append(payload)
        return b"RIFF\x00\x00\x00\x00WAVEfake"


@pytest.fixture
def world(monkeypatch, tmp_path):
    from backend import tts_client
    fake = Fake()
    for n in ("ensure_running", "engines", "generate"):
        monkeypatch.setattr(tts_client, n, getattr(fake, n))
    d = tmp_path / "voices"
    (d / "casting").mkdir(parents=True)
    for lang in ("de", "en"):
        (d / "casting" / f"warm-elder-male_{lang}.wav").write_bytes(f"RIFF-{lang}".encode())
        (d / "casting" / f"warm-elder-male_{lang}.txt").write_text(f"Transkript {lang}")
    (d / "synthetic.json").write_text(json.dumps({"voices": {"warm-elder-male": {
        "label": "Warm elder male", "description": "warm", "files": {
            "de": "casting/warm-elder-male_de.wav", "en": "casting/warm-elder-male_en.wav"}}}}))
    monkeypatch.setattr(media, "_VOICES_DIR", str(d))
    monkeypatch.setattr(media, "SYNTHETIC_VOICES_FILE", str(d / "synthetic.json"))
    monkeypatch.setattr(media, "_normalize_ref_audio", lambda p: (p, None))
    return fake


@pytest.mark.parametrize("model,variant", [("cosyvoice", "zero_shot"), ("chatterbox", None), ("qwen3", "base")])
def test_openai_speech_with_a_synthetic_voice(world, model, variant):
    r = client.post("/v1/audio/speech", json={"model": model, "input": "Hallo", "language": "de",
                                             "voice": "synthetic:warm-elder-male"})
    assert r.status_code == 200, r.text
    p = world.generated[-1]
    assert base64.b64decode(p["ref_audio_base64"]) == b"RIFF-de" and p["ref_text"] == "Transkript de"
    assert p.get("variant") == variant and "speaker" not in p and "instruct" not in p


def test_generate_with_a_synthetic_voice_in_another_language(world):
    r = client.post("/api/tts/generate", json={"text": "Hello", "engine": "chatterbox", "language": "fr",
                                               "voice": "synthetic:warm-elder-male"})
    assert r.status_code == 200, r.text
    assert base64.b64decode(world.generated[-1]["ref_audio_base64"]) == b"RIFF-en"   # no fr clip: English


@pytest.mark.parametrize("body,why", [
    ({"model": "cosyvoice", "voice": "synthetic:nobody"}, "unknown"),
    ({"model": "piper", "voice": "synthetic:warm-elder-male"}, "reference"),
])
def test_refused(world, body, why):
    r = client.post("/v1/audio/speech", json={"input": "x", **body})
    assert r.status_code == 400 and why in r.json()["detail"].lower()
    assert world.generated == []


def test_generate_refuses_a_voice_that_is_not_synthetic_and_a_second_reference(world):
    r = client.post("/api/tts/generate", json={"text": "x", "engine": "cosyvoice", "voice": "Real Actor"})
    assert r.status_code == 400
    r = client.post("/api/tts/generate", json={"text": "x", "engine": "cosyvoice", "voice": "synthetic:warm-elder-male",
                                               "ref_voice_path": "casting/warm-elder-male_en.wav"})
    assert r.status_code == 400 and world.generated == []


def test_the_catalog_lists_them(world, monkeypatch):
    from backend import stt_client

    async def down():
        return False
    monkeypatch.setattr(stt_client, "ensure_running", down)
    cat = client.get("/v1/audio/models").json()
    by = {e["id"]: e for e in cat["tts"]["engines"]}
    assert by["cosyvoice"]["modes"][0]["voices"][0]["id"] == "synthetic:warm-elder-male"
    assert "voices" not in by["piper"]["modes"][0]
