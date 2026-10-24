"""Each person's own voices (backend/voice_library.py, /v1/audio/voices): upload, list, delete, speak with
voice=my:<id> -- only the person who uploaded a voice sees and uses it."""
import base64
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import voice_library as vl
from backend.main import app
from backend.routers import media

client = TestClient(app, client=("127.0.0.1", 50000))


def clip(seconds=4.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\x10\x00\xf0\xff" * int(8000 * seconds))
    return buf.getvalue()


class FakeTts:
    def __init__(self):
        self.generated = []

    async def ensure_running(self):
        return True

    async def engines(self):
        return {"default_engine": "chatterbox", "engines": [
            {"id": "chatterbox", "aliases": [], "default_reference": False,
             "modes": [{"variant": None, "inputs": {"reference": "optional"}, "speakers": []}]}]}

    async def generate(self, payload):
        self.generated.append(payload)
        return b"RIFF\x00\x00\x00\x00WAVEfake"


@pytest.fixture
def world(monkeypatch, tmp_path):
    from backend import tts_client
    fake = FakeTts()
    for n in ("ensure_running", "engines", "generate"):
        monkeypatch.setattr(tts_client, n, getattr(fake, n))
    monkeypatch.setattr(media, "VOICE_LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setattr(media, "_normalize_ref_audio", lambda p: (p, None))

    async def transcribe(wav_bytes, language):
        return "automatisch abgeschrieben"
    monkeypatch.setattr(media, "_transcribe_clip", transcribe)
    return fake


def upload(person=None, **data):
    headers = {"X-AI-Serve-Person": person} if person else {}
    body = {"label": "Meine Stimme", "consent": "true", **data}
    return client.post("/v1/audio/voices", files={"file": ("me.wav", clip(), "audio/wav")}, data=body, headers=headers)


def test_upload_list_speak_delete(world):
    r = upload("anna", transcript="Hallo, das bin ich.")
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    assert vid.startswith("my:")
    listed = client.get("/v1/audio/voices", headers={"X-AI-Serve-Person": "anna"}).json()["voices"]
    assert [v["id"] for v in listed if v["own"]] == [vid]
    r = client.post("/v1/audio/speech", json={"model": "chatterbox", "input": "Hi", "voice": vid, "language": "de"},
                    headers={"X-AI-Serve-Person": "anna"})
    assert r.status_code == 200, r.text
    assert world.generated[-1]["ref_text"] == "Hallo, das bin ich." and world.generated[-1]["ref_audio_base64"]
    assert client.delete(f"/v1/audio/voices/{vid}", headers={"X-AI-Serve-Person": "anna"}).status_code == 204
    assert client.get("/v1/audio/voices", headers={"X-AI-Serve-Person": "anna"}).json()["voices"] == [] or \
        all(not v["own"] for v in client.get("/v1/audio/voices", headers={"X-AI-Serve-Person": "anna"}).json()["voices"])


def test_a_voice_is_only_its_persons(world):
    vid = upload("anna").json()["id"]
    bob = {"X-AI-Serve-Person": "bob"}
    assert not [v for v in client.get("/v1/audio/voices", headers=bob).json()["voices"] if v["own"]]
    r = client.post("/v1/audio/speech", json={"model": "chatterbox", "input": "Hi", "voice": vid}, headers=bob)
    assert r.status_code == 400 and world.generated == []
    assert client.delete(f"/v1/audio/voices/{vid}", headers=bob).status_code == 404
    assert not [v for v in client.get("/v1/audio/voices").json()["voices"] if v["own"]]   # nor the owner's


def test_missing_transcript_is_written_by_whisper(world):
    vid = upload("anna").json()["id"]
    client.post("/v1/audio/speech", json={"model": "chatterbox", "input": "Hi", "voice": vid},
                headers={"X-AI-Serve-Person": "anna"})
    assert world.generated[-1]["ref_text"] == "automatisch abgeschrieben"


@pytest.mark.parametrize("data,files,why", [
    ({"consent": "false"}, None, "consent"),
    ({"label": ""}, None, "label"),
    ({}, {"file": ("x.wav", b"not audio", "audio/wav")}, "audio"),
    ({}, {"file": ("x.wav", clip(1.0), "audio/wav")}, "3"),
])
def test_uploads_are_checked(world, data, files, why):
    body = {"label": "Meine Stimme", "consent": "true", **data}
    r = client.post("/v1/audio/voices", files=files or {"file": ("me.wav", clip(), "audio/wav")}, data=body)
    assert r.status_code == 400 and why in r.json()["detail"].lower()
