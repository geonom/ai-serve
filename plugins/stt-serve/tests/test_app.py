"""stt-serve over HTTP with a fake WhisperX model (real ffmpeg decoding): one-shot
transcription, validation, the GPU owner's refusal, live sessions (incl. the
segment-offset regression found live), unload/busy guard, status."""
import base64
import importlib.util
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from stt_serve import app as app_module
from stt_serve import engine

SR = 16000
# Audio is decoded through WhisperX (and ffmpeg) even with the fake model: those tests need whisperx, which setup.sh
# installs (with torch, several GB) -- the CI runs without it and skips them (.github/workflows/tests.yml).
needs_whisperx = pytest.mark.skipif(importlib.util.find_spec("whisperx") is None or importlib.util.find_spec("torch") is None,
                                    reason="needs whisperx + torch (stt-serve's setup.sh)")


def wav_b64(speech_s=1.0, silence_s=0.0):
    t = np.linspace(0, speech_s, int(SR * speech_s), endpoint=False)
    audio = np.concatenate([(0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                            np.zeros(int(SR * silence_s), dtype=np.float32)])
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, batch_size, language, task):
        self.calls.append({"samples": len(audio), "language": language, "task": task})
        return {"segments": [{"start": 0.0, "end": len(audio) / SR, "text": " hallo welt"}],
                "language": language or "de"}


@pytest.fixture
def ctx(monkeypatch):
    fake = FakeWhisper()
    preflights = []

    def preflight(purpose):
        preflights.append(purpose)

    monkeypatch.setattr(engine.state, "whisperx_model", fake)
    monkeypatch.setattr(engine.state, "whisperx_status", "idle")
    monkeypatch.setattr(engine, "preflight_sync", preflight)
    app = app_module.create_app()
    with TestClient(app) as c:
        yield c, fake, preflights
    engine.state.live_stt_sessions.clear()


def test_health(ctx):
    c, *_ = ctx
    assert c.get("/health").json() == {"status": "ok"}


@needs_whisperx
def test_transcribe(ctx):
    c, fake, _ = ctx
    r = c.post("/v1/transcribe", json={"audio_base64": wav_b64(1.0), "language": "de"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["transcript"] == "hallo welt" and d["segments"][0]["text"] == "hallo welt"
    assert d["duration_s"] == pytest.approx(1.0, abs=0.01)
    assert fake.calls[0]["task"] == "transcribe" and fake.calls[0]["language"] == "de"


@needs_whisperx
def test_audio_list_is_concatenated(ctx):
    c, fake, _ = ctx
    r = c.post("/v1/transcribe", json={"audio_base64_list": [wav_b64(1.0), wav_b64(0.5)]})
    assert r.status_code == 200 and fake.calls[0]["samples"] == pytest.approx(int(1.5 * SR), abs=10)


@needs_whisperx
@pytest.mark.parametrize("body, status", [
    ({}, 400),
    ({"audio_path": "/etc/passwd"}, 422),
    ({"audio_base64": "eA==", "diarize": True, "diarize_backend": "magic"}, 400),
    ({"audio_base64": "eA==", "diarize": True, "diarize_backend": "nemotron", "return_embeddings": True}, 400),
])
def test_validation(ctx, body, status):
    c, *_ = ctx
    assert c.post("/v1/transcribe", json=body).status_code == status


@needs_whisperx
def test_gpu_owner_refusal_loads_on_the_cpu(ctx, monkeypatch):
    """The GPU manager says no (no room, or Strata holds the GPUs): WhisperX loads on the CPU instead of failing
    (2026-10-24; it used to answer 507 and the spoken input was lost). Details: tests/test_gpu_robust.py."""
    c, fake, _ = ctx
    monkeypatch.setattr(engine.state, "whisperx_model", None)
    loaded = []

    def refuse(purpose, need_mb):
        raise HTTPException(status_code=507, detail=f"Strata holds the GPUs ({purpose})")

    def load(name, device, device_index, compute_type):
        loaded.append(device)
        return fake
    monkeypatch.setattr(engine, "reserve_sync", refuse)
    monkeypatch.setattr(engine, "_load_model", load)
    monkeypatch.setattr(engine, "_gpu_possible", lambda: True)
    r = c.post("/v1/transcribe", json={"audio_base64": wav_b64(0.5)})
    assert r.status_code == 200 and loaded == ["cpu"] and engine.state.whisperx_device == "cpu"


@needs_whisperx
def test_live_session_offsets(ctx):
    """Regression found live: a finalized segment was stamped with the offset AFTER
    consuming itself (first segment of a 10.27 s clip came back as 10.274-19.427)."""
    c, *_ = ctx
    sid = c.post("/v1/live/start", json={"language": "de", "silence_ms": 400}).json()["session_id"]
    first = c.post("/v1/live/chunk", json={"session_id": sid, "audio_base64": wav_b64(1.0, 1.0)}).json()
    assert first["new_segments"][0]["start"] == 0.0
    assert first["new_segments"][0]["end"] == pytest.approx(1.0, abs=0.05)
    second = c.post("/v1/live/chunk", json={"session_id": sid, "audio_base64": wav_b64(1.0, 1.0)}).json()
    assert second["new_segments"][0]["start"] == pytest.approx(2.0, abs=0.05)
    polled = c.get("/v1/live/poll", params={"session_id": sid, "since": 0}).json()
    assert polled["next_since"] == 2
    assert len(c.get("/v1/live/sessions").json()["sessions"]) == 1
    stopped = c.post("/v1/live/stop", json={"session_id": sid}).json()
    assert stopped["transcript"] == "hallo welt hallo welt"
    assert c.get("/v1/live/sessions").json()["sessions"] == []
    assert c.post("/v1/live/chunk", json={"session_id": sid, "audio_base64": wav_b64(0.2)}).status_code == 404


@needs_whisperx
def test_live_settings(ctx):
    c, *_ = ctx
    sid = c.post("/v1/live/start", json={}).json()["session_id"]
    r = c.patch("/v1/live/settings", json={"session_id": sid, "silence_ms": 300})
    assert r.json()["silence_ms"] == 300


def test_unload_and_busy_guard(ctx, monkeypatch):
    c, *_ = ctx
    monkeypatch.setattr(engine.state, "whisperx_status", "transcribing")
    assert c.post("/v1/unload").status_code == 409
    assert c.post("/v1/unload_all").status_code == 409
    monkeypatch.setattr(engine.state, "whisperx_status", "idle")
    assert c.post("/v1/unload").json()["status"] == "unloaded"
    assert c.post("/v1/unload").json()["status"] == "already unloaded"


def test_status(ctx):
    c, *_ = ctx
    s = c.get("/v1/status").json()
    assert s["whisperx"]["loaded"] is True and s["diarize"]["loaded"] is False
    assert s["nemotron"]["loaded"] is False and s["live_sessions"] == 0
    assert c.get("/v1/diarize/backends").json()["backends"]["whisperx"]["available"] is True


# --- engine "moss" (MOSS-Transcribe-Diarize) with a fake worker and a fake voice embedder ----------
@pytest.fixture
def moss(ctx, monkeypatch):
    c, _, preflights = ctx
    calls = {"worker": [], "embed": []}

    async def fake_ensure(cfg, cpu_only=False):
        return 1

    async def fake_worker(port, wav_path, prompt, max_new_tokens, duration_s):
        calls["worker"].append({"prompt": prompt, "max_new_tokens": max_new_tokens})
        return {"segments": [{"start": 0.0, "end": 6.0, "speaker": "S01", "text": "Hallo"},
                             {"start": 6.0, "end": 10.0, "speaker": "S02", "text": "Servus"}],
                "generated_tokens": 20, "max_new_tokens": max_new_tokens, "device": "cuda:0"}

    def fake_embedder():
        def embed(wave):
            calls["embed"].append(len(wave))
            # first speaker's windows start at 0, second's at 6 s -> tell them apart by length
            return np.array([1.0, 0.0]) if len(wave) == 6 * SR else np.array([0.0, 1.0])
        return embed

    monkeypatch.setattr(engine, "_ensure_worker", fake_ensure)
    monkeypatch.setattr(engine, "_moss_worker_transcribe", fake_worker)
    monkeypatch.setattr(engine, "_speaker_embedder", fake_embedder)
    monkeypatch.setattr(engine, "_reset_moss_keep_alive", lambda s: None)
    return c, calls


@needs_whisperx
def test_moss_diarized_transcript(moss):
    c, calls = moss
    r = c.post("/v1/transcribe", json={"audio_base64": wav_b64(10.0), "engine": "moss", "diarize": True,
                                       "hotwords": ["Liruna"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["engine"] == "moss" and d["transcript"] == "SPEAKER_00: Hallo\nSPEAKER_01: Servus"
    assert "热词提示" in calls["worker"][0]["prompt"] and "Liruna" in calls["worker"][0]["prompt"]
    assert calls["embed"] == [] and "speaker_embeddings" not in d


@needs_whisperx
def test_moss_returns_voice_embeddings(moss):
    c, calls = moss
    d = c.post("/v1/transcribe", json={"audio_base64": wav_b64(10.0), "engine": "moss", "diarize": True,
                                       "return_embeddings": True}).json()
    assert d["speaker_embeddings"] == {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
    assert sorted(calls["embed"]) == [4 * SR, 6 * SR]
    assert "warning" not in d


@needs_whisperx
def test_moss_matches_known_speakers(moss):
    c, _ = moss
    d = c.post("/v1/transcribe", json={"audio_base64": wav_b64(10.0), "engine": "moss", "diarize": True,
                                       "known_speakers": [{"label": "Anna", "embedding": [0.9, 0.1]}]}).json()
    assert d["transcript"].startswith("Anna: Hallo\nSPEAKER_01: Servus")
    assert d["segments"][0]["matched"] is True and d["segments"][1]["matched"] is False
    assert "speaker_embeddings" not in d


@pytest.mark.parametrize("body", [
    {"engine": "moss", "return_embeddings": True},                        # voices without diarize
    {"engine": "moss", "diarize": True, "diarize_backend": "nemotron"},  # MOSS diarizes itself
    {"engine": "whisperx", "hotwords": ["x"]},                           # hotwords are MOSS-only
    {"engine": "moss", "hotwords": ["a[b]"]},
])
def test_moss_refusals(moss, body):
    c, calls = moss
    r = c.post("/v1/transcribe", json={"audio_base64": wav_b64(1.0), **body})
    assert r.status_code == 400, r.text
    assert calls["worker"] == []


def test_unknown_engine_is_refused(ctx):
    c, *_ = ctx
    assert c.post("/v1/transcribe", json={"audio_base64": wav_b64(1.0), "engine": "whisper-large"}).status_code == 422


# --------------------------------------------------------------------------- plugin contract (2026-10-21)
# ai-serve-ui's GPU manager learns from /v1/status.workers which processes hold GPU memory and frees them with
# POST /v1/release (ai-serve-ui docs/plugins.md). WhisperX and the pyannote pipeline live in this process, so they
# are one worker ("stt-serve"); Nemotron and MOSS run as child processes of their own.

def test_status_workers_and_release(ctx, monkeypatch):
    import os
    c, fake, _ = ctx
    w = {x["id"]: x for x in c.get("/v1/status").json()["workers"]}
    assert set(w) == {"stt-serve"} and w["stt-serve"]["pid"] == os.getpid() and w["stt-serve"]["loaded"] is True
    monkeypatch.setattr(engine.state, "whisperx_status", "transcribing")
    assert c.post("/v1/release", json={"worker": "stt-serve"}).status_code == 409
    monkeypatch.setattr(engine.state, "whisperx_status", "idle")
    r = c.post("/v1/release", json={"worker": "stt-serve"})
    assert r.status_code == 200 and r.json()["released"] == ["stt-serve"] and engine.state.whisperx_model is None
    assert c.get("/v1/status").json()["workers"] == []
    assert c.post("/v1/release", json={}).json()["released"] == []
    assert c.post("/v1/release", json={"worker": "nope"}).status_code == 404
    assert c.post("/v1/release", json={"worker": 5}).status_code == 400
