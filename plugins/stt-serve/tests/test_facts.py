"""Short plain facts for ai-serve-ui's overview (GET /v1/status -> "facts")."""
from stt_serve.facts import status_facts


def _status(**over):
    s = {"whisperx": {"loaded": False, "status": "idle"},
         "diarize": {"loaded": False, "status": "idle"},
         "nemotron": {"loaded": False, "status": "idle"},
         "live_sessions": 0, "busy": False}
    s.update(over)
    return s


def test_nothing_loaded():
    assert status_facts(_status()) == [{"label": "Loaded", "value": "no models"}]


def test_loaded_busy_and_live():
    facts = {f["label"]: f["value"] for f in status_facts(_status(
        whisperx={"loaded": True, "status": "transcribing"},
        diarize={"loaded": True, "status": "idle"},
        live_sessions=2, busy=True))}
    assert facts["Loaded"] == "WhisperX, speaker diarization"
    assert facts["Working"] == "WhisperX: transcribing"
    assert facts["Live sessions"] == "2"


def test_nemotron_listed():
    facts = {f["label"]: f["value"] for f in status_facts(_status(nemotron={"loaded": True, "status": "idle"}))}
    assert facts["Loaded"] == "Nemotron diarization"
