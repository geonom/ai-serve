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
    assert status_facts(_status()) == [{"label": "Geladen", "value": "keine Modelle"}]


def test_loaded_busy_and_live():
    facts = {f["label"]: f["value"] for f in status_facts(_status(
        whisperx={"loaded": True, "status": "transcribing"},
        diarize={"loaded": True, "status": "idle"},
        live_sessions=2, busy=True))}
    assert facts["Geladen"] == "WhisperX, Sprechererkennung"
    assert facts["Arbeitet"] == "WhisperX: transcribing"
    assert facts["Live-Sitzungen"] == "2"


def test_nemotron_listed():
    facts = {f["label"]: f["value"] for f in status_facts(_status(nemotron={"loaded": True, "status": "idle"}))}
    assert facts["Geladen"] == "Nemotron-Sprechererkennung"
