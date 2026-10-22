"""What stt-serve offers, described for clients (owner, 2026-10-20: the Spielwiese should offer whatever
the gateway can do, open for future models). Pure: availability comes in as arguments.

Each engine lists its options as plain form fields -- `name` is an STTTranscribeRequest field, `type`
one of bool | choice | int | language | list, `show_if` = only relevant while other fields have these
values. A client renders these without knowing the engine; a new engine or option only changes this
file (tests: tests/test_catalog.py)."""
from typing import Any, Dict, Iterable

_LANGUAGE = {"name": "language", "type": "language", "label": "Language",
             "help": "Empty = detect automatically. If you know it, give it: that's more reliable."}


def engine_catalog(*, nemotron_ready: bool, moss_ready: bool, loaded: Iterable[str]) -> Dict[str, Any]:
    loaded = set(loaded)
    whisperx = {
        "id": "whisperx", "label": "WhisperX",
        "description": "Whisper with word timestamps; speaker separation with pyannote or Nemotron.",
        "available": True, "loaded": "whisperx" in loaded,
        "outputs": ["text", "segments", "words", "speakers"],
        "fields": [
            _LANGUAGE,
            {"name": "diarize", "type": "bool", "label": "Separate speakers", "default": False},
            {"name": "diarize_backend", "type": "choice", "label": "Separate speakers with", "default": "whisperx",
             "show_if": {"diarize": True},
             "choices": [{"value": "whisperx", "label": "pyannote (accurate, voices recognisable)", "available": True},
                         {"value": "nemotron", "label": "Nemotron (NVIDIA)", "available": bool(nemotron_ready)}]},
            {"name": "num_speakers", "type": "int", "label": "Number of speakers (if known)", "min": 1, "max": 20,
             "show_if": {"diarize": True}},
        ],
    }
    moss = {
        "id": "moss", "label": "MOSS-Transcribe-Diarize",
        "description": "Text, speakers and times in one pass, up to 90 minutes; no word timestamps."
                       + ("" if moss_ready else " Not set up yet -- it can be downloaded (dashboard or Spielwiese) on a machine with an NVIDIA GPU."),
        "available": bool(moss_ready), "loaded": "moss" in loaded,
        "outputs": ["text", "segments", "speakers"],
        "fields": [
            {"name": "diarize", "type": "bool", "label": "Mark speakers", "default": True},
            {"name": "hotwords", "type": "list", "label": "Spellings (names, technical terms)", "max_items": 50,
             "max_length": 40, "help": "One term per line; the model writes them like that."},
        ],
    }
    return {"engines": [whisperx, moss], "default_engine": "whisperx"}
