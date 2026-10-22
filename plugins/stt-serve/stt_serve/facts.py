"""A few plain facts for ai-serve-ui's overview: its manifest's status_path points at
/v1/status, which carries them as "facts". The dashboard measures VRAM itself."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

_NAMES = (("whisperx", "WhisperX"), ("diarize", "speaker diarization"), ("nemotron", "Nemotron diarization"),
          ("moss", "MOSS-Transcribe"))


def status_facts(status: Mapping[str, Any]) -> List[Dict[str, str]]:
    loaded = [name for key, name in _NAMES if (status.get(key) or {}).get("loaded")]
    facts = [{"label": "Loaded", "value": ", ".join(loaded) if loaded else "no models"}]
    working = [f"{name}: {(status.get(key) or {}).get('status')}" for key, name in _NAMES
               if (status.get(key) or {}).get("loaded") and (status.get(key) or {}).get("status") not in (None, "idle", "ready")]
    if working:
        facts.append({"label": "Working", "value": ", ".join(working)})
    if status.get("live_sessions"):
        facts.append({"label": "Live sessions", "value": str(status["live_sessions"])})
    return facts
