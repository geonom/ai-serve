"""A few plain facts for ai-serve-ui's overview: its manifest's status_path points at
/v1/status, which carries them as "facts". The dashboard measures VRAM itself."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

_NAMES = (("whisperx", "WhisperX"), ("diarize", "Sprechererkennung"), ("nemotron", "Nemotron-Sprechererkennung"),
          ("moss", "MOSS-Transcribe"))


def status_facts(status: Mapping[str, Any]) -> List[Dict[str, str]]:
    loaded = [name for key, name in _NAMES if (status.get(key) or {}).get("loaded")]
    facts = [{"label": "Geladen", "value": ", ".join(loaded) if loaded else "keine Modelle"}]
    working = [f"{name}: {(status.get(key) or {}).get('status')}" for key, name in _NAMES
               if (status.get(key) or {}).get("loaded") and (status.get(key) or {}).get("status") not in (None, "idle", "ready")]
    if working:
        facts.append({"label": "Arbeitet", "value": ", ".join(working)})
    if status.get("live_sessions"):
        facts.append({"label": "Live-Sitzungen", "value": str(status["live_sessions"])})
    return facts
