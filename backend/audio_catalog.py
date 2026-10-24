"""GET /v1/audio/models: every speech engine the gateway can drive (owner, 2026-10-20: the Playground should
offer whatever runs through the gateway, open for future models). Pure: tts-serve's /v1/engines and
stt-serve's /v1/engines come in, the client-facing catalog goes out. The engines describe themselves there
(modes/inputs/params resp. form fields); the gateway only adds what it knows itself -- a clone engine with
`default_reference` gets DEFAULT_TTS_VOICE from here, so its clip becomes optional. A service that's down is
reported with a reason, not hidden. Tests: tests/unit/test_audio_catalog.py."""
from typing import Any, Dict, List, Optional


def _down(service: str, url_hint: str) -> Dict[str, Any]:
    return {"available": False, "engines": [], "default_engine": None,
            "reason": f"{service} is unreachable ({url_hint}); start it in Apps & Services."}


def _tts_modes(e: Dict[str, Any], default_voice: str) -> List[Dict[str, Any]]:
    modes = e.get("modes") or []
    if not modes:   # an engine that doesn't describe itself yet: one bare mode per variant
        modes = [{"variant": v, "label": v, "description": "", "inputs": {}, "speakers": [], "params": []}
                 for v in (e.get("variants") or [None])]
        return [dict(m, described=False) for m in modes]
    out = []
    for m in modes:
        m = dict(m, inputs=dict(m.get("inputs") or {}), described=True)
        if e.get("default_reference") and m["inputs"].get("reference") == "required":
            m["inputs"]["reference"] = "optional"
            m["default_voice"] = default_voice
        out.append(m)
    return out


def audio_catalog(tts_view: Optional[Dict[str, Any]], stt_view: Optional[Dict[str, Any]],
                  default_voice: str) -> Dict[str, Any]:
    if tts_view is None:
        tts = _down("tts-serve", "speech output")
    else:
        tts = {"available": True, "default_engine": tts_view.get("default_engine"), "engines": [
            {"id": e["id"], "label": e.get("name") or e["id"], "description": e.get("description", ""),
             "license": e.get("license", ""), "aliases": list(e.get("aliases") or []),
             "languages": list(e.get("languages") or []), "params": list(e.get("params") or []),
             "loaded": bool(e.get("loaded")), "modes": _tts_modes(e, default_voice),
             # not set up on this machine yet: listed so the owner can download it (GET /api/plugins/installs)
             "installed": e.get("installed") is not False}
            for e in tts_view.get("engines") or []]}
    if stt_view is None:
        stt = _down("stt-serve", "transcription")
    else:
        stt = {"available": True, "default_engine": stt_view.get("default_engine"),
               "engines": list(stt_view.get("engines") or [])}
    return {"tts": tts, "stt": stt}
