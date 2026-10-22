"""Qwen3-TTS: variant base (clone: reference + transcript) | voice_design (voice
from `instruct`) | custom_voice (preset `speaker`, optional `instruct`). The worker
holds one checkpoint at a time and swaps on a variant change."""
from tts_serve.domain import RequestError

VARIANTS = ("base", "voice_design", "custom_voice")
# qwen_tts validates the full English name and raises on "de".
LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French",
    "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian", "auto": "Auto",
}


def build_payload(req, ref_audio_path):
    variant = (req.variant or "base").strip().lower()
    if variant not in VARIANTS:
        raise RequestError(f"engine='qwen3' variant must be one of {', '.join(VARIANTS)}")
    language = LANGUAGE_NAMES.get((req.language or "en").strip().lower(), req.language or "English")
    payload = {"text": req.text, "variant": variant, "language": language}
    if variant == "base":
        if not ref_audio_path:
            raise RequestError("engine='qwen3' variant='base' requires reference audio.")
        if not req.ref_text:
            raise RequestError("engine='qwen3' variant='base' requires ref_text (transcript of the reference).")
        payload["ref_audio_path"] = ref_audio_path
        payload["ref_text"] = req.ref_text
        if req.instruct:
            payload["instruct"] = req.instruct  # no-op in qwen-tts 0.1.1; the worker logs it
    elif variant == "voice_design":
        if not req.instruct:
            raise RequestError("engine='qwen3' variant='voice_design' requires instruct (a voice description).")
        payload["instruct"] = req.instruct
    else:
        if not req.speaker:
            raise RequestError("engine='qwen3' variant='custom_voice' requires speaker (a preset timbre name).")
        payload["speaker"] = req.speaker
        if req.instruct:
            payload["instruct"] = req.instruct
    return payload
