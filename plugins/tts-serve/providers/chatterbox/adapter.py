"""Chatterbox-Multilingual: optional reference clip; expressiveness via
exaggeration / cfg_weight / temperature. No free-form instruct channel."""


def build_payload(req, ref_audio_path):
    payload = {"text": req.text, "language_id": (req.language or "en").strip().lower()}
    if ref_audio_path:
        payload["audio_prompt_path"] = ref_audio_path
    for k in ("exaggeration", "cfg_weight", "temperature"):
        if k in req.params:
            payload[k] = req.params[k]
    return payload
