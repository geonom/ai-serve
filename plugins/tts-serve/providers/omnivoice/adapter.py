"""OmniVoice: clone from a reference clip (transcript optional -- OmniVoice
transcribes the reference itself when missing), OR, without a reference, a voice
from its instruct-design tags (accent / age / pitch / gender / whisper). Never both."""


def build_payload(req, ref_audio_path):
    payload = {"text": req.text, "language": req.language}
    if ref_audio_path:
        payload["ref_audio_path"] = ref_audio_path
        payload["ref_text"] = req.ref_text or None
    elif req.instruct:
        payload["instruct"] = req.instruct
    return payload
