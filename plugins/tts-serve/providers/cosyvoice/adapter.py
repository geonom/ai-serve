"""CosyVoice 3: variant zero_shot | instruct | cross_lingual -- all clone from a
reference clip; 'instruct' adds a free-form style description."""
from tts_serve.domain import RequestError

VARIANTS = ("zero_shot", "instruct", "cross_lingual")


def build_payload(req, ref_audio_path):
    variant = (req.variant or "zero_shot").strip().lower()
    if variant not in VARIANTS:
        raise RequestError("engine='cosyvoice' variant must be zero_shot | instruct | cross_lingual")
    if not ref_audio_path:
        raise RequestError("engine='cosyvoice' needs a reference clip (ref_audio_base64).")
    # inference_zero_shot crashes ("Calculated padded input size per channel: (3).
    # Kernel size: (4)") when BOTH the target text and the reference transcript are
    # short; cross_lingual clones from the audio alone and has no such minimum.
    if variant == "zero_shot" and (not req.ref_text or len(req.ref_text.split()) < 4):
        variant = "cross_lingual"
    payload = {"variant": variant, "text": req.text, "prompt_wav_path": ref_audio_path, "language": req.language}
    if variant == "zero_shot":
        payload["prompt_text"] = req.ref_text
    elif variant == "instruct":
        if not req.instruct:
            raise RequestError("engine='cosyvoice' variant='instruct' requires instruct "
                               "(a natural-language style description).")
        payload["instruct"] = req.instruct
    if "speed" in req.params:   # inference_* take speed; there is no temperature knob
        payload["speed"] = req.params["speed"]
    return payload
