"""VoxCPM2 on Nano-vLLM: variant 'reference' (clone from audio) | 'prompt' (audio +
its transcript, highest fidelity). Language is detected from the text."""
from tts_serve.domain import RequestError


def build_payload(req, ref_audio_path):
    variant = (req.variant or "reference").strip().lower()
    if variant not in ("reference", "prompt"):
        raise RequestError("engine='voxcpm' variant must be reference | prompt")
    if variant == "prompt" and not req.ref_text:
        raise RequestError("engine='voxcpm' variant='prompt' requires ref_text (transcript of the reference).")
    payload = {"text": req.text}
    if ref_audio_path:
        payload["ref_audio_path"] = ref_audio_path
    if variant == "prompt":
        payload["prompt_text"] = req.ref_text
    if "cfg_weight" in req.params:
        payload["cfg_value"] = req.params["cfg_weight"]
    if "temperature" in req.params:
        payload["temperature"] = req.params["temperature"]
    return payload
