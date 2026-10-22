"""Per-provider adapters (providers/<id>/adapter.py): the common TtsRequest -> the
worker's own payload. Behaviour carried over 1:1 from ai-serve-ui's former
generate_<engine>_tts() functions, so moving the engines out changes nothing for
callers."""
import base64

import pytest

from tts_serve.domain import RequestError, TtsRequest
from tts_serve.providers import PROVIDERS_DIR, load_adapter

REF = "/tmp/tts-serve-ref.wav"
WAV = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()


def _req(**kw):
    kw.setdefault("text", "Hallo Welt")
    return TtsRequest.from_dict(kw)


def _build(engine, ref=REF, **kw):
    return load_adapter(PROVIDERS_DIR / engine).build_payload(_req(**kw), ref)


# --------------------------------------------------------------------------- cosyvoice

class TestCosyVoice:
    LONG = "this is a long enough reference transcript"

    def test_zero_shot_with_transcript(self):
        p = _build("cosyvoice", ref_text=self.LONG, language="de")
        assert p == {"variant": "zero_shot", "text": "Hallo Welt", "prompt_wav_path": REF,
                     "language": "de", "prompt_text": self.LONG}

    @pytest.mark.parametrize("ref_text", [None, "too short"])
    def test_zero_shot_without_usable_transcript_falls_back_to_cross_lingual(self, ref_text):
        p = _build("cosyvoice", ref_text=ref_text)
        assert p["variant"] == "cross_lingual" and "prompt_text" not in p

    def test_instruct(self):
        # CosyVoice's inference_* take no temperature (found live 2026-10-20: TypeError); speed is what it has
        p = _build("cosyvoice", variant="instruct", instruct="langsam und ruhig", params={"speed": 0.8})
        assert p["variant"] == "instruct" and p["instruct"] == "langsam und ruhig" and p["speed"] == 0.8
        assert "temperature" not in _build("cosyvoice", ref_text=self.LONG, params={"temperature": 0.8})

    @pytest.mark.parametrize("kw, needle", [
        ({"variant": "nope"}, "variant must be"),
        ({"variant": "instruct"}, "requires instruct"),
    ])
    def test_validation(self, kw, needle):
        with pytest.raises(RequestError) as e:
            _build("cosyvoice", **kw)
        assert needle in e.value.message

    def test_reference_required(self):
        with pytest.raises(RequestError):
            _build("cosyvoice", ref=None)


# --------------------------------------------------------------------------- chatterbox

class TestChatterbox:
    def test_payload(self):
        p = _build("chatterbox", language="DE", params={"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8})
        assert p == {"text": "Hallo Welt", "language_id": "de", "audio_prompt_path": REF,
                     "exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 0.8}

    def test_reference_optional_and_instruct_ignored(self):
        p = _build("chatterbox", ref=None, instruct="whisper")
        assert p == {"text": "Hallo Welt", "language_id": "en"}


# --------------------------------------------------------------------------- voxcpm

class TestVoxCPM:
    def test_reference_variant(self):
        p = _build("voxcpm", params={"cfg_weight": 2.5, "temperature": 0.9})
        assert p == {"text": "Hallo Welt", "ref_audio_path": REF, "cfg_value": 2.5, "temperature": 0.9}

    def test_prompt_variant_needs_transcript(self):
        assert _build("voxcpm", variant="prompt", ref_text="ref words")["prompt_text"] == "ref words"
        with pytest.raises(RequestError):
            _build("voxcpm", variant="prompt")

    def test_unknown_variant(self):
        with pytest.raises(RequestError):
            _build("voxcpm", variant="nope")


# --------------------------------------------------------------------------- qwen3

class TestQwen3:
    def test_base_clone(self):
        p = _build("qwen3", ref_text="the transcript", language="de")
        assert p == {"text": "Hallo Welt", "variant": "base", "language": "German",
                     "ref_audio_path": REF, "ref_text": "the transcript"}

    @pytest.mark.parametrize("code, name", [("en", "English"), ("zh", "Chinese"), ("auto", "Auto"),
                                            ("pt", "Portuguese")])
    def test_language_names(self, code, name):
        assert _build("qwen3", variant="voice_design", instruct="warm", language=code, ref=None)["language"] == name

    def test_voice_design_and_custom_voice(self):
        assert _build("qwen3", variant="voice_design", instruct="a warm voice", ref=None)["instruct"] == "a warm voice"
        p = _build("qwen3", variant="custom_voice", speaker="Vivian", instruct="happy", ref=None)
        assert p["speaker"] == "Vivian" and p["instruct"] == "happy"

    @pytest.mark.parametrize("kw", [
        {"variant": "base", "ref": None, "ref_text": "x"},
        {"variant": "base"},  # no ref_text
        {"variant": "voice_design", "ref": None},
        {"variant": "custom_voice", "ref": None},
        {"variant": "nope", "ref": None},
    ])
    def test_validation(self, kw):
        with pytest.raises(RequestError):
            _build("qwen3", **kw)


# --------------------------------------------------------------------------- omnivoice

class TestOmniVoice:
    def test_clone_with_optional_transcript(self):
        p = _build("omnivoice", ref_text="ref words", language="de")
        assert p == {"text": "Hallo Welt", "language": "de", "ref_audio_path": REF, "ref_text": "ref words"}
        assert _build("omnivoice")["ref_text"] is None  # OmniVoice transcribes the reference itself

    def test_instruct_only_without_reference(self):
        assert _build("omnivoice", ref=None, instruct="female, low pitch") == {
            "text": "Hallo Welt", "language": "en", "instruct": "female, low pitch"}
        # clone wins -- OmniVoice never combines a reference with design tags
        assert "instruct" not in _build("omnivoice", instruct="whisper")

    def test_plain(self):
        assert _build("omnivoice", ref=None) == {"text": "Hallo Welt", "language": "en"}
