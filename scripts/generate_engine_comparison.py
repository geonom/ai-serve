#!/usr/bin/env python3
"""Comparison batch for the 4 "keeper" casting voices (gruff-older-male,
confident-executive-female, calm-meditation-voice, casual-young-male):

  1. Qwen3-TTS VoiceDesign, SAME instruct as the original casting-set clip,
     regenerated ("take2") -- tests whether identical instruct text gives a
     consistent speaker identity across calls (it shouldn't: do_sample=True,
     no seed pinning in qwen-tts 0.1.1's generate_voice_design()).
  2. OmniVoice, closest-matching closed-vocabulary instruct tags (gender/age/
     pitch/whisper -- OmniVoice has no free-text description, only 23 fixed
     tags) -- an alternative engine's take on roughly the same character.
  3. Chatterbox, generated once (not per-character): it has no voice-design
     capability at all (no tags, no free text) -- with no reference clip it
     always returns its single built-in default identity, only exaggeration/
     cfg_weight/temperature knobs on top. Included so the "fourth engine"
     comparison is complete, not because it can target these characters.

All output loudness-normalised the same way as the original casting set
(-18 LUFS, mono 24kHz) so levels are comparable. Talks to the already-running
gateway (POST /api/tts/generate), not the isolated venvs directly.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "casting", "compare")
BASE_URL = "http://127.0.0.1:8030"

EN_SCRIPT = ("The autumn rain drummed softly against the window as the old clock struck "
             "seven. Somewhere down the street, a dog barked twice, then fell silent. "
             "I wondered, quietly, whether tomorrow would bring anything different.")

KEEPERS = {
    "gruff-older-male": {
        "qwen_instruct": ("A gruff, raspy older male voice, low and rough-edged, speaking "
                           "slowly and deliberately with a slightly menacing undertone."),
        "omni_instruct": "male, elderly, very low pitch",
    },
    "confident-executive-female": {
        "qwen_instruct": ("A confident, articulate middle-aged female voice, clear and "
                           "composed, speaking at a measured, authoritative pace."),
        "omni_instruct": "female, middle-aged, moderate pitch",
    },
    "calm-meditation-voice": {
        "qwen_instruct": ("A calm, breathy, androgynous voice speaking very slowly and "
                           "softly, like a meditation guide, with long, relaxed pauses."),
        "omni_instruct": "female, middle-aged, low pitch, whisper",
    },
    "casual-young-male": {
        "qwen_instruct": ("A casual, friendly young male voice with a relaxed, easygoing "
                           "tone, speaking at a natural conversational pace with a hint of "
                           "playful energy."),
        "omni_instruct": "male, young adult, moderate pitch",
    },
}


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def loudnorm(raw_wav_bytes, out_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_wav_bytes)
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_path,
             "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", out_path],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            raise RuntimeError(f"ffmpeg loudnorm failed: {r.stderr.decode('utf-8', 'ignore')[:300]}")
    finally:
        os.remove(tmp_path)


def gen_and_save(tag, payload, out_path):
    print(f"{tag}: generating...", flush=True)
    try:
        raw = post("/api/tts/generate", payload)
        loudnorm(raw, out_path)
        print(f"{tag}: OK -> {out_path}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        print(f"{tag}: FAILED HTTP {e.code}: {body[:400]}", file=sys.stderr)
    except Exception as e:
        print(f"{tag}: FAILED: {e}", file=sys.stderr)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for slug, cfg in KEEPERS.items():
        # 1. Qwen3 VoiceDesign, same instruct, take2 (consistency probe).
        gen_and_save(
            f"{slug} qwen-take2",
            {"text": EN_SCRIPT, "engine": "qwen3", "variant": "voice_design",
             "instruct": cfg["qwen_instruct"], "language": "English", "response_format": "binary"},
            os.path.join(OUT_DIR, f"{slug}_en_qwen-take2.wav"),
        )
        # 2. OmniVoice, closest closed-vocab tags.
        gen_and_save(
            f"{slug} omnivoice",
            {"text": EN_SCRIPT, "engine": "omnivoice",
             "instruct": cfg["omni_instruct"], "language": "en", "response_format": "binary"},
            os.path.join(OUT_DIR, f"{slug}_en_omnivoice.wav"),
        )

    # 3. Chatterbox default identity, once -- no per-character targeting possible.
    gen_and_save(
        "chatterbox-default",
        {"text": EN_SCRIPT, "engine": "chatterbox", "language": "en", "response_format": "binary"},
        os.path.join(OUT_DIR, "chatterbox-default_en.wav"),
    )

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Engine-comparison batch for the 4 keeper casting voices. "
                     "qwen-take2 = same instruct as the original casting-set clip, "
                     "regenerated, to probe identity consistency. omnivoice = closest "
                     "closed-vocabulary tag equivalent (23 fixed tags, no free text; "
                     "CC-BY-NC, non-commercial only). chatterbox-default = the engine's "
                     "single built-in identity (no voice-design capability at all).",
            "script_en": EN_SCRIPT,
            "keepers": KEEPERS,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
