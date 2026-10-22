#!/usr/bin/env python3
"""Second OmniVoice take -- the SAME 60 tag combos as generate_omnivoice_pool.py,
regenerated. OmniVoice is diffusion/flow-matching (OmniVoiceGenerationConfig:
num_step/guidance_scale/denoise, no do_sample/temperature autoregressive-style
knob), so it samples from fresh noise every call -- identical instruct + text
should still yield a genuinely different voice each time, unlike Qwen's
VoiceDesign which shares the same underlying stochasticity story but was
already tested once (qwen-take2 in the keeper comparison). This take2 batch
tests that empirically at scale: how much *new* diversity comes out of
just re-rolling OmniVoice on the same 60 tag combos, versus how much
duplicates take1 or the Qwen pool.

Output: voices/casting/pool60/omni2-<gender>-<age_key>-<style_key>_en.wav
(+.txt), "omni2-" prefixed so take1/take2/Qwen all cluster together as one
flat set without collisions. Loudness-normalised the same way as the rest
of the casting set.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "casting", "pool60")
BASE_URL = "http://127.0.0.1:8030"

EN_SCRIPT = ("The autumn rain drummed softly against the window as the old clock struck "
             "seven. Somewhere down the street, a dog barked twice, then fell silent. "
             "I wondered, quietly, whether tomorrow would bring anything different.")

GENDERS = ["male", "female"]

AGES = [
    ("child", "child"),
    ("teen", "teenager"),
    ("youngadult", "young adult"),
    ("middleaged", "middle-aged"),
    ("elderly", "elderly"),
]

STYLES = [
    ("verylow", "very low pitch"),
    ("low", "low pitch"),
    ("moderate", "moderate pitch"),
    ("high", "high pitch"),
    ("veryhigh", "very high pitch"),
    ("whisper", "whisper"),
]


def build_pool():
    pool = []
    for gender in GENDERS:
        for age_key, age_tag in AGES:
            for style_key, style_tag in STYLES:
                slug = f"omni2-{gender}-{age_key}-{style_key}"
                instruct = f"{gender}, {age_tag}, {style_tag}"
                pool.append((slug, instruct))
    return pool


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = build_pool()
    manifest = {}
    total = len(pool)
    for i, (slug, instruct) in enumerate(pool, 1):
        wav_path = os.path.join(OUT_DIR, f"{slug}_en.wav")
        txt_path = os.path.join(OUT_DIR, f"{slug}_en.txt")
        manifest[slug] = {"instruct": instruct}
        tag = f"[{i}/{total}] {slug}"
        if os.path.exists(wav_path):
            print(f"{tag}: already exists, skipping")
            continue
        print(f"{tag}: generating...", flush=True)
        try:
            raw = post("/api/tts/generate", {
                "text": EN_SCRIPT, "engine": "omnivoice",
                "instruct": instruct, "language": "en", "response_format": "binary",
            })
            loudnorm(raw, wav_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(EN_SCRIPT)
            print(f"{tag}: OK")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            print(f"{tag}: FAILED HTTP {e.code}: {body[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"{tag}: FAILED: {e}", file=sys.stderr)

    manifest_path = os.path.join(OUT_DIR, "manifest_omni2.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "OmniVoice take2 -- same 60 tag combos as manifest_omni.json, "
                     "regenerated, to test how much new diversity a re-roll adds.",
            "script_en": EN_SCRIPT,
            "voices": manifest,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
