#!/usr/bin/env python3
"""Generate a larger pool of 60 systematically-varied Qwen3-TTS VoiceDesign
voices (EN only -- this is a discovery/dedup pass, not the final casting set)
so similarity analysis has enough material to find genuinely distinct voices
and flag near-duplicates.

60 = 2 genders x 5 ages x 6 style/pace combos, cartesian product, so the pool
has known, even coverage rather than being hand-picked -- lets the later
similarity clustering be checked against expectation (e.g. do the 2 clips
that share gender+age+style-but-nothing-else end up close in embedding
space, the way two random samples of the same instruct did in the small
keeper-comparison batch?).

Output: voices/casting/pool60/<slug>_en.wav + .txt, loudness-normalised the
same way as the rest of the casting set. manifest.json records the instruct
used per slug for reproducibility.
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
    ("child", "a young child's"),
    ("teen", "a teenage"),
    ("youngadult", "a young adult"),
    ("middleaged", "a middle-aged"),
    ("elderly", "an elderly"),
]

STYLES = [
    ("warm", "warm and gentle", "speaking slowly and softly"),
    ("bright", "bright and energetic", "speaking quickly with lively enthusiasm"),
    ("gravelly", "deep and gravelly", "speaking slowly and deliberately"),
    ("smooth", "smooth and velvety", "speaking in an unhurried, relaxed manner"),
    ("crisp", "crisp and clear", "speaking briskly and confidently"),
    ("raspy", "raspy and husky", "speaking with a rough, worn edge to every word"),
]


def build_pool():
    pool = []
    for gender in GENDERS:
        for age_key, age_phrase in AGES:
            for style_key, style_adj, pace in STYLES:
                slug = f"{gender}-{age_key}-{style_key}"
                instruct = f"{age_phrase.capitalize()} {gender} voice, {style_adj}, {pace}."
                pool.append((slug, instruct))
    return pool


def generate(text, instruct, language="English", timeout=600):
    payload = json.dumps({
        "text": text, "variant": "voice_design", "instruct": instruct,
        "language": language, "response_format": "binary",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/tts/qwen3/generate", data=payload,
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
            raw = generate(EN_SCRIPT, instruct)
            loudnorm(raw, wav_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(EN_SCRIPT)
            print(f"{tag}: OK")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            print(f"{tag}: FAILED HTTP {e.code}: {body[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"{tag}: FAILED: {e}", file=sys.stderr)

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "60-voice discovery pool (Qwen3-TTS VoiceDesign, EN only), systematic "
                     "2 gender x 5 age x 6 style cartesian product -- for similarity "
                     "clustering to find distinct voices / flag near-duplicates.",
            "script_en": EN_SCRIPT,
            "voices": manifest,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
