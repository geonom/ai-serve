#!/usr/bin/env python3
"""Generate a synthetic casting set: 12 distinct Qwen3-TTS VoiceDesign voices,
each reading the same original (non-copyrighted) short script in English,
German and French -- 36 WAV files total, each with a .txt sidecar transcript.

Every voice is described in natural language via `instruct` (Qwen3-TTS
VoiceDesign, variant="voice_design") -- none clones a reference clip, so
nothing here is a real person's voice. Output is loudness-normalised the
same way backend/routers/media.py normalises reference clips before
cloning (one-pass ffmpeg loudnorm to -18 LUFS, mono 24kHz), so the set is
usable directly as `base`-variant cloning references later.

Talks to the already-running gateway (POST /api/tts/qwen3/generate) rather
than importing qwen_tts directly, so it doesn't need the isolated
venv-qwen3tts interpreter -- just stdlib (urllib) + a system ffmpeg.

Usage: python3 scripts/generate_casting_voices.py [--base-url http://127.0.0.1:8030]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "casting")

VOICES = [
    ("warm-elder-male", "A warm, gravelly elderly male voice, slow and measured, with a "
     "gentle, grandfatherly warmth and a slight rasp of age."),
    ("bright-young-female", "A bright, youthful female voice, energetic and cheerful, "
     "speaking at a lively, quick pace with an upbeat, friendly tone."),
    ("deep-authoritative-male", "A deep, resonant male voice with strong authority and "
     "gravitas, speaking slowly and deliberately like a documentary narrator."),
    ("soft-elder-female", "A soft, gentle elderly female voice, warm and soothing, "
     "speaking slowly with a calm, motherly tenderness."),
    ("crisp-professional-female", "A crisp, clear female voice with neutral, professional "
     "diction, speaking briskly and confidently like a news presenter."),
    ("casual-young-male", "A casual, friendly young male voice with a relaxed, easygoing "
     "tone, speaking at a natural conversational pace with a hint of playful energy."),
    ("smooth-low-female", "A smooth, velvety female voice with a low, sultry register, "
     "speaking unhurried and richly, almost like a late-night radio host."),
    ("energetic-male-announcer", "An energetic, enthusiastic male voice, fast-paced and "
     "dynamic, like an excited sports commentator building up to a big moment."),
    ("calm-meditation-voice", "A calm, breathy, androgynous voice speaking very slowly and "
     "softly, like a meditation guide, with long, relaxed pauses."),
    ("confident-executive-female", "A confident, articulate middle-aged female voice, "
     "clear and composed, speaking at a measured, authoritative pace."),
    ("gruff-older-male", "A gruff, raspy older male voice, low and rough-edged, speaking "
     "slowly and deliberately with a slightly menacing undertone."),
    ("curious-child-female", "A bright, high-pitched young child's female voice, curious "
     "and giggly, speaking quickly with playful, sing-song intonation."),
]

# Same original (not copyrighted) short script read by every voice in every
# language, so the set doubles as an apples-to-apples casting reel.
SCRIPTS = {
    "en": ("The autumn rain drummed softly against the window as the old clock struck "
           "seven. Somewhere down the street, a dog barked twice, then fell silent. "
           "I wondered, quietly, whether tomorrow would bring anything different."),
    "de": ("Der Herbstregen trommelte leise gegen das Fenster, während die alte Uhr "
           "sieben Mal schlug. Irgendwo die Straße hinunter bellte ein Hund zweimal und "
           "wurde dann still. Ich fragte mich leise, ob der morgige Tag etwas anderes "
           "bringen würde."),
    "fr": ("La pluie d'automne tambourinait doucement contre la fenêtre pendant que la "
           "vieille horloge sonnait sept coups. Quelque part dans la rue, un chien "
           "aboya deux fois, puis se tut. Je me demandai, tout bas, si demain "
           "apporterait quelque chose de différent."),
}

LANG_NAMES = {"en": "English", "de": "German", "fr": "French"}


def generate(base_url, text, instruct, language, timeout=600):
    payload = json.dumps({
        "text": text, "variant": "voice_design", "instruct": instruct,
        "language": language, "response_format": "binary",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/tts/qwen3/generate", data=payload,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8030")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    total = len(VOICES) * len(SCRIPTS)
    done = 0
    for slug, instruct in VOICES:
        manifest[slug] = {"instruct": instruct}
        for lang_code, lang_name in LANG_NAMES.items():
            wav_path = os.path.join(OUT_DIR, f"{slug}_{lang_code}.wav")
            txt_path = os.path.join(OUT_DIR, f"{slug}_{lang_code}.txt")
            done += 1
            tag = f"[{done}/{total}] {slug} ({lang_code})"
            if os.path.exists(wav_path) and os.path.exists(txt_path):
                print(f"{tag}: already exists, skipping")
                continue
            print(f"{tag}: generating...", flush=True)
            try:
                raw = generate(args.base_url, SCRIPTS[lang_code], instruct, lang_name)
                loudnorm(raw, wav_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(SCRIPTS[lang_code])
                print(f"{tag}: OK -> {wav_path}")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")
                print(f"{tag}: FAILED HTTP {e.code}: {body[:300]}", file=sys.stderr)
            except Exception as e:
                print(f"{tag}: FAILED: {e}", file=sys.stderr)

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Synthetic Qwen3-TTS VoiceDesign voices (variant='voice_design', "
                     "instruct-described) -- no reference audio, no real person cloned. "
                     "Free to use for casting; not derived from any copyrighted or "
                     "living-actor recording.",
            "scripts": SCRIPTS,
            "voices": manifest,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
