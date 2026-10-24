"""Each person's own TTS voices (2026-10-24, owner via V-Noveller: voices per person belong in ai-serve, not in every
app). A person = the `X-AI-Serve-Person` header an app sends (the door's `sub` of whoever uses it); without it, the
owner -- like every gateway call from this machine. A voice = an uploaded clip (normalised to 24 kHz mono WAV) +
transcript + label, stored in VOICE_LIBRARY_DIR/<sha256(person)[:32]>/<id>.{wav,txt,json}; its id `my:<hex>` only
works for that person. Tests: tests/unit/test_voice_library.py."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional

PREFIX = "my:"
MAX_VOICES = 50
_ID = re.compile(r"^[0-9a-f]{16}$")


class VoiceLibraryError(ValueError):
    pass


def is_own(voice: Optional[str]) -> bool:
    return isinstance(voice, str) and voice.strip().lower().startswith(PREFIX)


def _dir(root: str, person: str) -> str:
    return os.path.join(root, hashlib.sha256(person.encode()).hexdigest()[:32])


def _vid(voice: str) -> str:
    vid = voice.strip().lower()[len(PREFIX):] if is_own(voice) else ""
    if not _ID.match(vid):
        raise VoiceLibraryError(f"unknown voice {voice!r}")
    return vid


def list_voices(root: str, person: str) -> List[Dict[str, Any]]:
    d = _dir(root, person)
    out = []
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith(".json"):
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            out.append({"id": PREFIX + name[:-5], "label": meta.get("label", ""), "language": meta.get("language"),
                        "created": meta.get("created"), "own": True, "synthetic": False})
    return sorted(out, key=lambda v: v["created"] or 0)


def add_voice(root: str, person: str, wav: bytes, label: str, transcript: str, language: Optional[str]) -> Dict[str, Any]:
    label = (label or "").strip()
    if not label or len(label) > 60:
        raise VoiceLibraryError("label: 1-60 characters")
    if len(list_voices(root, person)) >= MAX_VOICES:
        raise VoiceLibraryError(f"at most {MAX_VOICES} voices per person -- delete one first")
    d = _dir(root, person)
    os.makedirs(d, exist_ok=True)
    vid = secrets.token_hex(8)
    with open(os.path.join(d, vid + ".wav"), "wb") as f:
        f.write(wav)
    with open(os.path.join(d, vid + ".txt"), "w", encoding="utf-8") as f:
        f.write((transcript or "").strip())
    meta = {"label": label, "language": language, "created": time.time(), "consent": True}
    with open(os.path.join(d, vid + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return {"id": PREFIX + vid, "label": label, "language": language, "created": meta["created"], "own": True,
            "synthetic": False}


def delete_voice(root: str, person: str, voice: str) -> None:
    d, vid = _dir(root, person), _vid(voice)
    if not os.path.isfile(os.path.join(d, vid + ".json")):
        raise VoiceLibraryError(f"unknown voice {voice!r}")
    for ext in (".wav", ".txt", ".json"):
        try:
            os.remove(os.path.join(d, vid + ext))
        except FileNotFoundError:
            pass


def resolve(root: str, person: str, voice: str) -> tuple:
    """`my:<id>` of this person -> (clip path, transcript); someone else's id is unknown here."""
    d, vid = _dir(root, person), _vid(voice)
    wav = os.path.join(d, vid + ".wav")
    if not os.path.isfile(wav):
        raise VoiceLibraryError(f"unknown voice {voice!r}")
    with open(os.path.join(d, vid + ".txt"), encoding="utf-8") as f:
        return wav, f.read().strip()
