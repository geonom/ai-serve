"""Synthetic voices for the cloning TTS engines (tests: tests/unit/test_synthetic_voices.py).

Owner, 2026-10-24: six different voices for every engine where possible. Piper and Qwen3 bring named voices of their
own; Chatterbox, CosyVoice, VoxCPM, OmniVoice and Qwen3 `base` clone a reference clip instead. voices/ holds clips of
real people too (local data, never offered, never published) -- so apps only ever get *synthetic* references: clips
designed from a written description (Qwen3 VoiceDesign, voices/casting/manifest.json), listed in a registry
(SYNTHETIC_VOICES_FILE, default voices/synthetic.json) with one clip + transcript per language.

Rules: ids are plain (`synthetic:<a-z0-9->`); a registry entry's files must lie inside voices/ (realpath), be audio
files that exist and have their .txt transcript; anything else is dropped with a reason. A request names
`synthetic:<id>`; the clip in its language is used, else English, else any.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

PREFIX = "synthetic:"
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_LANG = re.compile(r"^[a-z]{2,3}$")
_AUDIO = (".wav", ".flac", ".mp3", ".ogg")


class SyntheticVoiceError(ValueError):
    pass


@dataclass(frozen=True)
class SyntheticVoice:
    id: str
    label: str
    description: str
    files: Dict[str, str]   # language -> absolute path of the clip (its transcript: same stem, .txt)


def is_synthetic(voice: Optional[str]) -> bool:
    return isinstance(voice, str) and voice.strip().lower().startswith(PREFIX)


def _text(v: Any, limit: int) -> str:
    return v.strip()[:limit] if isinstance(v, str) else ""


def load(voices_dir: str, registry_path: str) -> Tuple[List[SyntheticVoice], List[str]]:
    """The registry's voices whose clips pass the rules, and what was dropped (why)."""
    try:
        with open(registry_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return [], []
    except (OSError, ValueError) as e:
        return [], [f"{registry_path}: {e}"]
    root = os.path.realpath(voices_dir)
    out: List[SyntheticVoice] = []
    errors: List[str] = []
    entries = data.get("voices") if isinstance(data, dict) else None
    for vid, e in (entries.items() if isinstance(entries, dict) else []):
        if not isinstance(vid, str) or not _ID.match(vid) or not isinstance(e, dict):
            errors.append(f"{vid!r}: invalid entry")
            continue
        files: Dict[str, str] = {}
        raw = e.get("files")
        if not isinstance(raw, dict) or not raw:
            errors.append(f"{vid}: no files")
            continue
        for lang, rel in raw.items():
            if not isinstance(lang, str) or not _LANG.match(lang):
                errors.append(f"{vid}: invalid language {lang!r}")
                continue
            path = os.path.realpath(os.path.join(root, rel)) if isinstance(rel, str) else ""
            if not path or os.path.commonpath([root, path]) != root or path == root:
                errors.append(f"{vid}/{lang}: the clip must lie inside voices/")
                continue
            if os.path.splitext(path)[1].lower() not in _AUDIO or not os.path.isfile(path):
                errors.append(f"{vid}/{lang}: the clip doesn't exist (or isn't audio)")
                continue
            if not os.path.isfile(os.path.splitext(path)[0] + ".txt"):
                errors.append(f"{vid}/{lang}: no transcript next to the clip")
                continue
            files[lang] = path
        if files:
            out.append(SyntheticVoice(id=vid, label=_text(e.get("label"), 80) or vid,
                                      description=_text(e.get("description"), 300), files=files))
    return sorted(out, key=lambda v: v.id), errors


def catalog_entries(voices: List[SyntheticVoice]) -> List[Dict[str, Any]]:
    return [{"id": PREFIX + v.id, "label": v.label, "description": v.description, "languages": sorted(v.files),
             "synthetic": True} for v in voices]


def resolve(voices: List[SyntheticVoice], voice: str, language: Optional[str]) -> Tuple[str, str]:
    """`synthetic:<id>` -> (clip path, transcript) in `language` (else English, else any)."""
    if not is_synthetic(voice):
        raise SyntheticVoiceError(f"not a synthetic voice: {voice!r}")
    vid = voice.strip().lower()[len(PREFIX):]
    v = next((x for x in voices if x.id == vid), None) if _ID.match(vid) else None
    if v is None:
        raise SyntheticVoiceError(f"unknown voice {voice!r} -- GET /v1/audio/models lists the synthetic voices")
    lang = (language or "").lower().split("-")[0].split("_")[0]
    path = v.files.get(lang) or v.files.get("en") or next(iter(v.files.values()))
    with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as f:
        return path, f.read().strip()
