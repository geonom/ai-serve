"""Synthetic voices (backend/synthetic_voices.py, 2026-10-24; owner: six different voices for every engine where
possible): reference clips that were designed from a description (Qwen3 VoiceDesign), never recorded from a person,
offered to every engine that clones a reference. voices/ also holds clips of real people -- only voices listed in the
registry, with their clips inside voices/ and a transcript next to them, are ever offered as `synthetic:<id>`."""
import json

import pytest

from backend import synthetic_voices as sv


@pytest.fixture
def voices(tmp_path):
    d = tmp_path / "voices"
    (d / "casting").mkdir(parents=True)
    for vid in ("warm-elder-male", "bright-young-female"):
        for lang in ("de", "en"):
            (d / "casting" / f"{vid}_{lang}.wav").write_bytes(b"RIFF")
            (d / "casting" / f"{vid}_{lang}.txt").write_text(f"Text {vid} {lang}")
    (d / "Real Actor - Short.wav").write_bytes(b"RIFF")
    (d / "Real Actor - Short.txt").write_text("a real person")
    reg = {"voices": {
        "warm-elder-male": {"label": "Warm elder male", "description": "A warm, gravelly elderly male voice",
                            "files": {"de": "casting/warm-elder-male_de.wav", "en": "casting/warm-elder-male_en.wav"}},
        "bright-young-female": {"label": "Bright young female", "description": "A bright, youthful female voice",
                                "files": {"de": "casting/bright-young-female_de.wav",
                                          "en": "casting/bright-young-female_en.wav"}}}}
    (d / "synthetic.json").write_text(json.dumps(reg))
    return d


def test_the_registry_is_loaded_and_offered(voices):
    got, errors = sv.load(str(voices), str(voices / "synthetic.json"))
    assert errors == [] and [v.id for v in got] == ["bright-young-female", "warm-elder-male"]
    entries = sv.catalog_entries(got)
    assert entries[1] == {"id": "synthetic:warm-elder-male", "label": "Warm elder male",
                          "description": "A warm, gravelly elderly male voice", "languages": ["de", "en"],
                          "synthetic": True}


def test_the_clip_follows_the_language(voices):
    got, _ = sv.load(str(voices), str(voices / "synthetic.json"))
    wav, text = sv.resolve(got, "synthetic:warm-elder-male", "de")
    assert wav.endswith("warm-elder-male_de.wav") and text == "Text warm-elder-male de"
    assert sv.resolve(got, "synthetic:warm-elder-male", "fr")[0].endswith("_en.wav")   # else English
    assert sv.resolve(got, "SYNTHETIC:Warm-Elder-Male", "de-AT")[0].endswith("_de.wav")


def test_unknown_or_malformed_ids_are_refused(voices):
    got, _ = sv.load(str(voices), str(voices / "synthetic.json"))
    for bad in ("synthetic:nobody", "synthetic:../Real Actor - Short", "synthetic:", "Real Actor - Short",
                "synthetic:" + "x" * 60):
        with pytest.raises(sv.SyntheticVoiceError):
            sv.resolve(got, bad, "de")
    assert sv.is_synthetic("synthetic:warm-elder-male") and not sv.is_synthetic("warm-elder-male")


@pytest.mark.parametrize("files,why", [
    ({"de": "../outside.wav"}, "inside"),
    ({"de": "/etc/passwd"}, "inside"),
    ({"de": "Real Actor - Short.mp3"}, "exist"),
    ({"de": "casting/missing_de.wav"}, "exist"),
    ({"xx-evil": "casting/warm-elder-male_de.wav"}, "language"),
    ({}, "files"),
])
def test_entries_that_could_reach_other_files_are_dropped(voices, files, why):
    (voices / "synthetic.json").write_text(json.dumps({"voices": {"x": {"label": "X", "description": "d",
                                                                         "files": files}}}))
    got, errors = sv.load(str(voices), str(voices / "synthetic.json"))
    assert got == [] and len(errors) == 1 and why in errors[0]


def test_a_clip_without_a_transcript_is_dropped(voices):
    (voices / "casting" / "warm-elder-male_de.txt").unlink()
    got, errors = sv.load(str(voices), str(voices / "synthetic.json"))
    w = next(v for v in got if v.id == "warm-elder-male")
    assert list(w.files) == ["en"] and any("transcript" in e for e in errors)


def test_no_registry_means_no_voices(tmp_path):
    assert sv.load(str(tmp_path), str(tmp_path / "missing.json")) == ([], [])
    (tmp_path / "bad.json").write_text("{not json")
    got, errors = sv.load(str(tmp_path), str(tmp_path / "bad.json"))
    assert got == [] and errors
