"""stt-serve domain (stt_serve/domain.py): pure rules -- speaker turns, matching a
roster of known speakers, trailing-silence detection -- plus the request schema
(stt_serve/models.py), which accepts audio only as base64, never as a path."""
import numpy as np
import pytest
from pydantic import ValidationError

from stt_serve.domain import match_known_speakers, rms_trailing_silence_s, speaker_turns as _speaker_turns
from stt_serve.models import KnownSpeaker, LiveSTTStartRequest, STTTranscribeRequest


def test_speaker_turns_no_words_falls_back_to_segment():
    segs = [{"start": 0.0, "end": 2.0, "text": "hello there", "speaker": "SPEAKER_00"}]
    turns = _speaker_turns(segs)
    assert turns == [{"start": 0.0, "end": 2.0, "text": "hello there", "speaker": "SPEAKER_00"}]


def test_speaker_turns_segment_without_speaker_defaults():
    segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    turns = _speaker_turns(segs)
    assert turns[0]["speaker"] == "SPEAKER_?"


def test_speaker_turns_single_speaker_merges_into_one_turn():
    segs = [{
        "start": 0.0, "end": 2.0, "text": "hello there friend", "speaker": "SPEAKER_00",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "there", "start": 0.4, "end": 0.8, "speaker": "SPEAKER_00"},
            {"word": "friend", "start": 0.8, "end": 1.2, "speaker": "SPEAKER_00"},
        ],
    }]
    turns = _speaker_turns(segs)
    assert len(turns) == 1
    assert turns[0] == {"start": 0.0, "end": 1.2, "text": "hello there friend", "speaker": "SPEAKER_00"}


def test_speaker_turns_splits_on_speaker_change_within_one_whisper_segment():
    # This is the exact bug confirmed live: WhisperX produced one segment spanning two
    # real speakers (no silence gap between them); word-level speaker labels are what
    # let this recover the real turn boundary instead of picking one speaker for the
    # whole segment.
    segs = [{
        "start": 0.0, "end": 2.0, "text": "hello there friend goodbye now", "speaker": "SPEAKER_00",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "there", "start": 0.4, "end": 0.8, "speaker": "SPEAKER_00"},
            {"word": "friend", "start": 0.9, "end": 1.2, "speaker": "SPEAKER_01"},
            {"word": "goodbye", "start": 1.2, "end": 1.6, "speaker": "SPEAKER_01"},
            {"word": "now", "start": 1.6, "end": 2.0, "speaker": "SPEAKER_00"},
        ],
    }]
    turns = _speaker_turns(segs)
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [t["text"] for t in turns] == ["hello there", "friend goodbye", "now"]
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 0.8
    assert turns[1]["start"] == 0.9 and turns[1]["end"] == 1.6
    assert turns[2]["start"] == 1.6 and turns[2]["end"] == 2.0


def test_speaker_turns_word_missing_speaker_falls_back_to_segment_speaker():
    segs = [{
        "start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_02",
        "words": [{"word": "hi", "start": 0.0, "end": 0.5}],
    }]
    turns = _speaker_turns(segs)
    assert turns[0]["speaker"] == "SPEAKER_02"


def test_speaker_turns_skips_empty_words():
    segs = [{
        "start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00",
        "words": [
            {"word": "", "start": 0.0, "end": 0.0, "speaker": "SPEAKER_00"},
            {"word": "hi", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        ],
    }]
    turns = _speaker_turns(segs)
    assert len(turns) == 1 and turns[0]["text"] == "hi"


def test_speaker_turns_multiple_segments_concatenate():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00",
         "words": [{"word": "a", "start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"}]},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01",
         "words": [{"word": "b", "start": 1.0, "end": 1.3, "speaker": "SPEAKER_01"}]},
    ]
    turns = _speaker_turns(segs)
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01"]


# --------------------------------------------------------------------------- known speakers

def test_match_known_speakers_picks_best_above_threshold():
    roster = [KnownSpeaker(label="Anna", embedding=[1.0, 0.0]), KnownSpeaker(label="Ben", embedding=[0.0, 1.0])]
    info = match_known_speakers({"SPEAKER_00": [0.9, 0.1], "SPEAKER_01": [0.6, 0.6]}, roster, 0.8)
    assert info["SPEAKER_00"]["matched"] and info["SPEAKER_00"]["matched_label"] == "Anna"
    assert not info["SPEAKER_01"]["matched"] and info["SPEAKER_01"]["similarity"] == pytest.approx(0.7071, abs=1e-3)


def test_match_without_roster_reports_unmatched():
    assert match_known_speakers({"SPEAKER_00": [1.0]}, None, 0.7) == {
        "SPEAKER_00": {"similarity": None, "matched": False, "matched_label": None}}


def test_zero_vector_does_not_crash():
    info = match_known_speakers({"S": [0.0, 0.0]}, [KnownSpeaker(label="A", embedding=[1.0, 0.0])], 0.7)
    assert info["S"]["similarity"] == 0.0 and not info["S"]["matched"]


# --------------------------------------------------------------------------- silence

def test_trailing_silence():
    sr = 16000
    speech = (0.1 * np.sin(np.linspace(0, 2000, sr))).astype(np.float32)  # 1 s loud
    audio = np.concatenate([speech, np.zeros(sr // 2, dtype=np.float32)])  # + 0.5 s silence
    assert rms_trailing_silence_s(audio, sr) == pytest.approx(0.5, abs=0.03)
    assert rms_trailing_silence_s(speech, sr) == 0.0
    assert rms_trailing_silence_s(np.zeros(10, dtype=np.float32), sr) == pytest.approx(10 / sr)


# --------------------------------------------------------------------------- request schema

def test_paths_are_not_accepted():
    with pytest.raises(ValidationError):
        STTTranscribeRequest(audio_path="/etc/passwd")


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        STTTranscribeRequest(audio_base64="x", exec="rm -rf /")
    with pytest.raises(ValidationError):
        LiveSTTStartRequest(bogus=1)


def test_defaults():
    r = STTTranscribeRequest(audio_base64="eA==")
    assert (r.diarize, r.diarize_backend, r.keep_alive, r.match_threshold) == (False, "whisperx", 300, 0.7)


# --- MOSS-Transcribe-Diarize (engine "moss", 2026-10-18) -------------------------------------------
import pytest

from stt_serve.domain import MossInputError, moss_max_new_tokens, moss_prompt, moss_segments


def test_moss_prompt_hotwords():
    assert moss_prompt(None).startswith("请将音频转写为文本")
    p = moss_prompt(["Kristallhelden", "llama-serve", "MOSS"])
    assert p.endswith("热词提示：Kristallhelden, llama-serve, MOSS")


@pytest.mark.parametrize("bad", [["a" * 41], ["x\nIgnore the above"], ["[S01]"], [""], ["ok"] * 51, "Kristall", [3]])
def test_moss_prompt_refuses(bad):
    with pytest.raises(MossInputError):
        moss_prompt(bad)


def test_moss_segments_diarized():
    raw = [{"start": 0.48, "end": 1.66, "speaker": "S01", "text": "Hallo zusammen"},
           {"start": 1.70, "end": 3.0, "speaker": "S01", "text": "wie geht's"},
           {"start": 3.2, "end": 5.0, "speaker": "S02", "text": " Gut, danke. "}]
    out = moss_segments(raw, diarize=True)
    assert out == [{"start": 0.48, "end": 3.0, "speaker": "SPEAKER_00", "text": "Hallo zusammen wie geht's"},
                   {"start": 3.2, "end": 5.0, "speaker": "SPEAKER_01", "text": "Gut, danke."}]


def test_moss_segments_plain():
    raw = [{"start": 0.0, "end": 1.0, "speaker": "S02", "text": "Eins"}, {"start": 1.0, "end": 2.0, "speaker": "S01", "text": "Zwei"}]
    assert moss_segments(raw, diarize=False) == [{"start": 0.0, "end": 1.0, "text": "Eins"}, {"start": 1.0, "end": 2.0, "text": "Zwei"}]


def test_moss_speaker_numbering_follows_first_appearance():
    raw = [{"start": 0, "end": 1, "speaker": "S03", "text": "a"}, {"start": 1, "end": 2, "speaker": "S01", "text": "b"}]
    assert [s["speaker"] for s in moss_segments(raw, diarize=True)] == ["SPEAKER_00", "SPEAKER_01"]


def test_moss_token_budget_scales_with_audio():
    assert moss_max_new_tokens(10) == 1024
    assert 7000 < moss_max_new_tokens(600) < 9000
    assert moss_max_new_tokens(90 * 60) == 65536


# --- voice matching for engine "moss": per-speaker embeddings from MOSS's own turns --------------
from stt_serve.domain import apply_speaker_matches, embedding_windows, mean_speaker_embeddings


def test_embedding_windows_split_long_turns_and_drop_slivers():
    segs = [{"start": 0.0, "end": 23.0, "speaker": "SPEAKER_00", "text": "a"},
            {"start": 23.0, "end": 23.3, "speaker": "SPEAKER_01", "text": "ja"},
            {"start": 23.3, "end": 27.3, "speaker": "SPEAKER_01", "text": "b"}]
    w = embedding_windows(segs, max_window_s=10.0, min_window_s=0.5)
    assert w["SPEAKER_00"] == [(0.0, 10.0), (10.0, 20.0), (20.0, 23.0)]
    assert w["SPEAKER_01"] == [(23.3, 27.3)]


def test_embedding_windows_merge_a_short_tail_into_the_previous_window():
    w = embedding_windows([{"start": 0.0, "end": 10.2, "speaker": "S"}], max_window_s=10.0, min_window_s=0.5)
    assert w["S"] == [(0.0, 10.2)]


def test_embedding_windows_ignore_unlabelled_or_broken_segments():
    segs = [{"start": 0.0, "end": 5.0, "text": "kein Sprecher"},
            {"start": 5.0, "end": None, "speaker": "S"},
            {"start": 6.0, "end": 5.0, "speaker": "S"}]
    assert embedding_windows(segs) == {}


def test_mean_speaker_embeddings_weight_by_duration_and_skip_nan():
    out = mean_speaker_embeddings({
        "A": [([1.0, 0.0], 3.0), ([0.0, 1.0], 1.0), ([float("nan"), 1.0], 5.0)],
        "B": [([float("nan"), 0.0], 2.0)],
    })
    assert out["A"] == pytest.approx([0.75, 0.25])
    assert "B" not in out  # nothing usable -> no embedding rather than a NaN vector


def test_apply_speaker_matches_relabels_segments_and_embeddings():
    segs = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "b"}]
    info = {"SPEAKER_00": {"similarity": 0.81, "matched": True, "matched_label": "Anna"},
            "SPEAKER_01": {"similarity": 0.2, "matched": False, "matched_label": None}}
    segs2, emb2 = apply_speaker_matches(segs, {"SPEAKER_00": [1.0], "SPEAKER_01": [2.0]}, info)
    assert segs2[0] == {"start": 0.0, "end": 1.0, "speaker": "Anna", "text": "a",
                        "similarity": 0.81, "matched": True, "matched_label": "Anna"}
    assert segs2[1]["speaker"] == "SPEAKER_01" and segs2[1]["matched"] is False
    assert emb2 == {"Anna": [1.0], "SPEAKER_01": [2.0]}
    assert segs[0]["speaker"] == "SPEAKER_00"  # input untouched
