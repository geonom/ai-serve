"""stt-serve domain -- pure rules, no models, no I/O: regrouping aligned words into
speaker turns, matching detected speakers against a roster of known embeddings,
and the energy-based trailing-silence check that finalizes live segments. Moved
from ai-serve-ui's media.py (tests in tests/test_domain.py)."""
from typing import Any, Dict, List, Optional, Tuple


def speaker_turns(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Regroup whisper-native segments (already run through whisperx.align +
    assign_word_speakers) into contiguous same-speaker turns, using WORD-level speaker
    labels -- a whisper segment routinely spans a speaker change, so trusting the
    segment's own single 'speaker' field silently drops whichever speaker talked less
    within it (confirmed live). Falls back to the segment's own start/end/speaker
    as-is when it has no 'words' (alignment skipped or failed for that segment)."""
    turns: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            turns.append({
                "start": seg.get("start"), "end": seg.get("end"),
                "text": seg["text"].strip(), "speaker": seg.get("speaker", "SPEAKER_?"),
            })
            continue
        cur: Optional[Dict[str, Any]] = None
        for w in words:
            speaker = w.get("speaker", seg.get("speaker", "SPEAKER_?"))
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start = w.get("start", cur["end"] if cur else seg.get("start"))
            end = w.get("end", start)
            if cur is not None and cur["speaker"] == speaker:
                cur["end"] = end
                cur["text"] += " " + text
            else:
                if cur is not None:
                    turns.append(cur)
                cur = {"start": start, "end": end, "text": text, "speaker": speaker}
        if cur is not None:
            turns.append(cur)
    return turns


RELIABLE_EMBEDDING_MIN_SECONDS = 8.0


def match_known_speakers(
    speaker_embeddings: Dict[str, List[float]],
    known_speakers: Optional[List[Any]],
    match_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """Cosine-match each diarization-detected speaker label against a caller-supplied
    roster. Returns {detected_label: {similarity, matched, matched_label}} -- similarity
    and matched=False are populated even when nothing crosses match_threshold, so a
    caller can tell "no embedding" from "embedding computed, no match" (see the report
    that prompted this: a real bug on the caller's own side came from guessing a match
    from the label string alone instead of an explicit field)."""
    import numpy as np

    info: Dict[str, Dict[str, Any]] = {}
    if not known_speakers:
        return {label: {"similarity": None, "matched": False, "matched_label": None}
                for label in speaker_embeddings}

    roster = [(ks.label, np.asarray(ks.embedding, dtype=np.float64)) for ks in known_speakers]
    for label, emb in speaker_embeddings.items():
        v = np.asarray(emb, dtype=np.float64)
        v_norm = np.linalg.norm(v)
        best_label, best_sim = None, -1.0
        for known_label, known_vec in roster:
            denom = v_norm * np.linalg.norm(known_vec)
            sim = float(np.dot(v, known_vec) / denom) if denom > 0 else 0.0
            if sim > best_sim:
                best_label, best_sim = known_label, sim
        matched = best_sim >= match_threshold
        info[label] = {
            "similarity": round(best_sim, 4),
            "matched": matched,
            "matched_label": best_label if matched else None,
        }
    return info


def rms_trailing_silence_s(audio, sr: int, frame_ms: int = 20, threshold: float = 0.015) -> float:
    """How many trailing seconds of `audio` sit below an RMS energy threshold, walking
    backward from the end until a frame exceeds it (or the whole buffer is silent).
    Lightweight energy-based VAD, deliberately not the pyannote/silero VAD this project
    already uses elsewhere for batch diarization -- those are built to process a whole
    file at once, not to make a cheap synchronous per-chunk decision on every /chunk
    call of a live stream. threshold is a plain RMS amplitude cutoff (audio is float32
    in [-1, 1]) -- reasonable for normal-volume speech, not adaptive to ambient noise
    floor; a very noisy environment may need a caller-tunable threshold if this proves
    too sensitive in practice (not exposed yet -- silence_ms was the reported need)."""
    import numpy as np
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return len(audio) / sr
    trailing_frames = 0
    for i in range(n_frames - 1, -1, -1):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms >= threshold:
            break
        trailing_frames += 1
    return trailing_frames * frame_len / sr


# --- MOSS-Transcribe-Diarize (engine "moss") ------------------------------------------------------
# The model answers a prompt; hotwords are appended to it (the authors' convention, README
# "Custom Prompt and Hotwords"). Hotwords come from callers, so they are plain short terms only --
# nothing that could restate the instructions or mimic the output format ([S01], newlines).
MOSS_DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
MOSS_MAX_HOTWORDS = 50
MOSS_MAX_HOTWORD_LEN = 40


class MossInputError(ValueError):
    pass


def moss_prompt(hotwords: Optional[List[str]]) -> str:
    if not hotwords:
        return MOSS_DEFAULT_PROMPT
    if not isinstance(hotwords, list) or len(hotwords) > MOSS_MAX_HOTWORDS:
        raise MossInputError(f"hotwords: a list of at most {MOSS_MAX_HOTWORDS} terms")
    clean = []
    for h in hotwords:
        if not isinstance(h, str) or not h.strip() or len(h) > MOSS_MAX_HOTWORD_LEN \
                or any(c in h for c in "[]\n\r\t") or any(ord(c) < 32 for c in h):
            raise MossInputError(f"hotwords: plain terms of 1-{MOSS_MAX_HOTWORD_LEN} characters, no brackets or line breaks")
        clean.append(h.strip())
    return f"{MOSS_DEFAULT_PROMPT}热词提示：{', '.join(clean)}"


def moss_segments(raw: List[Dict[str, Any]], *, diarize: bool) -> List[Dict[str, Any]]:
    """MOSS segments ({start, end, speaker 'S01', text}) in stt-serve's usual shape: speakers renamed
    SPEAKER_00.. in order of first appearance (what WhisperX/pyannote callers already expect) and
    consecutive segments of the same speaker merged into one turn; without diarize, no speakers."""
    names: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for seg in raw:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if not diarize:
            out.append({"start": seg.get("start"), "end": seg.get("end"), "text": text})
            continue
        spk = names.setdefault(seg.get("speaker") or "?", f"SPEAKER_{len(names):02d}")
        if out and out[-1]["speaker"] == spk:
            out[-1]["end"] = seg.get("end")
            out[-1]["text"] = f"{out[-1]['text']} {text}"
        else:
            out.append({"start": seg.get("start"), "end": seg.get("end"), "speaker": spk, "text": text})
    return out


def moss_max_new_tokens(duration_s: float) -> int:
    """Enough tokens for the transcript of `duration_s` seconds (text + timestamps + speaker tags,
    ~13.5 tokens per second of speech), at least 1024, at most 65536 (90 minutes)."""
    return int(min(65536, max(1024, duration_s * 13.5)))


# --- voice matching for engine "moss" ---------------------------------------------------------------
# MOSS-Transcribe-Diarize only names speakers S01, S02 ... per recording, it has no voice embeddings.
# stt-serve computes them itself from MOSS's own speaker turns, with the embedding model of the
# pyannote pipeline the WhisperX path uses -- pyannote's speaker centroids are (weighted) means of
# exactly these raw embeddings, so vectors from both engines are comparable with the same roster.

def embedding_windows(segments: List[Dict[str, Any]], max_window_s: float = 10.0,
                      min_window_s: float = 0.5) -> Dict[str, List[Tuple[float, float]]]:
    """Audio windows per speaker label to embed: each turn cut into pieces of at most `max_window_s`
    (a tail shorter than `min_window_s` stays with the piece before it); turns shorter than
    `min_window_s` carry too little voice and are left out. Segments without a speaker or with
    broken times are ignored."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for seg in segments:
        label, start, end = seg.get("speaker"), seg.get("start"), seg.get("end")
        if not label or start is None or end is None or end - start < min_window_s:
            continue
        pieces: List[Tuple[float, float]] = []
        t = float(start)
        while end - t > max_window_s:
            pieces.append((t, t + max_window_s))
            t += max_window_s
        if pieces and end - t < min_window_s:
            pieces[-1] = (pieces[-1][0], float(end))
        else:
            pieces.append((t, float(end)))
        out.setdefault(label, []).extend(pieces)
    return out


def mean_speaker_embeddings(weighted: Dict[str, List[Tuple[List[float], float]]]) -> Dict[str, List[float]]:
    """Duration-weighted mean embedding per speaker. Vectors containing NaN (pyannote returns those
    for windows too short for its model) are skipped; a speaker with none left gets no entry."""
    import numpy as np

    out: Dict[str, List[float]] = {}
    for label, items in weighted.items():
        vecs, weights = [], []
        for vec, w in items:
            v = np.asarray(vec, dtype=np.float64)
            if w > 0 and v.size and not np.isnan(v).any():
                vecs.append(v)
                weights.append(w)
        if vecs:
            out[label] = (np.average(np.vstack(vecs), axis=0, weights=weights)).tolist()
    return out


def apply_speaker_matches(segments: List[Dict[str, Any]], speaker_embeddings: Dict[str, List[float]],
                          match_info: Dict[str, Dict[str, Any]]):
    """A matched speaker's generic label becomes its known_speakers label everywhere (segments and
    embeddings), and every segment carries similarity/matched/matched_label -- the same shape the
    WhisperX path returns. Returns new (segments, embeddings); the inputs stay untouched."""
    relabel = {label: m["matched_label"] for label, m in match_info.items() if m.get("matched")}
    new_segments = []
    for seg in segments:
        s = dict(seg)
        m = match_info.get(s.get("speaker"))
        if m is not None:
            s["speaker"] = relabel.get(s["speaker"], s["speaker"])
            s["similarity"], s["matched"], s["matched_label"] = m["similarity"], m["matched"], m["matched_label"]
        new_segments.append(s)
    return new_segments, {relabel.get(k, k): v for k, v in speaker_embeddings.items()}
