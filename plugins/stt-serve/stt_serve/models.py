"""Request schemas of stt-serve -- moved 1:1 from ai-serve-ui's media.py, with two
security changes: audio arrives only as base64 (no audio_path -- a request can't make
this service read a file), and unknown fields are rejected (extra="forbid")."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class KnownSpeaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    embedding: List[float]


class STTTranscribeRequest(BaseModel):
    engine: Literal["whisperx", "moss"] = Field(
        "whisperx",
        description="'whisperx' (default: WhisperX + word alignment, diarization via diarize_backend) or 'moss' "
                    "(MOSS-Transcribe-Diarize 0.9B: transcription, speakers and timestamps in one pass, up to 90 "
                    "minutes, 50+ languages; speakers always come out as SPEAKER_00.., no word timestamps; return_embeddings/"
                    "known_speakers work with diarize=true and use the same voice-embedding model as 'whisperx', so "
                    "embeddings from either engine can be matched against each other).",
    )
    hotwords: Optional[List[str]] = Field(
        None,
        description="engine 'moss' only: up to 50 terms (names, jargon) the model should spell this way; "
                    "1-40 characters each, no brackets or line breaks.",
    )
    model_config = ConfigDict(extra="forbid")

    audio_base64: Optional[str] = Field(
        None,
        description="Base64-encoded audio -- or video (e.g. MP4/MOV/MKV) -- data to transcribe. Decoding goes "
                     "through ffmpeg (via whisperx.load_audio()), which probes actual byte content rather than "
                     "any filename/extension, so a video container's audio track is extracted transparently; "
                     "no separate video-to-audio step is needed before calling this.",
    )
    keep_alive: Optional[int] = Field(
        300, description="Idle seconds before WhisperX auto-unloads. Reset on every transcribe call.",
    )
    diarize: bool = Field(
        False,
        description="Label each segment with a speaker ('SPEAKER_00', 'SPEAKER_01', ...) using "
                     "pyannote.audio speaker diarization (via whisperx.diarize). Lazy-loads a second "
                     "in-process model on first use -- same venv as WhisperX, no extra install "
                     "(both whisperx and pyannote-audio are already dependencies here). See HF_TOKEN "
                     "if the diarization model needs one.",
    )
    diarize_backend: str = Field(
        "whisperx",
        description="diarize=true only: 'whisperx' (default, pyannote.audio via whisperx.diarize) or "
                     "'nemotron' (nvidia/Nemotron-3-Diarization, an isolated-venv subprocess worker -- "
                     "see nemotron/setup.sh). Word-level "
                     "alignment (for per-word, not just per-segment, speaker labels) still runs via "
                     "WhisperX either way -- only the diarization step itself differs. The nemotron "
                     "backend does NOT support return_embeddings/known_speakers (Nemotron doesn't "
                     "produce comparable speaker embeddings); num_speakers/min_speakers/max_speakers "
                     "are honored only approximately (see nemotron/worker.py's docstring) since the model has no native "
                     "expect-exactly-N-speakers input, unlike pyannote's own num_speakers.",
    )
    num_speakers: Optional[int] = Field(
        None, description="diarize=true only: exact speaker count, if known -- most accurate.",
    )
    min_speakers: Optional[int] = Field(
        None, description="diarize=true only: lower bound on speaker count (ignored if num_speakers is set).",
    )
    max_speakers: Optional[int] = Field(
        None, description="diarize=true only: upper bound on speaker count (ignored if num_speakers is set).",
    )
    language: Optional[str] = Field(
        None,
        description="ISO 639-1 code (e.g. 'de', 'en') to pin the transcription language. Omit to "
                     "auto-detect from the first 30s of audio (WhisperX's own default) -- but "
                     "auto-detect is unreliable right after a cold model load (confirmed live: the "
                     "same German clip came back as a fluent-English translation on the first call "
                     "after load, correct German on the next) and small Whisper models can drift "
                     "into translating non-English speech into English even with task='transcribe' "
                     "forced. A caller that already knows the spoken language should always pass it.",
    )
    audio_base64_list: Optional[List[str]] = Field(
        None,
        description="Alternative to audio_base64: several independently-encoded audio containers, "
                     "each decoded on its own (ffmpeg content-sniffs each, so they needn't share a "
                     "format) then concatenated as PCM into one clip before transcribing/diarizing. "
                     "For a caller assembling audio from multiple short chunks (e.g. a live-streaming "
                     "session) where any single chunk is too short for a reliable speaker embedding -- "
                     "see return_embeddings. Takes priority over audio_base64 if given.",
    )
    return_embeddings: bool = Field(
        False,
        description="diarize=true only: return a speaker embedding (pyannote's own, same space its "
                     "diarization clustering already uses -- no extra model) per detected speaker, as "
                     "speaker_embeddings: {label: [floats]} in the response. Persist these and pass "
                     "them back as known_speakers on a later call to recognize the same person across "
                     "separate recordings -- diarization's own SPEAKER_00-style labels are only stable "
                     "*within* one call. Embeddings are unreliable from very short audio (confirmed: a "
                     "single real speaker gets missed entirely at 4s, clean separation from ~8s on) -- "
                     "see the duration_s/warning fields in the response.",
    )
    known_speakers: Optional[List[KnownSpeaker]] = Field(
        None,
        description="diarize=true only: a roster of previously-enrolled {label, embedding} pairs "
                     "(embedding from a prior call's return_embeddings=true response) to match "
                     "detected speakers against via cosine similarity. Each segment gets similarity "
                     "(best score against the roster, even below match_threshold), matched (bool), "
                     "and matched_label; a match also replaces that speaker's generic SPEAKER_00-style "
                     "label with matched_label everywhere in the response (segments and transcript).",
    )
    match_threshold: float = Field(
        0.7, description="known_speakers only: minimum cosine similarity to count as a match.",
    )


class LiveSTTStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Optional[str] = Field(None, description="Same as STTTranscribeRequest.language -- pin it if known.")
    diarize: bool = Field(False, description="Diarize each finalized segment, same as STTTranscribeRequest.diarize.")
    diarize_backend: str = Field("whisperx", description="Same as STTTranscribeRequest.diarize_backend.")
    silence_ms: int = Field(
        1200,
        description="Trailing silence (ms) that ends an utterance and finalizes it into a segment. "
                     "Lower it for lively conversation where a long fixed threshold never flushes "
                     "(a reported real case: 1200ms never flushed at all in fast back-and-forth "
                     "dialogue; 400ms did) -- there's no one right value, it trades off segment "
                     "responsiveness against clipping a speaker's own mid-sentence pause.",
    )
    keep_alive: int = Field(
        600,
        description="Idle seconds (no /chunk or /poll call) before this session force-finalizes "
                     "any pending audio and is dropped, same as the family="
                     "always-something-cleans-up-an-abandoned-resource convention elsewhere in this API "
                     "(TTS/STT model keep_alive). A client that calls /stop itself doesn't need to rely "
                     "on this at all.",
    )
    return_embeddings: bool = Field(False, description="Same as STTTranscribeRequest.return_embeddings, applied to every finalized segment's diarization.")
    known_speakers: Optional[List[KnownSpeaker]] = Field(None, description="Same as STTTranscribeRequest.known_speakers.")
    match_threshold: float = Field(0.7, description="Same as STTTranscribeRequest.match_threshold.")


class LiveSTTChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    audio_base64: str = Field(
        ..., description="One chunk of audio (any container ffmpeg can sniff). Appended to the "
                          "session's buffer; finalization into a segment happens automatically, "
                          "here and on /poll, whenever trailing silence crosses silence_ms.",
    )


class LiveSTTSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    silence_ms: Optional[int] = None
    keep_alive: Optional[int] = None


class LiveSTTStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
