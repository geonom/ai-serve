"""TTS, STT, voice listing, ComfyUI management, and image generation routes."""
import asyncio
import base64
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import aiohttp
import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticUndefined

from backend.config import DEFAULT_TTS_ENGINE, DEFAULT_TTS_VOICE, LOGS_DIR
from backend import stt_client, tts_client
from backend.state import state

logger = logging.getLogger("ai-serve-ui.media")
router = APIRouter()

_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VOICES_DIR = os.path.join(_PROJECT_DIR, "voices")
# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TTSGenerateRequest(BaseModel):
    text: str
    # Defaulted (was required) so the OpenAI-compatible /v1/audio/speech shim and any
    # caller that doesn't care about language can omit it. OmniVoice wants a short code
    # ("en", "de"); for engine="qwen3" a code here is mapped to the full name it needs.
    language: str = "en"
    engine: Optional[str] = Field(
        None,
        description="Which TTS engine. Omit to use the server default (DEFAULT_TTS_ENGINE, normally "
                     "'cosyvoice'). Every engine runs in tts-serve (503 if it is unavailable). "
                     "Values: 'cosyvoice' (CosyVoice 3 -- clone + "
                     "free-form style via `variant`; Apache-2.0), 'omnivoice' (23 style "
                     "tags OR clone, never both; weights CC-BY-NC / non-commercial), 'qwen3' (3 "
                     "checkpoints via `variant`; Apache-2.0), 'chatterbox' (clone + expressiveness "
                     "knobs; MIT, output watermarked), 'voxcpm' (VoxCPM2 on Nano-vLLM -- clone, batches "
                     "concurrent requests, fastest; Apache-2.0). Engine-specific fields below are only read "
                     "for the engine they name.",
    )
    instruct: Optional[str] = Field(
        None,
        description="Natural-language voice/style description, used when no reference clip is given. "
                     "OmniVoice: an instruct-design tag. engine='qwen3': required for "
                     "variant='voice_design', optional style layer for variant='custom_voice'.",
    )
    ref_audio_base64: Optional[str] = Field(
        None,
        description="Voice cloning: base64-encoded reference audio to clone. Supported by OmniVoice "
                     "and by engine='qwen3' variant='base'. Alternative to ref_voice_path.",
    )
    ref_voice_path: Optional[str] = Field(
        None,
        description="Voice cloning: path to a preset reference voice, must resolve inside voices/. "
                     "Supported by OmniVoice and by engine='qwen3' variant='base'.",
    )
    ref_text: Optional[str] = Field(
        None,
        description="Transcript of the reference audio used for cloning. OmniVoice: optional -- "
                     "WhisperX auto-transcribes when omitted. engine='qwen3' variant='base': REQUIRED.",
    )
    # --- engine='qwen3' / engine='cosyvoice' ---
    variant: Optional[str] = Field(
        None,
        description="engine='qwen3': 'base' (default) zero-shot cloning (ref_audio + ref_text; "
                     "delivery copied as-is, `instruct` is a no-op), 'voice_design' (voice from "
                     "`instruct`, no reference), 'custom_voice' (preset `speaker` + optional `instruct`). "
                     "engine='cosyvoice': 'zero_shot' (default) cloning (ref_audio + ref_text), "
                     "'instruct' cloning WITH free-form style `instruct` (langsam/ärgerlich/... "
                     "supported), 'cross_lingual' cloning (ref_audio only, no ref_text). "
                     "Ignored for OmniVoice and Chatterbox.",
    )
    speaker: Optional[str] = Field(
        None,
        description="engine='qwen3', variant='custom_voice' only, required there: one of the 9 "
                     "preset timbres (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, "
                     "Ono_Anna, Sohee). Ignored otherwise.",
    )
    # --- engine='chatterbox' only (and temperature is also forwarded to cosyvoice) ---
    exaggeration: Optional[float] = Field(
        None, description="engine='chatterbox' only: expressiveness, ~0..1 (default ~0.5). Higher = more dramatic/emotive.",
    )
    cfg_weight: Optional[float] = Field(
        None, description="engine='chatterbox': classifier-free-guidance weight (default ~0.5). Lower + higher exaggeration = more expressive, looser pacing. "
                          "engine='voxcpm': forwarded as cfg_value (default 2.0).",
    )
    temperature: Optional[float] = Field(
        None, description="engine='chatterbox' (and forwarded to engine='cosyvoice'): sampling temperature.",
    )
    params: Optional[Dict[str, float]] = Field(
        None,
        description="Numeric engine params by name, as the engine lists them in GET /v1/audio/models "
                    "(modes[].params); tts-serve refuses names the engine doesn't take. The older top-level "
                    "temperature/exaggeration/cfg_weight fields still work and are merged in.",
    )
    response_format: Optional[str] = "json"

    @field_validator("params")
    @classmethod
    def _param_names(cls, v):
        if v is not None:
            if len(v) > 16 or any(not re.match(r"^[a-z][a-z0-9_]{0,31}$", k) for k in v):
                raise ValueError("params: at most 16, names of lowercase letters, digits and _")
        return v
    # Mirrors llama-server instances' own keep_alive convention (see get_or_load_model()) --
    # idle seconds before the engine auto-unloads. Reset on every generate call, same pattern.
    # 60s (not llama-server's 300s default) -- a TTS call is a one-shot, seconds-long
    # request, not a multi-turn chat session someone keeps coming back to, so there's
    # much less reason to hold the VRAM/RAM warm on the chance of an imminent follow-up
    # call. Still fully overridable per-request for a caller who knows it'll batch many
    # calls back-to-back and wants to avoid repeated cold-load overhead in between.
    keep_alive: Optional[int] = 60


class Qwen3TTSGenerateRequest(BaseModel):
    text: str
    variant: str = Field(
        "base",
        description="Which Qwen3-TTS checkpoint to use: 'base' (Qwen3-TTS-12Hz-1.7B-Base -- 3-second "
                     "reference-audio voice cloning, needs ref_audio_base64/ref_voice_path + ref_text), "
                     "'voice_design' (Qwen3-TTS-12Hz-1.7B-VoiceDesign -- natural-language voice description "
                     "via instruct), or 'custom_voice' (Qwen3-TTS-12Hz-1.7B-CustomVoice -- one of 9 preset "
                     "premium timbres via speaker, with an optional style instruct on top). Only one variant "
                     "is held in VRAM at a time -- requesting a different variant than the one currently "
                     "loaded unloads it and loads the new one (see CLAUDE.md's Qwen3-TTS section).",
    )
    language: str = Field(
        "English",
        description="Full language name (English, Chinese, Japanese, Korean, German, French, Russian, "
                     "Portuguese, Spanish, Italian) or 'Auto'.",
    )
    ref_audio_base64: Optional[str] = Field(None, description="variant='base' only: base64-encoded reference audio to clone.")
    ref_voice_path: Optional[str] = Field(None, description="variant='base' only: path to a preset reference voice, must resolve inside voices/.")
    ref_text: Optional[str] = Field(None, description="variant='base' only: transcript of the reference audio -- required alongside ref_audio_base64/ref_voice_path.")
    instruct: Optional[str] = Field(
        None,
        description="variant='voice_design': required natural-language description of the target voice "
                     "(timbre/emotion/prosody). variant='custom_voice': optional style-control instruction "
                     "layered on top of the chosen speaker. Unused for variant='base'.",
    )
    speaker: Optional[str] = Field(
        None,
        description="variant='custom_voice' only, required: one of the 9 preset timbres -- Vivian, Serena, "
                     "Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee.",
    )
    response_format: Optional[str] = "json"
    # Same idle-unload convention and 60s default as TTSGenerateRequest.keep_alive --
    # see that field's comment for why 60s rather than llama-server's 300s. This
    # endpoint is deprecated in favor of /api/tts/generate (engine="qwen3"), which
    # always forwards its own TTSGenerateRequest.keep_alive here regardless of this
    # default -- this default only matters for a caller still hitting this endpoint
    # directly.
    keep_alive: Optional[int] = 60


class KnownSpeaker(BaseModel):
    label: str
    embedding: List[float]


class STTTranscribeRequest(BaseModel):
    engine: Optional[str] = Field(
        None, pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$",
        description="Omitted: DEFAULT_STT_ENGINE, else the transcription plugins' default (stt-serve: whisperx). "
                    "Any engine id from GET /v1/audio/models; stt-serve's: "
                    "'whisperx' (WhisperX + word alignment, diarization via diarize_backend) or 'moss' "
                    "(MOSS-Transcribe-Diarize 0.9B in stt-serve: text, speakers and timestamps in one pass, up to "
                    "90 minutes, no word timestamps; return_embeddings/known_speakers work with diarize=true and "
                    "use the same voice-embedding model as 'whisperx', so embeddings match across engines).",
    )
    hotwords: Optional[List[str]] = Field(
        None,
        description="engine 'moss' only: up to 50 terms (names, jargon) the model should spell this way; "
                    "1-40 characters each, no brackets or line breaks.",
    )
    audio_base64: Optional[str] = Field(
        None,
        description="Base64-encoded audio -- or video (e.g. MP4/MOV/MKV) -- data to transcribe. Decoding goes "
                     "through ffmpeg (via whisperx.load_audio()), which probes actual byte content rather than "
                     "any filename/extension, so a video container's audio track is extracted transparently; "
                     "no separate video-to-audio step is needed before calling this.",
    )
    audio_path: Optional[str] = Field(
        None,
        description="Path to a local audio or video file to transcribe (same video support as audio_base64 -- "
                     "ffmpeg content-sniffs regardless of extension). Must resolve inside the voices/ directory.",
    )
    keep_alive: Optional[int] = Field(
        300, description="Idle seconds before WhisperX auto-unloads. Reset on every transcribe call.",
    )
    diarize: bool = Field(
        False,
        description="Label each segment with a speaker ('SPEAKER_00', 'SPEAKER_01', ...) using "
                     "pyannote.audio speaker diarization (via whisperx.diarize). Lazy-loads a second "
                     "in-process model on first use -- same venv as WhisperX, no extra install "
                     "(runs in stt-serve, like everything STT). See stt-serve's HF_TOKEN "
                     "if the diarization model needs one.",
    )
    diarize_backend: str = Field(
        "whisperx",
        description="diarize=true only: 'whisperx' (default, pyannote.audio via whisperx.diarize) or "
                     "'nemotron' (nvidia/Nemotron-3-Diarization, an isolated-venv subprocess worker -- "
                     "see stt-serve's nemotron/setup.sh). Word-level "
                     "alignment (for per-word, not just per-segment, speaker labels) still runs via "
                     "WhisperX either way -- only the diarization step itself differs. The nemotron "
                     "backend does NOT support return_embeddings/known_speakers (Nemotron doesn't "
                     "produce comparable speaker embeddings); num_speakers/min_speakers/max_speakers "
                     "are honored only approximately (see stt-serve's nemotron/worker.py docstring) "
                     "since the model has no native "
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
                     "see return_embeddings. Takes priority over audio_base64/audio_path if given.",
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
    session_id: str
    audio_base64: str = Field(
        ..., description="One chunk of audio (any container ffmpeg can sniff). Appended to the "
                          "session's buffer; finalization into a segment happens automatically, "
                          "here and on /poll, whenever trailing silence crosses silence_ms.",
    )


class LiveSTTSettingsRequest(BaseModel):
    session_id: str
    silence_ms: Optional[int] = None
    keep_alive: Optional[int] = None


class LiveSTTStopRequest(BaseModel):
    session_id: str


class OpenAISpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    # Not part of OpenAI's schema -- an optional extension so callers can synthesize
    # non-English text on the language-aware engines (qwen3 maps it to its full
    # language name, chatterbox passes it as language_id). Omitted -> "en", as before.
    language: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    # extensions: layout = elements with a relative box [left, top, right, bottom] in 0..1 (models with `layout`,
    # e.g. MOSS-Image in the comfyui-moss plugin);
    # other models come from image plugins (backend/plugins/images.py; e.g. Iris in sd-serve)
    model: Optional[str] = None
    layout: Optional[List[Dict[str, Any]]] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    # extension: an aspect ratio from GET /v1/images/models instead of `size`; each provider turns it into a
    # size it was trained for (the image plugin decides)
    aspect_ratio: Optional[str] = None


class ImageGenerationResponseDataItem(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageGenerationResponseDataItem]


# ---------------------------------------------------------------------------
# Diffusion model / LoRA listing
# ---------------------------------------------------------------------------

@router.get("/api/voices", summary="List local reference voices")
async def list_voices():
    if not os.path.exists(_VOICES_DIR):
        return {"voices": []}
    files = glob.glob(os.path.join(_VOICES_DIR, "*.wav")) + glob.glob(os.path.join(_VOICES_DIR, "*.mp3"))
    voices = []
    for f in files:
        name = os.path.basename(f)
        base_name = os.path.splitext(name)[0]
        txt_path = os.path.join(_VOICES_DIR, f"{base_name}.txt")
        transcript = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as tf:
                    transcript = tf.read().strip()
            except Exception as e:
                logger.error(f"Failed to read transcription file {txt_path}: {e}")
        voices.append({"name": name, "path": os.path.abspath(f), "transcript": transcript})
    voices.sort(key=lambda x: x["name"].lower())
    return {"voices": voices}


# ---------------------------------------------------------------------------
# TTS reference audio (voices/ stays the gateway's)
# ---------------------------------------------------------------------------

def _resolve_ref_audio(req: "TTSGenerateRequest") -> tuple[Optional[str], Optional[str]]:
    """(path, temp_path_to_delete). ref_voice_path must resolve inside voices/;
    ref_audio_base64 is written to a NamedTemporaryFile the caller must clean up."""
    if req.ref_voice_path:
        voices_dir_abs = os.path.abspath(_VOICES_DIR)
        target_path = os.path.abspath(req.ref_voice_path)
        if not target_path.startswith(voices_dir_abs):
            raise HTTPException(status_code=400, detail="Invalid voice path.")
        return target_path, None
    if req.ref_audio_base64:
        import tempfile
        data = base64.b64decode(req.ref_audio_base64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name, tmp.name
    return None, None


def _default_ref_voice() -> tuple[Optional[str], Optional[str]]:
    """(path, transcript) for DEFAULT_TTS_VOICE -- the reference a clone engine uses
    when the caller supplies none. Transcript is read from the file's .txt sidecar
    (CosyVoice zero_shot needs it). Returns (None, None) if the file can't be found."""
    name = DEFAULT_TTS_VOICE
    p = name if os.path.isabs(name) else os.path.join(_VOICES_DIR, name)
    if not os.path.isfile(p) and os.path.isdir(_VOICES_DIR):
        stem = os.path.splitext(name)[0].lower()
        match = next((f for f in sorted(os.listdir(_VOICES_DIR)) if f.lower().startswith(stem)), None)
        p = os.path.join(_VOICES_DIR, match) if match else p
    if not os.path.isfile(p):
        return None, None
    txt_path = os.path.splitext(p)[0] + ".txt"
    transcript = None
    if os.path.isfile(txt_path):
        try:
            transcript = open(txt_path, encoding="utf-8").read().strip() or None
        except Exception:
            pass
    return p, transcript


def _normalize_ref_audio(path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Loudness-normalise a reference clip before cloning. Many voices/ presets are
    recorded well below the others -- the '- Interview' rips sit ~8-14 dB quieter
    than the audiobook clips -- and CosyVoice/Chatterbox match the reference's
    loudness and SNR, so a quiet reference produces a quiet, hissy clone that reads
    as pure noise. One-pass ffmpeg `loudnorm` to ~-18 LUFS, mono 24 kHz.

    Returns (path_to_use, temp_path_to_delete). On any failure (no ffmpeg, odd
    input) returns the original path with no temp, so cloning still proceeds."""
    if not path:
        return path, None
    import tempfile
    out = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out = tmp.name
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
             "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", "24000", "-ac", "1", out],
            capture_output=True, timeout=60,
        )
        if r.returncode == 0 and os.path.getsize(out) > 1024:
            return out, out
        logger.warning("reference-audio normalisation produced no output (%s); using the clip as-is",
                       (r.stderr or b"").decode("utf-8", "ignore")[:200])
    except Exception as e:
        logger.warning(f"reference-audio normalisation failed ({e}); using the clip as-is")
    if out and os.path.exists(out):
        try:
            os.remove(out)
        except Exception:
            pass
    return path, None


# ---------------------------------------------------------------------------
# STT: runs in stt-serve
# ---------------------------------------------------------------------------
# WhisperX transcription, word alignment, diarization (pyannote in-process there,
# Nemotron in its own worker), speaker matching and live sessions live in stt-serve
# (the `stt` plugin, see docs/plugins.md), started on demand. This gateway
# keeps the request models (validation + API docs) and voices/: audio_path is
# resolved here (inside voices/ only) and sent as base64 -- stt-serve accepts no
# paths -- and the plain transcript is written here as the clip's .txt sidecar.


def _voices_path(path: str) -> str:
    voices_dir_abs = os.path.realpath(_VOICES_DIR)
    target = os.path.realpath(path)
    if os.path.commonpath([voices_dir_abs, target]) != voices_dir_abs or not os.path.isfile(target):
        raise HTTPException(status_code=400, detail="Invalid audio path.")
    return target


@router.post("/api/stt/transcribe", summary="Transcribe audio via WhisperX STT, optionally with speaker diarization")
async def transcribe_stt(req: STTTranscribeRequest):
    """Forwarded to stt-serve's /v1/transcribe (see its docs for diarization,
    speaker matching, language pinning). audio_path must lie inside voices/; its
    plain transcript is written next to it as <name>.txt (a TTS reference
    transcript elsewhere -- hence never speaker-labelled)."""
    body = req.model_dump(exclude={"audio_path"}, exclude_none=True)
    if req.audio_path and not (req.audio_base64 or req.audio_base64_list):
        target = _voices_path(req.audio_path)
        with open(target, "rb") as f:
            body["audio_base64"] = base64.b64encode(f.read()).decode()
    if not (body.get("audio_base64") or body.get("audio_base64_list")):
        raise HTTPException(status_code=400, detail="Must provide audio_base64, audio_base64_list, or audio_path")
    result = await stt_client.call("transcribe", "POST", json=body)
    plain = result.pop("plain_transcript", None)
    if req.audio_path and plain:
        txt_path = os.path.splitext(_voices_path(req.audio_path))[0] + ".txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(plain)
        except OSError as write_err:
            logger.error(f"Failed to write transcript: {write_err}")
    return result


@router.get("/api/stt/diarize/backends", summary="List diarize_backend options and whether each is set up")
async def list_diarize_backends():
    """Derived from the plugin's catalog (the diarize_backend choices) and its status (what is loaded)."""
    view = await stt_client.call("catalog")
    st = await stt_client.status() or {}
    loaded = {w.get("id") for w in st.get("workers", []) if isinstance(w, dict)}
    held = {"whisperx": "stt-serve" in loaded, "nemotron": "nemotron" in loaded}
    backends: Dict[str, Any] = {}
    for e in (view or {}).get("engines", []):
        for f in e.get("fields", []) if isinstance(e, dict) else []:
            if isinstance(f, dict) and f.get("name") == "diarize_backend":
                for c in f.get("choices", []):
                    backends[c["value"]] = {"available": bool(c.get("available")), "loaded": held.get(c["value"], False),
                                            "description": c.get("label", "")}
    return {"backends": backends}


@router.post("/api/stt/unload", summary="Unload WhisperX (and diarization models), freeing VRAM/RAM")
async def unload_stt():
    return await stt_client.release("stt-serve")


@router.post("/api/stt/diarize/unload", summary="Unload just the pyannote diarization pipeline")
async def unload_diarize():
    return await stt_client.release("stt-serve")   # pyannote lives in the same worker as WhisperX


@router.post("/api/stt/diarize/nemotron/unload", summary="Unload the Nemotron-3-Diarization worker")
async def unload_nemotron_diarize_endpoint():
    return await stt_client.release("nemotron")


@router.post("/api/stt/live/start", summary="Start a live STT session (chunked, VAD-based incremental transcription)")
async def live_stt_start(req: LiveSTTStartRequest):
    return await stt_client.call("live", "POST", "/start", json=req.model_dump(exclude_none=True))


@router.post("/api/stt/live/chunk", summary="Append one audio chunk to a live STT session")
async def live_stt_chunk(req: LiveSTTChunkRequest):
    return await stt_client.call("live", "POST", "/chunk", json=req.model_dump())


@router.get("/api/stt/live/poll", summary="Poll a live STT session for newly finalized segments")
async def live_stt_poll(session_id: str, since: int = 0):
    return await stt_client.call("live", "GET", "/poll", params={"session_id": session_id, "since": since})


@router.post("/api/stt/live/stop", summary="Finalize and close a live STT session")
async def live_stt_stop(req: LiveSTTStopRequest):
    return await stt_client.call("live", "POST", "/stop", json=req.model_dump())


@router.patch("/api/stt/live/settings", summary="Update a live STT session's settings (e.g. silence_ms)")
async def live_stt_settings(req: LiveSTTSettingsRequest):
    return await stt_client.call("live", "PATCH", "/settings", json=req.model_dump(exclude_none=True))


@router.get("/api/stt/live/sessions", summary="List active live STT sessions")
async def live_stt_sessions_list():
    return await stt_client.call("live", "GET", "/sessions")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TTS: every engine runs in tts-serve
# ---------------------------------------------------------------------------
# OmniVoice, CosyVoice 3, Chatterbox, VoxCPM2 and Qwen3-TTS live in tts-serve, a
# separate service (the `tts` plugin, see docs/plugins.md) with its own API,
# its own worker lifecycle (on-demand start, keep-alive, eviction among its engines)
# and one provider folder per engine (manifest + adapter + worker + venv). Engine-
# specific request rules live in those adapters. This gateway forwards, keeps
# voices/ (presets, DEFAULT_TTS_VOICE), and stays the GPU owner: tts-serve asks
# POST /api/coordination/gpu-preflight before a cold start.
_TTS_STATUS_SEVERITY = ["error", "starting", "loading", "generating", "idle", "unloaded"]


def _tts_serve_engine(name: str, view: Dict[str, Any]) -> Dict[str, Any]:
    key = name.strip().lower()
    for e in view.get("engines", []):
        if key == e["id"] or key in e.get("aliases", []):
            return e
    known = ", ".join(e["id"] for e in view.get("engines", []))
    raise HTTPException(status_code=400, detail=f"Unknown TTS engine '{name}'. Use one of: {known}.")


@router.post(
    "/api/tts/generate",
    summary="Generate speech (engine-agnostic; the TTS plugin's default unless DEFAULT_TTS_ENGINE or `engine` says otherwise)",
)
async def generate_tts(req: TTSGenerateRequest):
    """Unified TTS entry point. `engine` picks the backend; omit it for the server
    default (DEFAULT_TTS_ENGINE, else the TTS plugin's own). Every engine is served by
    tts-serve, started on demand; unavailable -> 503. A clone engine (tts-serve's
    default_reference) with no reference clip gets DEFAULT_TTS_VOICE and its .txt
    transcript."""
    view = await tts_client.engines() if await tts_client.ensure_running() else None
    if view is None:
        raise HTTPException(status_code=503, detail=tts_client.unavailable_reason())
    name = (req.engine or DEFAULT_TTS_ENGINE or view.get("default_engine") or "").strip().lower()
    engine = _tts_serve_engine(name, view)

    ref_path, temp_path = _resolve_ref_audio(req)
    ref_text = req.ref_text
    if not ref_path and engine.get("default_reference"):
        ref_path, default_txt = _default_ref_voice()
        if ref_text is None:
            ref_text = default_txt
    norm_path, norm_temp = _normalize_ref_audio(ref_path)
    try:
        payload: Dict[str, Any] = {"engine": engine["id"], "text": req.text, "language": req.language,
                                   "response_format": "wav"}
        for key in ("variant", "instruct", "speaker"):
            value = getattr(req, key)
            if value:
                payload[key] = value
        if ref_text:
            payload["ref_text"] = ref_text
        if norm_path:
            with open(norm_path, "rb") as f:
                payload["ref_audio_base64"] = base64.b64encode(f.read()).decode()
        params = dict(req.params or {})
        params.update({k: getattr(req, k) for k in ("temperature", "exaggeration", "cfg_weight")
                       if getattr(req, k) is not None})
        if params:
            payload["params"] = params
        # Only an explicit keep_alive is forwarded; otherwise the engine's own
        # default from its provider manifest applies (VoxCPM: 360 s, others 60 s).
        if "keep_alive" in req.model_fields_set and req.keep_alive is not None:
            payload["keep_alive"] = req.keep_alive
        wav = await tts_client.generate(payload)
    finally:
        for p_ in (temp_path, norm_temp):
            if p_ and os.path.exists(p_):
                try:
                    os.remove(p_)
                except OSError:
                    pass
    if req.response_format == "binary":
        return Response(content=wav, media_type="audio/wav")
    return {"status": "success", "audio_base64": base64.b64encode(wav).decode("utf-8")}


async def tts_status_snapshot() -> Dict[str, Any]:
    """Engine-agnostic view of tts-serve's engines. `status` is the worst engine
    state (error > starting > loading > generating > idle > unloaded); 'unloaded'
    is fine -- engines lazy-load on the next call. Doesn't start tts-serve."""
    view = await tts_client.engines()
    engines = {e["id"]: {k: e.get(k) for k in ("status", "loaded", "variant", "device", "pid")}
               for e in (view or {}).get("engines", [])}
    agg = min((e["status"] for e in engines.values()),
              key=lambda s: _TTS_STATUS_SEVERITY.index(s) if s in _TTS_STATUS_SEVERITY else len(_TTS_STATUS_SEVERITY),
              default="unloaded")
    return {
        "status": agg,
        "ready": agg not in ("starting", "loading", "error"),
        "busy": agg in ("starting", "loading", "generating"),
        "default_engine": DEFAULT_TTS_ENGINE or (view or {}).get("default_engine"),
        "engines": engines,
        "tts_serve": {"available": view is not None, "plugin": getattr(tts_client.plugin(), "id", None)},
    }


@router.get("/api/tts/status", summary="Engine-agnostic TTS status (tts-serve's engines)")
async def get_tts_status():
    return await tts_status_snapshot()


@router.post("/api/tts/unload", summary="Unload OmniVoice (legacy path; same as /api/tts/omnivoice/unload)")
async def unload_tts(force: bool = False):
    return await tts_client.unload("omnivoice", force=force)


@router.post("/api/tts/{engine}/unload", summary="Unload a tts-serve engine (409 while busy)")
async def unload_tts_serve_engine(engine: str, force: bool = False):
    return await tts_client.unload(engine, force=force)


@router.post(
    "/api/tts/qwen3/generate",
    summary="DEPRECATED -- use POST /api/tts/generate with engine='qwen3'.",
    deprecated=True,
)
async def generate_qwen3_tts(req: Qwen3TTSGenerateRequest):
    fields = dict(text=req.text, engine="qwen3", variant=req.variant, language=req.language,
                  ref_audio_base64=req.ref_audio_base64, ref_voice_path=req.ref_voice_path, ref_text=req.ref_text,
                  instruct=req.instruct, speaker=req.speaker, response_format=req.response_format)
    if "keep_alive" in req.model_fields_set:
        fields["keep_alive"] = req.keep_alive
    return await generate_tts(TTSGenerateRequest(**fields))


# ---------------------------------------------------------------------------
# OpenAI audio compatibility
# ---------------------------------------------------------------------------

@router.post("/v1/audio/speech", summary="OpenAI-compatible speech generation")
async def openai_audio_speech(req: OpenAISpeechRequest):
    ref_voice_path = None
    instruct = None
    if os.path.exists(_VOICES_DIR) and req.voice:
        # Audio files only (the .txt transcript sidecars share the stem and used to win
        # whenever listdir() returned them first -> "Format not recognised" 500s); an
        # exact stem/filename match beats a mere prefix match.
        want = req.voice.lower()
        audio = sorted(f for f in os.listdir(_VOICES_DIR)
                       if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg"))
        exact = [f for f in audio if f.lower() == want or os.path.splitext(f)[0].lower() == want]
        prefix = [f for f in audio if f.lower().startswith(want)]
        if exact or prefix:
            ref_voice_path = os.path.join(_VOICES_DIR, (exact or prefix)[0])
    # OpenAI's 6 built-in voice names carry no meaning here -- treat them as "no voice
    # given" rather than as a style instruction / preset name.
    _OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "coral", "sage"}
    if not ref_voice_path and req.voice and req.voice.strip().lower() not in _OPENAI_VOICES:
        instruct = req.voice
    # OpenAI's `model` field selects the engine: an engine id or alias (exactly, or contained in it -- "qwen-tts",
    # "chatterbox-mlx" ...); anything else (incl. "tts-1") uses the server default.
    engine = await _speech_engine(req.model)
    eng = (engine or "").strip().lower()
    variant = None
    ref_text = None
    speaker, speaker_mode = await _engine_speaker(engine, req.voice)
    if speaker:   # `voice` names one of the engine's own speakers (Piper's voices, Qwen3's presets ...)
        ref_voice_path, instruct, variant = None, None, speaker_mode
    if ref_voice_path:
        # Transcript from the preset's .txt sidecar, for every engine: CosyVoice
        # zero_shot and Qwen3 base REQUIRE it (Qwen3 400s without), OmniVoice uses
        # it instead of auto-transcribing, Chatterbox ignores it.
        _t = os.path.splitext(ref_voice_path)[0] + ".txt"
        ref_text = open(_t, encoding="utf-8").read().strip() if os.path.isfile(_t) else None
    if eng in ("cosyvoice", "cosy", "cosyvoice3"):
        # A `voice` that matched a preset -> clone it (zero_shot, transcript from its
        # .txt sidecar). A free-form `voice` -> that's a style description: instruct2
        # on the default reference. Nothing -> zero_shot on the default reference.
        if ref_voice_path:
            variant = "zero_shot"
        else:
            variant = "instruct" if instruct else "zero_shot"
    elif eng == "qwen3" and instruct:
        # A bare `voice` description -> Qwen3's VoiceDesign checkpoint (free-form),
        # not the reference-clone default.
        variant = "voice_design"
    tts_req = TTSGenerateRequest(
        text=req.input, language=(req.language or "en"), engine=engine, variant=variant,
        ref_voice_path=ref_voice_path, ref_text=ref_text if not speaker else None, instruct=instruct,
        speaker=speaker, response_format="binary",
    )
    return await generate_tts(tts_req)


async def _engine_speaker(engine: Optional[str], voice: Optional[str]):
    """(speaker, mode variant) when `voice` is one of the engine's own speakers per its catalog, else (None, None)."""
    if not voice:
        return None, None
    view = await tts_client.engines() if await tts_client.ensure_running() else None
    name = (engine or DEFAULT_TTS_ENGINE or (view or {}).get("default_engine") or "").strip().lower()
    want = voice.strip().lower()
    for e in (view or {}).get("engines", []):
        if e.get("id") != name and name not in (e.get("aliases") or []):
            continue
        for m in e.get("modes") or []:
            for sp in m.get("speakers") or []:
                if isinstance(sp, str) and sp.lower() == want:
                    return sp, m.get("variant")
    return None, None


async def _transcription_engine(model: Optional[str]) -> Optional[str]:
    """/v1/audio/transcriptions' `model` -> an engine id of the transcription plugins, or None (= the default; e.g.
    OpenAI's "whisper-1"). Exact id/alias first, then the first engine whose id or alias the name contains
    ("moss-transcribe" -> moss)."""
    ml = (model or "").strip().lower()
    view = await stt_client.engines() if ml and await stt_client.ensure_running() else None
    engines = (view or {}).get("engines", [])
    for e in engines:
        if ml == e["id"] or ml in (e.get("aliases") or []):
            return e["id"]
    for e in engines:
        if any(n and n in ml for n in [e["id"], *(e.get("aliases") or [])]):
            return e["id"]
    return None


async def _speech_engine(model: Optional[str]) -> Optional[str]:
    """/v1/audio/speech's `model` -> an engine id of the TTS plugin, or None (= its default). Exact id/alias first,
    then the first engine whose id or alias the name contains (e.g. "qwen-tts" -> qwen3 via alias "qwen")."""
    ml = (model or "").strip().lower()
    if not ml:
        return DEFAULT_TTS_ENGINE or None
    view = await tts_client.engines() if await tts_client.ensure_running() else None
    engines = (view or {}).get("engines", [])
    for e in engines:
        if ml == e["id"] or ml in (e.get("aliases") or []):
            return e["id"]
    for e in engines:
        if any(n and n in ml for n in [e["id"], *(e.get("aliases") or [])]):
            return e["id"]
    return DEFAULT_TTS_ENGINE or None


def _subtitle_ts(seconds: float, sep: str) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def openai_transcription_response(result: Dict[str, Any], response_format: str,
                                  language: Optional[str]):
    """transcribe_stt()'s result in OpenAI's response_format shapes: json (default),
    text, verbose_json, srt, vtt. Nextcloud's OpenAI integration asks for verbose_json
    by default and returns the raw body for srt/vtt."""
    fmt = (response_format or "json").lower()
    text = result["transcript"]
    segments = result.get("segments") or []
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt in ("srt", "vtt"):
        sep = "," if fmt == "srt" else "."
        blocks = []
        for i, seg in enumerate(segments, 1):
            span = f"{_subtitle_ts(seg['start'], sep)} --> {_subtitle_ts(seg['end'], sep)}"
            blocks.append(f"{i}\n{span}\n{seg['text'].strip()}" if fmt == "srt"
                          else f"{span}\n{seg['text'].strip()}")
        body = "\n\n".join(blocks) + "\n"
        if fmt == "vtt":
            body = "WEBVTT\n\n" + body
        return PlainTextResponse(body, media_type="text/vtt" if fmt == "vtt" else "text/plain")
    if fmt == "verbose_json":
        out: Dict[str, Any] = {
            "task": "transcribe",
            "text": text,
            "segments": [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                         for i, s in enumerate(segments)],
        }
        if language:
            out["language"] = language
        if result.get("duration_s") is not None:
            out["duration"] = result["duration_s"]
        return out
    if fmt != "json":
        raise HTTPException(status_code=400, detail=f"Unsupported response_format '{response_format}' "
                            "(json, text, verbose_json, srt, vtt)")
    return {"text": text}


@router.post("/v1/audio/transcriptions", summary="OpenAI-compatible transcription")
async def openai_audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    """Same transcription as POST /api/stt/transcribe, wrapped in the OpenAI
    multipart-upload shape. `model` picks a transcription engine by id or alias (e.g. `moss-transcribe` ->
    moss); anything else (OpenAI's `whisper-1`) uses the default. `file` accepts video too, not just audio -- delegates to
    transcribe_stt(), whose ffmpeg-backed decoding content-sniffs regardless of container.
    `language` is forwarded (auto-detect is known-unreliable, see CLAUDE.md's STT notes).
    """
    content = await file.read()
    audio_base64 = base64.b64encode(content).decode("utf-8")
    engine = await _transcription_engine(model)
    stt_req = STTTranscribeRequest(audio_base64=audio_base64, language=language or None, engine=engine)
    result = await transcribe_stt(stt_req)
    return openai_transcription_response(result, response_format, language)


# ---------------------------------------------------------------------------
# ComfyUI management
# ---------------------------------------------------------------------------

@router.get("/v1/audio/models", summary="Speech engines (text-to-speech and transcription) with their modes and options")
async def openai_audio_models():
    """What tts-serve and stt-serve offer, as they describe it themselves (backend/audio_catalog.py).
    Starts either service if needed; one that stays down is reported with a reason."""
    from backend.audio_catalog import audio_catalog
    tts_view = await tts_client.engines() if await tts_client.ensure_running() else None
    stt_view = None
    if await stt_client.ensure_running():
        try:
            stt_view = await stt_client.call("catalog")
        except HTTPException:
            stt_view = None
    return audio_catalog(tts_view, stt_view, default_voice=DEFAULT_TTS_VOICE)


@router.get("/v1/images/models", summary="Image models from the image plugins (ComfyUI, MOSS, Iris ...) for apps to pick from")
async def image_models(task: str = "generate"):
    """task: generate (default) | edit | style | all. Every image model comes from an image plugin
    (backend/plugins/images.py); `default` is the plugin's suggestion for that task."""
    from backend.plugins import images as image_plugins
    if task not in ("generate", "edit", "style", "all"):
        raise HTTPException(status_code=400, detail="task: generate | edit | style | all")
    entries, errors = await image_plugins.catalog()
    rows = [image_plugins.row(e) for e in entries if task == "all" or task in e["tasks"]]
    default = next((r["id"] for r in rows if r.get("default") and r.get("available")), rows[0]["id"] if rows else None)
    out = {"object": "list", "default": default, "data": rows}
    if errors:
        out["errors"] = errors
    return out


def _plugin_answer(r) -> Dict[str, Any]:
    """An image plugin's answer: its JSON on success, its status and message otherwise."""
    try:
        data = r.json()
    except ValueError:
        data = {"detail": r.text[:500]}
    if r.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        raise HTTPException(status_code=r.status_code if r.status_code in (400, 404, 409, 422, 429, 503, 507) else 502,
                            detail=detail if isinstance(detail, str) else f"image plugin answered {r.status_code}")
    return data


async def _image_target(model: Optional[str], task: str):
    """(plugin, entry) for `model`; an unknown or missing name gets the default model for `task` (OpenAI clients
    send e.g. dall-e-3 -- that has always meant "the default")."""
    from backend.plugins import images as image_plugins
    found = await image_plugins.find(model)
    if found is not None:
        if task not in found[1]["tasks"]:
            verb = "edits images; use /v1/images/edits" if task == "generate" else f"can't do {task}"
            raise HTTPException(status_code=400, detail=f"{found[1]['label']} {verb}")
        return found
    entries, _ = await image_plugins.catalog()
    usable = [e for e in entries if task in e["tasks"] and e.get("available")]
    entry = next((e for e in usable if e.get("default")), usable[0] if usable else None)
    if entry is None:
        raise HTTPException(status_code=503, detail="No image plugin is set up or running (see Apps & Services).")
    return await image_plugins.find(entry["id"])


@router.post("/v1/images/generations", summary="OpenAI-compatible image generation (routed to the image plugin of the model)")
async def openai_image_generation(req: ImageGenerationRequest, x_request_id: Optional[str] = Header(None)):
    from backend.plugins import images as image_plugins
    rid = x_request_id if x_request_id and re.fullmatch(r"[A-Za-z0-9_-]{16,64}", x_request_id) else None
    p, entry = await _image_target(req.model, "generate")
    # only what the caller sent: the schema's defaults (size 1024x1024, response_format url) aren't theirs
    body = {**req.model_dump(exclude_unset=True, exclude_none=True), "model": entry["id"]}
    try:
        r = await image_plugins.forward(p, "generate", request_id=rid, json=body)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return _plugin_answer(r)


class ImageCancelRequest(BaseModel):
    request_id: str


@router.post("/v1/images/cancel", summary="Stop one running image generation (the one sent with this X-Request-Id)")
async def cancel_image_generation(req: ImageCancelRequest):
    from backend.plugins import images as image_plugins
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", req.request_id or ""):
        raise HTTPException(status_code=400, detail="request_id: 16-64 of A-Z, a-z, 0-9, _ and -")
    return {"cancelled": await image_plugins.cancel(req.request_id)}


@router.post("/v1/images/edits", summary="OpenAI-compatible image editing (routed to the image plugin of the model)")
async def openai_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),   # extension: same seed + image + prompt = same result (Figurenwerkstatt)
    task: str = Form("edit"),           # extension: edit | style (style: `reference` = the style image)
    reference: Optional[UploadFile] = File(None),
    x_request_id: Optional[str] = Header(None),
):
    from backend.plugins import images as image_plugins
    if seed is not None and not 0 <= seed < 2 ** 50:
        raise HTTPException(status_code=400, detail="seed must be 0 .. 2^50")
    if task not in ("edit", "style"):
        raise HTTPException(status_code=400, detail="task: edit | style")
    p, entry = await _image_target(model, task)
    fields = {"prompt": prompt, "model": entry["id"], "task": task, **({"seed": str(seed)} if seed is not None else {})}
    files = {"image": (image.filename or "image.png", await image.read(), image.content_type or "application/octet-stream")}
    if reference is not None:
        files["reference"] = (reference.filename or "reference.png", await reference.read(),
                              reference.content_type or "application/octet-stream")
    rid = x_request_id if x_request_id and re.fullmatch(r"[A-Za-z0-9_-]{16,64}", x_request_id) else None
    try:
        r = await image_plugins.forward(p, "edit", request_id=rid, data=fields, files=files)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return _plugin_answer(r)
