"""Music and sound effects through ComfyUI -- the `audio` capability (ai-serve-ui docs/plugins.md "audio" and
"Long work: jobs"), added 2026-10-30 (owner: "bau das mit SFX und ACE-Step"; V-Noveller wants background music per
location and sound effects).

Models:
- `sa3-sfx`  Stable Audio 3 Small SFX (Stability AI Community License: free below $1M revenue, outputs are yours):
             sound effects and ambiences, 1-120 s. English prompts -- with `enhance` (default on) a chat model behind
             the gateway first turns the caller's words (German too) into the dense English sound description the
             model was trained on (Comfy-Org's own SFX reprompt, which also suggests a length).
- `ace-step` ACE-Step 1.5 Turbo (MIT): songs with lyrics in many languages, or instrumental, 10-300 s. `prompt` is
             the style ("tags"), `lyrics` the words ([Verse]/[Chorus] sections); no lyrics = instrumental.
Both graphs are Comfy-Org's templates (audio_stable_audio_3_medium.json without its Qwen subgraph,
audio_ace_step_1_5_split.json) in API form; 8 steps each.

Jobs: POST -> 202 {id}; the job queues its workflow (ComfyUI started and the GPU asked like for every image) and waits;
GET shows status/progress; cancel removes or interrupts exactly its prompt. Results are MP3 files kept in
GENERATED_AUDIO_DIR (never deleted by this plugin); job records live in memory for JOB_TTL_S.
Pure parts (catalog, request checks, workflows, enhance parsing) are tested in tests/test_audio.py."""
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("image-serve.audio")

TASKS = ("music", "sound")
JOB_TTL_S = 6 * 3600
_LANGS = ('ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'he', 'hi', 'hr', 'ht',
          'hu', 'id', 'is', 'it', 'ja', 'ko', 'la', 'lt', 'ms', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sa',
          'sk', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'yue', 'zh', 'unknown')
_KEYS = tuple(f"{r} {q}" for q in ("major", "minor")
              for r in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"))


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class AudioModel:
    id: str
    label: str
    task: str
    files: Tuple[Tuple[str, str], ...]     # (models-root subdir, file)
    description: str
    license: str
    min_seconds: int
    max_seconds: int
    default_seconds: int
    bundle: str
    lyrics: bool = False


CATALOG: Tuple[AudioModel, ...] = (
    AudioModel("sa3-sfx", "Stable Audio 3 Small SFX", "sound",
               (("checkpoints", "stable_audio_3_small_sfx.safetensors"), ("text_encoders", "t5gemma_b_b_ul2.safetensors")),
               "Sound effects and ambiences from a description: rain on a tin roof, a creaking door, a busy tavern. "
               "Any language -- it is turned into an English sound description first.",
               "Stability AI Community License (free below $1M annual revenue)", 1, 120, 8, "sa3-sfx"),
    AudioModel("ace-step", "ACE-Step 1.5 Turbo", "music",
               (("diffusion_models", "acestep_v1.5_turbo.safetensors"), ("text_encoders", "qwen_0.6b_ace15.safetensors"),
                ("text_encoders", "qwen_1.7b_ace15.safetensors"), ("vae", "ace_1.5_vae.safetensors")),
               "Songs and background music from a style description, with lyrics (many languages, German too) or "
               "instrumental.", "MIT", 10, 300, 60, "ace-step", lyrics=True),
)
DEFAULTS = {"sound": "sa3-sfx", "music": "ace-step"}


def resolve(name: Optional[str]) -> Optional[AudioModel]:
    key = (name or "").strip().lower()
    return next((m for m in CATALOG if m.id == key), None)


def available(m: AudioModel, models_root: str) -> bool:
    return all(os.path.isfile(os.path.join(models_root, d, f)) for d, f in m.files)


def listing(models_root: str) -> List[Dict[str, Any]]:
    return [{"id": m.id, "label": m.label, "description": m.description, "tasks": [m.task],
             "max_seconds": m.max_seconds, "min_seconds": m.min_seconds, "default_seconds": m.default_seconds,
             "lyrics": m.lyrics, "license": m.license, "available": available(m, models_root),
             "default": DEFAULTS[m.task] == m.id, "bundle": m.bundle} for m in CATALOG]


# ------------------------------------------------------------------ requests

@dataclass
class AudioRequest:
    model: AudioModel
    prompt: str
    seconds: Optional[float]
    seed: int
    lyrics: str = ""
    enhance: bool = True
    bpm: int = 120
    language: str = "en"
    keyscale: str = "C major"
    timesignature: str = "4"


def _num(v: Any, name: str, lo: float, hi: float, integer: bool = False) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AudioError(f"{name} must be a number")
    if not lo <= v <= hi:
        raise AudioError(f"{name}: {lo:g}..{hi:g}")
    return int(v) if integer else float(v)


def parse_request(body: Any) -> AudioRequest:
    """The jobs body ({model?, task?, prompt, lyrics?, seconds?, seed?} + bpm, language, keyscale, timesignature,
    enhance) -> a checked request. A missing model takes the default of `task` (default: music)."""
    if not isinstance(body, dict):
        raise AudioError("the body must be a JSON object")
    task = body.get("task")
    if task is not None and task not in TASKS:
        raise AudioError(f"task: {' | '.join(TASKS)}")
    model = resolve(body.get("model")) if body.get("model") else resolve(DEFAULTS[task or "music"])
    if model is None:
        raise AudioError(f"unknown audio model {body.get('model')!r} -- {', '.join(m.id for m in CATALOG)}")
    if task is not None and model.task != task:
        raise AudioError(f"{model.label} makes {model.task}, not {task}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AudioError("prompt: describe the sound or the music")
    if len(prompt) > 2000:
        raise AudioError("prompt: at most 2000 characters")
    lyrics = body.get("lyrics") or ""
    if not isinstance(lyrics, str) or len(lyrics) > 6000:
        raise AudioError("lyrics: text, at most 6000 characters")
    if lyrics and not model.lyrics:
        raise AudioError(f"{model.label} takes no lyrics")
    seconds = _num(body.get("seconds"), "seconds", model.min_seconds, model.max_seconds)
    seed = _num(body.get("seed"), "seed", 0, 2 ** 53, integer=True)
    enhance = body.get("enhance", True)
    if not isinstance(enhance, bool):
        raise AudioError("enhance: true or false")
    bpm = _num(body.get("bpm"), "bpm", 40, 220, integer=True)
    language = body.get("language") or "en"
    if language not in _LANGS:
        raise AudioError("language: a short code such as de, en, fr")
    keyscale = body.get("keyscale") or "C major"
    if keyscale not in _KEYS:
        raise AudioError("keyscale: e.g. 'E minor', 'F# major'")
    ts = str(body.get("timesignature") or "4")
    if ts not in ("2", "3", "4", "6"):
        raise AudioError("timesignature: 2 | 3 | 4 | 6")
    return AudioRequest(model, prompt.strip(), seconds, int(seed) if seed is not None else random.randint(1, 2 ** 31),
                        lyrics.strip(), enhance, int(bpm or 120), language, keyscale, ts)


# ------------------------------------------------------------------ enhance (sound prompts)

SFX_SYSTEM = (
    "You are a professional sound design expert. Convert the user's input (any language) into a precise, vivid "
    "English sound effects description suitable for generative audio models.\n\n"
    "Describe clearly: sound source; physical character (texture, timbre, material); spatial qualities "
    "(indoor/outdoor, dry/reverberant, close/distant); temporal evolution (attack, decay, movement); motion if any.\n\n"
    "Audio length rules: very short sounds (impacts, clicks, gunshots) 1-3 seconds; medium actions (footsteps, "
    "object movement) 3-6 seconds; ambience / environments 6-15 seconds.\n\n"
    "Output: 1-2 dense sentences in English, nothing else -- no explanations, no quotes -- and always append "
    "'Length: X seconds' (integer).\n\n"
    "Examples:\n"
    "- Heavy rain hitting a metal roof during a thunderstorm, distant thunder rumbles, stereo, realistic ambience. "
    "Length: 45 seconds\n"
    "- Wooden door creaking open slowly in an old house, echoing interior, eerie tone. Length: 3 seconds\n"
    "- Footsteps on gravel, steady walking pace, close perspective. Length: 8 seconds\n"
    "- Glass bottle shattering on concrete, sharp impact, scattered fragments. Length: 2 seconds\n"
    "- Busy medieval tavern at night, murmuring crowd, clinking mugs, crackling fireplace, distant lute. "
    "Length: 15 seconds")

_LENGTH = re.compile(r"\s*Length:\s*(\d{1,3})\s*seconds?\.?\s*$", re.IGNORECASE)


def parse_enhanced(answer: Any, fallback: str) -> Tuple[str, Optional[int]]:
    """The chat model's rewrite -> (prompt, suggested seconds). Anything unusable -> the caller's own prompt."""
    if not isinstance(answer, str):
        return fallback, None
    text = re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip().strip('"').strip()
    if not text or len(text) > 1500:
        return fallback, None
    m = _LENGTH.search(text)
    secs = int(m.group(1)) if m else None
    text = _LENGTH.sub("", text).strip()
    return (text or fallback), secs


async def enhance_sound(prompt: str, gateway: str, model: str, timeout: float = 120.0) -> Tuple[str, Optional[int]]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{gateway}/v1/chat/completions", json={
                "model": model, "max_tokens": 200, "temperature": 0.7, "thinking": False,
                "messages": [{"role": "system", "content": SFX_SYSTEM}, {"role": "user", "content": prompt}]})
        if r.status_code != 200:
            logger.warning(f"sound prompt not enhanced: the gateway answered {r.status_code}")
            return prompt, None
        return parse_enhanced(r.json()["choices"][0]["message"].get("content"), prompt)
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
        logger.warning(f"sound prompt not enhanced: {type(e).__name__}")
        return prompt, None


# ------------------------------------------------------------------ workflows (ComfyUI API form)

def sa3_workflow(prompt: str, seconds: float, seed: int, prefix: str) -> Dict[str, Any]:
    ckpt, te = (f for _, f in resolve("sa3-sfx").files)
    return {
        "25": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "26": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "stable_audio", "device": "default"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["26", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["26", 0], "text": ""}},
        "11": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": float(seconds), "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["25", 0], "positive": ["6", 0], "negative": ["7", 0],
                                                    "latent_image": ["11", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                                                    "sampler_name": "lcm", "scheduler": "simple", "denoise": 1.0}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["25", 2]}},
        "57": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["12", 0], "filename_prefix": prefix, "quality": "V0"}},
    }


def ace_workflow(r: AudioRequest, seconds: float, prefix: str) -> Dict[str, Any]:
    unet, te1, te2, vae = (f for _, f in resolve("ace-step").files)
    return {
        "104": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "105": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": te1, "clip_name2": te2, "type": "ace",
                                                           "device": "default"}},
        "106": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "94": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip": ["105", 0], "tags": r.prompt, "lyrics": r.lyrics or "[Instrumental]", "seed": r.seed,
            "bpm": r.bpm, "duration": float(seconds), "timesignature": r.timesignature, "language": r.language,
            "keyscale": r.keyscale, "generate_audio_codes": True, "cfg_scale": 2.0, "temperature": 0.85,
            "top_p": 0.9, "top_k": 0, "min_p": 0.0}},
        "47": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["94", 0]}},
        "98": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": float(seconds), "batch_size": 1}},
        "78": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["104", 0], "shift": 3.0}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["78", 0], "positive": ["94", 0], "negative": ["47", 0],
                                                    "latent_image": ["98", 0], "seed": r.seed, "steps": 8, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "18": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["106", 0]}},
        "107": {"class_type": "SaveAudioMP3", "inputs": {"audio": ["18", 0], "filename_prefix": prefix, "quality": "V0"}},
    }


# ------------------------------------------------------------------ jobs

@dataclass
class Job:
    id: str
    model: str
    created: float = field(default_factory=time.time)
    status: str = "queued"           # queued | running | done | failed | cancelled
    message: str = ""
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    file: Optional[str] = None       # path of the result
    prompt_used: Optional[str] = None
    seconds: Optional[float] = None
    seed: Optional[int] = None
    task: Optional[asyncio.Task] = None


_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Jobs:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def new(self, model: str) -> Job:
        now = time.time()
        for k in [k for k, j in self._jobs.items() if now - j.created > JOB_TTL_S and j.status != "running"]:
            del self._jobs[k]
        job = Job(id="aud-" + secrets.token_hex(12), model=model)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id) if isinstance(job_id, str) and _ID.match(job_id) else None


JOBS = Jobs()


def view(job: Job, progress: Optional[float], jobs_path: str) -> Dict[str, Any]:
    """A job in the contract's status shape; the result file is only a path under this plugin's jobs endpoint."""
    out: Dict[str, Any] = {"id": job.id, "status": job.status, "progress": progress, "message": job.message,
                           "error": job.error}
    if job.status == "done" and job.file:
        name = os.path.basename(job.file)
        out["result"] = {"files": [{"name": name, "type": "audio/mpeg", "url": f"{jobs_path}/{job.id}/files/{name}"}],
                         "data": {"model": job.model, "prompt": job.prompt_used, "seconds": job.seconds,
                                  "seed": job.seed}}
    return out
