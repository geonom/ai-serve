"""stt-serve engine: WhisperX transcription, word alignment, speaker diarization
(pyannote in-process, or Nemotron in its own worker), speaker matching and live
sessions. Moved from ai-serve-ui's backend/routers/media.py with these changes:
audio only as base64 (no paths), the Strata guard is asked of the gateway
(coordination.py) instead of imported, routes live under /v1, and the plain
transcript is returned for the caller to store. Rules: domain.py."""
import asyncio
import base64
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException

from stt_serve import coordination
from stt_serve.coordination import preflight, preflight_sync, release_sync, reserve_sync, whisper_need_mb
from stt_serve.domain import (RELIABLE_EMBEDDING_MIN_SECONDS, apply_speaker_matches, embedding_windows, match_known_speakers,
                              mean_speaker_embeddings, rms_trailing_silence_s, speaker_turns)
from stt_serve.models import (
    KnownSpeaker,
    LiveSTTChunkRequest,
    LiveSTTSettingsRequest,
    LiveSTTStartRequest,
    LiveSTTStopRequest,
    STTTranscribeRequest,
)
from stt_serve.state import state

logger = logging.getLogger("stt-serve")
router = APIRouter()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.getenv("STT_SERVE_LOGS_DIR", os.path.join(ROOT, "logs"))
MOSS_TRANSCRIBE_PYTHON_BIN = os.getenv("MOSS_TRANSCRIBE_PYTHON_BIN", os.path.join(ROOT, "moss", "venv", "bin", "python"))
NEMOTRON_DIARIZE_PYTHON_BIN = os.getenv("NEMOTRON_DIARIZE_PYTHON_BIN", os.path.join(ROOT, "nemotron", "venv", "bin", "python"))
DIARIZE_MODEL = os.getenv("DIARIZE_MODEL", "")
# WhisperX model: a size (tiny, base, small, medium, large-v3, turbo ...), a Hugging Face repo of a CTranslate2
# conversion, or a local folder -- whatever whisperx.load_model takes. Was always "small" (until 2026-10-22).
WHISPER_MODEL = os.getenv("STT_WHISPER_MODEL") or "small"
HF_TOKEN = os.getenv("HF_TOKEN", "")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _release_cuda_memory() -> None:
    """Hand VRAM of an unloaded in-process model back to the driver. gc + empty_cache
    alone isn't enough: torch allocates its cuBLAS workspace (~8 MiB) from the caching
    allocator, and that small live block pins the whole segment it landed in --
    measured after an OmniVoice unload (when it still ran in this process): 8 MiB allocated, 766 MiB reserved and
    unreleasable. Clearing the workspaces first lets empty_cache() free it all
    (924 -> 158 MiB, i.e. back to the bare CUDA context). cuBLAS re-creates the
    workspace lazily on its next use."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _drop_whisperx_model() -> None:
    """The model out of memory (no keep-alive change) -- also between a failed GPU try and the next one."""
    if state.whisperx_model is not None:
        state.whisperx_model = None
        state.whisperx_device = None
        _release_cuda_memory()


def _unload_whisperx():
    if state.whisperx_keep_alive_task:
        state.whisperx_keep_alive_task.cancel()
        state.whisperx_keep_alive_task = None
    release_sync(state.whisperx_reservation)
    state.whisperx_reservation = None
    if state.whisperx_model is not None:
        _drop_whisperx_model()
        state.whisperx_status = "idle"
        logger.info("WhisperX model unloaded.")


def _reset_whisperx_keep_alive(keep_alive_seconds: int):
    if state.whisperx_keep_alive_task:
        state.whisperx_keep_alive_task.cancel()

    async def _check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            while state.whisperx_status == "transcribing":
                await asyncio.sleep(5)
            logger.info(f"Keep-alive expired for WhisperX ({keep_alive_seconds}s idle). Unloading...")
            _unload_whisperx()
        except asyncio.CancelledError:
            pass

    state.whisperx_keep_alive_task = asyncio.create_task(_check())


def _unload_diarize():
    if state.diarize_keep_alive_task:
        state.diarize_keep_alive_task.cancel()
        state.diarize_keep_alive_task = None
    if state.diarize_pipeline is not None or state.align_model is not None:
        was_cuda = bool(state.diarize_device and state.diarize_device.startswith("cuda"))
        state.diarize_pipeline = None
        state.diarize_device = None
        state.diarize_status = "idle"
        state.align_model = None
        state.align_metadata = None
        state.align_language = None
        if was_cuda:
            _release_cuda_memory()
        logger.info("Diarization pipeline unloaded.")


def _reset_diarize_keep_alive(keep_alive_seconds: int):
    if state.diarize_keep_alive_task:
        state.diarize_keep_alive_task.cancel()

    async def _check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            while state.diarize_status == "diarizing":
                await asyncio.sleep(5)
            logger.info(f"Keep-alive expired for diarization ({keep_alive_seconds}s idle). Unloading...")
            _unload_diarize()
        except asyncio.CancelledError:
            pass

    state.diarize_keep_alive_task = asyncio.create_task(_check())


@dataclass
class _WorkerCfg:
    name: str            # human label, e.g. "CosyVoice"
    prefix: str          # state.<prefix>_process / _port / _pid / _status / ...
    python_bin: str
    script: str          # abs path to the *_server.py
    log_name: str        # file under LOGS_DIR
    setup_hint: str
    extra_env: Optional[Dict[str, str]] = None
    startup_timeout: float = 60.0


_worker_locks: Dict[str, asyncio.Lock] = {}


def _worker_lock(prefix: str) -> asyncio.Lock:
    lock = _worker_locks.get(prefix)
    if lock is None:
        lock = _worker_locks[prefix] = asyncio.Lock()
    return lock


def _unload_worker(prefix: str, name: str) -> None:
    task = getattr(state, f"{prefix}_keep_alive_task", None)
    if task:
        task.cancel()
        setattr(state, f"{prefix}_keep_alive_task", None)
    proc = getattr(state, f"{prefix}_process", None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        logger.info(f"{name} worker stopped.")
    release_sync(getattr(state, f"{prefix}_reservation", None))
    setattr(state, f"{prefix}_reservation", None)
    log_file = getattr(state, f"{prefix}_log_file", None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass
    for attr, val in (
        (f"{prefix}_process", None), (f"{prefix}_port", None), (f"{prefix}_pid", None),
        (f"{prefix}_log_file", None), (f"{prefix}_variant", None), (f"{prefix}_device", None),
        (f"{prefix}_status", "idle"),
    ):
        setattr(state, attr, val)


def _worker_purpose(cfg: _WorkerCfg) -> str:
    return f"stt:{cfg.prefix}"


def _worker_env(grant: Optional["coordination.Grant"]) -> Dict[str, str]:
    """The worker sees only the GPU the manager picked (it uses cuda:0 = that one); no grant = the CPU;
    a grant without a GPU (gateway unreachable, standalone) leaves the default."""
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if grant is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif grant.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(grant.gpu)
    return env


async def _ensure_worker(cfg: _WorkerCfg, cpu_only: bool = False) -> int:
    """Start the worker if needed -- on the GPU the GPU manager picks (tests/test_gpu_robust.py), on the CPU when it
    says no (or cpu_only). The reservation is bound to the worker's PID and kept until its first job is done
    (_finish_worker_first_use)."""
    async with _worker_lock(cfg.prefix):
        proc = getattr(state, f"{cfg.prefix}_process", None)
        if proc is not None and proc.poll() is None:
            return getattr(state, f"{cfg.prefix}_port")

        if not os.path.exists(cfg.python_bin):
            raise HTTPException(
                status_code=500,
                detail=f"{cfg.name} python interpreter not found at {cfg.python_bin}. {cfg.setup_hint}",
            )

        grant = None
        if not cpu_only:
            try:
                grant = await coordination.reserve(_worker_purpose(cfg), coordination.NEEDS_MB.get(cfg.prefix, 4000))
            except HTTPException as e:
                logger.warning(f"{cfg.name}: no room on the GPUs ({e.detail}) -- starting it on the CPU")
        port = _free_port()
        log_path = os.path.join(LOGS_DIR, cfg.log_name)
        log_file = open(log_path, "w", encoding="utf-8")
        env = _worker_env(grant)
        if cfg.extra_env:
            env.update(cfg.extra_env)
        where = "the CPU" if grant is None else (f"GPU {grant.gpu}" if grant.gpu is not None else "the default GPU")
        logger.info(f"Starting {cfg.name} worker on port {port} ({where})...")
        proc = subprocess.Popen(
            [cfg.python_bin, cfg.script, "--port", str(port)],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        if grant is not None:
            await coordination.confirm(grant.rid, proc.pid)
        setattr(state, f"{cfg.prefix}_reservation", grant.rid if grant else None)
        setattr(state, f"{cfg.prefix}_process", proc)
        setattr(state, f"{cfg.prefix}_port", port)
        setattr(state, f"{cfg.prefix}_pid", proc.pid)
        setattr(state, f"{cfg.prefix}_log_file", log_file)
        setattr(state, f"{cfg.prefix}_status", "starting")

        deadline = time.time() + cfg.startup_timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if proc.poll() is not None:
                    setattr(state, f"{cfg.prefix}_status", "error")
                    raise HTTPException(
                        status_code=500,
                        detail=f"{cfg.name} worker process exited during startup; check logs/{cfg.log_name}",
                    )
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/health")
                    if resp.status_code == 200:
                        setattr(state, f"{cfg.prefix}_status", "idle")
                        return port
                except Exception:
                    pass
                await asyncio.sleep(1)
        setattr(state, f"{cfg.prefix}_status", "error")
        raise HTTPException(status_code=500, detail=f"{cfg.name} worker did not become ready in time.")


_NEMOTRON_CFG = _WorkerCfg(
    name="Nemotron Diarization",
    prefix="nemotron_diarize",
    python_bin=NEMOTRON_DIARIZE_PYTHON_BIN,
    script=os.path.join(ROOT, "nemotron", "worker.py"),
    log_name="nemotron_diarize.log",
    setup_hint="Run nemotron/setup.sh (builds nemotron/venv with torch + transformers from git main).",
    # Downloads its checkpoint on first use like the other workers; also the very
    # first request against a git-main transformers install can be slower than usual
    # while it imports/JIT-compiles bits the pinned releases wouldn't need to.
    startup_timeout=90.0,
)


def _unload_nemotron_diarize() -> None:
    _unload_worker("nemotron_diarize", "Nemotron Diarization")


_MOSS_CFG = _WorkerCfg(
    name="MOSS-Transcribe-Diarize",
    prefix="moss_transcribe",
    python_bin=MOSS_TRANSCRIBE_PYTHON_BIN,
    script=os.path.join(ROOT, "moss", "worker.py"),
    log_name="moss_transcribe.log",
    setup_hint="Run moss/setup.sh (builds moss/venv: Python 3.12, torch cu130, the authors' package).",
    startup_timeout=90.0,
)


def _unload_moss_transcribe() -> None:
    _unload_worker("moss_transcribe", "MOSS-Transcribe-Diarize")


def _reset_moss_keep_alive(keep_alive_seconds: int) -> None:
    if state.moss_transcribe_keep_alive_task:
        state.moss_transcribe_keep_alive_task.cancel()

    async def _check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            while state.moss_transcribe_status in ("transcribing", "loading"):
                await asyncio.sleep(5)
            logger.info("MOSS-Transcribe-Diarize idle -- unloading")
            _unload_moss_transcribe()
        except asyncio.CancelledError:
            pass
    state.moss_transcribe_keep_alive_task = asyncio.create_task(_check())


def _finish_worker_first_use(prefix: str) -> None:
    rid = getattr(state, f"{prefix}_reservation", None)
    if rid:
        release_sync(rid)
        setattr(state, f"{prefix}_reservation", None)


async def _moss_job(wav_path: str, prompt: str, max_new_tokens: int, duration_s: float) -> Dict[str, Any]:
    """One MOSS job. Out of memory -> restart the worker where the GPU manager has room (it may unload an idle model
    or pick the other GPU) and retry; again -> on the CPU; out of memory there too -> 507 (retryable)."""
    for cpu_only in (False, False, True):
        port = await _ensure_worker(_MOSS_CFG, cpu_only=cpu_only)
        state.moss_transcribe_status = "transcribing"
        try:
            return await _moss_worker_transcribe(port, wav_path, prompt, max_new_tokens, duration_s)
        except HTTPException as e:
            if not _is_oom(Exception(str(e.detail))):
                raise
            logger.warning(f"MOSS-Transcribe-Diarize ran out of memory -- restarting it (cpu={cpu_only})")
            _unload_moss_transcribe()
        finally:
            state.moss_transcribe_status = "idle"
            _finish_worker_first_use("moss_transcribe")
    raise HTTPException(status_code=507, detail="MOSS-Transcribe-Diarize ran out of memory (GPU and CPU) -- "
                                                "try again in a moment")


async def _transcribe_moss(req: "STTTranscribeRequest") -> Dict[str, Any]:
    """engine 'moss': one pass through MOSS-Transcribe-Diarize. Audio is decoded the same way as for
    WhisperX (ffmpeg content-sniffing, so video works too), written as one 16 kHz WAV for the worker."""
    import tempfile
    import numpy as np
    import soundfile as sf
    from stt_serve.domain import MossInputError, moss_max_new_tokens, moss_prompt, moss_segments

    wants_voices = bool(req.return_embeddings or req.known_speakers)
    if wants_voices and not req.diarize:
        raise HTTPException(status_code=400, detail="return_embeddings/known_speakers need diarize=true.")
    if req.diarize and req.diarize_backend != "whisperx":
        raise HTTPException(status_code=400, detail="engine 'moss' diarizes by itself -- leave diarize_backend unset.")
    try:
        prompt = moss_prompt(req.hotwords)
    except MossInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    entries = req.audio_base64_list or ([req.audio_base64] if req.audio_base64 else [])
    if not entries:
        raise HTTPException(status_code=400, detail="Must provide audio_base64 or audio_base64_list")
    chunks = [await asyncio.to_thread(_decode_audio_base64_to_array, e) for e in entries]
    audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    duration_s = len(audio) / 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
        data = await _moss_job(wav_path, prompt, moss_max_new_tokens(duration_s), duration_s)
        _reset_moss_keep_alive(req.keep_alive or 300)
        state.moss_transcribe_device = data.get("device")
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    segments = moss_segments(data.get("segments") or [], diarize=req.diarize)
    speaker_embeddings: Dict[str, List[float]] = {}
    if wants_voices:
        speaker_embeddings = await asyncio.to_thread(_moss_speaker_embeddings, audio, segments)
        match_info = match_known_speakers(speaker_embeddings, req.known_speakers, req.match_threshold)
        segments, speaker_embeddings = apply_speaker_matches(segments, speaker_embeddings, match_info)
    plain = " ".join(s["text"] for s in segments).strip()
    transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments) if req.diarize else plain
    out: Dict[str, Any] = {"status": "success", "engine": "moss", "transcript": transcript, "segments": segments,
                           "plain_transcript": plain, "duration_s": round(duration_s, 3)}
    if req.return_embeddings:
        out["speaker_embeddings"] = speaker_embeddings
    warnings = []
    if data.get("generated_tokens", 0) >= data.get("max_new_tokens", 1 << 30):
        warnings.append("The transcript may be cut off (token budget reached).")
    if wants_voices and duration_s < RELIABLE_EMBEDDING_MIN_SECONDS:
        warnings.append(f"Audio is {duration_s:.1f}s, below the ~{RELIABLE_EMBEDDING_MIN_SECONDS:.0f}s floor for "
                        f"reliable speaker embeddings -- treat similarity/matched results with caution.")
    if warnings:
        out["warning"] = " ".join(warnings)
    return out


async def _moss_worker_transcribe(port: int, wav_path: str, prompt: str, max_new_tokens: int,
                                  duration_s: float) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=max(300.0, duration_s * 2)) as c:
        r = await c.post(f"http://127.0.0.1:{port}/transcribe",
                         json={"audio_path": wav_path, "prompt": prompt, "max_new_tokens": max_new_tokens})
    data = r.json()
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"MOSS-Transcribe-Diarize: {data.get('error', r.text)[:500]}")
    return data


def _speaker_embedder():
    """The raw speaker-embedding model inside the WhisperX path's pyannote pipeline (loaded on demand,
    same GPU preflight): a callable 16 kHz mono float32 array -> vector. pyannote's own speaker
    centroids are weighted means of exactly these vectors, so MOSS's embeddings match the same roster."""
    import torch

    _ensure_diarize_loaded()
    model = state.diarize_pipeline.model._embedding

    def embed(waveform):
        with torch.inference_mode():
            return model(torch.from_numpy(waveform).float()[None, None, :])[0]
    return embed


def _moss_speaker_embeddings(audio, segments: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Blocking: one voice embedding per MOSS speaker from its own turns (cut into <=10 s windows,
    duration-weighted mean)."""
    windows = embedding_windows(segments)
    if not windows:
        return {}
    embed = _speaker_embedder()
    state.diarize_status = "diarizing"
    try:
        weighted = {label: [(embed(audio[int(a * 16000):int(b * 16000)]), b - a) for a, b in spans]
                    for label, spans in windows.items()}
    finally:
        state.diarize_status = "idle"
    return mean_speaker_embeddings(weighted)


def _reset_nemotron_keep_alive(keep_alive_seconds: int) -> None:
    if state.nemotron_diarize_keep_alive_task:
        state.nemotron_diarize_keep_alive_task.cancel()

    async def _check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            while state.nemotron_diarize_status in ("diarizing", "loading"):
                await asyncio.sleep(5)
            logger.info(f"Keep-alive expired for Nemotron Diarization ({keep_alive_seconds}s idle). Stopping worker...")
            _unload_nemotron_diarize()
        except asyncio.CancelledError:
            pass

    state.nemotron_diarize_keep_alive_task = asyncio.create_task(_check())


def _nemotron_diarize_sync(port: int, audio_data, num_speakers, min_speakers, max_speakers):
    """Blocking: write audio_data (16kHz float32 PCM, same array WhisperX itself
    works with) to a temp WAV and POST it to the already-running worker's /diarize.
    Called from inside a thread-pool executor (see transcribe_stt()/
    _transcribe_live_chunk_sync()), alongside the whisperx.diarize.DiarizationPipeline
    call it's a drop-in alternative to -- hence the plain sync httpx.Client here
    rather than the async client the rest of this worker-plumbing module uses.

    Returns a pandas DataFrame with columns {start, end, speaker}, the same shape
    whisperx.diarize.DiarizationPipeline.__call__() returns -- assign_word_speakers()
    and the live-session dominant-speaker grouping only ever read those three
    columns, so both call sites work unmodified regardless of which backend
    produced the DataFrame."""
    import tempfile

    import pandas as pd
    import soundfile as sf
    from whisperx.audio import SAMPLE_RATE

    state.nemotron_diarize_status = "diarizing"
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, audio_data, SAMPLE_RATE, subtype="PCM_16")
        resp = httpx.post(
            f"http://127.0.0.1:{port}/diarize",
            json={
                "audio_path": tmp.name,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
            timeout=300.0,
        )
        if resp.status_code != 200:
            detail = resp.json().get("error", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            raise HTTPException(status_code=500, detail=f"Nemotron diarization failed: {detail}")
        segments = resp.json().get("segments", [])
        return pd.DataFrame(segments, columns=["start", "end", "speaker"])
    finally:
        state.nemotron_diarize_status = "idle"
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def _gpu_possible() -> bool:
    import torch
    return sys.platform != "darwin" and torch.cuda.is_available()


def _load_model(name: str, device: str, device_index: int, compute_type: str):
    """whisperx.load_model on one device. Its VAD model loads on torch's current CUDA device, so that is set too."""
    import whisperx
    if device == "cuda":
        import torch
        torch.cuda.set_device(device_index)
    return whisperx.load_model(name, device=device, device_index=device_index, compute_type=compute_type)


def _whisper_purpose() -> str:
    return f"stt:whisperx:{os.path.basename(WHISPER_MODEL.rstrip('/'))}"


def _ensure_whisperx_loaded(cpu_only: bool = False):
    """Have state.whisperx_model resident. Shared by the one-shot transcribe_stt() and the live sessions.
    Robust against a full GPU (tests/test_gpu_robust.py): the GPU manager picks the GPU and reserves what this
    model size needs; its reservation is kept (state.whisperx_reservation) until _whisper_transcribe() has finished
    the first transcription. A no from the manager, or a failed CUDA load, -> the CPU. A model on the CPU moves to a
    GPU as soon as the manager has room again."""
    if state.whisperx_model is not None:
        if state.whisperx_device != "cpu" or cpu_only or not _gpu_possible():
            return
        try:   # on the CPU since the GPUs were full: room now?
            grant = reserve_sync(_whisper_purpose(), whisper_need_mb(WHISPER_MODEL))
        except HTTPException:
            return
        logger.info("WhisperX: room on a GPU again -- moving it off the CPU")
        _drop_whisperx_model()
    else:
        grant = None
        if not cpu_only and _gpu_possible():
            try:
                grant = reserve_sync(_whisper_purpose(), whisper_need_mb(WHISPER_MODEL))
            except HTTPException as e:
                logger.warning(f"WhisperX: no room on the GPUs ({e.detail}) -- loading on the CPU")
                cpu_only = True
    state.whisperx_status = "loading"
    try:
        if not cpu_only and _gpu_possible():
            gpu = grant.gpu if grant and grant.gpu is not None else 0
            logger.info(f"Lazy loading WhisperX model ({WHISPER_MODEL}) onto cuda:{gpu} (float16)...")
            try:
                state.whisperx_model = _load_model(WHISPER_MODEL, "cuda", gpu, "float16")
                state.whisperx_device = f"cuda:{gpu}"
                state.whisperx_reservation = grant.rid if grant else None
            except Exception as cuda_err:
                logger.warning(f"WhisperX CUDA load failed, falling back to CPU: {cuda_err}")
                release_sync(grant.rid if grant else None)
                _release_cuda_memory()
                cpu_only = True
        if state.whisperx_model is None:
            logger.info(f"Lazy loading WhisperX model ({WHISPER_MODEL}) onto cpu (int8)...")
            state.whisperx_model = _load_model(WHISPER_MODEL, "cpu", 0, "int8")
            state.whisperx_device = "cpu"
        logger.info(f"WhisperX model loaded ({state.whisperx_device}).")
        state.whisperx_status = "idle"
    except Exception as e:
        state.whisperx_status = "error"
        release_sync(state.whisperx_reservation)
        state.whisperx_reservation = None
        logger.error(f"Failed to load WhisperX: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def _is_oom(e: BaseException) -> bool:
    text = str(e).lower()
    return isinstance(e, MemoryError) or "out of memory" in text or "cuda_error_out_of_memory" in text


def _is_cuda_error(e: BaseException) -> bool:
    text = str(e).lower()
    return "cuda" in text and ("error" in text or "failed" in text)


# One worker thread per device for WhisperX (found live 2026-10-24: after a load on GPU 1 and a later one on GPU 0
# in the same process every transcription failed with "CUDA failed with error invalid argument" -- CTranslate2 keeps
# per-thread CUDA state, and asyncio's executor ran the GPU 0 model on a thread that had worked on GPU 1; setting
# torch's device in that thread wasn't enough). A thread here never switches devices.
_DEVICE_THREADS: Dict[str, Any] = {}
_thread_device = threading.local()


def _device_executor(device: str):
    ex = _DEVICE_THREADS.get(device)
    if ex is None:
        from concurrent.futures import ThreadPoolExecutor

        def init():
            _thread_device.device = device
            if device.startswith("cuda:"):
                try:
                    import torch
                    torch.cuda.set_device(int(device.split(":")[1]))
                except Exception:
                    pass
        ex = _DEVICE_THREADS[device] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"whisper-{device}",
                                                          initializer=init)
    return ex


def _transcribe_on_device(audio, language: Optional[str]):
    """model.transcribe on the worker thread of the device the model is on (directly when already there)."""
    model, device = state.whisperx_model, state.whisperx_device or "cpu"

    def go():
        return model.transcribe(audio, batch_size=4, language=language, task="transcribe")
    if getattr(_thread_device, "device", None) == device:
        return go()
    return _device_executor(device).submit(go).result()


def _finish_first_use() -> None:
    """The first transcription is done: the GPU manager counts the measured usage from now on."""
    if state.whisperx_reservation:
        release_sync(state.whisperx_reservation)
        state.whisperx_reservation = None


def _whisper_transcribe(audio, language: Optional[str]):
    """The one place WhisperX transcribes (one-shot and live). Out of memory on a GPU -> ask the manager again (it may
    unload an idle model or pick the other GPU), reload, retry once; again -> the CPU; out of memory there too ->
    507, retryable. task="transcribe" always (see transcribe_stt)."""
    state.whisperx_last_used = time.time()
    oom = True
    try:
        try:
            return _transcribe_on_device(audio, language)
        except Exception as e:
            if not (_is_oom(e) or _is_cuda_error(e)):
                raise
            oom = _is_oom(e)
            logger.warning(f"WhisperX failed on {state.whisperx_device} ({e}) -- reloading and asking for room again")
        for cpu_only in (False, True):
            if cpu_only and state.whisperx_device == "cpu":
                break   # already tried there
            _finish_first_use()
            _drop_whisperx_model()
            _ensure_whisperx_loaded(cpu_only=cpu_only)
            try:
                return _transcribe_on_device(audio, language)
            except Exception as e:
                if not (_is_oom(e) or _is_cuda_error(e)):
                    raise
                oom = _is_oom(e)
                logger.warning(f"WhisperX failed again on {state.whisperx_device} ({e})")
        _drop_whisperx_model()
        if oom:
            raise HTTPException(status_code=507, detail="Speech recognition ran out of memory (GPU and CPU) -- "
                                                        "try again in a moment")
        raise HTTPException(status_code=503, detail="Speech recognition hit a GPU error -- try again in a moment")
    finally:
        state.whisperx_last_used = time.time()
        _finish_first_use()


def _ensure_diarize_loaded():
    """Lazy-load state.diarize_pipeline if not already resident. Shared the same way
    _ensure_whisperx_loaded() is."""
    import torch
    from whisperx.diarize import DiarizationPipeline

    if state.diarize_pipeline is not None:
        return
    # same process as WhisperX: on its GPU (torch's current device, set when WhisperX loaded)
    gpu = int(state.whisperx_device.split(":")[1]) if (state.whisperx_device or "").startswith("cuda:") else 0
    rid = preflight_sync("diarize", gpu=gpu)
    try:
        state.diarize_status = "loading"
        ddevice = f"cuda:{gpu}" if (sys.platform != "darwin" and torch.cuda.is_available()) else "cpu"
        logger.info(f"Lazy loading diarization pipeline ({DIARIZE_MODEL or 'default'}) onto {ddevice}...")
        state.diarize_pipeline = DiarizationPipeline(
            model_name=DIARIZE_MODEL or None, token=(HF_TOKEN or None), device=ddevice,
        )
        state.diarize_device = ddevice
        logger.info("Diarization pipeline loaded.")
        state.diarize_status = "idle"
        state.diarize_reservation, rid = rid, None   # kept until the transcription that needed it is done
    except Exception as e:
        state.diarize_status = "error"
        logger.error(f"Failed to load diarization pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load diarization pipeline: {e}")
    finally:
        release_sync(rid)


@router.get("/v1/engines", summary="The transcription engines, whether each is set up, and the options each takes")
async def list_engines():
    from stt_serve.catalog import engine_catalog
    moss_proc = getattr(state, "moss_transcribe_process", None)
    loaded = {name for name, on in (("whisperx", state.whisperx_model is not None),
                                    ("moss", moss_proc is not None and moss_proc.poll() is None)) if on}
    return engine_catalog(nemotron_ready=os.path.exists(NEMOTRON_DIARIZE_PYTHON_BIN),
                          moss_ready=os.path.exists(MOSS_TRANSCRIBE_PYTHON_BIN), loaded=loaded)


@router.get("/v1/diarize/backends", summary="List available diarize_backend options and whether each is actually set up")
async def list_diarize_backends():
    """Surfaces what's actually usable right now rather than making a caller guess or
    discover a missing isolated venv via a 500 mid-transcription -- same reasoning as
    /api/comfyui/models existing instead of a hardcoded/guessed model list (see
    TODO.md's design principle for pluggable backends)."""
    nemotron_ready = os.path.exists(NEMOTRON_DIARIZE_PYTHON_BIN)
    return {
        "backends": {
            "whisperx": {
                "available": True,
                "loaded": state.diarize_pipeline is not None,
                "description": "pyannote.audio via whisperx.diarize.DiarizationPipeline. In-process, "
                               "same venv as WhisperX. Supports return_embeddings/known_speakers and a "
                               "native num_speakers/min_speakers/max_speakers clustering constraint.",
            },
            "nemotron": {
                "available": nemotron_ready,
                "loaded": state.nemotron_diarize_process is not None and state.nemotron_diarize_process.poll() is None,
                "description": "nvidia/Nemotron-3-Diarization, an isolated-venv subprocess worker. "
                               "Does NOT support return_embeddings/known_speakers. num_speakers/"
                               "min_speakers/max_speakers are approximated (rank-by-duration "
                               "truncation), not a native constraint." + (
                                   "" if nemotron_ready else
                                   " Not set up yet -- run nemotron/setup.sh."
                               ),
            },
        },
    }


@router.post("/v1/transcribe", summary="Transcribe audio via WhisperX STT, optionally with speaker diarization")
async def transcribe_stt(req: STTTranscribeRequest):
    """Transcribe audio -- or video -- via WhisperX. Accepts audio_base64 or audio_base64_list
    (see their own field descriptions for the video-container support: ffmpeg content-sniffs
    the actual bytes, so a video file's audio track is extracted transparently, no separate
    conversion step needed by the caller). Long recordings are handled by WhisperX's own
    VAD-based batched transcription (model.transcribe(..., batch_size=4)), not a fixed-window
    chunking scheme implemented here.

    `diarize=true` labels each segment with a speaker. `diarize_backend` picks which model
    does it: 'whisperx' (default) via pyannote.audio (whisperx.diarize.DiarizationPipeline,
    a second in-process model, see state.diarize_*/_unload_diarize()), or 'nemotron' via
    nvidia/Nemotron-3-Diarization, an isolated-venv subprocess worker (see state.
    nemotron_diarize_*/_unload_nemotron_diarize(), backend/nemotron_diarize_server.py).
    Either way, word-level alignment always runs first (see below) so speakers get
    assigned per-word, not just per-segment -- the two backends only differ in who
    produces the {start, end, speaker} diarization dataframe assign_word_speakers()
    then consumes. The nemotron backend doesn't support return_embeddings/known_speakers
    (see diarize_backend's own field description) and approximates num_speakers/
    min_speakers/max_speakers rather than natively constraining on them.

    Pass `language` when the caller already knows it (see STTTranscribeRequest.language) --
    left unset, WhisperX auto-detects from the first 30s, and that auto-detect is unreliable
    on a just-loaded model (confirmed live: same German clip, cold call came back as a
    fluent English translation, the very next call on the same audio came back correct
    German). `task="transcribe"` is always forced explicitly for the same reason -- never
    left to whatever the pipeline's cached tokenizer defaults to.
    """
    if req.engine not in ("whisperx", "moss"):
        raise HTTPException(status_code=400, detail="engine must be 'whisperx' or 'moss'.")
    if req.engine == "moss":
        return await _transcribe_moss(req)
    if req.hotwords:
        raise HTTPException(status_code=400, detail="hotwords are for engine 'moss' only.")
    import whisperx
    import tempfile

    if req.diarize and req.diarize_backend not in ("whisperx", "nemotron"):
        raise HTTPException(status_code=400, detail="diarize_backend must be 'whisperx' or 'nemotron'.")
    if req.diarize and req.diarize_backend == "nemotron" and (req.return_embeddings or req.known_speakers):
        raise HTTPException(
            status_code=400,
            detail="diarize_backend='nemotron' does not support return_embeddings/known_speakers -- "
                   "Nemotron-3-Diarization doesn't produce comparable speaker embeddings. Use the "
                   "default 'whisperx' backend for cross-recording speaker matching.",
        )

    nemotron_port: Optional[int] = None
    _ensure_whisperx_loaded()
    if req.diarize:
        if req.diarize_backend == "nemotron":
            nemotron_port = await _ensure_worker(_NEMOTRON_CFG)
        else:
            _ensure_diarize_loaded()

    _reset_whisperx_keep_alive(req.keep_alive or 300)
    if req.diarize:
        if req.diarize_backend == "nemotron":
            _reset_nemotron_keep_alive(req.keep_alive or 300)
        else:
            _reset_diarize_keep_alive(req.keep_alive or 300)
    temp_wav_paths: List[str] = []
    try:
        state.whisperx_status = "transcribing"
        if req.audio_base64_list:
            # Each entry is decoded (ffmpeg content-sniffs, independently of the others)
            # then concatenated as PCM inside _run() -- lets a caller assemble one clip
            # out of several short chunks (e.g. a live-streaming session) rather than
            # needing to pre-concatenate containers client-side.
            for entry in req.audio_base64_list:
                audio_bytes = base64.b64decode(entry)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    temp_wav_paths.append(tmp.name)
            audio_files = temp_wav_paths
        elif req.audio_base64:
            audio_bytes = base64.b64decode(req.audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_wav_paths.append(tmp.name)
            audio_files = temp_wav_paths
        else:
            raise HTTPException(
                status_code=400, detail="Must provide audio_base64 or audio_base64_list")

        logger.info(
            f"Transcribing {len(audio_files)} file(s){' with diarization' if req.diarize else ''}"
            f"{' + speaker embeddings' if req.return_embeddings else ''}...")

        def _run():
            import numpy as np
            from whisperx import align, load_align_model
            from whisperx.audio import SAMPLE_RATE
            from whisperx.diarize import assign_word_speakers

            chunks = [whisperx.load_audio(f) for f in audio_files]
            audio_data = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            duration_s = len(audio_data) / SAMPLE_RATE
            # task="transcribe" is forced explicitly (not just relying on WhisperX's own
            # "transcribe" default) so a cold-load auto-detect misfire can never silently
            # produce a translation instead of a transcript -- see STTTranscribeRequest.language.
            result = _whisper_transcribe(audio_data, req.language)
            result["duration_s"] = duration_s
            if req.diarize:
                align_device = state.diarize_device or state.nemotron_diarize_device or "cpu"
                # Word-level alignment first, regardless of diarize_backend: whisper's own
                # segments can span a speaker change (confirmed live -- two speakers with no
                # silence gap between them came back as ONE whisper segment), so assigning a
                # speaker per *segment* silently picks whichever speaker dominates that whole
                # span. Aligning gives per-word timestamps, so assign_word_speakers() (below)
                # can label each word and the turns get rebuilt correctly (speaker_turns()).
                # assign_word_speakers() only ever reads {start, end, speaker} off diarize_df
                # (see whisperx/diarize.py) -- it doesn't care which backend produced it, so
                # everything from here down is identical for both.
                lang = result.get("language", "en")
                try:
                    if state.align_model is None or state.align_language != lang:
                        logger.info(f"Lazy loading word-alignment model for language '{lang}'...")
                        state.align_model, state.align_metadata = load_align_model(
                            language_code=lang, device=align_device)
                        state.align_language = lang
                    result = align(
                        result["segments"], state.align_model, state.align_metadata,
                        audio_data, align_device, return_char_alignments=False,
                    )
                    result["language"] = lang
                    result["duration_s"] = duration_s
                except Exception as align_err:
                    # No alignment model for this language, or alignment itself failed --
                    # fall back to segment-level speaker assignment rather than erroring
                    # the whole request; speaker_turns() below degrades gracefully when
                    # segments carry no 'words'.
                    logger.warning(f"Word alignment failed ({align_err}); diarization falls back to segment-level.")

                if req.diarize_backend == "nemotron":
                    diarize_df = _nemotron_diarize_sync(
                        nemotron_port, audio_data, req.num_speakers, req.min_speakers, req.max_speakers)
                    speaker_embeddings = None  # not supported -- rejected upfront, see transcribe_stt()
                else:
                    state.diarize_status = "diarizing"
                    diarize_out = state.diarize_pipeline(
                        audio_data, num_speakers=req.num_speakers,
                        min_speakers=req.min_speakers, max_speakers=req.max_speakers,
                        return_embeddings=req.return_embeddings,
                    )
                    if req.return_embeddings:
                        diarize_df, speaker_embeddings = diarize_out
                    else:
                        diarize_df, speaker_embeddings = diarize_out, None
                result = assign_word_speakers(diarize_df, result, speaker_embeddings=speaker_embeddings)
                result["duration_s"] = duration_s
                state.diarize_status = "idle"
                state.nemotron_diarize_status = "idle"
            return result

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _run)
        finally:   # the first use is done: the GPU manager counts what was really allocated from now on
            _finish_worker_first_use("nemotron_diarize")
            release_sync(state.diarize_reservation)
            state.diarize_reservation = None
        state.whisperx_status = "idle"
        _reset_whisperx_keep_alive(req.keep_alive or 300)   # a retry on another device dropped the model meanwhile

        duration_s = result.get("duration_s")
        speaker_embeddings = result.get("speaker_embeddings") if req.return_embeddings else None
        match_info: Dict[str, Dict[str, Any]] = {}
        if req.diarize and speaker_embeddings:
            match_info = match_known_speakers(speaker_embeddings, req.known_speakers, req.match_threshold)
            # "bester Treffer ersetzt das generische Label" -- a matched speaker's
            # SPEAKER_00-style label becomes its known_speakers label everywhere
            # (segments and the derived transcript below), not just an extra field.
            relabel = {label: m["matched_label"] for label, m in match_info.items() if m["matched"]}
            if relabel:
                for seg in result["segments"]:
                    if seg.get("speaker") in relabel:
                        seg["speaker"] = relabel[seg["speaker"]]
                    for w in (seg.get("words") or []):
                        if w.get("speaker") in relabel:
                            w["speaker"] = relabel[w["speaker"]]
                speaker_embeddings = {relabel.get(k, k): v for k, v in speaker_embeddings.items()}
                match_info = {relabel.get(k, k): v for k, v in match_info.items()}

        if req.diarize:
            segments = speaker_turns(result["segments"])
            for seg in segments:
                m = match_info.get(seg.get("speaker"))
                if m is not None:
                    seg["similarity"] = m["similarity"]
                    seg["matched"] = m["matched"]
                    seg["matched_label"] = m["matched_label"]
        else:
            segments = [
                {"start": seg.get("start"), "end": seg.get("end"), "text": seg["text"].strip()}
                for seg in result["segments"]
            ]
        # The plain, unlabeled transcript -- this is what's written to the .txt sidecar
        # below, which elsewhere in this file (_default_ref_voice()) doubles as a TTS
        # reference clip's ref_text; a "SPEAKER_00: " prefix baked in there would be
        # read aloud by a clone, so speaker labels stay JSON-response-only.
        plain_transcript = " ".join(s["text"] for s in segments).strip()
        if req.diarize:
            transcript = "\n".join(f"{s.get('speaker', 'SPEAKER_?')}: {s['text']}" for s in segments).strip()
        else:
            transcript = plain_transcript

        # plain_transcript (no speaker labels) is returned separately so a caller that
        # stores it -- ai-serve-ui writes it as a voices/ .txt sidecar, which doubles
        # as a TTS reference transcript -- never gets "SPEAKER_00: " baked in.
        response: Dict[str, Any] = {"status": "success", "transcript": transcript, "segments": segments,
                                    "plain_transcript": plain_transcript}
        if duration_s is not None:
            response["duration_s"] = round(duration_s, 3)
        if req.return_embeddings:
            response["speaker_embeddings"] = speaker_embeddings or {}
        if (req.return_embeddings or req.known_speakers) and duration_s is not None \
                and duration_s < RELIABLE_EMBEDDING_MIN_SECONDS:
            response["warning"] = (
                f"Audio is {duration_s:.1f}s, below the ~{RELIABLE_EMBEDDING_MIN_SECONDS:.0f}s floor for "
                f"reliable speaker embeddings (confirmed: a real speaker can be missed entirely at 4s; "
                f"clean separation starts around 8s) -- treat similarity/matched results with caution, "
                f"or send more audio (see audio_base64_list to combine several short chunks in one call)."
            )
        return response
    except HTTPException:
        # Pre-existing bug, fixed in passing: without this, a client error raised
        # above (bad path, missing audio) fell into the broad `except Exception`
        # below and came back as a 500 wrapping "400: ...", not a real 400.
        raise
    except Exception as e:
        state.whisperx_status = "error"
        if req.diarize:
            if req.diarize_backend == "nemotron":
                state.nemotron_diarize_status = "error"
            else:
                state.diarize_status = "error"
        logger.error(f"STT transcription failed: {e}")
        if _is_oom(e):   # retryable: memory may be free again in a moment
            raise HTTPException(status_code=507, detail=f"Speech recognition ran out of memory -- try again ({e})")
        if _is_cuda_error(e):
            raise HTTPException(status_code=503, detail=f"Speech recognition hit a GPU error -- try again ({e})")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if state.whisperx_status == "transcribing":
            state.whisperx_status = "idle"
        if state.diarize_status == "diarizing":
            state.diarize_status = "idle"
        if state.nemotron_diarize_status == "diarizing":
            state.nemotron_diarize_status = "idle"
        for p in temp_wav_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@router.post("/v1/unload", summary="Unload the WhisperX STT model (and diarization pipeline, if loaded), freeing VRAM/RAM")
async def unload_stt():
    nemotron_running = state.nemotron_diarize_process is not None and state.nemotron_diarize_process.poll() is None
    if state.whisperx_model is None and state.diarize_pipeline is None and not nemotron_running:
        return {"status": "already unloaded"}
    if state.whisperx_status == "transcribing" or state.diarize_status == "diarizing" \
            or state.nemotron_diarize_status == "diarizing":
        raise HTTPException(status_code=409, detail="STT is currently transcribing; try again shortly.")
    _unload_whisperx()
    _unload_diarize()
    if nemotron_running:
        _unload_nemotron_diarize()
    return {"status": "unloaded"}


@router.post("/v1/diarize/unload", summary="Unload just the diarization pipeline (whisperx backend), freeing VRAM/RAM")
async def unload_diarize():
    if state.diarize_pipeline is None:
        return {"status": "already unloaded"}
    if state.diarize_status == "diarizing":
        raise HTTPException(status_code=409, detail="Diarization is currently running; try again shortly.")
    _unload_diarize()
    return {"status": "unloaded"}


@router.post("/v1/moss/unload", summary="Unload the MOSS-Transcribe-Diarize worker, freeing VRAM/RAM")
async def unload_moss_transcribe_endpoint():
    if state.moss_transcribe_process is None or state.moss_transcribe_process.poll() is not None:
        return {"status": "already unloaded"}
    if state.moss_transcribe_status == "transcribing":
        raise HTTPException(status_code=409, detail="MOSS-Transcribe-Diarize is transcribing; try again shortly.")
    _unload_moss_transcribe()
    return {"status": "unloaded"}


@router.post("/v1/diarize/nemotron/unload", summary="Unload the Nemotron-3-Diarization worker, freeing VRAM/RAM")
async def unload_nemotron_diarize_endpoint():
    if state.nemotron_diarize_process is None or state.nemotron_diarize_process.poll() is not None:
        return {"status": "already unloaded"}
    if state.nemotron_diarize_status == "diarizing":
        raise HTTPException(status_code=409, detail="Diarization is currently running; try again shortly.")
    _unload_nemotron_diarize()
    return {"status": "unloaded"}


_LIVE_SAMPLE_RATE = 16000  # whisperx.audio.SAMPLE_RATE -- fixed, hardcoded rather than


_LIVE_SESSION_KEEP_ALIVE_DEFAULT = 600


_LIVE_MIN_FLUSH_SPEECH_S = 0.3


def _decode_audio_base64_to_array(audio_b64: str):
    """Same ffmpeg-content-sniffing decode transcribe_stt() uses for a one-shot
    request, but for one live-session chunk: base64 -> temp file -> whisperx.load_audio
    -> numpy array, temp file cleaned up immediately rather than held for the request's
    lifetime (a live session outlives any single /chunk call)."""
    import tempfile
    import whisperx

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return whisperx.load_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _new_live_session(req: "LiveSTTStartRequest") -> Dict[str, Any]:
    import numpy as np
    sid = uuid.uuid4().hex[:12]
    session: Dict[str, Any] = {
        "id": sid,
        "created_at": time.time(),
        "last_activity": time.time(),
        "language": req.language,
        "diarize": req.diarize,
        "diarize_backend": req.diarize_backend,
        "silence_ms": req.silence_ms,
        "keep_alive": req.keep_alive,
        "return_embeddings": req.return_embeddings,
        "known_speakers": req.known_speakers,
        "match_threshold": req.match_threshold,
        "pending": np.zeros(0, dtype=np.float32),
        "session_offset_s": 0.0,
        "segments": [],
        "closed": False,
        "keep_alive_task": None,
    }
    state.live_stt_sessions[sid] = session
    return session


def _reset_live_keep_alive(session_id: str):
    session = state.live_stt_sessions.get(session_id)
    if session is None:
        return
    if session.get("keep_alive_task"):
        session["keep_alive_task"].cancel()
    keep_alive_seconds = session.get("keep_alive") or _LIVE_SESSION_KEEP_ALIVE_DEFAULT

    async def _check():
        try:
            await asyncio.sleep(keep_alive_seconds)
            logger.info(f"Live STT session {session_id} idle-expired after {keep_alive_seconds}s; force-finalizing.")
            await _finalize_and_close_live_session(session_id)
        except asyncio.CancelledError:
            pass

    session["keep_alive_task"] = asyncio.create_task(_check())


def _transcribe_live_chunk_sync(session: Dict[str, Any], audio) -> Optional[Dict[str, Any]]:
    """Blocking: transcribe one finalized live segment, diarize it if the session asked
    for that. Segment-level only, no word-level alignment (unlike the one-shot
    endpoint's diarization path) -- each live chunk is short, usually one speaker, and
    already paying for an alignment-model call on every few seconds of audio would add
    real latency to a live session for a case (a speaker change mid-chunk) that word
    alignment exists to fix, at a cost this path doesn't need to pay by default."""
    result = _whisper_transcribe(audio, session["language"])
    text = " ".join(seg["text"].strip() for seg in result["segments"]).strip()
    if not text:
        return None

    start_s = session["session_offset_s"]
    end_s = start_s + len(audio) / _LIVE_SAMPLE_RATE
    seg_out: Dict[str, Any] = {"start": round(start_s, 3), "end": round(end_s, 3), "text": text}

    if session["diarize"]:
        if session.get("diarize_backend") == "nemotron":
            # Worker is already ensured resident at /v1/live/start time (an async
            # call, unlike this function) -- state.nemotron_diarize_port is read directly
            # here rather than re-ensuring it, same as _ensure_diarize_loaded() isn't
            # re-called per-chunk on the whisperx side either.
            diarize_df = _nemotron_diarize_sync(state.nemotron_diarize_port, audio, None, None, None)
            speaker_embeddings = None  # not supported -- rejected upfront in live_stt_start()
        else:
            _ensure_diarize_loaded()
            diarize_out = state.diarize_pipeline(audio, return_embeddings=session["return_embeddings"])
            diarize_df, speaker_embeddings = diarize_out if session["return_embeddings"] else (diarize_out, None)
        if len(diarize_df) > 0:
            # Dominant speaker for this whole chunk by total overlap duration -- the
            # per-word regrouping speaker_turns() does for the one-shot endpoint isn't
            # applicable here since this chunk has no word-level alignment (see above).
            durations = diarize_df.groupby("speaker").apply(lambda d: (d["end"] - d["start"]).sum())
            dominant_speaker = durations.idxmax()
            seg_out["speaker"] = dominant_speaker
            if speaker_embeddings and dominant_speaker in speaker_embeddings:
                emb = speaker_embeddings[dominant_speaker]
                match_info = match_known_speakers({dominant_speaker: emb}, session["known_speakers"], session["match_threshold"])
                m = match_info[dominant_speaker]
                seg_out["similarity"] = m["similarity"]
                seg_out["matched"] = m["matched"]
                seg_out["matched_label"] = m["matched_label"]
                if m["matched"]:
                    seg_out["speaker"] = m["matched_label"]
                if session["return_embeddings"]:
                    seg_out["embedding"] = emb
    return seg_out


def _flush_live_session_sync(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Blocking: check session['pending'] for enough trailing silence to finalize a
    segment; if so, transcribe everything before that silence, clear pending, return
    the new segment (as a 0- or 1-element list). Called from both /chunk (so a session
    that never explicitly polls still finalizes promptly) and /poll (so a poll forces a
    fresh check even with no new /chunk since the last one)."""
    import numpy as np

    pending = session["pending"]
    if len(pending) == 0:
        return []
    trailing_s = rms_trailing_silence_s(pending, _LIVE_SAMPLE_RATE)
    if trailing_s * 1000 < session["silence_ms"]:
        return []

    speech_len = len(pending) - int(trailing_s * _LIVE_SAMPLE_RATE)
    # session_offset_s must stay at its PRE-flush value while _transcribe_live_chunk_sync
    # reads it as the new segment's start time -- confirmed live: advancing it first (the
    # original version of this function) double-counted, since the just-finalized chunk's
    # own start got stamped with the offset *after* consuming itself.
    total_consumed_s = len(pending) / _LIVE_SAMPLE_RATE
    session["pending"] = np.zeros(0, dtype=np.float32)
    if speech_len / _LIVE_SAMPLE_RATE < _LIVE_MIN_FLUSH_SPEECH_S:
        session["session_offset_s"] += total_consumed_s
        return []  # nothing but a short blip before the silence -- drop it, don't transcribe

    _ensure_whisperx_loaded()
    new_seg = _transcribe_live_chunk_sync(session, pending[:speech_len])
    session["session_offset_s"] += total_consumed_s
    if new_seg is None:
        return []
    session["segments"].append(new_seg)
    return [new_seg]


async def _finalize_and_close_live_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Force-finalize whatever's pending regardless of trailing silence, then remove
    the session. Shared by POST /stop (explicit) and idle keep-alive expiry (implicit)
    -- the same "something always cleans this up" guarantee every other keep_alive
    field in this API already gives, so an abandoned session that never calls /stop
    doesn't leak memory (or a loaded model no other caller still needs) forever."""
    import numpy as np

    session = state.live_stt_sessions.get(session_id)
    if session is None or session.get("closed"):
        return None

    def _run():
        pending = session["pending"]
        if len(pending) / _LIVE_SAMPLE_RATE >= _LIVE_MIN_FLUSH_SPEECH_S:
            _ensure_whisperx_loaded()
            new_seg = _transcribe_live_chunk_sync(session, pending)
            if new_seg is not None:
                session["segments"].append(new_seg)
        session["pending"] = np.zeros(0, dtype=np.float32)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run)

    session["closed"] = True
    if session.get("keep_alive_task"):
        session["keep_alive_task"].cancel()
        session["keep_alive_task"] = None
    state.live_stt_sessions.pop(session_id, None)
    return session


@router.post("/v1/live/start", summary="Start a live STT session (chunked, VAD-based incremental transcription)")
async def live_stt_start(req: LiveSTTStartRequest):
    if req.diarize and req.diarize_backend not in ("whisperx", "nemotron"):
        raise HTTPException(status_code=400, detail="diarize_backend must be 'whisperx' or 'nemotron'.")
    if req.diarize and req.diarize_backend == "nemotron" and (req.return_embeddings or req.known_speakers):
        raise HTTPException(
            status_code=400,
            detail="diarize_backend='nemotron' does not support return_embeddings/known_speakers -- "
                   "see STTTranscribeRequest.diarize_backend.",
        )
    _ensure_whisperx_loaded()
    if req.diarize:
        if req.diarize_backend == "nemotron":
            await _ensure_worker(_NEMOTRON_CFG)
        else:
            _ensure_diarize_loaded()
    session = _new_live_session(req)
    _reset_live_keep_alive(session["id"])
    return {"session_id": session["id"], "silence_ms": session["silence_ms"], "keep_alive": session["keep_alive"]}


@router.post("/v1/live/chunk", summary="Append one audio chunk to a live STT session")
async def live_stt_chunk(req: LiveSTTChunkRequest):
    session = state.live_stt_sessions.get(req.session_id)
    if session is None or session["closed"]:
        raise HTTPException(status_code=404, detail=f"No active live STT session '{req.session_id}'.")
    session["last_activity"] = time.time()
    _reset_live_keep_alive(req.session_id)

    def _run():
        import numpy as np
        chunk = _decode_audio_base64_to_array(req.audio_base64)
        session["pending"] = np.concatenate([session["pending"], chunk]) if len(session["pending"]) else chunk
        return _flush_live_session_sync(session)

    loop = asyncio.get_running_loop()
    new_segments = await loop.run_in_executor(None, _run)
    return {
        "status": "ok",
        "new_segments": new_segments,
        "pending_duration_s": round(len(session["pending"]) / _LIVE_SAMPLE_RATE, 3),
    }


@router.get("/v1/live/poll", summary="Poll a live STT session for newly finalized segments")
async def live_stt_poll(session_id: str, since: int = 0):
    session = state.live_stt_sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"No live STT session '{session_id}' (never started, already /stop'd, or idle-expired).",
        )
    session["last_activity"] = time.time()
    _reset_live_keep_alive(session_id)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _flush_live_session_sync, session)

    return {
        "segments": session["segments"][since:],
        "next_since": len(session["segments"]),
        "pending_duration_s": round(len(session["pending"]) / _LIVE_SAMPLE_RATE, 3),
        "closed": session["closed"],
    }


@router.post("/v1/live/stop", summary="Finalize and close a live STT session")
async def live_stt_stop(req: LiveSTTStopRequest):
    session = await _finalize_and_close_live_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active live STT session '{req.session_id}'.")
    transcript = " ".join(s["text"] for s in session["segments"]).strip()
    return {"status": "stopped", "transcript": transcript, "segments": session["segments"]}


@router.patch("/v1/live/settings", summary="Update a live STT session's settings (e.g. silence_ms)")
async def live_stt_settings(req: LiveSTTSettingsRequest):
    session = state.live_stt_sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No active live STT session '{req.session_id}'.")
    if req.silence_ms is not None:
        session["silence_ms"] = req.silence_ms
    if req.keep_alive is not None:
        session["keep_alive"] = req.keep_alive
        _reset_live_keep_alive(req.session_id)
    return {"session_id": req.session_id, "silence_ms": session["silence_ms"], "keep_alive": session["keep_alive"]}


@router.get("/v1/live/sessions", summary="List active live STT sessions")
async def live_stt_sessions_list():
    now = time.time()
    return {"sessions": [
        {
            "session_id": sid,
            "created_at": s["created_at"],
            "idle_seconds": round(now - s["last_activity"], 1),
            "language": s["language"],
            "diarize": s["diarize"],
            "silence_ms": s["silence_ms"],
            "keep_alive": s["keep_alive"],
            "segment_count": len(s["segments"]),
            "pending_duration_s": round(len(s["pending"]) / _LIVE_SAMPLE_RATE, 3),
        }
        for sid, s in state.live_stt_sessions.items()
    ]}

