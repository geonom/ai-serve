"""Runtime state of stt-serve's models (moved from ai-serve-ui's State)."""
from typing import Any, Dict, Optional


class State:
    whisperx_model: Optional[Any] = None
    whisperx_status: str = "idle"
    whisperx_keep_alive_task: Optional[Any] = None
    whisperx_device: Optional[str] = None        # "cuda:<n>" or "cpu"
    whisperx_reservation: Optional[str] = None   # GPU reservation kept until the first transcription is done
    whisperx_last_used: float = 0.0              # reported to the GPU manager (least recently used goes first)
    # the workers' GPU reservations, kept until their first job is done
    nemotron_diarize_reservation: Optional[str] = None
    moss_transcribe_reservation: Optional[str] = None
    diarize_reservation: Optional[str] = None
    # whisperx-backend diarization (pyannote) + the word-alignment model it needs
    diarize_pipeline: Optional[Any] = None
    diarize_device: Optional[str] = None
    diarize_status: str = "idle"
    diarize_keep_alive_task: Optional[Any] = None
    align_model: Optional[Any] = None
    align_metadata: Optional[Any] = None
    align_language: Optional[str] = None
    # nemotron-backend diarization: isolated-venv worker (nemotron/worker.py)
    nemotron_diarize_process: Optional[Any] = None
    nemotron_diarize_port: Optional[int] = None
    nemotron_diarize_pid: Optional[int] = None
    nemotron_diarize_log_file: Optional[Any] = None
    nemotron_diarize_variant: Optional[str] = None
    nemotron_diarize_status: str = "idle"
    nemotron_diarize_device: Optional[str] = None
    nemotron_diarize_keep_alive_task: Optional[Any] = None
    # engine "moss": MOSS-Transcribe-Diarize, isolated-venv worker (moss/worker.py)
    moss_transcribe_process: Optional[Any] = None
    moss_transcribe_port: Optional[int] = None
    moss_transcribe_pid: Optional[int] = None
    moss_transcribe_log_file: Optional[Any] = None
    moss_transcribe_variant: Optional[str] = None
    moss_transcribe_status: str = "idle"
    moss_transcribe_device: Optional[str] = None
    moss_transcribe_keep_alive_task: Optional[Any] = None
    live_stt_sessions: Dict[str, Dict[str, Any]] = {}


state = State()
