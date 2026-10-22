from typing import Dict, Any, Optional, List


class State:
    managed_processes: Dict[int, Dict[str, Any]] = {}
    smoke_test_running: bool = False
    smoke_test_progress: Dict[str, Any] = {"current": 0, "total": 0, "model": "", "status": "idle"}
    smoke_test_results: List[Dict[str, Any]] = []



    # Every TTS engine (OmniVoice, CosyVoice, Chatterbox, VoxCPM, Qwen3-TTS) runs in
    # tts-serve, a separate service with its own state -- see backend/tts_client.py.


    # Speech-to-text (WhisperX, diarization, alignment, live sessions) runs in
    # stt-serve with its own state -- see backend/stt_client.py.

    ocr_batch: Optional[Dict[str, Any]] = None


state = State()
