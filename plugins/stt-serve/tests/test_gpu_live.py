"""Real GPUs, real WhisperX -- the failure of 2026-10-24 18:01 that the faked tests could not see: the same process
loads Whisper on GPU 1, later on GPU 0, and transcribes from a reused worker thread (as asyncio's executor does);
before the fix every transcription then failed with "CUDA failed with error invalid argument".
Opt-in, needs two NVIDIA GPUs and the cached model:  STT_GPU_TESTS=1 .venv/bin/python -m pytest -q tests/test_gpu_live.py"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("STT_GPU_TESTS") != "1", reason="real-GPU test: set STT_GPU_TESTS=1")


def test_whisper_moves_between_gpus_in_one_process(monkeypatch):
    import torch
    from stt_serve import coordination, engine
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    grants = iter([1, 0, 1])
    monkeypatch.setattr(engine, "reserve_sync", lambda purpose, need: coordination.Grant(None, next(grants)))
    monkeypatch.setattr(engine, "release_sync", lambda rid: None)
    monkeypatch.setattr(engine.state, "whisperx_model", None)
    t = np.linspace(0, 2, 32000, endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pool = ThreadPoolExecutor(max_workers=1)        # like asyncio's default executor: one thread, reused
    try:
        for want in ("cuda:1", "cuda:0", "cuda:1"):
            engine._ensure_whisperx_loaded()
            assert engine.state.whisperx_device == want
            pool.submit(engine._whisper_transcribe, audio, "de").result()   # raised before the fix
            engine._drop_whisperx_model()
    finally:
        pool.shutdown()
        engine._drop_whisperx_model()
