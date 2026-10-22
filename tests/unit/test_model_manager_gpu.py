"""Unit tests for backend/model_manager.py's Apple/Darwin GPU-status path —
no network, no subprocesses, no real macOS/ioreg dependency (mocked)."""
import asyncio

from backend import model_manager


class _FakeVirtualMemory:
    def __init__(self, total, used, available, percent):
        self.total = total
        self.used = used
        self.available = available
        self.percent = percent


def test_detect_apple_gpu_used_mb_parses_ioreg_output(monkeypatch):
    # A trimmed but structurally real ioreg -r -d 1 -c IOAccelerator excerpt --
    # the one key this parser cares about, "In use system memory", in bytes.
    fake_output = (
        '+-o AGXAccelerator  <class AGXAccelerator, id 0x100000123>\n'
        '    "PerformanceStatistics" = {"In use system memory"=4294967296,'
        '"Alloc system memory"=6442450944}\n'
    )
    monkeypatch.setattr(
        model_manager.subprocess, "check_output",
        lambda *a, **k: fake_output,
    )
    result = model_manager._detect_apple_gpu_used_mb()
    assert result == 4294967296 // (1024 * 1024)  # 4096 MB


def test_detect_apple_gpu_used_mb_returns_none_when_ioreg_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("ioreg not found")
    monkeypatch.setattr(model_manager.subprocess, "check_output", _raise)
    assert model_manager._detect_apple_gpu_used_mb() is None


def test_detect_apple_gpu_used_mb_returns_none_when_key_absent(monkeypatch):
    # Simulates a future/older macOS whose ioreg output doesn't carry the key
    # this parser looks for -- must degrade to None, not raise or misparse.
    monkeypatch.setattr(
        model_manager.subprocess, "check_output",
        lambda *a, **k: '+-o AGXAccelerator <class AGXAccelerator>\n',
    )
    assert model_manager._detect_apple_gpu_used_mb() is None


def test_fetch_gpu_status_darwin_prefers_ioreg_over_psutil(monkeypatch):
    monkeypatch.setattr(model_manager.sys, "platform", "darwin")
    monkeypatch.setattr(model_manager, "_apple_wired_limit_mb", lambda: 0)
    monkeypatch.setattr(
        model_manager.psutil, "virtual_memory",
        lambda: _FakeVirtualMemory(
            total=32 * 1024 * 1024 * 1024,
            used=20 * 1024 * 1024 * 1024,  # whole-system figure -- should be ignored
            available=12 * 1024 * 1024 * 1024,
            percent=62.5,
        ),
    )
    monkeypatch.setattr(model_manager, "_detect_apple_gpu_used_mb", lambda: 8192)
    monkeypatch.setattr(model_manager, "_apple_wired_limit_mb", lambda: 0)

    status = asyncio.run(model_manager.fetch_gpu_status())

    gpu = status["gpus"][0]
    assert gpu["used"] == 8192
    assert gpu["memory_source"] == "ioreg_gpu"
    assert gpu["total"] == 32768
    # free = Metal's limit (2/3 of 32 GB) minus what the GPU holds -- not psutil's available RAM
    assert gpu["gpu_limit_mb"] == 21845 and gpu["free"] == 21845 - 8192
    assert gpu["usage_percent"] == round(8192 / 32768 * 100, 1)


def test_fetch_gpu_status_darwin_falls_back_to_psutil_when_ioreg_unavailable(monkeypatch):
    monkeypatch.setattr(model_manager.sys, "platform", "darwin")
    monkeypatch.setattr(model_manager, "_apple_wired_limit_mb", lambda: 0)
    monkeypatch.setattr(
        model_manager.psutil, "virtual_memory",
        lambda: _FakeVirtualMemory(
            total=32 * 1024 * 1024 * 1024,
            used=20 * 1024 * 1024 * 1024,
            available=12 * 1024 * 1024 * 1024,
            percent=62.5,
        ),
    )
    monkeypatch.setattr(model_manager, "_detect_apple_gpu_used_mb", lambda: None)

    status = asyncio.run(model_manager.fetch_gpu_status())

    gpu = status["gpus"][0]
    assert gpu["used"] == 20480  # whole-system psutil figure, unchanged fallback
    assert gpu["memory_source"] == "psutil_system_ram"
