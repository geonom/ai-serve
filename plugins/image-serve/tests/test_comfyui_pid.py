"""ComfyUI's pid as the GPU manager sees it (2026-11-24, V-Noveller's painting run): two jobs auto-started ComfyUI at
once, the second process died on the taken port and its dead pid stayed in state. The gateway then counted the live
ComfyUI's 10-16 GB as "other programs" -- never freed, so the next model never fit."""
import asyncio

import pytest

from image_serve import comfy
from image_serve.state import state


class _Proc:
    def __init__(self, pid, alive):
        self.pid, self.alive = pid, alive

    def poll(self):
        return None if self.alive else 1


class _Found:
    def __init__(self, pid):
        self.pid = pid

    def cmdline(self):
        return ["python", "main.py", "--port", "8188", "--cuda-device", "0"]

    def is_running(self):
        return True


@pytest.fixture
def clean(monkeypatch):
    saved = (state.comfyui_process, state.comfyui_pid, state.comfyui_gpu_id)
    yield monkeypatch
    state.comfyui_process, state.comfyui_pid, state.comfyui_gpu_id = saved


def test_dead_tracked_pid_gives_way_to_the_live_process(clean):
    state.comfyui_process, state.comfyui_pid = _Proc(947913, alive=False), 947913
    clean.setattr(comfy, "find_comfyui_process", lambda: _Found(947751))
    assert comfy.live_pid() == 947751
    assert state.comfyui_pid == 947751 and state.comfyui_process.pid == 947751 and state.comfyui_gpu_id == 0


def test_live_tracked_pid_is_kept(clean):
    state.comfyui_process, state.comfyui_pid = _Proc(5, alive=True), 5
    clean.setattr(comfy, "find_comfyui_process", lambda: _Found(6))
    assert comfy.live_pid() == 5


def test_nothing_running(clean):
    state.comfyui_process, state.comfyui_pid = None, None
    clean.setattr(comfy, "find_comfyui_process", lambda: None)
    assert comfy.live_pid() is None


def test_two_starts_at_once_spawn_one_comfyui(clean, tmp_path):
    state.comfyui_process, state.comfyui_pid = None, None
    spawned = []

    class _Popen(_Proc):
        def __init__(self, *a, **k):
            super().__init__(1000 + len(spawned), alive=True)
            spawned.append(self)

    async def no_room():
        return {}

    async def ok(*a, **k):
        await asyncio.sleep(0.01)      # the await that let the second start slip in before the first Popen

    clean.setattr(comfy.subprocess, "Popen", _Popen)
    clean.setattr(comfy, "find_comfyui_process", lambda: None)
    clean.setattr(comfy, "LOGS_DIR", str(tmp_path))
    from image_serve.gpu import MANAGER
    clean.setattr(MANAGER, "raise_if_exclusive", ok)

    async def both():
        req = comfy.ComfyUIStartRequest(port=state.comfyui_port, gpu_id=0)
        return await asyncio.gather(comfy.start_comfyui(req), comfy.start_comfyui(req))

    a, b = asyncio.run(both())
    assert len(spawned) == 1
    assert {a["status"], b["status"]} == {"started", "already running"}
    assert state.comfyui_pid == spawned[0].pid
