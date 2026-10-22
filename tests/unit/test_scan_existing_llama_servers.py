"""Unit tests for backend/model_manager.py's scan_existing_llama_servers() --
no real subprocesses, no real system process scan (psutil.process_iter mocked)."""
import asyncio

from backend import model_manager
from backend.state import state


class _FakeProc:
    """Stands in for a psutil.Process the way process_iter() would yield one --
    only .as_dict() (via backend.procs), .pid and .environ() are touched by
    scan_existing_llama_servers()."""

    def __init__(self, pid, cmdline, cuda_visible_devices):
        self.pid = pid
        self.info = {"pid": pid, "name": "llama-server", "cmdline": cmdline}
        self._cuda_visible_devices = cuda_visible_devices

    def as_dict(self, attrs, ad_value=None):
        return {a: self.info.get(a, ad_value) for a in attrs}

    def environ(self):
        if self._cuda_visible_devices is None:
            return {}
        return {"CUDA_VISIBLE_DEVICES": self._cuda_visible_devices}


def _run_scan_and_collect(monkeypatch, fake_procs):
    monkeypatch.setattr(model_manager.psutil, "process_iter", lambda attrs=None: iter(fake_procs))
    saved = dict(state.managed_processes)
    state.managed_processes.clear()
    try:
        async def _go():
            model_manager.scan_existing_llama_servers()
            # Let any keep_alive_task's create_task() actually schedule before we
            # inspect/cancel it -- create_task() only *schedules* synchronously.
            await asyncio.sleep(0)
            for info in state.managed_processes.values():
                task = info.get("keep_alive_task")
                if task:
                    task.cancel()
        asyncio.run(_go())
        return dict(state.managed_processes)
    finally:
        state.managed_processes.clear()
        state.managed_processes.update(saved)


def test_adopts_instance_and_recovers_gpu_id_from_environ_not_a_cli_flag(monkeypatch):
    # Confirmed live (2026-08-27): llama-server is never launched with a "--gpu-id"/
    # "--gpu_id" CLI flag anywhere in this codebase -- GPU pinning is exclusively via
    # a CUDA_VISIBLE_DEVICES env var on the subprocess. A prior version of this
    # function searched cmdline for that nonexistent flag and always got gpu_id=None.
    cmdline = [
        "/home/me/llama.cpp/build/bin/llama-server",
        "-m", "~/gguf_models/some-model.gguf",
        "--port", "39979", "-c", "8192", "-t", "8", "-ngl", "-1",
        "--host", "127.0.0.1", "--jinja",
    ]
    procs = [_FakeProc(pid=999, cmdline=cmdline, cuda_visible_devices="1")]
    result = _run_scan_and_collect(monkeypatch, procs)

    assert 39979 in result
    assert result[39979]["gpu_id"] == 1
    assert result[39979]["model"].endswith("some-model.gguf")


def test_adopted_instance_gets_a_real_keep_alive_task_not_just_a_number(monkeypatch):
    # Confirmed live: adoption used to store "keep_alive": 300 as a bare number
    # without ever calling reset_keep_alive(), so the asyncio task that actually
    # enforces the timeout was never created -- an adopted instance ran forever
    # regardless of how long it sat idle.
    cmdline = [
        "llama-server", "-m", "~/gguf_models/other-model.gguf",
        "--port", "12345",
    ]
    procs = [_FakeProc(pid=1000, cmdline=cmdline, cuda_visible_devices="0")]
    result = _run_scan_and_collect(monkeypatch, procs)

    assert result[12345]["keep_alive"] == 300
    assert result[12345]["keep_alive_task"] is not None


def test_missing_cuda_visible_devices_falls_back_to_none_gpu_id(monkeypatch):
    # No CUDA_VISIBLE_DEVICES set at all (e.g. adopted from a differently-launched
    # process) should degrade to None, same as before, not raise.
    cmdline = ["llama-server", "-m", "~/gguf_models/x.gguf", "--port", "22222"]
    procs = [_FakeProc(pid=1001, cmdline=cmdline, cuda_visible_devices=None)]
    result = _run_scan_and_collect(monkeypatch, procs)

    assert result[22222]["gpu_id"] is None
