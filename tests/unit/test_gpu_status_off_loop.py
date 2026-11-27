"""fetch_gpu_status runs nvidia-smi off the event loop (2026-11-26: the loop watch caught it blocking the gateway
for 0.8 s at a time under the dashboards' polling)."""
import asyncio
import platform
import time

import pytest

from backend import model_manager


@pytest.mark.skipif(platform.system() == "Darwin", reason="the NVIDIA path")
def test_nvidia_smi_does_not_block_the_loop(monkeypatch):
    def slow_smi(cmd, **kw):
        time.sleep(0.5)
        return "0, RTX, 580.1, 24576, 1000, 23576, GPU-abc\n"

    monkeypatch.setattr(model_manager.subprocess, "check_output", slow_smi)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.02)

        t = asyncio.create_task(ticker())
        out = await model_manager.fetch_gpu_status()
        t.cancel()
        return ticks, out

    ticks, out = asyncio.run(main())
    assert ticks >= 10          # the loop kept running while nvidia-smi was busy
    assert out["gpus"][0]["index"] == 0
