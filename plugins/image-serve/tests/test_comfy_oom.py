"""ComfyUI out of memory mid-job (2026-10-24, owner: nothing may fail just because another model fills the GPU):
its models are freed, the job queued again -- which asks ai-serve's GPU manager for the full need again (it may
unload an idle model or move ComfyUI to the other GPU) -- and on a second out-of-memory the answer is 507
(retryable), not 500. Other failures pass through unchanged."""
import asyncio

import pytest
from fastapi import HTTPException

from image_serve import app as plugin
from image_serve import comfy


@pytest.fixture
def world(monkeypatch):
    w = {"submits": 0, "frees": 0, "results": []}

    async def submit(req):
        w["submits"] += 1
        return {"task_id": f"t{w['submits']}"}

    async def wait(pid):
        r = w["results"].pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def free(force=False):
        w["frees"] += 1
        return {"status": "unloaded"}
    monkeypatch.setattr(comfy, "_await_comfyui_task", wait)
    monkeypatch.setattr(comfy, "unload_comfyui", free)
    w["submit"] = submit
    return w


OOM = HTTPException(status_code=500, detail="Image generation failed: ComfyUI failed in KSampler: Allocation on device")
TORCH_OOM = HTTPException(status_code=500, detail="Image generation failed: ComfyUI failed in VAEDecode: "
                                                  "CUDA out of memory. Tried to allocate 2.00 GiB")


@pytest.mark.parametrize("err", [OOM, TORCH_OOM])
def test_out_of_memory_frees_and_retries_once(world, err):
    world["results"] = [err, "UE5H"]
    assert asyncio.run(plugin._run(world["submit"], None, None)) == "UE5H"
    assert world["submits"] == 2 and world["frees"] == 1


def test_out_of_memory_twice_is_a_retryable_507(world):
    world["results"] = [OOM, OOM]
    with pytest.raises(HTTPException) as e:
        asyncio.run(plugin._run(world["submit"], None, None))
    assert e.value.status_code == 507 and "try again" in e.value.detail.lower()


def test_other_failures_pass_through(world):
    world["results"] = [HTTPException(status_code=500, detail="Image generation failed: ComfyUI failed in X: bad")]
    with pytest.raises(HTTPException) as e:
        asyncio.run(plugin._run(world["submit"], None, None))
    assert e.value.status_code == 500 and world["submits"] == 1 and world["frees"] == 0
