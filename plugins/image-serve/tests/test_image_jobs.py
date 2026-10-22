"""Cancelling one running image (POST /v1/images/cancel): a caller tags its generation with an unguessable
X-Request-Id; cancel stops exactly that job in ComfyUI / MOSS's ComfyUI (queue delete + targeted interrupt),
never anyone else's. User 2026-10-19: "Abbrechen works, but only after the next image is generated fully"."""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from image_serve import jobs
from image_serve import comfy as media
from image_serve.state import state


def test_request_ids_are_checked():
    assert jobs.valid_request_id("a" * 32) == "a" * 32
    for bad in (None, "", "short", "../../x" + "a" * 20, "a" * 65, "a b" * 10, 5):
        assert jobs.valid_request_id(bad) is None


def test_cancel_runs_the_registered_stop_once():
    reg = jobs.Jobs()
    stopped = []

    async def stop():
        stopped.append(1)
    reg.register("r" * 32, stop)
    assert asyncio.run(reg.cancel("r" * 32)) is True
    assert stopped == [1] and reg.cancelled("r" * 32)
    assert asyncio.run(reg.cancel("x" * 32)) is False            # unknown / someone else's id: nothing happens
    reg.unregister("r" * 32)
    assert asyncio.run(reg.cancel("r" * 32)) is False


def test_cancel_before_the_job_registered_still_counts():
    """The click may arrive while the request is still on its way to ComfyUI."""
    reg = jobs.Jobs()
    assert asyncio.run(reg.cancel("e" * 32)) is False
    assert reg.cancelled("e" * 32)


@pytest.fixture
def client(monkeypatch):
    from image_serve.app import app
    interrupts = []

    async def fake_generate(req):
        state.comfyui_tasks["p1"] = {"status": "processing"}
        return {"task_id": "p1"}

    async def fake_interrupt(base, prompt_id):
        interrupts.append((base, prompt_id))
    monkeypatch.setattr(media, "comfyui_generate", fake_generate)
    monkeypatch.setattr(jobs, "interrupt_comfy", fake_interrupt)
    monkeypatch.setattr(jobs, "JOBS", jobs.Jobs())
    c = TestClient(app, client=("127.0.0.1", 50000))
    c.interrupts = interrupts
    return c


def test_cancelling_a_comfy_job_ends_the_waiting_call_with_409(client, monkeypatch):
    async def cancel_soon(prompt_id, timeout_s=10):
        await jobs.JOBS.cancel("c" * 32)                          # as if the cancel arrived mid-way
        return await original(prompt_id, timeout_s=3)
    original = media._await_comfyui_task
    monkeypatch.setattr(media, "_await_comfyui_task", cancel_soon)
    r = client.post("/v1/images/generations", json={"model": "z-image-turbo", "prompt": "x"},
                    headers={"X-Request-Id": "c" * 32})
    assert r.status_code == 409 and "cancel" in r.json()["detail"].lower()
    assert client.interrupts and client.interrupts[0][1] == "p1"
    assert state.comfyui_tasks["p1"]["status"] == "failed"
    state.comfyui_tasks.pop("p1", None)


def test_cancel_endpoint(client):
    assert client.post("/v1/images/cancel", json={"request_id": "z" * 32}).json() == {"cancelled": False}
    assert client.post("/v1/images/cancel", json={"request_id": "../x"}).status_code == 400
