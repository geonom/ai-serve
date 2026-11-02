"""Music and sound effects (image_serve/audio.py, 2026-10-30): request checks, the two ComfyUI graphs, the sound prompt
rewrite, and the job flow through the plugin's endpoints with a faked ComfyUI."""
import asyncio
import base64
import time

import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import audio, comfy, config
from image_serve.state import state

client = TestClient(plugin.app, client=("127.0.0.1", 50000))


@pytest.fixture(autouse=True)
def installed(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "available", lambda m, root: True)
    monkeypatch.setattr(config, "GENERATED_AUDIO_DIR", str(tmp_path / "out"))


# ------------------------------------------------------------------ requests

def test_defaults_per_task_and_checks():
    r = audio.parse_request({"prompt": "Regen auf einem Blechdach", "task": "sound"})
    assert r.model.id == "sa3-sfx" and r.seconds is None and r.enhance is True and r.seed > 0
    r = audio.parse_request({"prompt": "Ruhige Tavernenmusik, Laute und Flöte"})
    assert r.model.id == "ace-step"                                        # no task: music
    for bad in ({"prompt": ""}, {"prompt": "x", "task": "speech"}, {"prompt": "x", "model": "dall-e"},
                {"prompt": "x", "model": "sa3-sfx", "task": "music"}, {"prompt": "x", "model": "sa3-sfx", "lyrics": "la"},
                {"prompt": "x", "model": "sa3-sfx", "seconds": 500}, {"prompt": "x", "seconds": "10"},
                {"prompt": "x", "language": "deutsch"}, {"prompt": "x", "keyscale": "H dur"},
                {"prompt": "x", "enhance": "yes"}, {"prompt": "x", "seed": True}, ["x"]):
        with pytest.raises(audio.AudioError):
            audio.parse_request(bad)


def test_the_graphs_carry_the_request():
    w = audio.sa3_workflow("Heavy rain on a tin roof", 12, 7, "audio/aud-1")
    assert w["6"]["inputs"]["text"] == "Heavy rain on a tin roof" and w["11"]["inputs"]["seconds"] == 12.0
    assert w["3"]["inputs"]["seed"] == 7 and w["26"]["inputs"]["type"] == "stable_audio"
    assert comfy.comfy_model_files(w) == {"stable_audio_3_small_sfx.safetensors", "t5gemma_b_b_ul2.safetensors"}
    r = audio.parse_request({"prompt": "Folk, warm", "lyrics": "[Verse]\nIm Hafen brennt ein Licht", "language": "de",
                             "bpm": 96, "seed": 3})
    w = audio.ace_workflow(r, 90, "audio/aud-2")
    enc = w["94"]["inputs"]
    assert enc["tags"] == "Folk, warm" and enc["lyrics"].startswith("[Verse]") and enc["language"] == "de"
    assert enc["bpm"] == 96 and enc["duration"] == 90.0 and w["98"]["inputs"]["seconds"] == 90.0
    assert audio.ace_workflow(audio.parse_request({"prompt": "Ambient"}), 30, "p")["94"]["inputs"]["lyrics"] == "[Instrumental]"
    assert len(comfy.comfy_model_files(w)) == 4


def test_the_rewrite_suggests_a_length_and_bad_answers_keep_the_prompt():
    assert audio.parse_enhanced("Wooden door creaking open slowly, echoing interior. Length: 3 seconds", "Tür") == \
        ("Wooden door creaking open slowly, echoing interior.", 3)
    assert audio.parse_enhanced("<think>hm</think>Rain on a tin roof, close.", "Regen") == ("Rain on a tin roof, close.", None)
    assert audio.parse_enhanced(None, "Regen") == ("Regen", None)
    assert audio.parse_enhanced("x" * 2000, "Regen") == ("Regen", None)


# ------------------------------------------------------------------ jobs through the endpoints

@pytest.fixture
def fake_comfy(monkeypatch):
    seen = {}

    async def queue(workflow, label):
        seen["workflow"], seen["label"] = workflow, label
        state.comfyui_tasks["p1"] = {"status": "processing", "progress": {"value": 50}}
        return {"status": "queued", "task_id": "p1"}

    async def wait(pid, timeout_s=0):
        await asyncio.sleep(0.05)
        state.comfyui_tasks[pid]["status"] = "completed"
        return base64.b64encode(b"ID3fake-mp3").decode()

    async def enhance(prompt, gateway, model, timeout=0):
        seen["enhanced"] = prompt
        return "Heavy rain hitting a metal roof, distant thunder.", 20
    monkeypatch.setattr(comfy, "queue_workflow", queue)
    monkeypatch.setattr(comfy, "_await_comfyui_task", wait)
    monkeypatch.setattr(audio, "enhance_sound", enhance)
    return seen


def test_a_sound_job_runs_and_its_file_can_be_fetched(fake_comfy):
    with TestClient(plugin.app, client=("127.0.0.1", 50000)) as c:
        r = c.post("/v1/audio/jobs", json={"task": "sound", "prompt": "Regen auf einem Blechdach, Gewitter"})
        assert r.status_code == 202 and r.json()["status"] == "queued"
        jid = r.json()["id"]
        for _ in range(100):
            st = c.get(f"/v1/audio/jobs/{jid}").json()
            if st["status"] == "done":
                break
            time.sleep(0.02)
        assert st["status"] == "done" and st["progress"] == 1.0
        f = st["result"]["files"][0]
        assert f["type"] == "audio/mpeg" and f["url"] == f"/v1/audio/jobs/{jid}/files/{jid}.mp3"
        assert st["result"]["data"]["prompt"] == "Heavy rain hitting a metal roof, distant thunder."
        assert st["result"]["data"]["seconds"] == 20                          # the rewrite's suggestion
        assert fake_comfy["workflow"]["6"]["inputs"]["text"].startswith("Heavy rain")
        got = c.get(f["url"])
        assert got.status_code == 200 and got.content == b"ID3fake-mp3"
        assert c.get(f"/v1/audio/jobs/{jid}/files/other.mp3").status_code == 404
        assert c.get("/v1/audio/jobs/aud-nothing").status_code == 404


def test_without_enhance_and_with_seconds_the_prompt_is_used_as_is(fake_comfy):
    with TestClient(plugin.app, client=("127.0.0.1", 50000)) as c:
        jid = c.post("/v1/audio/jobs", json={"model": "sa3-sfx", "prompt": "door creak", "seconds": 4,
                                             "enhance": False}).json()["id"]
        for _ in range(100):
            if c.get(f"/v1/audio/jobs/{jid}").json()["status"] == "done":
                break
            time.sleep(0.02)
    assert "enhanced" not in fake_comfy and fake_comfy["workflow"]["11"]["inputs"]["seconds"] == 4.0


def test_bad_requests_and_missing_models_are_400(monkeypatch):
    assert client.post("/v1/audio/jobs", json={"prompt": ""}).status_code == 400
    monkeypatch.setattr(audio, "available", lambda m, root: False)
    r = client.post("/v1/audio/jobs", json={"prompt": "Ambient"})
    assert r.status_code == 400 and "ace-step" in r.text


def test_catalog_rows():
    rows = {r["id"]: r for r in client.get("/v1/audio/models").json()["data"]}
    assert rows["sa3-sfx"]["tasks"] == ["sound"] and rows["ace-step"]["tasks"] == ["music"]
    assert rows["sa3-sfx"]["max_seconds"] == 120 and rows["ace-step"]["lyrics"] is True
    assert rows["sa3-sfx"]["default"] and rows["ace-step"]["default"]
