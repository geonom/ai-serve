"""Video (image_serve/video.py, 2026-11-13): catalog, request checks, the ComfyUI graphs (LTX one/two stages, Wan
A14B, Wan 5B, LoRA wiring) and the job flow through the plugin's endpoints with a faked ComfyUI."""
import asyncio
import base64
import time

import pytest
from fastapi.testclient import TestClient

from image_serve import app as plugin
from image_serve import comfy, config, video
from image_serve.state import state

client = TestClient(plugin.app, client=("127.0.0.1", 50000))
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 64).decode()


@pytest.fixture
def root(tmp_path):
    """A models root with every catalog file, the upscaler and one LoRA per family."""
    for m in video.CATALOG:
        for d, f in m.files:
            (tmp_path / d / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / d / f).write_bytes(b"x")
    (tmp_path / "latent_upscale_models").mkdir()
    (tmp_path / "latent_upscale_models" / video.UPSCALER).write_bytes(b"x")
    for fam, name in (("ltx-2.5", "style A"), ("wan-2.2", "motion")):
        (tmp_path / "loras" / fam).mkdir(parents=True)
        (tmp_path / "loras" / fam / f"{name}.safetensors").write_bytes(b"x")
    (tmp_path / "loras" / "ltx-2.5" / "notes.txt").write_text("not a lora")
    return str(tmp_path)


# ------------------------------------------------------------------ catalog

def test_listing_marks_installed_models_and_their_loras(root, tmp_path):
    rows = {r["id"]: r for r in video.listing(root)}
    assert rows["ltx-2.5"]["available"] and rows["ltx-2.5"]["audio"] and rows["ltx-2.5"]["tasks"] == ["text_to_video", "image_to_video"]
    assert rows["ltx-2.5"]["loras"] == ["style A"]          # .txt ignored
    assert rows["wan-2.2"]["loras"] == ["motion"] and not rows["wan-2.2"]["audio"]
    assert rows["wan-2.2-5b"]["loras"] == []                # its own family folder
    assert rows["ltx-2.5"]["default"] and not rows["wan-2.2"]["default"]          # LTX is the default for both tasks
    assert rows["ltx-2.5"]["prompt_guide"] == rows["ltx-2.5-q3"]["prompt_guide"] == "ltx-2"
    assert rows["wan-2.2"]["prompt_guide"] == rows["wan-2.2-5b"]["prompt_guide"] == "wan-2.2"
    assert rows["minimax-h3"]["prompt_guide"] == rows["minimax-h3-4step"]["prompt_guide"] == "minimax-h3"
    assert not any(r["available"] for r in video.listing(str(tmp_path / "empty")))


# ------------------------------------------------------------------ requests

def test_default_model_and_the_picture(root):
    r = video.parse_request({"prompt": "a cat"}, root)
    assert r.model.id == "ltx-2.5" and r.image is None
    r = video.parse_request({"prompt": "the cat jumps", "image_base64": PNG}, root)
    assert r.model.id == "ltx-2.5" and r.image.startswith(b"\x89PNG")                # LTX does pictures too
    assert video.parse_request({"prompt": "x", "task": "image_to_video", "image_base64": "data:image/png;base64," + PNG},
                               root).model.id == "ltx-2.5"
    with pytest.raises(video.VideoError, match="image_base64"):
        video.parse_request({"prompt": "x", "task": "image_to_video"}, root)       # asked for, but no picture


@pytest.mark.parametrize("body, msg", [
    ({"prompt": ""}, "prompt"),
    ({"prompt": "x", "model": "sora"}, "unknown video model"),
    ({"prompt": "x", "model": "wan-2.2", "task": "text_to_video"}, "not text_to_video"),
    ({"prompt": "x", "model": "wan-2.2"}, "image_base64"),
    ({"prompt": "x", "model": "wan-2.2", "image_base64": base64.b64encode(b"GIF89a....").decode()}, "PNG, JPEG"),
    ({"prompt": "x", "model": "wan-2.2", "image_base64": "data:text/html;base64," + PNG}, "data:image"),
    ({"prompt": "x", "seconds": 60}, "seconds"),
    ({"prompt": "x", "resolution": "1920x1080"}, "resolution"),
    ({"prompt": "x", "lora": "motion"}, "lora for"),            # a Wan LoRA on LTX
    ({"prompt": "x", "lora": "../../etc/passwd"}, "lora for"),
    ({"prompt": "x", "lora": "style A", "lora_strength": 5}, "lora_strength"),
    ({"prompt": "x", "seed": "1"}, "seed"),
])
def test_refusals(root, body, msg):
    with pytest.raises(video.VideoError, match=msg):
        video.parse_request(body, root)


def test_frames_fit_each_family(root):
    assert video.parse_request({"prompt": "x", "seconds": 4}, root).frames == 97              # LTX: 8n+1 at 24 fps
    r = video.parse_request({"prompt": "x", "image_base64": PNG, "seconds": 5, "model": "wan-2.2"}, root)
    assert r.frames == 81                                                                     # Wan: 4n+1 at 16 fps


def test_big_ltx_needs_the_upscaler(root, tmp_path):
    (tmp_path / "latent_upscale_models" / video.UPSCALER).unlink()
    assert video.parse_request({"prompt": "x", "resolution": "960x544"}, root)
    for res in ("1280x704", "1024x576"):
        with pytest.raises(video.VideoError, match="upscaler"):
            video.parse_request({"prompt": "x", "resolution": res}, root)


def test_default_ltx_size_is_two_stage(root):
    g = video.workflow(video.parse_request({"prompt": "x"}, root), None, "p")
    assert g["vlat"]["inputs"]["width"] == 512 and g["vlat"]["inputs"]["height"] == 288 and "up" in g


# ------------------------------------------------------------------ workflows

def _types(g):
    return [n["class_type"] for n in g.values()]


def test_ltx_one_stage_with_sound(root):
    g = video.workflow(video.parse_request({"prompt": "rain", "seed": 7, "resolution": "960x544"}, root), None, "video/x")
    assert _types(g).count("SamplerCustomAdvanced") == 1 and "LTXVLatentUpsampler" not in _types(g)
    assert g["vlat"]["inputs"]["width"] == 960 and g["vlat"]["inputs"]["length"] == 97
    assert g["vid"]["inputs"]["audio"] == ["adec", 0] and g["save"]["inputs"]["filename_prefix"] == "video/x"
    assert g["unet"]["class_type"] == "UNETLoader" and g["clip"]["inputs"]["type"] == "ltxv"     # int8 safetensors
    assert "lora" not in g


def test_ltx_two_stages_and_lora(root):
    r = video.parse_request({"prompt": "rain", "resolution": "1280x704", "lora": "style A", "lora_strength": 0.6}, root)
    g = video.workflow(r, None, "video/x")
    assert g["vlat"]["inputs"]["width"] == 640 and g["vlat"]["inputs"]["height"] == 352   # half size, then doubled
    assert g["up"]["inputs"]["samples"] == ["sep", 0] and g["av2"]["inputs"]["audio_latent"] == ["sep", 1]
    assert g["dec"]["inputs"]["samples"] == ["sep2", 0] and g["upm"]["inputs"]["model_name"] == video.UPSCALER
    assert g["lora"]["inputs"]["lora_name"] == "ltx-2.5/style A.safetensors"
    assert g["lora"]["inputs"]["strength_model"] == 0.6
    assert g["guide"]["inputs"]["model"] == ["lora", 0] and g["guide2"]["inputs"]["model"] == ["lora", 0]
    assert comfy.comfy_model_files(g) >= {video.UPSCALER, "ltx-2.5/style A.safetensors"}


def test_ltx_picture_to_video_pins_the_first_frame(root):
    r = video.parse_request({"prompt": "x", "image_base64": PNG}, root)                # 1024x576: two stages
    g = video.workflow(r, "start.png", "p")
    assert g["img"]["inputs"]["image"] == "start.png"
    assert g["img_fit"]["inputs"]["width"] == 1024 and g["img_fit"]["inputs"]["crop"] == "center"
    assert g["i2v"]["inputs"]["latent"] == ["vlat", 0] and g["i2v"]["inputs"]["strength"] == 0.7
    assert g["av"]["inputs"]["video_latent"] == ["i2v", 0]
    assert g["i2v2"]["inputs"]["latent"] == ["up", 0] and g["av2"]["inputs"]["video_latent"] == ["i2v2", 0]
    one = video.workflow(video.parse_request({"prompt": "x", "image_base64": PNG, "resolution": "960x544"}, root),
                         "s.png", "p")
    assert one["i2v"]["inputs"]["strength"] == 1.0 and "i2v2" not in one
    assert "img" not in video.workflow(video.parse_request({"prompt": "x"}, root), None, "p")


def test_wan_two_experts_and_lora(root):
    r = video.parse_request({"prompt": "x", "image_base64": PNG, "lora": "motion", "model": "wan-2.2"}, root)
    g = video.workflow(r, "start.png", "video/x")
    assert g["img"]["inputs"]["image"] == "start.png"
    assert g["k1"]["inputs"]["end_at_step"] == 2 and g["k2"]["inputs"]["start_at_step"] == 2
    assert g["hs"]["inputs"]["model"] == ["lora_hi", 0] and g["ls"]["inputs"]["model"] == ["lora_lo", 0]
    assert video._loader("x.safetensors")["class_type"] == "UNETLoader"     # safetensors, not GGUF


def test_sage_attention_default_on_for_wan_off_for_ltx(root):
    wan = video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2"}, root)
    g = video.workflow(wan, "s.png", "p", sage=True)
    assert g["sage_hi"]["class_type"] == "PathchSageAttentionKJ" and g["hs"]["inputs"]["model"] == ["sage_hi", 0]
    assert g["ls"]["inputs"]["model"] == ["sage_lo", 0]
    assert "sage_hi" not in video.workflow(wan, "s.png", "p", sage=False)        # not installed: never
    off = video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2", "sage_attention": False}, root)
    assert "sage_hi" not in video.workflow(off, "s.png", "p", sage=True)          # asked off
    ltx = video.workflow(video.parse_request({"prompt": "x"}, root), None, "p", sage=True)
    assert "PathchSageAttentionKJ" not in _types(ltx)                              # LTX: off by default
    ltx_on = video.workflow(video.parse_request({"prompt": "x", "sage_attention": True}, root), None, "p", sage=True)
    assert ltx_on["guide"]["inputs"]["model"] == ["sage", 0]
    with pytest.raises(video.VideoError, match="sage_attention"):
        video.parse_request({"prompt": "x", "sage_attention": "yes"}, root)
    rows = {r["id"]: r for r in video.listing(root, sage=True)}
    assert rows["wan-2.2"]["sage_attention"] is True and rows["ltx-2.5"]["sage_attention"] is False
    assert video.listing(root, sage=False)[0]["sage_attention"] is None
    small = video.workflow(video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2-5b"}, root),
                           "s.png", "p", sage=True)
    assert small["ms"]["inputs"]["model"] == ["sage", 0]


def test_wan_5b_single_model(root):
    r = video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2-5b"}, root)
    g = video.workflow(r, "s.png", "p")
    assert g["lat"]["class_type"] == "Wan22ImageToVideoLatent" and g["lat"]["inputs"]["width"] == 1280
    assert g["k"]["inputs"]["steps"] == 4 and g["vid"]["inputs"]["fps"] == 24


def test_steps_are_listed_and_checked(root):
    rows = {r["id"]: r for r in video.listing(root)}
    assert rows["ltx-2.5"]["steps"] == {"default": 8, "min": 4, "max": 20, "adjustable": True}
    assert rows["wan-2.2"]["steps"]["default"] == 4 and rows["wan-2.2-5b"]["steps"]["default"] == 4
    assert video.parse_request({"prompt": "x"}, root).steps == 8
    for bad in (3, 21, 6.5, True, "8"):
        with pytest.raises(video.VideoError, match="steps"):
            video.parse_request({"prompt": "x", "steps": bad}, root)


def test_ltx_steps_stretch_the_distilled_schedule():
    assert video.ltx_sigmas(8) == video.LTX_STAGE1_SIGMAS                      # the default is Lightricks' own
    for n in (4, 12, 20):
        s = [float(x) for x in video.ltx_sigmas(n).split(",")]
        assert len(s) == n + 1 and s[0] == 1.0 and s[-1] == 0.0
        assert all(a > b for a, b in zip(s, s[1:]))                            # strictly falling
    s12 = [float(x) for x in video.ltx_sigmas(12).split(",")]
    assert s12[6] < 0.98 and s12[3] > 0.97                                     # keeps the slow start of the curve


def test_steps_reach_every_workflow(root):
    ltx = video.workflow(video.parse_request({"prompt": "x", "steps": 12}, root), None, "p")
    assert len(ltx["sig"]["inputs"]["sigmas"].split(",")) == 13
    assert ltx["sig2"]["inputs"]["sigmas"] == video.LTX_STAGE2_SIGMAS           # the refine stage stays 3 steps
    wan = video.workflow(video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2", "steps": 7},
                                             root), "s.png", "p")
    assert wan["k1"]["inputs"]["steps"] == wan["k2"]["inputs"]["steps"] == 7
    assert wan["k1"]["inputs"]["end_at_step"] == 4 and wan["k2"]["inputs"]["start_at_step"] == 4   # high noise: half, rounded up
    small = video.workflow(video.parse_request({"prompt": "x", "image_base64": PNG, "model": "wan-2.2-5b",
                                                "steps": 10}, root), "s.png", "p")
    assert small["k"]["inputs"]["steps"] == 10


def test_minimax_h3_frames_follow_its_17k_plus_5_grid(root):
    m = video.resolve("minimax-h3")
    assert m.audio and m.fps == 24 and m.tasks == ("text_to_video", "image_to_video")
    assert video.frame_count(m, 5) == 124 and video.frame_count(m, 3) == 73     # Comfy-Org's template formula
    assert all((video.frame_count(m, s) - 5) % 17 == 0 for s in (2, 4, 7, 10))


def test_minimax_h3_text_to_video_follows_comfy_orgs_template(root):
    r = video.parse_request({"prompt": "rain on a tin roof", "model": "minimax-h3", "seed": 3}, root)
    assert r.steps == 6                                             # the 8-step turbo LoRA at the template's 6
    g = video.workflow(r, None, "video/x")
    assert g["unet"]["class_type"] == "UnetLoaderGGUF" and g["clip"]["class_type"] == "CLIPLoader"
    assert g["clip"]["inputs"]["type"] == "minimax"
    assert g["speed"]["inputs"]["lora_name"].startswith("minimax-h3-speed/") and g["speed"]["inputs"]["model"] == ["unet", 0]
    assert g["cond"]["class_type"] == "MiniMaxH3ImageToVideo" and g["cond"]["inputs"]["length"] == 124
    assert "first_frame" not in g["cond"]["inputs"] and g["cond"]["inputs"]["width"] == 864
    assert g["sched"]["inputs"] == {"model": ["speed", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}
    assert g["samp"]["inputs"]["sampler_name"] == "res_multistep" and g["guide"]["class_type"] == "BasicGuider"
    assert g["run"]["inputs"]["latent_image"] == ["cond", 1]
    assert g["dec"]["inputs"]["samples"] == ["run", 0] and g["adec"]["class_type"] == "VAEDecodeAudio"
    assert g["vid"]["inputs"]["audio"] == ["adec", 0] and g["vid"]["inputs"]["fps"] == 24
    assert comfy.comfy_model_files(g) >= {f for _, f in r.model.files}


def test_minimax_h3_picture_is_the_first_frame_and_the_fast_row_uses_four_steps(root):
    g = video.workflow(video.parse_request({"prompt": "x", "model": "minimax-h3", "image_base64": PNG}, root),
                       "start.png", "p")
    assert g["cond"]["inputs"]["first_frame"] == ["img", 0] and g["img"]["inputs"]["image"] == "start.png"
    fast = video.parse_request({"prompt": "x", "model": "minimax-h3-4step"}, root)
    assert fast.steps == 4 and "4step" in video.workflow(fast, None, "p")["speed"]["inputs"]["lora_name"]


# ------------------------------------------------------------------ jobs through the endpoints

def test_job_flow_with_a_fake_comfyui(monkeypatch, root, tmp_path):
    monkeypatch.setattr(config, "COMFYUI_MODELS_ROOT", root)
    monkeypatch.setattr(config, "GENERATED_VIDEO_DIR", str(tmp_path / "out"))
    seen = {}

    async def ensure(*a, **k):
        return None

    async def upload(data):
        seen["upload"] = data[:4]
        return "uploaded.png"

    monkeypatch.setattr(plugin, "_sage_available", lambda: False)

    async def queue(workflow, label):
        seen["workflow"] = workflow
        state.comfyui_tasks["p1"] = {"status": "processing", "progress": {"value": 50}}
        return {"status": "queued", "task_id": "p1"}

    async def wait(pid, timeout_s=0):
        return base64.b64encode(b"MP4DATA").decode()

    monkeypatch.setattr(comfy, "_ensure_comfyui_running", ensure)
    monkeypatch.setattr(comfy, "_upload_image_to_comfyui", upload)
    monkeypatch.setattr(comfy, "queue_workflow", queue)
    monkeypatch.setattr(comfy, "_await_comfyui_task", wait)

    r = client.post("/v1/video/jobs", json={"prompt": "the cat jumps", "image_base64": PNG, "seed": 3, "model": "wan-2.2"})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(50):
        st = client.get(f"/v1/video/jobs/{jid}").json()
        if st["status"] == "done":
            break
        time.sleep(0.02)
    assert st["status"] == "done" and seen["upload"] == b"\x89PNG"
    assert seen["workflow"]["img"]["inputs"]["image"] == "uploaded.png"
    f = st["result"]["files"][0]
    assert f["type"] == "video/mp4" and f["url"] == f"/v1/video/jobs/{jid}/files/{jid}.mp4"
    assert st["result"]["data"]["frames"] == 81 and st["result"]["data"]["seed"] == 3
    assert client.get(f["url"]).content == b"MP4DATA"
    assert client.get(f"/v1/video/jobs/{jid}/files/other.mp4").status_code == 404


def test_endpoint_refusals(monkeypatch, root):
    monkeypatch.setattr(config, "COMFYUI_MODELS_ROOT", root)
    assert client.post("/v1/video/jobs", json={"prompt": "x", "model": "nope"}).status_code == 400
    assert client.get("/v1/video/jobs/vid-unknown").status_code == 404
    # the client normalises this to /etc, which lands on the ComfyUI pass-through: 404 where ComfyUI runs, 503
    # where it isn't installed (the public tree) -- never a file from below the jobs folder
    assert client.get("/v1/video/jobs/../../etc").status_code in (404, 503)
    rows = client.get("/v1/video/models").json()["data"]
    assert {r["id"] for r in rows} >= {"ltx-2.5", "wan-2.2"}


def test_no_gpu_room_is_a_507_before_any_job_exists(monkeypatch, root):
    """Fast yes/no (2026-11-15): a job that can't get room is refused at once, not accepted and failed later."""
    from fastapi import HTTPException
    monkeypatch.setattr(config, "COMFYUI_MODELS_ROOT", root)
    asked = {}

    async def no(workflow, label):
        asked["files"], asked["label"] = comfy.comfy_model_files(workflow), label
        raise HTTPException(status_code=507, detail="No room for the image model on GPU 0: busy")
    monkeypatch.setattr(comfy, "check_room", no)
    before = len(video.JOBS._jobs)
    r = client.post("/v1/video/jobs", json={"prompt": "the cat jumps", "image_base64": PNG, "model": "wan-2.2"})
    assert r.status_code == 507 and "No room" in r.json()["detail"]
    assert asked["label"] == "Wan 2.2 14B (4 steps, Q4)" and asked["files"]            # the model's own files
    assert len(video.JOBS._jobs) == before


def test_sulphur_3_is_minimax_h3_with_its_own_main_model(root):
    g = video.workflow(video.parse_request({"prompt": "x", "model": "sulphur-3", "seed": 1}, root), None, "p")
    assert g["unet"]["inputs"]["unet_name"] == "Sulphur-3-Q4_K_M.gguf"
    h3 = video.workflow(video.parse_request({"prompt": "x", "model": "minimax-h3", "seed": 1}, root), None, "p")
    assert {k: v for k, v in g.items() if k != "unet"} == {k: v for k, v in h3.items() if k != "unet"}
    row = next(r for r in video.listing(str(root)) if r["id"] == "sulphur-3")
    assert row["audio"] and row["prompt_guide"] == "minimax-h3" and not row["default"]
