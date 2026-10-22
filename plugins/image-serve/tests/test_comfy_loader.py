"""The generate workflow's UNet loader follows the checkpoint's file type: a .gguf needs ComfyUI-GGUF's
UnetLoaderGGUF, a .safetensors (e.g. Qwen-Image 2.1's official int8 file) the core UNETLoader."""
from image_serve import comfy as media


def build(model_name):
    req = media.ComfyUIGenerateRequest(prompt="a fox", model_name=model_name, aspect_ratio="4:3", base_resolution=1024)
    return media._build_patched_generate_workflow(req)


def loaders(wf):
    return [(n["class_type"], n["inputs"]) for n in wf.values() if n["class_type"] in ("UNETLoader", "UnetLoaderGGUF")]


def test_safetensors_uses_the_core_loader():
    [(cls, inputs)] = loaders(build("qwen_image_2.1_int8_convrot.safetensors"))
    assert cls == "UNETLoader" and inputs == {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}


def test_gguf_uses_the_gguf_loader():
    [(cls, inputs)] = loaders(build("qwen-image-2.1-Q4_K_M.gguf"))
    assert cls == "UnetLoaderGGUF" and inputs == {"unet_name": "qwen-image-2.1-Q4_K_M.gguf"}
    [(cls, _)] = loaders(build("z_image_turbo-Q6_K.gguf"))
    assert cls == "UnetLoaderGGUF"


def test_slow_jobs_are_awaited_not_cut_off(monkeypatch):
    """/v1/images/generations gave up after 120 s while ComfyUI kept computing (Qwen-Image 2.1 int8 at 40 steps
    takes ~150 s here) -- the caller got "Timeout" and the GPU stayed busy for the next request (2026-10-19)."""
    import asyncio
    from image_serve.state import state
    ticks = {"n": 0}

    async def fast_sleep(_):
        ticks["n"] += 1
        if ticks["n"] == 150:
            state.comfyui_tasks["slow"] = {"status": "completed", "image_base64": "UE5H"}
    monkeypatch.setattr(media.asyncio, "sleep", fast_sleep)
    state.comfyui_tasks.pop("slow", None)
    assert asyncio.run(media._await_comfyui_task("slow")) == "UE5H"
    state.comfyui_tasks.pop("slow", None)
