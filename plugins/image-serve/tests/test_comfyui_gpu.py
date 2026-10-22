"""ComfyUI asks the central GPU manager before a job -- but only when the job needs
other models than the ones already loaded (user, 2026-10-16: "wenn das Modell
schon geladen ist, muss es ja nicht geschätzt und gefragt werden").

- The models a job needs = the files its loader nodes name (UNet/GGUF, CLIP/text
  encoder, VAE, LoRA, checkpoint).
- Same set as already loaded -> no question asked.
- Otherwise need = their file sizes + a margin for sampling, minus what ComfyUI
  already holds (it frees its old models itself when switching); a no from the
  manager is a fast 507 and the job isn't queued.
- After /free (idle timer, Unload, the manager) nothing counts as loaded."""
import asyncio

import pytest
from fastapi import HTTPException

from image_serve import comfy as media
from image_serve.state import state

WF = {
    "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "z_image_turbo-Q6_K.gguf"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "z-image-turbo/pen.safetensors",
                                                          "model": ["1", 0]}},
    "5": {"class_type": "KSampler", "inputs": {"seed": 1, "model": ["4", 0]}},
}


def test_model_files_from_loader_nodes():
    assert media.comfy_model_files(WF) == frozenset({
        "z_image_turbo-Q6_K.gguf", "qwen_3_4b.safetensors", "ae.safetensors", "z-image-turbo/pen.safetensors"})


def test_bypassed_lora_not_counted():
    wf = {k: v for k, v in WF.items()}
    wf["5"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}  # LoRA node not wired in
    assert "z-image-turbo/pen.safetensors" not in media.comfy_model_files(wf)


@pytest.fixture
def fake(monkeypatch):
    calls = []

    async def reserve(gpu, need, purpose):
        calls.append((gpu, need, purpose))
        return media.Reservation(id="c1", gpu=gpu, mb=need, purpose=purpose, created=0), [], ""
    from image_serve import gpu as local
    monkeypatch.setattr(local.MANAGER, "reserve", reserve)

    async def room():
        return {0: 0, 1: 99999}     # by default ComfyUI's own GPU (1) has room: no move
    monkeypatch.setattr(local.MANAGER, "room", room)
    monkeypatch.setattr(media, "_comfy_file_mb", lambda name: {"z_image_turbo-Q6_K.gguf": 5000,
                                                               "qwen_3_4b.safetensors": 7000,
                                                               "ae.safetensors": 300}.get(name, 100))

    async def held(gpu):
        return 0
    monkeypatch.setattr(media, "_comfy_held_mb", held)
    monkeypatch.setattr(state, "comfyui_gpu_id", 1)
    monkeypatch.setattr(state, "comfyui_loaded_models", None)
    monkeypatch.setattr(state, "comfyui_reservation_id", None)
    monkeypatch.setattr(state, "comfyui_process", None)
    return calls


def test_asks_when_models_change_and_not_again(fake):
    asyncio.run(media._comfyui_gpu_preflight(WF, "z-image-turbo"))
    # ComfyUI computes with one model at a time and offloads the others itself under pressure: what it
    # can't do without is the largest one plus the sampling margin (the sum made ERNIE -- 3B text encoder
    # 7.4 GB + UNet 6.5 GB -- impossible on the 16 GB card, found live 2026-10-18)
    assert fake == [(1, 7000 + media.COMFY_SAMPLING_MARGIN_MB, "image:z-image-turbo")]
    asyncio.run(media._comfyui_gpu_preflight(WF, "z-image-turbo"))
    assert len(fake) == 1                      # same models already loaded -> no question


def test_after_free_asks_again(fake):
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    media._comfyui_models_freed()
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    assert len(fake) == 2


def test_what_comfyui_holds_counts(fake, monkeypatch):
    async def held(gpu):
        return 5000
    monkeypatch.setattr(media, "_comfy_held_mb", held)
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    assert fake[0][1] == 7000 + media.COMFY_SAMPLING_MARGIN_MB - 5000


def test_refusal_is_a_fast_507(monkeypatch, fake):
    from image_serve import gpu as local

    async def no(gpu, need, purpose):
        return None, [], "GPU 1: held by gemma (busy, 9000 MB)"
    monkeypatch.setattr(local.MANAGER, "reserve", no)
    with pytest.raises(HTTPException) as e:
        asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    assert e.value.status_code == 507 and "gemma" in e.value.detail
    assert state.comfyui_loaded_models is None


def test_reservation_is_released_when_the_job_is_done(fake, monkeypatch):
    """Found live 2026-10-18: the reservation stayed open up to 2 min after each image, so ComfyUI counted as
    'still loading' (busy, not unloadable) and Iris/MOSS right after a Z-Image picture got a 507. Once a job
    has finished, the models are allocated and the measured VRAM counts -- like llama-server after /health."""
    from image_serve import gpu as local
    released = []
    monkeypatch.setattr(local.MANAGER, "release", lambda rid: released.append(rid))
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    assert state.comfyui_reservation_id == "c1" and released == []
    media._release_comfyui_reservation()
    assert released == ["c1"] and state.comfyui_reservation_id is None
    media._release_comfyui_reservation()        # nothing open -> nothing to do
    assert released == ["c1"]


def test_free_also_releases(fake, monkeypatch):
    from image_serve import gpu as local
    released = []
    monkeypatch.setattr(local.MANAGER, "release", lambda rid: released.append(rid))
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    media._comfyui_models_freed()
    assert released == ["c1"]



def test_moves_to_a_free_gpu_instead_of_unloading_others(fake, monkeypatch):
    """2026-10-19: ComfyUI was pinned to GPU 1 and pushed the other image services out while GPU 0 was free.
    An idle ComfyUI whose GPU is short now restarts on the GPU with room."""
    from image_serve import gpu as local

    async def room():
        return {0: 20000, 1: 1000}
    monkeypatch.setattr(local.MANAGER, "room", room)

    class P:
        pid = 777
    monkeypatch.setattr(state, "comfyui_process", P())
    moved = []

    async def restart_on(gpu):
        moved.append(gpu)
        state.comfyui_gpu_id = gpu
    monkeypatch.setattr(media, "_restart_comfyui_on", restart_on)

    async def idle():
        return None
    monkeypatch.setattr(media, "_comfyui_busy_reason", idle)
    asyncio.run(media._comfyui_gpu_preflight(WF, "z"))
    assert moved == [0]
    assert fake[0][0] == 0


def test_a_pinned_request_never_moves_it(fake, monkeypatch):
    from image_serve import gpu as local

    async def room():
        return {0: 20000, 1: 1000}
    monkeypatch.setattr(local.MANAGER, "room", room)
    moved = []

    async def restart_on(gpu):
        moved.append(gpu)
    monkeypatch.setattr(media, "_restart_comfyui_on", restart_on)
    asyncio.run(media._comfyui_gpu_preflight(WF, "z", pinned=True))
    assert moved == [] and fake[0][0] == 1
