"""ComfyUI's refusal of a queued workflow (POST /prompt) reaches the caller as what it is (2026-11-27: the Playground
sent model_name "undefined", ComfyUI answered 400 naming the bad input, the caller got a bare 500)."""
from image_serve import comfy


def test_a_validation_refusal_is_the_callers_400_with_comfyuis_reason():
    e = comfy.prompt_refusal(400, '{"error": {"message": "Prompt outputs failed validation", "details": "unet_name: \'undefined\' not in list"}}', "image")
    assert e.status_code == 400
    assert "undefined" in e.detail and "image" in e.detail


def test_anything_else_from_comfyui_is_a_server_error():
    e = comfy.prompt_refusal(500, "boom", "MOSS")
    assert e.status_code == 502 and "boom" in e.detail


def test_long_answers_are_cut():
    assert len(comfy.prompt_refusal(400, "x" * 5000, "image").detail) < 500
