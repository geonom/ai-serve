"""Upscaling (2026-11-24, owner via V-Noveller: reader backgrounds shown zoomed in got soft): Real-ESRGAN x4plus and
x4plus-anime on ComfyUI's own UpscaleModelLoader + ImageUpscaleWithModel; task `upscale`, `scale` 2 (default) or 4,
2x = the 4x result halved with lanczos; results above ~16.8 MP are refused before anything is queued."""
import struct
import zlib

import pytest

from image_serve import catalog, comfy


def _png(w, h):
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def _jpeg(w, h):
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def test_catalog_rows():
    rows = {r["id"]: r for r in catalog.listing(task="upscale")}
    assert set(rows) == {"realesrgan-x4", "realesrgan-x4-anime"}
    assert rows["realesrgan-x4"]["default"] and rows["realesrgan-x4"]["tasks"] == ["upscale"]
    assert catalog.resolve("realesrgan").id == "realesrgan-x4"
    assert catalog.for_task(None, "upscale").id == "realesrgan-x4"
    with pytest.raises(catalog.CatalogError):
        catalog.for_task("realesrgan-x4", "edit")


def test_workflow_4x_and_2x():
    wf4 = comfy.upscale_workflow("in.png", "RealESRGAN_x4plus.pth", 4)
    assert wf4["2"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    assert wf4["5"]["inputs"]["images"] == ["3", 0] and "4" not in wf4
    wf2 = comfy.upscale_workflow("in.png", "RealESRGAN_x4plus.pth", 2)
    assert wf2["4"]["inputs"]["scale_by"] == 0.5 and wf2["5"]["inputs"]["images"] == ["4", 0]
    assert comfy.comfy_model_files(wf2) == frozenset({"RealESRGAN_x4plus.pth"})   # the GPU manager sees its file
    with pytest.raises(ValueError):
        comfy.upscale_workflow("in.png", "RealESRGAN_x4plus.pth", 3)


def test_image_size():
    assert comfy.image_size(_png(1024, 576)) == (1024, 576)
    assert comfy.image_size(_jpeg(1360, 768)) == (1360, 768)
    assert comfy.image_size(b"not a picture") is None


def test_size_limit():
    comfy.check_upscale_size(_png(1024, 576), 4)          # -> 4096x2304
    comfy.check_upscale_size(_png(2720, 1536), 2)         # H3's size -> 5440x3072
    with pytest.raises(ValueError):
        comfy.check_upscale_size(_png(2720, 1536), 4)
    with pytest.raises(ValueError):
        comfy.check_upscale_size(b"junk", 2)


def test_endpoint_checks_before_queueing(monkeypatch):
    from fastapi.testclient import TestClient
    from image_serve import app as app_mod
    from image_serve import config
    monkeypatch.setattr(config, "comfy_ready", lambda: True)
    monkeypatch.setattr(catalog, "available", lambda m, root=None: True)
    queued = []

    async def fake(data, mid, mfile, scale):
        queued.append((mid, mfile, scale))
        return "b64"
    monkeypatch.setattr(comfy, "comfyui_upscale", fake)

    async def run(fn, req, rid):
        return await fn(req)
    monkeypatch.setattr(app_mod, "_run", run)
    c = TestClient(app_mod.app)
    post = lambda img, **f: c.post("/v1/images/edits", data={"task": "upscale", **f},
                                   files={"image": ("a.png", img, "image/png")})
    r = post(_png(1024, 576))
    assert r.status_code == 200 and r.json()["data"][0]["b64_json"] == "b64"
    assert queued[-1] == ("realesrgan-x4", "RealESRGAN_x4plus.pth", 2)
    assert post(_png(1024, 576), scale="4", model="realesrgan-x4-anime").status_code == 200
    assert queued[-1] == ("realesrgan-x4-anime", "RealESRGAN_x4plus_anime_6B.pth", 4)
    assert post(_png(1024, 576), scale="3").status_code == 400
    assert post(_png(2720, 1536), scale="4").status_code == 400
    assert post(_png(64, 64), model="z-image-turbo").status_code == 400
    assert c.post("/v1/images/edits", data={"task": "edit", "prompt": "p", "scale": "2"},
                  files={"image": ("a.png", _png(8, 8), "image/png")}).status_code == 400
    assert len(queued) == 2
