"""Faces in photos (vision_plugin/faces.py). The fixtures are generated pictures (Z-Image / Qwen-Image-Edit through
ai-serve, 2026-10-25) -- no real people: a portrait, the same woman in another scene, another person, a faded b/w
group photo of six, a landscape. The model tests run on the CPU and are skipped when the models aren't installed."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_plugin import install
from vision_plugin.faces import ImageError, cosine, decode, order_points

FIX = Path(__file__).parent / "fixtures"


def jpeg(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


class TestDecode:
    def test_grey_and_sepia_become_three_channels(self):
        assert decode(jpeg(Image.new("L", (40, 30), 128))).shape == (30, 40, 3)

    def test_sixteen_bit_scans(self):
        buf = io.BytesIO()
        Image.fromarray(np.linspace(0, 65535, 1200, dtype=np.uint16).reshape(30, 40)).save(buf, "PNG")
        img = decode(buf.getvalue())
        assert img.shape == (30, 40, 3) and img.max() == 255

    def test_exif_rotation_is_applied(self):
        im = Image.new("RGB", (40, 30))
        exif = im.getexif()
        exif[0x0112] = 6      # rotate 90
        assert decode(jpeg(im, exif=exif)).shape[:2] == (40, 30)

    def test_garbage_and_huge_images_are_refused(self):
        with pytest.raises(ImageError):
            decode(b"not an image")
        buf = io.BytesIO()
        Image.new("1", (12000, 12000)).save(buf, "PNG")          # 144 MP, tiny file
        with pytest.raises(ImageError, match="too large"):
            decode(buf.getvalue())


def test_points_are_ordered_left_to_right():
    lmk = np.array([[70, 50], [40, 50], [55, 70], [68, 90], [42, 90]], dtype=np.float32)
    out = order_points(lmk)
    assert out[0, 0] < out[1, 0] and out[3, 0] < out[4, 0]


@pytest.fixture(scope="module")
def model():
    if not (install.present("yunet") and install.present("auraface")):
        pytest.skip("models not installed (python -m vision_plugin.install)")
    from vision_plugin.faces import FaceModel
    return FaceModel(str(install.path_of("yunet")), str(install.path_of("auraface")), ["CPUExecutionProvider"])


def faces(model, name, embed=True):
    return model.analyze(decode((FIX / name).read_bytes()), embed=embed)["faces"]


def test_finds_the_faces(model):
    assert len(faces(model, "portrait.jpg")) == 1
    assert len(faces(model, "group_bw.jpg", embed=False)) == 6        # ~18 px faces on a faded print
    assert faces(model, "landscape.jpg") == []


def test_face_shape(model):
    f = faces(model, "portrait.jpg")[0]
    assert len(f["box"]) == 4 and len(f["landmarks"]) == 5 and 0 < f["score"] <= 1
    assert len(f["embedding"]) == 512 and abs(np.linalg.norm(f["embedding"]) - 1) < 1e-3
    assert f["quality"]["size"] > 100
    assert "embedding" not in faces(model, "portrait.jpg", embed=False)[0]


def test_same_person_is_closer_than_another(model):
    a = faces(model, "portrait.jpg")[0]["embedding"]
    same = cosine(a, faces(model, "portrait_same.jpg")[0]["embedding"])
    other = cosine(a, faces(model, "other.jpg")[0]["embedding"])
    from vision_plugin.app import catalog
    th = catalog()["data"][0]["embedding"]["thresholds"]
    assert same > th["likely"] and other < th["maybe"]
