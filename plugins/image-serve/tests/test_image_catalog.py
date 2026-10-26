"""image_serve/catalog.py (moved from ai-serve-ui) -- one list of every built-in image model (ComfyUI; Iris and MOSS are image plugins since 2026-10-21),
so apps pick from it instead of typing GGUF names."""
import pytest

from image_serve import bundles as _bundles, catalog

REAL_PRESENT = _bundles.present   # taken before conftest patches it


def test_only_comfyui_models_here():
    """MOSS and Iris are their own plugins (a leftover MOSS row once clashed with comfyui-moss's catalog)."""
    assert {m.provider for m in catalog.CATALOG} == {"comfyui"}


def test_ids_are_unique_and_lowercase():
    ids = [m.id for m in catalog.CATALOG]
    assert len(ids) == len(set(ids)) and all(i == i.lower() for i in ids)
    assert catalog.DEFAULT == "z-image-turbo" and catalog.resolve(catalog.DEFAULT)


@pytest.mark.parametrize("name, want", [
    ("z-image-turbo", "z-image-turbo"),
    ("Z-Image-Turbo", "z-image-turbo"),
    ("z_image_turbo-Q6_K.gguf", "z-image-turbo"),        # what StoryTime stored before
    ("ernie-image-turbo-Q6_K.gguf", "ernie-turbo"),
    ("qwen_image_2.1_int8_convrot.safetensors", "qwen-image-2.1-int8"),
])
def test_resolve(name, want):
    assert catalog.resolve(name).id == want


@pytest.mark.parametrize("name", [None, "", "dall-e-3", "../../etc/passwd", "z_image"])
def test_resolve_unknown_is_none(name):
    assert catalog.resolve(name) is None


def test_qwen_image_21_does_every_aspect():
    """It was listed as square only -- a limit of our workflow (its text-encode node only shapes the latent
    from a reference image), not of the model; an empty latent of any size works (2026-10-18)."""
    assert catalog.resolve("qwen-image-2.1").aspects == catalog.ASPECTS


def test_listing_marks_missing_files_unavailable(tmp_path, monkeypatch, real_on_disk):
    from image_serve import bundles
    monkeypatch.setattr(catalog, "_on_disk", real_on_disk)
    monkeypatch.setattr(bundles, "present", REAL_PRESENT)
    for f in bundles.files_of("z-image-turbo"):                 # every file of the bundle, with its size
        p = tmp_path / f.dir / f.name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as out:
            out.truncate(f.size)
    rows = {r["id"]: r for r in catalog.listing(models_root=str(tmp_path))}
    assert rows["z-image-turbo"]["available"] is True and rows["z-image-turbo"]["default"] is True
    assert "ideogram-4" not in rows                              # neither here nor downloadable: not offered
    assert rows["ernie-turbo"]["available"] is False             # downloadable: listed, not available
    assert set(rows["ernie-turbo"]) >= {"id", "label", "provider", "aspect_ratios", "description", "text_in_image",
                                        "layout", "available", "default"}
    assert "file" not in rows["ernie-turbo"]   # paths stay internal
    assert rows["z-image-turbo"]["aliases"] == ["z_image_turbo-Q6_K.gguf"]   # settings saved before the catalog
    assert not any("/" in a for r in rows.values() for a in r["aliases"])


def test_steps_per_model():
    """Compare by steps needs to know each model's default and whether it can change."""
    rows = {r["id"]: r for r in catalog.listing()}
    assert rows["z-image-turbo"]["steps"] == {"default": 8, "adjustable": True, "max": 60}
    assert rows["ernie-turbo"]["steps"]["default"] == 4
    assert rows["qwen-image-2.1"]["steps"]["default"] == 40 and rows["qwen-image-2.1-int8"]["steps"]["default"] == 40


# ------------------------------------------------------------------ editing and style transfer
# 2026-10-20: edit models joined the catalog. 2026-10-21: the older pre-2511 Qwen-Image-Edit files were removed
# (owner: the old workflow was broken and shouldn't be usable); their names now mean the 2511 Q2_K, so callers that
# stored them (language_learning, Munin) keep working.

def test_edit_and_style_models_are_in_the_catalog():
    edit = {m.id for m in catalog.CATALOG if "edit" in m.tasks}
    style = {m.id for m in catalog.CATALOG if "style" in m.tasks}
    assert style == {"qwen-edit-2511", "qwen-edit-2511-q3"}
    assert edit == style | {"flux2-klein-4b"}
    assert catalog.DEFAULTS == {"generate": "z-image-turbo", "edit": "qwen-edit-2511", "style": "qwen-edit-2511",
                                "remove_background": "birefnet"}


@pytest.mark.parametrize("old", ["Qwen_Image_Edit-Q2_K.gguf", "Qwen_Image_Edit-Q4_K_S.gguf", "qwen-edit", "qwen-edit-q4"])
def test_old_edit_names_mean_2511(old):
    assert catalog.resolve(old).id == "qwen-edit-2511"
    assert catalog.for_task(old, "edit").file == "qwen-image-edit-2511-Q2_K.gguf"


def test_listing_by_task():
    gen = {r["id"] for r in catalog.listing()}
    assert "z-image-turbo" in gen and "qwen-edit-2511" not in gen          # generate stays as it was
    rows = {r["id"]: r for r in catalog.listing(task="edit")}
    assert set(rows) == {"qwen-edit-2511", "qwen-edit-2511-q3", "flux2-klein-4b"} and rows["qwen-edit-2511"]["default"]
    assert {r["id"] for r in catalog.listing(task="style") if r["default"]} == {"qwen-edit-2511"}
    assert len(catalog.listing(task="all")) == len(catalog.CATALOG)
    with pytest.raises(catalog.CatalogError):
        catalog.listing(task="nope")


def test_for_task():
    assert catalog.for_task(None, "style").id == "qwen-edit-2511"
    for name, task in (("qwen-edit-2511", "generate"), ("z-image-turbo", "edit")):
        with pytest.raises(catalog.CatalogError):
            catalog.for_task(name, task)
    assert catalog.for_task("something-unknown.gguf", "edit") is None   # not ours to judge: passed on as before


def test_edit_templates_use_the_2511_node_and_lora():
    import json, os
    root = os.path.join(os.path.dirname(__file__), "..", "workflows")
    for f in ("Qwen-Image-Edit.json", "Qwen-Image-Style-Transfer.json"):
        wf = json.load(open(os.path.join(root, f)))
        kinds = {n["class_type"] for n in wf.values()}
        assert "TextEncodeQwenImageEditPlus" in kinds and "TextEncodeQwenImageEdit" not in kinds, f
        assert [n["inputs"]["lora_name"] for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"] == [
            "Qwen Image/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"], f
        assert all("2511" in n["inputs"]["unet_name"] for n in wf.values() if n["class_type"] == "UnetLoaderGGUF"), f
