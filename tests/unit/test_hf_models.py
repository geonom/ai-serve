"""Downloading GGUF language models from Hugging Face (backend/hf_models.py, pure rules). The owner pastes an address;
only huggingface.co and GGUF files in a repo are accepted, every file is checked against Hugging Face's sha256, a
vision projector is saved under a name that pairs it with its model, nothing is overwritten or deleted."""
import pytest

from backend import hf_models as hf
from backend.hf_models import HfLinkError


# ---------------------------------------------------------------- which addresses


@pytest.mark.parametrize("text,repo,rev,path,quant", [
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF", "main", None, None),
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/", "unsloth/Qwen3-4B-GGUF", "main", None, None),
    ("huggingface.co/unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF", "main", None, None),
    ("unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF", "main", None, None),
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf",
     "unsloth/Qwen3-4B-GGUF", "main", "Qwen3-4B-Q4_K_M.gguf", None),
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true",
     "unsloth/Qwen3-4B-GGUF", "main", "Qwen3-4B-Q4_K_M.gguf", None),
    ("https://huggingface.co/org/Big-GGUF/blob/main/Q4_K_M/Big-Q4_K_M-00001-of-00002.gguf",
     "org/Big-GGUF", "main", "Q4_K_M/Big-Q4_K_M-00001-of-00002.gguf", None),
    ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main", "unsloth/Qwen3-4B-GGUF", "main", None, None),
    ("hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", "unsloth/Qwen3-4B-GGUF", "main", None, "Q4_K_M"),
    ("https://hf.co/unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF", "main", None, None),
])
def test_addresses_people_copy(text, repo, rev, path, quant):
    link = hf.parse_link(text)
    assert (link.repo, link.revision, link.path, link.quant) == (repo, rev, path, quant)


@pytest.mark.parametrize("text,why", [
    ("https://evil.example/unsloth/Qwen3-4B-GGUF", "huggingface.co"),
    ("https://huggingface.co.evil.example/a/b", "huggingface.co"),
    ("https://user@huggingface.co/a/b", "huggingface.co"),
    ("ftp://huggingface.co/a/b", "huggingface.co"),
    ("https://huggingface.co/datasets/a/b", "model"),
    ("https://huggingface.co/spaces/a/b", "model"),
    ("https://huggingface.co/unsloth", "owner/name"),
    ("https://huggingface.co/a/b/blob/main/../../etc/passwd.gguf", "path"),
    ("https://huggingface.co/a/b/blob/main/model.safetensors", ".gguf"),
    ("https://huggingface.co/a/b/blob/main/%2e%2e/x.gguf", "path"),
    ("", "address"),
    (None, "address"),
])
def test_everything_else_is_refused(text, why):
    with pytest.raises(HfLinkError) as e:
        hf.parse_link(text)
    assert why in str(e.value)


# ---------------------------------------------------------------- what's in a repo

TREE = [
    {"type": "file", "path": "README.md", "size": 100},
    {"type": "file", "path": "Qwen3-4B-Q4_K_M.gguf", "size": 2_500_000_000, "lfs": {"oid": "a" * 64, "size": 2_500_000_000}},
    {"type": "file", "path": "Qwen3-4B-Q8_0.gguf", "size": 4_300_000_000, "lfs": {"oid": "b" * 64, "size": 4_300_000_000}},
    {"type": "file", "path": "mmproj-F16.gguf", "size": 800_000_000, "lfs": {"oid": "c" * 64, "size": 800_000_000}},
    {"type": "file", "path": "BF16/Qwen3-4B-BF16-00001-of-00002.gguf", "size": 5_000_000_000,
     "lfs": {"oid": "d" * 64, "size": 5_000_000_000}},
    {"type": "file", "path": "BF16/Qwen3-4B-BF16-00002-of-00002.gguf", "size": 3_000_000_000,
     "lfs": {"oid": "e" * 64, "size": 3_000_000_000}},
    {"type": "file", "path": "no-checksum.gguf", "size": 10},
    {"type": "directory", "path": "BF16"},
]


def test_the_repo_lists_its_models_with_split_files_as_one():
    cat = hf.catalog(TREE)
    models = {m["path"]: m for m in cat["models"]}
    assert set(models) == {"Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q8_0.gguf", "BF16/Qwen3-4B-BF16-00001-of-00002.gguf"}
    assert models["BF16/Qwen3-4B-BF16-00001-of-00002.gguf"]["parts"] == 2
    assert models["BF16/Qwen3-4B-BF16-00001-of-00002.gguf"]["size"] == 8_000_000_000
    assert models["Qwen3-4B-Q4_K_M.gguf"]["quant"] == "Q4_K_M"
    assert [p["path"] for p in cat["mmproj"]] == ["mmproj-F16.gguf"]
    assert "no-checksum.gguf" not in models          # not offered: can't be checked


def test_ollama_style_quant_picks_the_file():
    assert hf.pick(hf.catalog(TREE), hf.parse_link("hf.co/unsloth/Qwen3-4B-GGUF:Q8_0")) == "Qwen3-4B-Q8_0.gguf"
    assert hf.pick(hf.catalog(TREE), hf.parse_link("unsloth/Qwen3-4B-GGUF")) is None      # the owner chooses
    with pytest.raises(HfLinkError):
        hf.pick(hf.catalog(TREE), hf.parse_link("hf.co/unsloth/Qwen3-4B-GGUF:Q2_K"))


# ---------------------------------------------------------------- what gets saved where


def test_plan_saves_the_model_and_pairs_the_projector(tmp_path):
    plan = hf.plan(hf.catalog(TREE), "unsloth/Qwen3-4B-GGUF", "main", "Qwen3-4B-Q4_K_M.gguf", "mmproj-F16.gguf",
                   str(tmp_path), free_bytes=10**12)
    saves = {f["save_as"]: f for f in plan["files"]}
    assert set(saves) == {"Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M-mmproj-F16.gguf"}
    assert saves["Qwen3-4B-Q4_K_M.gguf"]["sha256"] == "a" * 64
    assert saves["Qwen3-4B-Q4_K_M.gguf"]["url"] == \
        "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    from backend.model_manager import find_mmproj_path
    paths = [str(tmp_path / n) for n in saves]
    assert find_mmproj_path(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"), paths).endswith("Qwen3-4B-Q4_K_M-mmproj-F16.gguf")


def test_a_split_model_brings_all_its_parts(tmp_path):
    plan = hf.plan(hf.catalog(TREE), "r/x", "main", "BF16/Qwen3-4B-BF16-00001-of-00002.gguf", None,
                   str(tmp_path), free_bytes=10**12)
    assert [f["save_as"] for f in plan["files"]] == ["Qwen3-4B-BF16-00001-of-00002.gguf",
                                                     "Qwen3-4B-BF16-00002-of-00002.gguf"]
    assert plan["model"] == "Qwen3-4B-BF16-00001-of-00002.gguf"


def test_files_already_here_are_kept_not_overwritten(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"x")
    plan = hf.plan(hf.catalog(TREE), "r/x", "main", "Qwen3-4B-Q4_K_M.gguf", "mmproj-F16.gguf", str(tmp_path),
                   free_bytes=10**12)
    assert [f["save_as"] for f in plan["files"]] == ["Qwen3-4B-Q4_K_M-mmproj-F16.gguf"]
    assert plan["already"] == ["Qwen3-4B-Q4_K_M.gguf"]


def test_no_room_is_refused_and_nothing_is_deleted(tmp_path):
    with pytest.raises(HfLinkError) as e:
        hf.plan(hf.catalog(TREE), "r/x", "main", "Qwen3-4B-Q8_0.gguf", None, str(tmp_path), free_bytes=3 * 10**9)
    assert "room" in str(e.value)


def test_only_files_of_the_listing_can_be_chosen(tmp_path):
    for model, mmproj in (("../x.gguf", None), ("README.md", None), ("Qwen3-4B-Q8_0.gguf", "../../evil.gguf"),
                          ("no-checksum.gguf", None)):
        with pytest.raises(HfLinkError):
            hf.plan(hf.catalog(TREE), "r/x", "main", model, mmproj, str(tmp_path), free_bytes=10**12)
