"""Sending models to nodes (backend/cluster/models.py) -- pure rules.

User (2026-10-16): models come from whichever source is faster. The coordinator
offers every source it can vouch for -- itself (over the overlay network) and/or
Hugging Face -- with ONE sha256; the agent measures and picks, and falls back if a
source delivers the wrong bytes.

Rules under test:
- Hugging Face refs: owner/repo, a plain file path (no ..), a revision; the sha256
  comes from HF's own metadata (lfs.oid), never from the caller.
- A local file is offered only by plain .gguf basename; its sha256 is computed here.
- Both known: if the sizes differ they aren't the same file -> only Hugging Face.
- Neither -> a clear error."""
import pytest

from backend.cluster.models import HfRef, ModelPlanError, plan_model

COORD = "http://my-pc.example-tailnet.ts.net:8030"
SHA_L = "1" * 64
SHA_H = "2" * 64


class TestHfRef:
    def test_valid(self):
        r = HfRef.parse("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", None)
        assert r.url == "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
        assert r.name == "Qwen3-0.6B-Q4_K_M.gguf"

    def test_subdir_file(self):
        assert HfRef.parse("u/r", "Q4/model.gguf", "abc123").name == "model.gguf"

    @pytest.mark.parametrize("repo,file,rev", [
        ("nouser", "m.gguf", None), ("u/r/x", "m.gguf", None), ("u/r", "../m.gguf", None),
        ("u/r", "m.bin", None), ("u/r", "m.gguf", "../main"), ("u/r", "/abs.gguf", None),
    ])
    def test_refused(self, repo, file, rev):
        with pytest.raises(ModelPlanError):
            HfRef.parse(repo, file, rev)


class TestPlan:
    hf = HfRef.parse("u/r", "m.gguf", None)

    def test_both_same_size(self):
        p = plan_model(COORD, local=("m.gguf", 100, SHA_L), hf=self.hf, hf_meta=(100, SHA_H))
        # HF's hash is authoritative when HF is a source -- a local copy with other
        # bytes simply fails verification on the agent and it falls back
        assert p == {"name": "m.gguf", "sha256": SHA_H, "size": 100,
                     "sources": [f"{COORD}/api/cluster/models/m.gguf", self.hf.url]}

    def test_sizes_differ_hf_only(self):
        p = plan_model(COORD, local=("m.gguf", 99, SHA_L), hf=self.hf, hf_meta=(100, SHA_H))
        assert p["sources"] == [self.hf.url]

    def test_local_only(self):
        p = plan_model(COORD, local=("m.gguf", 100, SHA_L), hf=None, hf_meta=None)
        assert p["sha256"] == SHA_L and p["sources"] == [f"{COORD}/api/cluster/models/m.gguf"]

    def test_local_only_without_coordinator_url(self):
        with pytest.raises(ModelPlanError):
            plan_model(None, local=("m.gguf", 100, SHA_L), hf=None, hf_meta=None)

    def test_hf_only(self):
        assert plan_model(None, local=None, hf=self.hf, hf_meta=(100, SHA_H))["sources"] == [self.hf.url]

    def test_nothing(self):
        with pytest.raises(ModelPlanError):
            plan_model(COORD, local=None, hf=None, hf_meta=None)
