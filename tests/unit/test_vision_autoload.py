"""get_or_load_model attaches a model's vision module (mmproj) by itself -- but never to embedding/reranking loads."""
import asyncio

import pytest

from backend import model_manager as mm


@pytest.fixture
def captured(monkeypatch, tmp_path):
    model = tmp_path / "gemma-4-E2B-it-Q4_K_M.gguf"
    proj = tmp_path / "gemma-4-E2B-it-mmproj-F16.gguf"
    model.write_bytes(b"x")
    proj.write_bytes(b"x")
    monkeypatch.setattr(mm, "scan_all_gguf_paths", lambda: [str(model), str(proj)])
    monkeypatch.setattr(mm, "find_draft_model_path", lambda p: None)
    monkeypatch.setattr(mm, "gguf_has_bundled_mtp", lambda p: False)
    seen = {}

    async def fake_load(model_path, gpu_id, ctx, threads, ngl, keep, mmproj, *rest):
        seen["mmproj"] = mmproj
        return 1234
    monkeypatch.setattr(mm, "_load_locked", fake_load)
    return model, proj, seen


def test_chat_load_gets_the_vision_module(captured):
    model, proj, seen = captured
    asyncio.run(mm.get_or_load_model(str(model)))
    assert seen["mmproj"] == str(proj)


@pytest.mark.parametrize("kw", [{"embedding": True}, {"reranking": True}])
def test_embedding_and_rerank_loads_dont(captured, kw):
    model, _, seen = captured
    asyncio.run(mm.get_or_load_model(str(model), **kw))
    assert seen["mmproj"] is None


def test_an_explicit_one_wins(captured, tmp_path):
    model, _, seen = captured
    asyncio.run(mm.get_or_load_model(str(model), mmproj_path="/x/other.gguf"))
    assert seen["mmproj"] == "/x/other.gguf"
