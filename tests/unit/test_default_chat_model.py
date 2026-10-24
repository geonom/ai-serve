"""'Auto' / 'default' in a chat picks a running instance -- but only one that can chat (2026-10-20: with
bge-m3 and the reranker running, the Spielwiese's Auto chat went to the reranker and got
"the current context does not logits computation")."""
import pytest

from backend import model_manager
from backend.state import state


class Proc:
    def poll(self):
        return None


@pytest.fixture
def running(monkeypatch):
    saved = dict(state.managed_processes)
    state.managed_processes.clear()
    kinds = {"/m/bge-reranker-v2-m3-Q8_0.gguf": "rerank", "/m/bge-m3-Q8_0.gguf": "embedding",
             "/m/YuE-s1-7B.gguf": "base", "/m/gemma-4-12b-it-Q4_K_M.gguf": "chat"}
    monkeypatch.setattr(model_manager, "get_model_traits", lambda p: {"kind": kinds.get(p, "chat")})
    _files(monkeypatch, ["gemma-4-12b-it-qat-q4_0.gguf"])   # never this machine's model folders
    yield state.managed_processes
    state.managed_processes.clear()
    state.managed_processes.update(saved)


def test_auto_skips_embedding_rerank_and_base_instances(running):
    running[1] = {"process": Proc(), "model": "/m/bge-reranker-v2-m3-Q8_0.gguf", "reranking": True, "embedding": True}
    running[2] = {"process": Proc(), "model": "/m/bge-m3-Q8_0.gguf", "embedding": True}
    running[3] = {"process": Proc(), "model": "/m/YuE-s1-7B.gguf"}
    running[4] = {"process": Proc(), "model": "/m/gemma-4-12b-it-Q4_K_M.gguf"}
    assert model_manager.resolve_model_path("default") == "/m/gemma-4-12b-it-Q4_K_M.gguf"


def test_auto_with_only_non_chat_instances_falls_back_to_the_default_model(running):
    running[1] = {"process": Proc(), "model": "/m/bge-reranker-v2-m3-Q8_0.gguf", "reranking": True}
    picked = model_manager.resolve_model_path(None)
    assert "rerank" not in picked and picked.endswith(".gguf")


def test_auto_skips_reading_models(running):
    # 2026-10-20: NuExtract3 was still loaded from "Dokumente lesen"; an Auto chat went there and came back odd
    running[1] = {"process": Proc(), "model": "/m/NuExtract3-Q8_0.gguf"}
    running[2] = {"process": Proc(), "model": "/m/RolmOCR.i1-IQ4_XS.gguf"}
    running[3] = {"process": Proc(), "model": "/m/gemma-4-12b-it-Q4_K_M.gguf"}
    assert model_manager.resolve_model_path("default") == "/m/gemma-4-12b-it-Q4_K_M.gguf"


def _files(monkeypatch, names):
    import glob as _glob
    monkeypatch.setattr(model_manager.glob, "glob", lambda pattern, recursive=False: [f"/m/{n}" for n in names])
    monkeypatch.setattr(model_manager.os.path, "isdir", lambda d: True)


def test_nothing_running_takes_the_configured_default(running, monkeypatch):
    # 2026-10-20: the fallback took "the first Gemma 4 12B file" -- a 2-bit IQ2_M that thought in circles until
    # max_tokens ran out, so Auto chats came back without an answer
    _files(monkeypatch, ["gemma-4-12b-it-UD-IQ2_M.gguf", "gemma-4-12b-it-qat-q4_0.gguf", "gemma-4-12b-it-UD-Q6_K_XL.gguf"])
    monkeypatch.setattr(model_manager, "DEFAULT_CHAT_MODEL", "gemma-4-12b-it-qat-q4_0.gguf")
    assert model_manager.resolve_model_path("default") == "/m/gemma-4-12b-it-qat-q4_0.gguf"


def test_without_the_configured_file_tiny_quants_are_avoided(running, monkeypatch):
    _files(monkeypatch, ["gemma-4-12b-it-UD-IQ2_M.gguf", "gemma-4-12b-it-UD-Q3_K_XL.gguf", "gemma-4-12b-it-UD-Q6_K_XL.gguf"])
    monkeypatch.setattr(model_manager, "DEFAULT_CHAT_MODEL", "missing.gguf")
    assert model_manager.resolve_model_path("default") == "/m/gemma-4-12b-it-UD-Q6_K_XL.gguf"


def test_default_skips_files_on_a_drive_that_is_not_mounted(running, monkeypatch, tmp_path):
    # 2026-10-24: after a reboot the external SSD wasn't mounted; the configured default (a symlink onto it) was
    # still picked, and every "default" chat failed -- V-Noveller found it. A real file of the family is used instead.
    gone = tmp_path / "unplugged"
    (tmp_path / "gemma-4-12b-it-qat-q4_0.gguf").symlink_to(gone / "gemma-4-12b-it-qat-q4_0.gguf")
    (tmp_path / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf").symlink_to(gone / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    (tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf").write_bytes(b"GGUF")
    import os
    monkeypatch.setattr(model_manager.glob, "glob", lambda pattern, recursive=False: sorted(
        str(tmp_path / n) for n in os.listdir(tmp_path) if n.endswith(".gguf")))
    monkeypatch.setattr(model_manager, "DEFAULT_CHAT_MODEL", "gemma-4-12b-it-qat-q4_0.gguf")
    assert model_manager.resolve_model_path("default") == str(tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
    # asked for by name, the symlink is still what it resolves to -- so the caller gets the "is the drive mounted?"
    # message instead of some other model
    assert model_manager.resolve_model_path("gemma-4-12b-it-qat-q4_0.gguf") == str(tmp_path / "gemma-4-12b-it-qat-q4_0.gguf")
