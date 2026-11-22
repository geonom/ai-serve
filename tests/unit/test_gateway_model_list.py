"""GET /v1/models scans the model folders and reads each file's header (bundled MTP). That must not run on the
event loop: it took ~4 s with ~60 files, and the backend stood still meanwhile -- the owner gates live in the same
process, so the Playground behind its gate timed out and showed "gateway not running" / 502 (2026-11-22). Header
answers come from the shared HEADER_CACHE (path + size + mtime, kept on disk), so known files aren't read again."""
import asyncio

from backend import model_manager
from backend.header_cache import HeaderCache
from backend.routers import gateway


def _model(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"GGUF" + b"\0" * 64)
    return p


def _setup(tmp_path, monkeypatch, header):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(gateway, "MODEL_DIRS", [str(models)])
    monkeypatch.setattr(model_manager, "MODEL_DIRS", [str(models)])
    store = str(tmp_path / "headers.json")
    monkeypatch.setattr(model_manager, "HEADER_CACHE", HeaderCache(store))
    monkeypatch.setattr(gateway, "HEADER_CACHE", model_manager.HEADER_CACHE)
    monkeypatch.setattr(model_manager, "gguf_has_bundled_mtp", header)
    monkeypatch.setattr(model_manager, "gguf_has_bundled_vision", lambda path: False)
    return models, store


def test_the_scan_runs_off_the_event_loop_and_known_files_are_not_read_again(tmp_path, monkeypatch):
    reads = []

    def header(path):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        reads.append((path, on_loop))
        return path.endswith("b-model-Q8_0.gguf")

    models, store = _setup(tmp_path, monkeypatch, header)
    _model(models, "a-model-Q4_K_M.gguf")
    _model(models, "b-model-Q8_0.gguf")
    first = asyncio.run(gateway.list_openai_models())
    monkeypatch.setattr(model_manager, "HEADER_CACHE", HeaderCache(store))   # as after a restart
    monkeypatch.setattr(gateway, "HEADER_CACHE", model_manager.HEADER_CACHE)
    second = asyncio.run(gateway.list_openai_models())

    assert {m["id"]: m["mtp_available"] for m in first["data"]} == {"a-model-Q4_K_M.gguf": False,
                                                                     "b-model-Q8_0.gguf": True}
    assert second == first
    assert len(reads) == 2 and not any(on_loop for _, on_loop in reads)


def test_a_changed_file_is_read_again(tmp_path, monkeypatch):
    reads = []
    models, _ = _setup(tmp_path, monkeypatch, lambda path: reads.append(path) or False)
    p = _model(models, "a-model-Q4_K_M.gguf")
    asyncio.run(gateway.list_openai_models())
    p.write_bytes(b"GGUF" + b"\0" * 128)
    asyncio.run(gateway.list_openai_models())
    assert len(reads) == 2


def test_the_list_points_to_the_other_catalogs(tmp_path, monkeypatch):
    """/v1/models is where an OpenAI client looks first, but it only lists local language models; image, sound,
    video and vision models have their own catalogs (owner, 2026-11-22: a client found moss-turbo-4 only by
    guessing). An extra top-level field -- OpenAI clients read only `data`."""
    _setup(tmp_path, monkeypatch, lambda path: False)
    out = asyncio.run(gateway.list_openai_models())
    assert out["object"] == "list" and out["data"] == []
    cats = out["ai_serve"]["catalogs"]
    assert cats["images"] == "/v1/images/models?task=all"
    assert cats["video"] == "/v1/video/models"
    assert {"llm", "images", "speech", "music_and_sound", "video", "vision"} <= set(cats)
    assert out["ai_serve"]["docs"] == "/docs"
