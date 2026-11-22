"""What a model file's header says is kept on disk, keyed by path + size + mtime (owner, 2026-11-22: known models
shouldn't be read again; a sha256 would mean reading every multi-GB file in full -- size + mtime change whenever a
file is replaced or edited). A restart then starts with every known model answered."""
import json
import os

from backend.header_cache import HeaderCache


def _file(tmp_path, name="m.gguf", data=b"GGUF1234"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_a_known_file_is_not_read_again_even_after_a_restart(tmp_path):
    model, store = _file(tmp_path), str(tmp_path / "cache.json")
    calls = []
    first = HeaderCache(store)
    assert first.get(model, "flags", lambda: calls.append(1) or [True, False]) == [True, False]
    first.save()
    again = HeaderCache(store)   # a new process
    assert again.get(model, "flags", lambda: calls.append(1) or [False, False]) == [True, False]
    assert len(calls) == 1


def test_a_replaced_file_is_read_again(tmp_path):
    model, store = _file(tmp_path), str(tmp_path / "cache.json")
    cache = HeaderCache(store)
    cache.get(model, "flags", lambda: "old")
    with open(model, "wb") as f:
        f.write(b"GGUF-a-different-and-longer-file")
    assert cache.get(model, "flags", lambda: "new") == "new"


def test_kinds_are_kept_apart(tmp_path):
    model = _file(tmp_path)
    cache = HeaderCache(str(tmp_path / "cache.json"))
    cache.get(model, "flags", lambda: "f")
    assert cache.get(model, "traits", lambda: {"kind": "chat"}) == {"kind": "chat"}


def test_a_missing_file_is_computed_but_not_kept(tmp_path):
    cache = HeaderCache(str(tmp_path / "cache.json"))
    assert cache.get(str(tmp_path / "gone.gguf"), "flags", lambda: None) is None
    assert cache.get(str(tmp_path / "gone.gguf"), "flags", lambda: "x") == "x"


def test_a_broken_store_starts_empty_and_vanished_files_are_dropped(tmp_path):
    store = tmp_path / "cache.json"
    store.write_text("{not json")
    model, gone = _file(tmp_path, "a.gguf"), _file(tmp_path, "b.gguf")
    cache = HeaderCache(str(store))
    cache.get(model, "flags", lambda: 1)
    cache.get(gone, "flags", lambda: 2)
    os.remove(gone)
    cache.save()
    kept = json.loads(store.read_text())
    assert list(kept["files"]) == [model]


def test_save_writes_only_when_something_changed(tmp_path):
    model, store = _file(tmp_path), tmp_path / "cache.json"
    cache = HeaderCache(str(store))
    cache.save()
    assert not store.exists()
    cache.get(model, "flags", lambda: 1)
    cache.save()
    assert store.exists()
