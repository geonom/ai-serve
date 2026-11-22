"""What a model file's GGUF header says, kept on disk (tests tests/unit/test_header_cache.py).

Reading the headers of ~100 models took 4-15 s; after a restart every list (/v1/models, /v1/llm/models, /api/models)
paid it again. Entries are keyed by path + size + mtime: replacing or editing a file changes those, so it is read
again. A sha256 would be no faster check -- it means reading every multi-GB file in full (owner asked, 2026-11-22).
Entries of files that are gone are dropped on save. The store is a plain JSON file (gitignored)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Callable, Dict, Optional


class HeaderCache:
    def __init__(self, store: str):
        self.store = store
        self._lock = threading.Lock()
        self._dirty = False
        self._files: Dict[str, Dict[str, Any]] = {}
        try:
            with open(store) as f:
                files = json.load(f).get("files", {})
            if isinstance(files, dict):
                self._files = {p: e for p, e in files.items() if isinstance(e, dict)}
        except (OSError, ValueError, AttributeError):
            pass

    @staticmethod
    def _key(path: str) -> Optional[list]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return [st.st_size, st.st_mtime]

    def get(self, path: str, kind: str, compute: Callable[[], Any]) -> Any:
        """The cached answer of `kind` for this file, else compute() (kept only while the file exists)."""
        key = self._key(path)
        if key is None:
            return compute()
        with self._lock:
            entry = self._files.get(path)
            if entry and entry.get("key") == key and kind in entry:
                return entry[kind]
        value = compute()
        with self._lock:
            entry = self._files.get(path)
            if not entry or entry.get("key") != key:
                entry = self._files[path] = {"key": key}
            entry[kind] = value
            self._dirty = True
        return value

    def save(self) -> None:
        """Write the store if anything changed (atomically); entries of vanished files are dropped."""
        with self._lock:
            if not self._dirty:
                return
            self._files = {p: e for p, e in self._files.items() if os.path.exists(p)}
            data = json.dumps({"files": self._files})
            self._dirty = False
        folder = os.path.dirname(os.path.abspath(self.store))
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".header-cache-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
            os.replace(tmp, self.store)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
