"""sha256 of local model files, cached by (path, size, mtime) -- hashing a 20 GB GGUF
takes about a minute, so it's done once per file version."""
import hashlib
import json
import os
import threading
from typing import Dict


class HashCache:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, ValueError):
            return {}

    def sha256(self, file_path: str) -> str:
        real = os.path.realpath(file_path)
        st = os.stat(real)
        key = f"{real}|{st.st_size}|{int(st.st_mtime)}"
        with self._lock:
            hit = self._load().get(key)
        if hit:
            return hit["sha256"]
        h = hashlib.sha256()
        with open(real, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with self._lock:
            data = self._load()
            data[key] = {"sha256": digest}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        return digest
