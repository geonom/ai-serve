"""Verified downloads for the installer: every file is checked against a sha256 it got from a source it trusts
(GitHub's release API, Hugging Face's paths-info API) before it is moved into place. stdlib only."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

UA = {"User-Agent": "ai-serve-installer"}


def get_json(url: str, data: Optional[bytes] = None):
    req = urllib.request.Request(url, data=data, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def hf_file(repo: str, path: str, revision: str = "main"):
    """(sha256, size) of a file on Hugging Face (its LFS oid), from HF's own API."""
    info = get_json(f"https://huggingface.co/api/models/{repo}/paths-info/{revision}",
                    data=urllib.parse.urlencode({"paths": path}).encode())
    lfs = (info[0].get("lfs") or {}) if isinstance(info, list) and info else {}
    if not lfs.get("oid"):
        raise SystemExit(f"{repo}/{path}: Hugging Face gives no checksum -- not downloading it unchecked")
    return lfs["oid"], int(lfs.get("size") or 0)


def fetch(url: str, dest: Path, sha256: str, label: str) -> None:
    """Stream to a temp file next to dest, hash while writing, move into place only if the hash matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    tty = sys.stdout.isatty()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, \
            tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, prefix=".part-") as f:
        total, done, shown = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else 0
            if total and (tty or pct // 10 > shown):
                shown = pct // 10
                print(f"\r  {label}: {done >> 20} / {total >> 20} MB ({pct} %)", end="" if tty else "\n", flush=True)
        tmp = Path(f.name)
    if tty:
        print()
    if h.hexdigest() != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{label}: checksum mismatch -- nothing installed")
    tmp.chmod(0o644)       # temp files are created 0600
    tmp.replace(dest)


def hf_download(repo: str, path: str, dest: Path) -> None:
    if dest.exists():
        print(f"  {dest.name}: already there")
        return
    sha, _ = hf_file(repo, path)
    fetch(f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(path)}", dest, sha, dest.name)
