"""Set up sd-serve on a fresh machine: the prebuilt stable-diffusion.cpp for this platform (GitHub release, sha256 from
GitHub's API checked while downloading) and, with --models, Iris-3B Turbo + its text encoder from Hugging Face (sha256
from HF's API). A local CUDA build in sdcpp/build/bin/ (this project's CLAUDE.md) is kept and used instead.

    .venv/bin/python -m sd_plugin.install [--models]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from sd_plugin import rules

HERE = Path(__file__).resolve().parent.parent
BIN = HERE / "bin"
RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
MODELS = [  # (Hugging Face repo, file, where under the models dir)
    ("wbruna/Iris-3b-sdcpp-GGUF", "iris-3b-turbo-Q4_K.gguf", "diffusion_models/iris/iris-3b-turbo-Q4_K.gguf"),
    ("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
]


def _json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "sd-serve-setup", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url: str, dest: Path, sha256: str, label: str, progress=None) -> None:
    """Stream to a temp file, hash while writing, move into place only if the hash matches. `progress(n)` is told
    every chunk's size (the plugin's install endpoint shows how far it got)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "sd-serve-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown, tty = 0, -1, sys.stdout.isatty()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(len(chunk))
            pct = done * 100 // total if total else 0
            if total and (tty or pct // 10 > shown):   # a terminal: live; a log: every 10 %
                shown = pct // 10
                print(f"\r  {label}: {done >> 20} / {total >> 20} MB ({pct} %)", end="" if tty else "\n", flush=True)
        tmp = Path(f.name)
    print()
    if h.hexdigest() != sha256.lower():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{label}: checksum mismatch -- nothing installed")
    tmp.chmod(0o644)       # temp files are created 0600
    tmp.replace(dest)


def _has_vulkan() -> bool:
    return any(Path(p).exists() for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1", "/usr/lib64/libvulkan.so.1",
                                          "/usr/lib/libvulkan.so.1"))


def install_binary() -> None:
    if (HERE / "sdcpp" / "build" / "bin" / "sd-server").exists():
        print("sd-server: using the local build in sdcpp/build/bin")
        return
    rel = _json(RELEASES)
    assets = {a["name"]: a for a in rel.get("assets", [])}
    name = rules.release_asset(platform.system(), platform.machine(), _has_vulkan(), list(assets))
    if name is None:
        raise SystemExit(f"No prebuilt stable-diffusion.cpp for {platform.system()} {platform.machine()} -- "
                         "build it (see CLAUDE.md) into sdcpp/build/bin")
    a = assets[name]
    digest = (a.get("digest") or "").removeprefix("sha256:")
    if not digest:
        raise SystemExit(f"{name}: GitHub gives no checksum -- not installing it unchecked")
    with tempfile.TemporaryDirectory() as tmp:
        z = Path(tmp) / name
        _download(a["browser_download_url"], z, digest, name)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():   # plain relative files only
                if m.filename.startswith("/") or ".." in Path(m.filename).parts:
                    raise SystemExit(f"{name}: unsafe path {m.filename!r} in the archive")
            if BIN.exists():
                shutil.rmtree(BIN)
            zf.extractall(BIN)
    for exe in ("sd-server", "sd-cli"):
        p = BIN / exe
        if p.exists():
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"sd-server: {rel.get('tag_name')} ({name}) in bin/")


def install_models(models_dir: Path, progress=None) -> None:
    for repo, path, rel in MODELS:
        dest = models_dir / rel
        if dest.exists():
            print(f"{rel}: already there")
            continue
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/paths-info/main",
                                     data=urllib.parse.urlencode({"paths": path}).encode(),
                                     headers={"User-Agent": "sd-serve-setup"})
        with urllib.request.urlopen(req, timeout=60) as r:
            info = json.load(r)
        sha = (info[0].get("lfs") or {}).get("oid") if isinstance(info, list) and info else None
        if not sha:
            raise SystemExit(f"{repo}/{path}: no checksum from Hugging Face -- not downloading it unchecked")
        _download(f"https://huggingface.co/{repo}/resolve/main/{path}", dest, sha, rel, progress)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="also download Iris-3B Turbo (~2.9 GB) + text encoder (~2.5 GB)")
    a = ap.parse_args()
    install_binary()
    if a.models:
        install_models(Path(os.environ.get("SD_MODELS_DIR") or HERE / "models"))
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
