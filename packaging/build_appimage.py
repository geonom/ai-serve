#!/usr/bin/env python3
"""Build ai-serve's AppImage for Linux: a double-click file that installs ai-serve into ~/.local/share/ai-serve and
opens the setup page (packaging/linux/AppRun). Inside: a standalone Python for the setup page (python-build-standalone),
uv for ai-serve's own environments, and the ai-serve tree. Every download is checked against the sha256 GitHub's
release API gives for it. stdlib only.

    packaging/build_appimage.py --tree <ai-serve tree> [--out dist] [--arch x86_64|aarch64]

The tree is a public ai-serve checkout (a git repo: what HEAD holds; else the folder as it is, without .git and
__pycache__). Needs mksquashfs (squashfs-tools).
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from installer import download as D  # noqa: E402
from node_agent.llama_serve_node import safe_extract  # noqa: E402   checks every member before extracting

UV_TAG = "0.13.7"
PYTHON_TAG, PYTHON_VERSION = "20261026", "3.12.16"   # python-build-standalone


def asset(repo: str, tag: str, name: str):
    """(url, sha256) of a release asset, from GitHub's API; refused when GitHub gives no checksum."""
    rel = D.get_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    for a in rel.get("assets", []):
        if a["name"] == name:
            digest = a.get("digest") or ""
            if not digest.startswith("sha256:"):
                raise SystemExit(f"{repo} {tag}/{name}: GitHub gives no checksum -- not using it unchecked")
            return a["browser_download_url"], digest.split(":", 1)[1]
    raise SystemExit(f"{repo} {tag}: no asset {name}")


def get(cache: Path, repo: str, tag: str, name: str) -> Path:
    dest = cache / tag / name
    if not dest.exists():
        url, sha = asset(repo, tag, name)
        D.fetch(url, dest, sha, name)
    return dest


def unpack(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    safe_extract(str(archive), str(into))


def copy_tree(tree: Path, dest: Path) -> str:
    """The ai-serve tree into dest; returns a short id of what was copied."""
    if (tree / ".git").exists():
        tar = dest.parent / "tree.tar"
        subprocess.run(["git", "-C", str(tree), "archive", "-o", str(tar), "HEAD"], check=True)
        unpack(tar, dest)
        tar.unlink()
        return subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    shutil.copytree(tree, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"), symlinks=True)
    return "folder"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, type=Path, help="the ai-serve tree to package")
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--arch", default=platform.machine(), choices=["x86_64", "aarch64"])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "ai-serve-build")
    a = ap.parse_args(argv)
    tree = a.tree.resolve()
    if not (tree / "installer" / "wizard.py").is_file():
        raise SystemExit(f"{tree} doesn't look like an ai-serve tree (installer/wizard.py is missing)")
    if not shutil.which("mksquashfs"):
        raise SystemExit("mksquashfs is missing (package squashfs-tools)")
    triple = f"{a.arch}-unknown-linux-gnu"
    uv = get(a.cache, "astral-sh/uv", UV_TAG, f"uv-{triple}.tar.gz")
    py = get(a.cache, "astral-sh/python-build-standalone", PYTHON_TAG,
             f"cpython-{PYTHON_VERSION}+{PYTHON_TAG}-{triple}-install_only_stripped.tar.gz")
    # the AppImage tools follow their "continuous" releases; the checksum still comes from GitHub
    tool = get(a.cache, "AppImage/appimagetool", "continuous", f"appimagetool-{a.arch}.AppImage")
    runtime = get(a.cache, "AppImage/type2-runtime", "continuous", f"runtime-{a.arch}")
    tool.chmod(0o755)

    with tempfile.TemporaryDirectory(prefix="ai-serve-appdir-") as tmp:
        app = Path(tmp) / "ai-serve.AppDir"
        unpack(py, app)                                  # -> app/python
        unpack(uv, Path(tmp) / "uv")
        (app / "bin").mkdir()
        shutil.copy2(Path(tmp) / "uv" / f"uv-{triple}" / "uv", app / "bin" / "uv")
        rev = copy_tree(tree, app / "ai-serve")
        stamp = f"{rev} {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        (app / "ai-serve" / ".bundle").write_text(stamp + "\n")
        shutil.copy2(HERE / "linux" / "AppRun", app / "AppRun")
        shutil.copy2(HERE / "linux" / "ai-serve.desktop", app / "ai-serve.desktop")
        shutil.copy2(HERE / "ai-serve.svg", app / "ai-serve.svg")
        os.symlink("ai-serve.svg", app / ".DirIcon")
        a.out.mkdir(parents=True, exist_ok=True)
        target = (a.out / f"ai-serve-{a.arch}.AppImage").resolve()
        env = {**os.environ, "ARCH": a.arch, "APPIMAGE_EXTRACT_AND_RUN": "1"}   # also where FUSE is missing
        subprocess.run([str(tool), "--no-appstream", "--runtime-file", str(runtime), str(app), str(target)],
                       check=True, env=env)
    print(f"{target}  ({target.stat().st_size >> 20} MB, {stamp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
