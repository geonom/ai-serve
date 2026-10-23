"""Apps and plugins still running old code after an update (owner on the Mac, 2026-10-23: `./ai-serve restart` only
restarted the gateway; image-serve and the Spielwiese kept running the code from before `git pull`, so a fixed
download stayed hidden). Rules here, pure except `newest_code_mtime` (tests: tests/unit/test_service_updates.py):

- only what belongs to this installation -- apps/ and plugins/ inside the ai-serve folder; projects in other service
  roots belong to someone else and are never restarted;
- outdated = a code file in its folder is newer than the running process (environments, models, logs, data don't
  count);
- the caller restarts it only when it doesn't report itself busy (status `busy`)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CODE_SUFFIXES = {".py", ".js", ".mjs", ".ts", ".html", ".css", ".json", ".sh", ".toml", ".txt", ".lock"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "logs", "models", "voices", "data", "output", "runtimes",
             "generated_images", "ComfyUI", ".pytest_cache", "dist", "build"}


def _skip_dir(name: str) -> bool:
    return (name in SKIP_DIRS or name.startswith((".venv", "venv")) or name.endswith(("-venv", "_venv"))
            or name in (".ci", "piper-venv"))


def newest_code_mtime(project_dir: str) -> float:
    newest = 0.0
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not _skip_dir(d)]
        for f in files:
            if os.path.splitext(f)[1] in CODE_SUFFIXES or f in ("run.sh", "setup.sh"):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return newest


def part_of_installation(project_dir: str, install_root: str) -> bool:
    d, root = Path(project_dir).resolve(), Path(install_root).resolve()
    return any(d.is_relative_to(root / sub) and d != root / sub for sub in ("apps", "plugins"))


def outdated(instances: Iterable[Tuple[str, str, int, float]], newest: Dict[str, float],
             install_root: str) -> List[Tuple[str, int]]:
    """instances: (service id, project dir, root pid, started_at) -> [(service id, pid)] to restart."""
    out = []
    for sid, pdir, pid, started in instances:
        if part_of_installation(pdir, install_root) and newest.get(pdir, 0.0) > started:
            out.append((sid, pid))
    return out
