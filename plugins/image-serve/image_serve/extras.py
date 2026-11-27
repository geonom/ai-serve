"""Local extra modules (2026-11-27, owner): models kept on this machine only -- their licence doesn't allow
publishing them -- live in modules of their own, image_serve/extra_<kind>_<name>.py, that the public tree leaves out.
video.py, catalog.py, comfy.py and app.py load whatever is there; without them everything else works the same.
Tests: tests/test_extras.py."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import List

_PACKAGE = Path(__file__).resolve().parent


def _names() -> List[str]:
    return [m.name for m in pkgutil.iter_modules([str(_PACKAGE)])]


def load(prefix: str) -> List[ModuleType]:
    """Every module of this package whose name starts with `prefix`, in name order (an error inside one is raised)."""
    return [importlib.import_module(f"image_serve.{n}") for n in sorted(_names()) if n.startswith(prefix)]
