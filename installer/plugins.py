"""Adding plugins by git URL (`./ai-serve plugin add <url>`) -- the owner: other repos must be as easy to use as the
bundled plugins, "angegeben als git url, um sie einfach selber zu installieren".

Threat model: a plugin is someone else's code that ai-serve will start. Adding one is always the owner's explicit
act at the terminal; these rules make sure the URL can only mean "clone this repository" -- https:// or ssh (git@host:
path) only, never a local path or file://, nothing that git would read as an option, no whitespace or control
characters -- that the folder name is plain, that nothing existing is overwritten, that the plugin's manifest is
checked like any app's (backend/services/domain.py) before it counts, and that its setup.sh runs only after the owner
said yes (it is code from the internet). Tests: tests/unit/test_installer_plugins.py."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

_HTTPS = re.compile(r"^https://[A-Za-z0-9.-]+(:\d{1,5})?(/[A-Za-z0-9._~%+-]+)+/?$")
_SSH = re.compile(r"^(ssh://)?[A-Za-z0-9._-]+@[A-Za-z0-9.-]+[:/][A-Za-z0-9._~/+-]+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class PluginError(ValueError):
    pass


def check_url(url: str) -> str:
    url = url or ""          # no stripping: a URL with whitespace or a newline is refused, not repaired
    if not url or url.startswith("-") or any(ord(c) < 33 for c in url):
        raise PluginError("give a git URL (https://… or git@host:owner/repo)")
    if url.startswith(("file:", "/", ".", "~")) or "::" in url or "/../" in url + "/" or "/./" in url + "/":
        raise PluginError("only remote repositories (https:// or ssh), never local paths")
    if not (_HTTPS.match(url) or _SSH.match(url)):
        raise PluginError("only https://host/owner/repo or git@host:owner/repo")
    return url


def name_for(url: str, name: Optional[str] = None) -> str:
    n = (name or re.split(r"[/:]", url.rstrip("/"))[-1]).lower()
    n = n[:-4] if n.endswith(".git") else n
    if not _NAME.match(n) or n in (".", ".."):
        raise PluginError(f"plugin folder name {n!r}: lowercase letters, digits, . _ - (pass --name)")
    return n


def _git_clone(url: str, dest: Path) -> None:
    # `--` ends options: the URL can't be read as one; no submodules, shallow
    subprocess.run(["git", "clone", "--depth", "1", "--", url, str(dest)], check=True)


def add(url: str, plugins_dir: Path, name: Optional[str] = None,
        clone: Callable[[str, Path], None] = _git_clone) -> Dict:
    """Clone a plugin into plugins/<name> and check its manifest. Returns what it is (for the owner to confirm the
    setup step). A clone without a valid manifest is removed again (only what this call just created)."""
    url = check_url(url)
    n = name_for(url, name)
    dest = plugins_dir / n
    if dest.exists():
        raise PluginError(f"{dest} exists already")
    plugins_dir.mkdir(parents=True, exist_ok=True)
    clone(url, dest)
    try:
        info = describe(dest)
    except PluginError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return info


def describe(d: Path) -> Dict:
    from backend.services.domain import ManifestError, ServiceManifest   # the same checks as Apps & Services
    import json
    f = d / "llama-serve.service.json"
    if not f.is_file():
        raise PluginError(f"{d.name}: no llama-serve.service.json -- not an ai-serve app or plugin")
    try:
        m = ServiceManifest.from_dict(json.loads(f.read_text(encoding="utf-8")), d)
    except (ManifestError, ValueError) as e:
        raise PluginError(f"{d.name}: manifest not valid: {e}")
    return {"id": m.id, "name": m.name, "dir": str(d), "port": m.port, "command": list(m.command),
            "provides": sorted(m.provides), "setup": (d / "setup.sh").is_file()}


def listing(*dirs: Path) -> List[Dict]:
    out = []
    for root in dirs:
        for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            try:
                out.append(describe(d))
            except PluginError as e:
                out.append({"dir": str(d), "error": str(e)})
    return out
