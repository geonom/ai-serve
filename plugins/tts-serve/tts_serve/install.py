"""Setting up a provider on demand (plugin contract endpoint `install`, ai-serve-ui docs/plugins.md): the owner asks
for an engine by id; tts-serve runs that provider's own setup.sh (shipped in its folder, builds its environment and
fetches its model) in the background and reports progress. Nothing else from the request is used: no URLs, no
paths, no arguments. A provider this machine can't run (its `requires`) isn't offered."""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tts_serve.domain import ProviderManifest, ProviderRegistry, supported


class Installer:
    def __init__(self, registry: Callable[[], ProviderRegistry], logs_dir: Path, has_cuda: Callable[[], bool],
                 system: str, machine: str, run: Optional[Callable[..., int]] = None):
        self.registry = registry
        self.logs_dir = Path(logs_dir)
        self.has_cuda = has_cuda
        self.system, self.machine = system, machine
        self._run = run or self._run_setup
        self.state: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def _offered(self) -> List[ProviderManifest]:
        cuda = self.has_cuda()
        return [m for m in self.registry().providers if supported(m, cuda, self.system, self.machine)]

    def item(self, m: ProviderManifest) -> Dict[str, Any]:
        st = self.state.get(m.id, {})
        state = "installed" if m.installed and st.get("state") != "downloading" else (st.get("state") or "missing")
        return {"id": m.id, "label": m.name, "description": m.description, "size_mb": m.install_mb,
                "state": state, "license": m.license, "error": st.get("error", "") if state == "failed" else ""}

    def items(self) -> List[Dict[str, Any]]:
        return [self.item(m) for m in self._offered()]

    def start(self, engine_id: Any) -> Dict[str, Any]:
        m = next((p for p in self._offered() if p.id == engine_id), None) if isinstance(engine_id, str) else None
        if m is None:
            raise LookupError("no engine to set up under that id on this machine")
        with self._lock:
            if not m.installed and self.state.get(m.id, {}).get("state") != "downloading":
                self.state[m.id] = {"state": "downloading"}
                threading.Thread(target=self._work, args=(m,), daemon=True).start()
        return self.item(m)

    def _work(self, m: ProviderManifest) -> None:
        log = self.logs_dir / f"install-{m.id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            code = self._run(m, log)
        except Exception as e:   # never leave it "downloading" forever
            code, err = -1, f"{type(e).__name__}: {e}"
        else:
            err = ""
        fresh = next((p for p in self.registry().providers if p.id == m.id), None)
        if code == 0 and fresh is not None and fresh.installed:
            self.state[m.id] = {"state": "installed"}
        else:
            self.state[m.id] = {"state": "failed", "error": err or f"setup.sh exited with {code} -- " + _tail(log)}

    @staticmethod
    def _run_setup(m: ProviderManifest, log: Path) -> int:
        script = m.dir / "setup.sh"
        if not script.is_file() or script.is_symlink():
            raise FileNotFoundError(f"{m.id} has no setup.sh")
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as out:
            return subprocess.run(["bash", str(script), "--download"], cwd=m.dir, stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=dict(os.environ)).returncode


def _tail(log: Path, n: int = 3) -> str:
    try:
        lines = [l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[:250] or "no output"
    except OSError:
        return "no log"
