"""Setting up stt-serve's optional engines on demand (plugin contract endpoint `install`, ai-serve-ui
docs/plugins.md): MOSS-Transcribe-Diarize and the Nemotron diarization worker each have their own setup.sh that
builds their environment. The owner asks by id; the script runs in the background; nothing else from the request is
used. Both need an NVIDIA GPU (CUDA builds of PyTorch) -- elsewhere they aren't offered. WhisperX itself comes with
stt-serve's own setup and fetches its model on first use."""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ITEMS = {
    "moss": {"label": "MOSS-Transcribe-Diarize", "dir": "moss", "size_mb": 7000, "license": "Apache-2.0",
             "description": "Text, speakers and times in one pass, up to 90 minutes"},
    "nemotron": {"label": "Nemotron diarization", "dir": "nemotron", "size_mb": 5500, "license": "NVIDIA Open Model",
                 "description": "Speaker separation with NVIDIA's Nemotron model (an alternative to pyannote)"},
}


class Installer:
    def __init__(self, root: Path, ready: Callable[[str], bool], has_cuda: Callable[[], bool], logs_dir: Path,
                 run: Optional[Callable[[Path, Path], int]] = None):
        self.root, self.ready, self.has_cuda, self.logs_dir = Path(root), ready, has_cuda, Path(logs_dir)
        self._run = run or self._run_setup
        self.state: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def items(self) -> List[Dict[str, Any]]:
        if not self.has_cuda():
            return []
        return [self.item(k) for k in ITEMS]

    def item(self, key: str) -> Dict[str, Any]:
        spec, st = ITEMS[key], self.state.get(key, {})
        state = "installed" if self.ready(key) and st.get("state") != "downloading" else (st.get("state") or "missing")
        return {"id": key, "label": spec["label"], "description": spec["description"], "size_mb": spec["size_mb"],
                "license": spec["license"], "state": state, "error": st.get("error", "") if state == "failed" else ""}

    def start(self, key: Any) -> Dict[str, Any]:
        if not isinstance(key, str) or key not in ITEMS or not self.has_cuda():
            raise LookupError("nothing to set up under that id on this machine")
        with self._lock:
            if not self.ready(key) and self.state.get(key, {}).get("state") != "downloading":
                self.state[key] = {"state": "downloading"}
                threading.Thread(target=self._work, args=(key,), daemon=True).start()
        return self.item(key)

    def _work(self, key: str) -> None:
        log = self.logs_dir / f"install-{key}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            code, err = self._run(self.root / ITEMS[key]["dir"] / "setup.sh", log), ""
        except Exception as e:
            code, err = -1, f"{type(e).__name__}: {e}"
        if code == 0 and self.ready(key):
            self.state[key] = {"state": "installed"}
        else:
            self.state[key] = {"state": "failed", "error": err or f"setup.sh exited with {code} -- {_tail(log)}"}

    @staticmethod
    def _run_setup(script: Path, log: Path) -> int:
        if not script.is_file() or script.is_symlink():
            raise FileNotFoundError(f"{script.parent.name} has no setup.sh")
        with open(log, "w", encoding="utf-8") as out:
            return subprocess.run(["bash", str(script)], cwd=script.parent, stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=dict(os.environ)).returncode


def has_cuda() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _tail(log: Path, n: int = 3) -> str:
    try:
        lines = [l.strip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[:250] or "no output"
    except OSError:
        return "no log"
