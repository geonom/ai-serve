"""The setup's steps, one path for the terminal (`./ai-serve setup`) and the setup page (installer/wizard.py): ai-serve's
own Python environment, llama.cpp for this machine, each chosen offer (model downloads, plugin setup), the settings
file. Progress and output go to a `Report`; the terminal's prints them, the page's keeps them for the browser."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from installer import offers as O
from installer import plugins as P

HERE = Path(__file__).resolve().parent.parent
ENV_FILE = HERE / "ai-serve.local.env"


class Report:
    """Prints to the terminal. The setup page subclasses it."""

    def step(self, label: str) -> None:
        print(f"\n== {label}", flush=True)

    def log(self, text: str) -> None:
        print(text, flush=True)

    def progress(self, label: str, done: int, total: int) -> None:
        if total:
            print(f"\r  {label}: {done >> 20} / {total >> 20} MB ({done * 100 // total} %)", end="", flush=True)
            if done >= total:
                print()


def venv_python() -> Path:
    return HERE / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def find_uv() -> Optional[str]:
    """uv named by AI_SERVE_UV, the one a package put into bin/, or the one on the PATH."""
    for v in (os.environ.get("AI_SERVE_UV"), str(HERE / "bin" / ("uv.exe" if sys.platform == "win32" else "uv"))):
        if v and os.path.isfile(v):
            return v
    return shutil.which("uv")


def run(cmd: List[str], cwd: Path, report: Report, env: Optional[dict] = None) -> None:
    """A command with its output passed on line by line; CalledProcessError when it fails."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         text=True, errors="replace", bufsize=1, env=env)
    for line in p.stdout:
        report.log(line.rstrip("\n"))
    if p.wait() != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def core_environment(report: Report) -> None:
    """ai-serve's own Python environment (uv venv + backend/requirements.txt); quick when it is there already."""
    uv = find_uv()
    if not uv:
        raise SystemExit("uv is needed: https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)")
    report.step("ai-serve's Python environment")
    if not venv_python().exists():
        run([uv, "venv", "-q", "--python", "3.12", str(HERE / "venv")], HERE, report)
    run([uv, "pip", "install", "-q", "--python", str(venv_python()), "-r", str(HERE / "backend" / "requirements.txt")],
        HERE, report)
    example = HERE / "ai-serve.example.env"
    if not ENV_FILE.exists() and example.exists():
        shutil.copyfile(example, ENV_FILE)


def setup_plugin(offer: O.Offer, report: Report) -> None:
    d = HERE / "plugins" / offer.plugin
    if not d.is_dir() and offer.git:   # an external plugin: cloned and its manifest checked like `plugin add`
        report.log(f"{offer.git} -> plugins/{offer.plugin}")
        P.add(offer.git, HERE / "plugins", offer.plugin)
    if not d.is_dir():
        raise SystemExit(f"plugins/{offer.plugin} is missing -- is this a complete ai-serve checkout?")
    script, *args = offer.setup
    report.log(f"plugins/{offer.plugin}/{script} {' '.join(args)}")
    env = dict(os.environ)
    uv = find_uv()
    if uv:   # the plugins' setup scripts call `uv`; a bundled one is put first on their PATH
        env["PATH"] = os.path.dirname(uv) + os.pathsep + env.get("PATH", "")
    run(["bash", str(d / script), *args], d, report, env)


def install(picked: List[O.Offer], report: Report, skip_llamacpp: bool = False,
            before_each: Callable[[str], None] = lambda _id: None) -> None:
    """llama.cpp + every chosen offer + the settings file. `before_each(id)` lets the page mark the current offer."""
    llama_server = None
    if not skip_llamacpp:
        report.step("llama.cpp for this machine")
        from installer import llamacpp
        binary = llamacpp.install(HERE / "runtimes", report=report)
        llama_server = "./" + binary.relative_to(HERE).as_posix()
    for o in picked:
        before_each(o.id)
        report.step(o.label)
        if o.files:
            from installer.download import hf_download
            for f in o.files:
                hf_download(f.repo, f.path, HERE / "models" / f.save_as, report)
        if o.plugin:
            setup_plugin(o, report)
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else \
        "# this machine's settings (ai-serve.example.env lists them all)\n"
    ENV_FILE.write_text(O.merge_env(text, O.env_updates(picked, llama_server)), encoding="utf-8")
