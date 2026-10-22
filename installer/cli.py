"""`./ai-serve` -- set up and run ai-serve.

    ./ai-serve setup                      llama.cpp for this machine + the basic services you choose (asks)
    ./ai-serve setup --offers llm-gemma-e2b,image-iris --yes
    ./ai-serve offers                     what setup can install, with sizes
    ./ai-serve plugin add <git-url> [--name N]   another app or plugin from a git repository
    ./ai-serve plugin list
    ./ai-serve start                      the gateway (http://localhost:8030)
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from installer import offers as O  # noqa: E402
from installer import plugins as P  # noqa: E402

ENV_FILE = HERE / "ai-serve.local.env"


def machine() -> O.Machine:
    nvidia = bool(shutil.which("nvidia-smi")) and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    return O.Machine(platform.system(), platform.machine(), nvidia=nvidia)


def ask(question: str, default: bool = False) -> bool:
    hint = "[J/n]" if default else "[j/N]"
    try:
        a = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return default
    return default if not a else a in ("j", "ja", "y", "yes")


def choose(m: O.Machine) -> List[str]:
    by_id = {o.id: o for o in O.offers(m)}
    print("\nGrundausstattung -- alles optional, später jederzeit nachholbar (./ai-serve setup):\n")
    picked: List[str] = []
    e4b, e2b = by_id["llm-gemma-e4b"], by_id["llm-gemma-e2b"]
    print(f"  Sprachmodell, liest auch Bilder: Gemma 4 E4B (~{e4b.size_gb} GB) oder E2B, besonders klein (~{e2b.size_gb} GB)")
    try:
        a = input("  [4] E4B  [2] E2B  [n] keins  (Standard 4): ").strip().lower() or "4"
    except EOFError:
        a = "4"
    picked += ["llm-gemma-e4b"] if a == "4" else ["llm-gemma-e2b"] if a == "2" else []
    for oid in ("tts-chatterbox", "stt-whisper", "image-iris"):
        o = by_id[oid]
        print(f"\n  {o.label}: {o.what}\n  ~{o.size_gb} GB, Lizenz {o.license}")
        if ask("  Einrichten?", default=True):
            picked.append(oid)
    return picked


def run_setup(offer: O.Offer) -> None:
    d = HERE / "plugins" / offer.plugin
    if not d.is_dir():
        raise SystemExit(f"plugins/{offer.plugin} is missing -- is this a complete ai-serve checkout?")
    script, *args = offer.setup
    print(f"\n== {offer.label}: plugins/{offer.plugin}/{script} {' '.join(args)}")
    subprocess.run(["bash", str(d / script), *args], cwd=d, check=True)


def setup(chosen: Optional[List[str]], yes: bool, skip_llamacpp: bool) -> None:
    m = machine()
    print(f"ai-serve setup -- {m.system} {m.machine}" + (" with NVIDIA GPU" if m.nvidia else ""))
    if not shutil.which("uv"):
        raise SystemExit("uv is needed for the services' Python environments: https://docs.astral.sh/uv/ "
                         "(curl -LsSf https://astral.sh/uv/install.sh | sh), then run this again.")
    ids = O.validate_choice(chosen, m) if chosen is not None else choose(m)
    picked = [o for o in O.offers(m) if o.id in ids]
    total = sum(o.size_gb for o in picked)
    print(f"\nGewählt: {', '.join(o.label for o in picked) or 'nur ai-serve selbst'} -- zusammen ~{total:.1f} GB")
    if not yes and not ask("Los?", default=True):
        raise SystemExit("abgebrochen")
    llama_server = None
    if not skip_llamacpp:
        print("\n== llama.cpp für diesen Rechner")
        from installer import llamacpp
        binary = llamacpp.install(HERE / "runtimes")
        llama_server = "./" + str(binary.relative_to(HERE))
    for o in picked:
        if o.files:
            print(f"\n== {o.label}")
            from installer.download import hf_download
            for f in o.files:
                hf_download(f.repo, f.path, HERE / "models" / f.save_as)
        if o.plugin:
            run_setup(o)
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else "# this machine's settings (ai-serve.example.env lists them all)\n"
    ENV_FILE.write_text(O.merge_env(text, O.env_updates(picked, llama_server)), encoding="utf-8")
    port = os.environ.get("AI_SERVE_PORT") or "8030"
    print(f"\nFertig. Einstellungen: {ENV_FILE.name}. Starten: ./ai-serve start  -> http://localhost:{port}"
          + ("  (Spielwiese: Apps & Services)" if (HERE / "apps" / "spielwiese").is_dir() else ""))


def plugin_add(url: str, name: Optional[str], yes: bool) -> None:
    info = P.add(url, HERE / "plugins", name)
    print(f"\n{info['name']} ({info['id']}) in {info['dir']}\n  Port {info['port']}, startet mit {' '.join(info['command'])}"
          f"\n  Stellt bereit: {', '.join(info['provides']) or '-- (eine App)'}")
    if info["setup"]:
        print("\n  setup.sh richtet seine Umgebung ein -- das ist Code aus dem Internet; vorher ansehen ist eine gute Idee.")
        if yes or ask("  setup.sh jetzt ausführen?"):
            subprocess.run(["bash", "setup.sh"], cwd=info["dir"], check=True)
    print("\nErscheint in Apps & Services; ai-serve startet es, wenn es gebraucht wird.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ai-serve", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--offers", help="comma-separated offer ids (see `offers`); without: ask")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--skip-llamacpp", action="store_true")
    sub.add_parser("offers")
    p = sub.add_parser("plugin")
    psub = p.add_subparsers(dest="pcmd", required=True)
    pa = psub.add_parser("add")
    pa.add_argument("url")
    pa.add_argument("--name")
    pa.add_argument("--yes", action="store_true", help="also run its setup.sh without asking")
    psub.add_parser("list")
    sub.add_parser("start")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "setup":
            setup(a.offers.split(",") if a.offers else None, a.yes, a.skip_llamacpp)
        elif a.cmd == "offers":
            for o in O.offers(machine()):
                print(f"{o.id:16} ~{o.size_gb:4.1f} GB  {o.label}\n{'':24}{o.what}")
        elif a.cmd == "plugin" and a.pcmd == "add":
            plugin_add(a.url, a.name, a.yes)
        elif a.cmd == "plugin":
            for r in P.listing(HERE / "apps", HERE / "plugins"):
                print(f"{r.get('id', '?'):20} {r['dir']}" + (f"  ({r['error']})" if "error" in r else f"  {', '.join(r['provides'])}"))
        elif a.cmd == "start":
            os.execv("/bin/bash", ["bash", str(HERE / "launcher.sh")])
    except (P.PluginError, ValueError) as e:
        print(f"ai-serve: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"ai-serve: step failed ({e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
