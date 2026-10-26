"""`./ai-serve` -- set up and run ai-serve.

    ./ai-serve setup                      llama.cpp for this machine + the basic services you choose (asks; the
                                          defaults follow the GPU, RAM and disk space found)
    ./ai-serve setup --recommended --yes  exactly what fits this machine, without asking
    ./ai-serve setup --offers llm-gemma-e2b,image-iris --yes
    ./ai-serve offers                     what setup can install, with sizes and what is suggested here
    ./ai-serve plugin add <git-url> [--name N]   another app or plugin from a git repository
    ./ai-serve plugin list
    ./ai-serve start [-d]                 the gateway (http://localhost:8030); -d: in the background
    ./ai-serve stop [--all]               stop it (--all: also the apps it started and every loaded model)
    ./ai-serve restart | status
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from installer import offers as O  # noqa: E402
from installer import plugins as P  # noqa: E402

ENV_FILE = HERE / "ai-serve.local.env"


def gateway_port() -> int:
    """AI_SERVE_PORT from the environment, else from ai-serve.local.env, else 8030."""
    v = os.environ.get("AI_SERVE_PORT")
    if not v and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("AI_SERVE_PORT="):
                v = line.split("=", 1)[1].strip()
    return int(v) if v and v.isdigit() else 8030


def call(path: str, method: str = "GET", timeout: float = 5.0):
    """The local gateway's answer as JSON, or None when it doesn't answer (this machine counts as the owner)."""
    req = urllib.request.Request(f"http://127.0.0.1:{gateway_port()}{path}", method=method,
                                 data=b"{}" if method == "POST" else None, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def running() -> bool:
    return call("/api/version", timeout=2) is not None


def wait_for(up: bool, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if running() == up:
            return True
        time.sleep(0.5)
    return running() == up


def start(background: bool) -> int:
    url = f"http://localhost:{gateway_port()}"
    if running():
        print(f"ai-serve is already running -> {url}  (./ai-serve stop | restart)")
        return 0
    if not background:
        os.execv("/bin/bash", ["bash", str(HERE / "launcher.sh")])
    log = HERE / "logs" / "launcher.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen(["bash", str(HERE / "launcher.sh")], cwd=HERE, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print("Starting ai-serve in the background (log: logs/launcher.log) …")
    if wait_for(True, 180):
        print(f"ai-serve runs -> {url}")
        refresh_outdated()
        return 0
    print("ai-serve didn't come up within 3 minutes -- see logs/launcher.log", file=sys.stderr)
    return 1


def stop(everything: bool) -> int:
    if not running():
        print("ai-serve isn't running.")
        (HERE / "kill.flag").unlink(missing_ok=True)
        return 0
    if everything:
        services = (call("/api/services") or {}).get("services", [])
        for svc in services:
            for inst in svc.get("instances", []):
                req = urllib.request.Request(f"http://127.0.0.1:{gateway_port()}/api/services/{svc['id']}/stop",
                                             data=json.dumps({"pid": inst["pid"]}).encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(req, timeout=30).close()
                    print(f"  stopped {svc.get('name') or svc['id']}")
                except OSError as e:
                    print(f"  {svc['id']}: not stopped ({e})", file=sys.stderr)
        r = call("/api/models/unload_all", "POST", timeout=60)
        if r is not None:
            print(f"  models: {r.get('message') or 'unloaded'}")
    (HERE / "kill.flag").touch()            # launcher.sh stops the gateway and ends
    if wait_for(False, 30):
        print("ai-serve stopped." + ("" if everything else
              "  (Apps and loaded models keep running; the next start takes them over. --all stops them too.)"))
        return 0
    print("ai-serve is still answering -- was it started without ./ai-serve start (launcher.sh)? "
          "Then stop that process yourself.", file=sys.stderr)
    return 1


def refresh_outdated() -> None:
    """Apps and plugins of this folder still running code from before an update are restarted (when idle)."""
    r = call("/api/services/restart-outdated", "POST", timeout=300)
    if not r:
        return
    for sid in r.get("restarted", []):
        print(f"  restarted {sid} (new code)")
    for sid in r.get("busy", []):
        print(f"  {sid} runs older code but is busy -- restart it later (Apps & Services)")


def restart() -> int:
    if not running():
        print("ai-serve isn't running -- starting it in the background.")
        return start(True)
    (HERE / "restart.flag").touch()         # launcher.sh restarts the gateway
    wait_for(False, 20)
    if wait_for(True, 180):
        print(f"ai-serve restarted -> http://localhost:{gateway_port()}")
        refresh_outdated()
        return 0
    print("ai-serve didn't come back -- see logs/launcher.log or the terminal it runs in", file=sys.stderr)
    return 1


def status() -> int:
    v = call("/api/version", timeout=2)
    if v is None:
        print(f"ai-serve isn't running (port {gateway_port()}).")
        return 1
    svcs = (call("/api/services") or {}).get("services", [])
    up = [s.get("name") or s["id"] for s in svcs if s.get("instances")]
    loaded = [i.get("model") for i in (call("/api/status") or {}).get("instances", []) if i.get("model")]
    print(f"ai-serve runs -> http://localhost:{gateway_port()}  (backend {v.get('version', '?')})")
    print(f"  apps running: {', '.join(up) or 'none'}")
    print(f"  models loaded: {', '.join(loaded) or 'none'}")
    return 0


def machine() -> O.Machine:
    """What this machine has -- each figure 0/empty when it can't be read (the suggestions then fall back)."""
    gpus: List[int] = []
    nvidia = False
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=15)
            nvidia = r.returncode == 0
            gpus = [int(float(x)) for x in r.stdout.split() if x.strip()] if nvidia else []
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        ram_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >> 20
    except (ValueError, OSError, AttributeError):
        ram_mb = 0
    try:
        free_gb = shutil.disk_usage(HERE).free / 1e9
    except OSError:
        free_gb = 0
    return O.Machine(platform.system(), platform.machine(), nvidia=nvidia, gpus_mb=tuple(gpus), ram_mb=ram_mb,
                     disk_free_gb=free_gb)


def ask(question: str, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        a = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return default
    return default if not a else a in ("y", "yes", "j", "ja")


def choose(m: O.Machine) -> List[str]:
    """Asks offer by offer; the defaults are the suggestions for this machine (offers.recommend)."""
    by_id = {o.id: o for o in O.offers(m)}
    rec = {oid: (yes, why) for oid, yes, why in O.recommend(m)}
    print("\nBasic set -- all optional, can be added any time later (./ai-serve setup).")
    print("The defaults are suggestions for this machine.\n")
    picked: List[str] = []
    keys = {"12": "llm-gemma-12b", "4": "llm-gemma-e4b", "2": "llm-gemma-e2b"}
    print("  Language model, also reads images: " + ", ".join(
        f"Gemma 4 {by_id[i].label.split('Gemma 4 ')[1].split(' ')[0]} (~{by_id[i].size_gb} GB)" for i in keys.values()))
    suggested = next(k for k, i in keys.items() if rec[i][0])
    print(f"  Suggested: {by_id[keys[suggested]].label.split(' (')[0]} -- {rec[keys[suggested]][1]}")
    try:
        a = input(f"  [12] 12B  [4] E4B  [2] E2B  [n] none  (default {suggested}): ").strip().lower() or suggested
    except EOFError:
        a = suggested
    picked += [keys[a]] if a in keys else []
    for oid in ("tts-chatterbox", "tts-piper", "stt-whisper", "image-iris", "image-comfyui"):
        o = by_id[oid]
        yes, why = rec[oid]
        print(f"\n  {o.label}: {o.what}\n  ~{o.size_gb} GB, license {o.license}\n"
              f"  {'Suggested' if yes else 'Not suggested'}: {why}")
        if ask("  Set it up?", default=yes):
            picked.append(oid)
    return picked


def run_setup(offer: O.Offer) -> None:
    d = HERE / "plugins" / offer.plugin
    if not d.is_dir() and offer.git:   # an external plugin: cloned and its manifest checked like `plugin add`
        print(f"\n== {offer.label}: {offer.git} -> plugins/{offer.plugin}")
        P.add(offer.git, HERE / "plugins", offer.plugin)
    if not d.is_dir():
        raise SystemExit(f"plugins/{offer.plugin} is missing -- is this a complete ai-serve checkout?")
    script, *args = offer.setup
    print(f"\n== {offer.label}: plugins/{offer.plugin}/{script} {' '.join(args)}")
    subprocess.run(["bash", str(d / script), *args], cwd=d, check=True)


def setup(chosen: Optional[List[str]], yes: bool, skip_llamacpp: bool, recommended: bool = False) -> None:
    m = machine()
    print(f"ai-serve setup -- {O.describe(m)}")
    if recommended:
        if chosen is not None:
            raise ValueError("give either --offers or --recommended")
        chosen = [oid for oid, ok, _ in O.recommend(m) if ok]
    if not shutil.which("uv"):
        raise SystemExit("uv is needed for the services' Python environments: https://docs.astral.sh/uv/ "
                         "(curl -LsSf https://astral.sh/uv/install.sh | sh), then run this again.")
    ids = O.validate_choice(chosen, m) if chosen is not None else choose(m)
    picked = [o for o in O.offers(m) if o.id in ids]
    total = sum(o.size_gb for o in picked)
    print(f"\nChosen: {', '.join(o.label for o in picked) or 'only ai-serve itself'} -- together ~{total:.1f} GB")
    if m.disk_free_gb and total > m.disk_free_gb - O.DISK_RESERVE_GB:
        print(f"Note: only {m.disk_free_gb:.0f} GB free on the disk -- that is tight or too little.")
    if not yes and not ask("Go?", default=True):
        raise SystemExit("cancelled")
    llama_server = None
    if not skip_llamacpp:
        print("\n== llama.cpp for this machine")
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
    print(f"\nDone. Settings: {ENV_FILE.name}. Start: ./ai-serve start  -> http://localhost:{port}"
          + ("  (Playground: Apps & Services)" if (HERE / "apps" / "playground").is_dir() else ""))


def plugin_add(url: str, name: Optional[str], yes: bool) -> None:
    info = P.add(url, HERE / "plugins", name)
    print(f"\n{info['name']} ({info['id']}) in {info['dir']}\n  Port {info['port']}, started with {' '.join(info['command'])}"
          f"\n  Provides: {', '.join(info['provides']) or '-- (an app)'}")
    if info["setup"]:
        print("\n  setup.sh sets up its environment -- that is code from the internet; looking at it first is a good idea.")
        if yes or ask("  Run setup.sh now?"):
            subprocess.run(["bash", "setup.sh"], cwd=info["dir"], check=True)
    print("\nShows up in Apps & Services; ai-serve starts it when it is needed.")


def model_add(url: str, file: Optional[str], mmproj: str, yes: bool) -> None:
    """A GGUF model from Hugging Face into models/ -- the same rules and checks as the dashboard's form
    (backend/hf_models.py): only huggingface.co, sha256 from Hugging Face, nothing overwritten or deleted."""
    from backend import hf_download, hf_models
    try:
        info = hf_download.inspect(url)
        model = file or info["selected"]
        if not model:
            print(f"{info['repo']} -- models:")
            for i, m in enumerate(info["models"], 1):
                parts = f", {m['parts']} parts" if m.get("parts", 1) > 1 else ""
                print(f"  {i:2}. {m['path']}  ({m['size'] / 2**30:.1f} GB{parts})")
            if not sys.stdin.isatty():
                raise ValueError("several models in this repo -- name one with --file <path from the list>")
            pick = input("Number: ").strip()
            if not pick.isdigit() or not 1 <= int(pick) <= len(info["models"]):
                raise ValueError("no model chosen")
            model = info["models"][int(pick) - 1]["path"]
        proj = None
        if mmproj not in ("none", "auto"):
            proj = mmproj
        elif mmproj == "auto" and info["mmproj"]:
            names = [p["path"] for p in info["mmproj"]]
            proj = next((n for n in names if re.search(r"(?<!b)f16", n.lower())), names[0])   # F16, not BF16
            print(f"Vision projector (lets the model see images): {proj}  -- --mmproj none to leave it out")
        plan = hf_download.make_plan(url, model, proj)
    except hf_models.HfLinkError as e:
        raise ValueError(str(e))
    for name in plan["already"]:
        print(f"  {name}: already there")
    if not plan["files"]:
        print(f"Nothing to download -- {plan['model']} is ready.")
        return
    print(f"Into {plan['dest']} ({plan['size'] / 2**30:.1f} GB):")
    for f in plan["files"]:
        print(f"  {f['save_as']}  ({f['size'] / 2**30:.1f} GB)")
    if not yes and not ask("Download?"):
        return
    for f in plan["files"]:
        done = [0]

        def progress(n, f=f, done=done):
            done[0] += n
            print(f"\r  {f['save_as']}: {done[0] >> 20} / {f['size'] >> 20} MB", end="", flush=True)
        try:
            hf_download.fetch(f, plan["dest"], progress)
        except hf_models.HfLinkError as e:
            print()
            raise ValueError(str(e))
        print("  -- checked")
    print(f"Ready: {plan['model']} -- use it with \"model\": \"{plan['model']}\" (or pick it in the Playground).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ai-serve", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--offers", help="comma-separated offer ids (see `offers`); without: ask")
    s.add_argument("--recommended", action="store_true", help="the suggestions for this machine (see `offers`)")
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
    m = sub.add_parser("model")
    msub = m.add_subparsers(dest="mcmd", required=True)
    ma = msub.add_parser("add", help="download a GGUF model from Hugging Face (docs/models.md)")
    ma.add_argument("url", help="the repo's page (https://huggingface.co/<owner>/<repo>), a file's page, or owner/repo[:QUANT]")
    ma.add_argument("--file", help="the model file's path in the repo (else: from the address, or a list to choose from)")
    ma.add_argument("--mmproj", default="auto", help="vision projector: auto (default), none, or its path in the repo")
    ma.add_argument("--yes", action="store_true")
    st = sub.add_parser("start")
    st.add_argument("-d", "--background", action="store_true", help="run in the background (log: logs/launcher.log)")
    sp = sub.add_parser("stop")
    sp.add_argument("--all", action="store_true", help="also stop the apps it started and unload every model")
    sub.add_parser("restart")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "setup":
            setup(a.offers.split(",") if a.offers else None, a.yes, a.skip_llamacpp, a.recommended)
        elif a.cmd == "offers":
            mach = machine()
            rec = {oid: (yes, why) for oid, yes, why in O.recommend(mach)}
            print(f"This machine: {O.describe(mach)}\n")
            for o in O.offers(mach):
                yes, why = rec[o.id]
                print(f"{'*' if yes else ' '} {o.id:16} ~{o.size_gb:4.1f} GB  {o.label}\n{'':26}{o.what}\n"
                      f"{'':26}{'suggested' if yes else 'not suggested'}: {why}")
            print("\n* = suggested here; take them with ./ai-serve setup --recommended")
        elif a.cmd == "plugin" and a.pcmd == "add":
            plugin_add(a.url, a.name, a.yes)
        elif a.cmd == "plugin":
            for r in P.listing(HERE / "apps", HERE / "plugins"):
                print(f"{r.get('id', '?'):20} {r['dir']}" + (f"  ({r['error']})" if "error" in r else f"  {', '.join(r['provides'])}"))
        elif a.cmd == "model":
            model_add(a.url, a.file, a.mmproj, a.yes)
        elif a.cmd == "start":
            return start(a.background)
        elif a.cmd == "stop":
            return stop(a.all)
        elif a.cmd == "restart":
            return restart()
        elif a.cmd == "status":
            return status()
    except (P.PluginError, ValueError) as e:
        print(f"ai-serve: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"ai-serve: step failed ({e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
