"""Setting up an engine on demand (tts_serve/install.py, plugin contract `install`): only providers this machine can
run are offered; a request names just the id; the provider's own setup.sh runs in the background; a failure says why;
an engine that isn't set up is refused for generating, not started."""
import json
import os
import time
from pathlib import Path

import pytest

from tts_serve.domain import ProviderRegistry, supported
from tts_serve.install import Installer


def provider(root: Path, pid: str, requires=None, installed=False):
    d = root / pid
    (d / "venv" / "bin").mkdir(parents=True)
    m = {"id": pid, "name": pid.title(), "capability": "tts", "command": ["./venv/bin/python", "worker.py"],
         "gpu": 0, "vram_mb": 1000, "install_mb": 3000}
    if requires:
        m["requires"] = requires
    (d / "llama-serve.provider.json").write_text(json.dumps(m))
    (d / "setup.sh").write_text("#!/bin/sh\necho ok\n")
    if installed:
        exe = d / "venv" / "bin" / "python"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    return d


def wait(inst, pid, state, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if inst.state.get(pid, {}).get("state") == state:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def root(tmp_path):
    provider(tmp_path, "plain")
    provider(tmp_path, "gpu-only", requires=["cuda"])
    provider(tmp_path, "mac-only", requires=["darwin-arm64"])
    provider(tmp_path, "ready", installed=True)
    return tmp_path


def make(root, ran, cuda=False, system="Linux", machine="x86_64", code=0):
    def run(m, log):
        ran.append(m.id)
        if code == 0:
            exe = m.dir / "venv" / "bin" / "python"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
        log.write_text("step 1\nboom: no space left\n")
        return code
    return Installer(lambda: ProviderRegistry.load(root), root / "logs", lambda: cuda, system, machine, run=run)


def test_only_what_this_machine_can_run(root):
    assert [i["id"] for i in make(root, []).items()] == ["plain", "ready"]
    assert [i["id"] for i in make(root, [], cuda=True).items()] == ["gpu-only", "plain", "ready"]
    assert [i["id"] for i in make(root, [], system="Darwin", machine="arm64").items()] == ["mac-only", "plain", "ready"]
    items = {i["id"]: i for i in make(root, []).items()}
    assert (items["plain"]["state"], items["plain"]["size_mb"]) == ("missing", 3000)
    assert items["ready"]["state"] == "installed"


def test_setup_runs_in_the_background_once(root):
    ran = []
    inst = make(root, ran)
    assert inst.start("plain")["state"] == "downloading"
    assert wait(inst, "plain", "installed")
    assert ran == ["plain"]
    assert {i["id"]: i["state"] for i in inst.items()}["plain"] == "installed"


@pytest.mark.parametrize("bad", ["gpu-only", "nope", "../plain", None, 5])
def test_refused(root, bad):
    with pytest.raises(LookupError):
        make(root, []).start(bad)


def test_a_failure_says_why(root):
    ran = []
    inst = make(root, ran, code=1)
    inst.start("plain")
    assert wait(inst, "plain", "failed")
    item = {i["id"]: i for i in inst.items()}["plain"]
    assert item["state"] == "failed" and "no space left" in item["error"]


def test_requires_is_checked():
    from tts_serve.domain import ProviderError, ProviderManifest
    with pytest.raises(ProviderError):
        ProviderManifest.from_dict({"id": "x", "name": "X", "capability": "tts", "command": ["./run"], "gpu": 0,
                                    "vram_mb": 1, "requires": ["quantum"]}, Path("/tmp"))
