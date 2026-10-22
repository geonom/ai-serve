"""Setting up MOSS-Transcribe / Nemotron on demand (stt_serve/install.py): only on machines with an NVIDIA GPU, only
the two known ids, their own setup.sh in the background, a failure says why."""
import time
from pathlib import Path

import pytest

from stt_serve.install import Installer


def make(tmp_path, ran, cuda=True, code=0):
    ready = set()

    def run(script: Path, log: Path):
        ran.append(script.parent.name)
        log.write_text("downloading\nerror: CUDA 13 wheels not found\n")
        if code == 0:
            ready.add(script.parent.name)
        return code
    return Installer(tmp_path, lambda k: k in ready, lambda: cuda, tmp_path / "logs", run=run)


def wait(inst, key, state):
    for _ in range(100):
        if inst.state.get(key, {}).get("state") == state:
            return True
        time.sleep(0.05)
    return False


def test_offered_only_with_cuda(tmp_path):
    assert [i["id"] for i in make(tmp_path, []).items()] == ["moss", "nemotron"]
    assert make(tmp_path, [], cuda=False).items() == []
    with pytest.raises(LookupError):
        make(tmp_path, [], cuda=False).start("moss")


def test_runs_its_setup_once(tmp_path):
    ran = []
    inst = make(tmp_path, ran)
    assert inst.start("moss")["state"] == "downloading"
    assert wait(inst, "moss", "installed") and ran == ["moss"]
    assert inst.item("moss")["state"] == "installed"


@pytest.mark.parametrize("bad", ["whisperx", "../moss", None, {"id": "moss"}])
def test_refused(tmp_path, bad):
    with pytest.raises(LookupError):
        make(tmp_path, []).start(bad)


def test_failure_says_why(tmp_path):
    inst = make(tmp_path, [], code=2)
    inst.start("nemotron")
    assert wait(inst, "nemotron", "failed")
    assert "CUDA 13 wheels" in inst.item("nemotron")["error"]
