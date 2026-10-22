"""backend/procs.py: fields come from the call, not from psutil's shared `p.info`."""
import psutil

from backend import procs
from backend.services import runtime


class FakeProc:
    def __init__(self, pid, gone=False):
        self.pid, self.gone = pid, gone
        self.info = {"pid": pid}          # what another thread's process_iter left behind

    def as_dict(self, attrs, ad_value=None):
        if self.gone:
            raise psutil.NoSuchProcess(self.pid)
        return {a: {"pid": self.pid, "ppid": 1, "uids": None, "cmdline": ["x"], "create_time": 5.0}.get(a, ad_value)
                for a in attrs}


def test_fields_read_per_call_and_vanished_skipped(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter([FakeProc(1), FakeProc(2, gone=True)]))
    got = list(procs.processes(["pid", "create_time"]))
    assert [(p.pid, info) for p, info in got] == [(1, {"pid": 1, "create_time": 5.0})]


def test_snapshot_survives_foreign_p_info(monkeypatch):
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter([FakeProc(1)]))
    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [])
    assert runtime.snapshot() == []   # uids None -> not ours; must not raise KeyError
