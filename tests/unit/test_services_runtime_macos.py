"""macOS: psutil.net_connections() needs root there and raises AccessDenied -- the snapshot must still see which of
this user's processes listen on which port (per-process lookups work for one's own processes). Found on a Mac:
the Playground ran but Apps & Services showed "starting" for an hour."""
import socket
import subprocess
import sys
import time

import psutil

from backend.services import runtime


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_listening_ports_without_permission_for_all_connections(monkeypatch):
    port = _free_port()
    # a plain listening socket -- http.server would first spend ~35 s in getfqdn() on a Mac CI runner
    code = ("import socket,sys,time; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(); "
            "time.sleep(120)")
    proc = subprocess.Popen([sys.executable, "-c", code, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)

        def denied(*a, **k):
            raise psutil.AccessDenied()
        monkeypatch.setattr(runtime.psutil, "net_connections", denied)
        mine = [p for p in runtime.snapshot() if p.pid == proc.pid]
        assert mine and mine[0].listening_ports == frozenset({port})
        assert ("127.0.0.1", port) in mine[0].listen_addresses
    finally:
        proc.terminate()
        proc.wait(5)


def test_parse_lsof():
    text = "p100\nn127.0.0.1:8037\nn[::1]:8037\np200\nn*:8030\nnlocalhost:x\n"
    assert runtime.parse_lsof(text) == [(100, "127.0.0.1", 8037), (100, "::1", 8037), (200, "0.0.0.0", 8030)]


def test_without_lsof_each_process_is_asked(monkeypatch):
    monkeypatch.setattr(runtime, "_lsof_listening", lambda: None)
    test_listening_ports_without_permission_for_all_connections(monkeypatch)
