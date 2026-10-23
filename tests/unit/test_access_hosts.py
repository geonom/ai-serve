"""This machine's names and addresses for the Host/Origin check (backend/access/hosts.py)."""
from backend.access import hosts


def test_names_addresses_and_extra_names(monkeypatch):
    monkeypatch.setattr(hosts.socket, "gethostname", lambda: "Box-One")
    monkeypatch.setattr(hosts, "_network_hostname", lambda: "my-pc.example-tailnet.ts.net.")
    monkeypatch.setattr(hosts, "_addresses", lambda: {"127.0.0.1", "100.101.102.103"})
    monkeypatch.setenv("AI_SERVE_ALLOWED_HOSTS", "ai.example.org, Other-Name")
    hosts._cache.clear()
    names, ips = hosts.known_hosts()
    assert {"localhost", "box-one", "box-one.local", "my-pc.example-tailnet.ts.net", "ai.example.org", "other-name"} <= names
    assert "100.101.102.103" in ips


def test_a_failing_lookup_still_gives_localhost(monkeypatch):
    def boom():
        raise OSError("no network")
    monkeypatch.setattr(hosts, "_network_hostname", boom)
    monkeypatch.setattr(hosts, "_addresses", boom)
    monkeypatch.delenv("AI_SERVE_ALLOWED_HOSTS", raising=False)
    hosts._cache.clear()
    names, ips = hosts.known_hosts()
    assert "localhost" in names and "127.0.0.1" in ips
