"""Enrolling a node (backend/cluster/enrollment.py) -- instead of copying tokens and
config by hand (user, 2026-10-16: the node agent should install what it needs itself).

Flow: the dashboard creates a one-time code -> on the new device the agent runs
`--enroll <coordinator> --code <code>`, generates its own token and calls
POST /api/cluster/enroll -> the coordinator stores the node.

Rules under test:
- A code is random, single-use and expires (15 min); unknown/used/expired codes are
  refused the same way (no oracle); comparison is constant-time.
- The agent URL must point at the address the enrollment came from, and that
  address must be in the private overlay network (loopback only for a local
  agent) -- nobody can register a third party's address as a "node".
- Node names are a strict charset; the agent token must be >= 32 chars; the URL is
  plain http://<ip>:<port> (no hostnames, credentials, paths).
- Re-enrolling an existing name needs a fresh code and replaces the old record.
- The registry file holds the tokens: written 0600; listings never include tokens."""
import json
import os
import stat

import pytest

from backend.cluster.enrollment import (
    EnrollError, EnrollmentBook, NodeRegistry, validate_enrollment,
)

TOKEN = "n" * 40


def tailnet(ip):
    return ip.startswith("100.")


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestCodes:
    def test_single_use(self):
        book = EnrollmentBook(clock=Clock())
        code = book.create().code
        assert len(code) >= 20
        assert book.redeem(code) and not book.redeem(code)

    def test_expiry(self):
        clock = Clock()
        book = EnrollmentBook(clock=clock, ttl_s=900)
        code = book.create().code
        clock.t += 901
        assert not book.redeem(code)

    def test_unknown(self):
        book = EnrollmentBook(clock=Clock())
        book.create()
        assert not book.redeem("guess") and not book.redeem("")

    def test_codes_differ(self):
        book = EnrollmentBook(clock=Clock())
        assert book.create().code != book.create().code


class TestValidate:
    def ok(self, **over):
        body = {"node_name": "macbook", "agent_url": "http://100.64.7.11:8040", "token": TOKEN}
        body.update(over)
        return body

    def test_valid(self):
        rec = validate_enrollment(self.ok(), peer="100.64.7.11", is_private=tailnet, now=5)
        assert (rec.id, rec.url, rec.token, rec.enrolled_at) == ("macbook", "http://100.64.7.11:8040", TOKEN, 5)

    def test_url_must_match_peer(self):
        with pytest.raises(EnrollError):
            validate_enrollment(self.ok(agent_url="http://100.90.1.2:8040"), peer="100.64.7.11",
                                is_private=tailnet, now=0)

    def test_peer_must_be_private_or_loopback(self):
        with pytest.raises(EnrollError):
            validate_enrollment(self.ok(agent_url="http://192.168.1.9:8040"), peer="192.168.1.9",
                                is_private=tailnet, now=0)
        assert validate_enrollment(self.ok(agent_url="http://127.0.0.1:8040"), peer="127.0.0.1",
                                   is_private=tailnet, now=0)

    @pytest.mark.parametrize("url", [
        "https://100.64.7.11:8040", "http://user:pw@100.64.7.11:8040", "http://100.64.7.11:8040/x",
        "http://macbook:8040", "http://100.64.7.11", "http://100.64.7.11:80", "http://100.64.7.11:99999",
        "http://100.64.7.11:8040?x=1",
    ])
    def test_bad_urls(self, url):
        with pytest.raises(EnrollError):
            validate_enrollment(self.ok(agent_url=url), peer="100.64.7.11", is_private=tailnet, now=0)

    @pytest.mark.parametrize("over", [{"node_name": "../x"}, {"node_name": ""}, {"node_name": "a" * 64},
                                      {"token": "short"}, {"token": 5}, {"extra": 1}])
    def test_bad_fields(self, over):
        with pytest.raises(EnrollError):
            validate_enrollment(self.ok(**over), peer="100.64.7.11", is_private=tailnet, now=0)


class TestRegistry:
    def test_file_is_private_and_listing_hides_tokens(self, tmp_path):
        f = tmp_path / "nodes.json"
        reg = NodeRegistry(str(f))
        rec = validate_enrollment({"node_name": "mac", "agent_url": "http://100.64.7.11:8040", "token": TOKEN},
                                  peer="100.64.7.11", is_private=tailnet, now=1)
        reg.add(rec)
        assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
        assert TOKEN not in json.dumps(reg.listing())
        assert NodeRegistry(str(f)).get("mac").token == TOKEN

    def test_replace_and_remove(self, tmp_path):
        reg = NodeRegistry(str(tmp_path / "nodes.json"))
        a = validate_enrollment({"node_name": "mac", "agent_url": "http://100.64.7.11:8040", "token": TOKEN},
                                peer="100.64.7.11", is_private=tailnet, now=1)
        b = validate_enrollment({"node_name": "mac", "agent_url": "http://100.64.7.11:8041", "token": "m" * 40},
                                peer="100.64.7.11", is_private=tailnet, now=2)
        reg.add(a)
        reg.add(b)
        assert reg.get("mac").url.endswith(":8041") and len(reg.listing()) == 1
        assert reg.remove("mac") and not reg.remove("mac") and reg.listing() == []
