"""Who may use the dashboard and the local APIs (backend/access/policy.py).

Before (architecture review 2026-10-16): the backend listened on 0.0.0.0:8030 with no
key -- every device on the LAN or tailnet could load/stop models, start/stop
services and pause/cancel/DELETE training jobs.

Rules under test:
- /public/ is the internet-facing API with its own key and guard -> passes here.
- /api/training/ has its own fail-closed token -> passes here (the token-less
  /api/training-monitor/ does not).
- Tunnel-marked requests (Tailscale Funnel, Cloudflare) never get anything else.
- This machine (loopback) is trusted -- unless the request came in through
  Tailscale Serve (identity header present), then only the owner.
- Tailnet peers: only devices of the machine's Tailscale owner (login via
  `tailscale whois`), not other users' shared-in devices.
- Everything else (LAN, ...) only with LLAMA_SERVE_API_KEY (Bearer, constant-time,
  >= 32 chars, else disabled). The key is an alternative credential, never an extra
  requirement for the owner's own devices."""
import pytest

from backend.access.policy import AccessPolicy, ApiKey, PeerKind, classify_peer

OWNER = "owner@github"
KEY = "k" * 40


def policy(key=KEY):
    return AccessPolicy(api_key=ApiKey(key))


def decide(peer, path="/api/status", headers=None, login=None, owner=OWNER, key=KEY):
    return policy(key).decide(peer=peer, path=path, headers=headers or {}, owner=owner, peer_login=login)


class TestClassify:
    @pytest.mark.parametrize("peer,kind", [
        ("127.0.0.1", PeerKind.LOOPBACK), ("::1", PeerKind.LOOPBACK),
        ("100.64.7.11", PeerKind.TAILNET), ("fd7a:115c:a1e0::7536:4a31", PeerKind.TAILNET),
        ("192.168.1.40", PeerKind.OTHER), ("85.13.165.151", PeerKind.OTHER),
        ("testclient", PeerKind.OTHER), (None, PeerKind.OTHER), ("", PeerKind.OTHER),
    ])
    def test_kinds(self, peer, kind):
        assert classify_peer(peer) == kind


class TestDecide:
    def test_loopback_allowed(self):
        assert decide("127.0.0.1").allowed

    def test_owner_tailnet_device_allowed(self):
        assert decide("100.64.7.11", login=OWNER).allowed
        assert decide("100.64.7.11", login="Owner@GitHub").allowed

    def test_other_tailnet_user_refused(self):
        d = decide("100.64.7.11", login="someone@example.com")
        assert not d.allowed and "owner" in d.reason

    def test_tailnet_without_identity_refused(self):
        assert not decide("100.64.7.11", login=None).allowed

    def test_owner_unknown_refuses_tailnet(self):
        assert not decide("100.64.7.11", login=OWNER, owner=None).allowed
        assert decide("127.0.0.1", owner=None).allowed

    def test_lan_refused_without_key(self):
        d = decide("192.168.1.40")
        assert not d.allowed and "key" in d.reason.lower()

    def test_lan_with_key(self):
        assert decide("192.168.1.40", headers={"authorization": f"Bearer {KEY}"}).allowed

    @pytest.mark.parametrize("auth", [f"Bearer {KEY[:-1]}", "Bearer ", KEY, f"Basic {KEY}", f"Bearer {KEY}x"])
    def test_wrong_key(self, auth):
        assert not decide("192.168.1.40", headers={"authorization": auth}).allowed

    def test_short_key_disables_key_access(self):
        assert not decide("192.168.1.40", headers={"authorization": "Bearer short"}, key="short").allowed
        assert not decide("192.168.1.40", headers={"authorization": "Bearer "}, key="").allowed

    @pytest.mark.parametrize("path", ["/", "/app.js", "/api/services", "/v1/chat/completions",
                                      "/api/training-monitor/jobs", "/docs", "/openapi.json"])
    def test_everything_but_public_is_guarded(self, path):
        assert not decide("192.168.1.40", path=path).allowed

    def test_public_passes(self):
        assert decide("85.13.165.151", path="/public/v1/models").allowed

    def test_training_has_own_token(self):
        assert decide("192.168.1.40", path="/api/training/jobs").allowed

    @pytest.mark.parametrize("h", [{"tailscale-funnel-request": "?1"}, {"cf-connecting-ip": "1.2.3.4"}])
    def test_tunnel_never_gets_more_than_public(self, h):
        assert not decide("127.0.0.1", headers=h).allowed
        assert not decide("127.0.0.1", path="/api/training/jobs", headers=h).allowed
        assert not decide("127.0.0.1", headers={**h, "authorization": f"Bearer {KEY}"}).allowed
        assert decide("127.0.0.1", path="/public/v1/models", headers=h).allowed

    def test_via_tailscale_serve_only_owner(self):
        # Serve proxies from loopback and sets the peer's identity (stripping
        # client-supplied copies)
        assert decide("127.0.0.1", headers={"tailscale-user-login": OWNER}).allowed
        assert not decide("127.0.0.1", headers={"tailscale-user-login": "x@y"}).allowed

    @pytest.mark.parametrize("path", ["/public", "/publicx", "/public/../api/status", "/api/trainingx/jobs"])
    def test_prefix_tricks(self, path):
        assert not decide("192.168.1.40", path=path).allowed


class TestApiKey:
    def test_matches(self):
        assert ApiKey(KEY).matches(f"Bearer {KEY}")
        assert not ApiKey(KEY).matches(None)

    def test_disabled(self):
        assert not ApiKey("").enabled and not ApiKey("x" * 31).enabled and ApiKey(KEY).enabled


class TestDnsRebindingAndCrossSite:
    """Found 2026-10-23 while building V-Noveller's MCP endpoint: this machine and the owner's devices need no
    credential, so a web page could act as them –
    - DNS rebinding: evil.example re-points its name at 127.0.0.1 (or this machine's tailnet IP). The browser treats
      the requests as same-origin: GETs then carry no Origin, only Host = evil.example – it could read logs and
      lists, and change things.
    - Cross-site requests: any page can POST a form to http://localhost:8030/api/… – Host is fine, Origin is not.
    Rules: requests trusted for where they come from (loopback, the owner's tailnet devices) must name this machine
    in Host (an IP address, localhost, this machine's names) and, if they carry an Origin, come from this machine
    (its names or its own addresses). Credential holders (/public/, training token, API key) are not affected: a page
    can't forge those."""
    NAMES = frozenset({"localhost", "my-pc", "my-pc.example-tailnet.ts.net"})
    OWN_IPS = frozenset({"127.0.0.1", "::1", "100.101.102.103", "192.168.1.40"})

    def hosted(self, peer, headers, path="/api/status", login=OWNER):
        p = AccessPolicy(api_key=ApiKey(KEY), known_hosts=lambda: (self.NAMES, self.OWN_IPS))
        return p.decide(peer=peer, path=path, headers=headers, owner=OWNER, peer_login=login)

    @pytest.mark.parametrize("host", ["localhost:8030", "127.0.0.1:8030", "[::1]:8030", "my-pc:8030",
                                      "my-pc.example-tailnet.ts.net".upper(), "100.101.102.103:8030", "10.0.0.7:8030"])
    def test_this_machines_names_and_any_ip_are_fine_as_host(self, host):
        assert self.hosted("127.0.0.1", {"host": host}).allowed

    def test_a_rebinding_host_is_refused_also_for_the_owners_tailnet_device(self):
        for peer in ("127.0.0.1", "100.64.0.9"):
            d = self.hosted(peer, {"host": "evil.example:8030"})
            assert not d.allowed and "host" in d.reason.lower()

    @pytest.mark.parametrize("origin", ["https://evil.example", "http://192.168.1.50:8000", "null",
                                        "http://localhost.evil.example"])
    def test_a_foreign_origin_is_refused(self, origin):
        d = self.hosted("127.0.0.1", {"host": "localhost:8030", "origin": origin})
        assert not d.allowed and "origin" in d.reason.lower()

    @pytest.mark.parametrize("origin", ["http://localhost:8030", "http://127.0.0.1:5173", "http://100.101.102.103:8030",
                                        "https://my-pc.example-tailnet.ts.net:18030", "http://[::1]:8030"])
    def test_this_machines_own_pages_are_fine_as_origin(self, origin):
        assert self.hosted("127.0.0.1", {"host": "localhost:8030", "origin": origin}).allowed

    def test_credentials_are_not_affected(self):
        evil = {"host": "evil.example", "origin": "https://evil.example"}
        assert self.hosted("8.8.8.8", {**evil, "authorization": f"Bearer {KEY}"}).allowed
        assert self.hosted("127.0.0.1", evil, path="/public/v1/models").allowed

    def test_without_known_hosts_nothing_changes(self):
        """The check is switched on by passing known_hosts (main.py does); other uses of the policy keep working."""
        assert decide("127.0.0.1", headers={"host": "evil.example"}).allowed
