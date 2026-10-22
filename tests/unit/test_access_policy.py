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
