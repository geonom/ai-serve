"""sdk/identity/python/llama_serve_identity.py -- the reference "who is this?" for apps
(docs/identity.md section 2). Security-relevant: an app trusts exactly this.

Rules under test: a valid ES256 token from the door names the person; no header from
loopback = the owner; anything else -- no header from elsewhere, or any header that
doesn't verify (also from loopback) -- is nobody."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

_spec = importlib.util.spec_from_file_location(
    "llama_serve_identity", Path(__file__).parents[2] / "sdk/identity/python/llama_serve_identity.py")
ident = importlib.util.module_from_spec(_spec)
sys.modules["llama_serve_identity"] = ident   # dataclasses look their module up there
_spec.loader.exec_module(ident)

KEY = ec.generate_private_key(ec.SECP256R1())
OTHER = ec.generate_private_key(ec.SECP256R1())
JWK = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(KEY.public_key()))
JWK.update(kid="k1", alg="ES256", use="sig")
JWKS = {"keys": [JWK]}
APP = "pv-calc"
ISS = "https://apps.example"


def token(key=KEY, **over):
    now = int(time.time())
    claims = {"sub": "u-anna", "name": "Anna", "groups": ["familie"], "aud": APP, "iss": ISS,
              "iat": now, "exp": now + 300}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": "k1"})


def who(headers, peer="100.64.0.5", **kw):
    kw.setdefault("jwks", JWKS)
    kw.setdefault("issuer", ISS)
    kw.setdefault("owner_sub", "u-owner")
    return ident.identify(headers, peer, app_id=APP, **kw)


def test_valid_token_names_the_person():
    p = who({"X-Llama-Serve-Identity": token()})
    assert (p.sub, p.name, p.groups, p.via) == ("u-anna", "Anna", ("familie",), "door")
    assert not p.is_owner


def test_header_name_case_insensitive():
    assert who({"x-llama-serve-identity": token()}).sub == "u-anna"


def test_owner_on_loopback_without_header():
    for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        p = who({}, peer=peer)
        assert (p.sub, p.via, p.is_owner) == ("u-owner", "local", True)


def test_owner_logging_in_through_the_door_is_the_owner():
    assert who({"X-Llama-Serve-Identity": token(sub="u-owner")}).is_owner


def test_owner_sub_from_environment(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVE_OWNER_SUB", "u-env")
    assert ident.identify({}, "127.0.0.1", app_id=APP, jwks=JWKS).sub == "u-env"
    monkeypatch.delenv("LLAMA_SERVE_OWNER_SUB")
    assert ident.identify({}, "127.0.0.1", app_id=APP, jwks=JWKS).sub == "owner"


def test_no_header_from_elsewhere_is_nobody():
    with pytest.raises(ident.Unauthenticated):
        who({}, peer="100.64.0.5")


@pytest.mark.parametrize("bad", [
    lambda: token(key=OTHER),                       # signed by someone else
    lambda: token(aud="storytime"),                 # meant for another app
    lambda: token(exp=int(time.time()) - 120),      # expired (beyond the 30 s clock-skew leeway)
    lambda: token(iss="https://evil.example"),      # wrong issuer
    lambda: token(sub=None),                        # no person
    lambda: token(sub=""),
    lambda: token() + "x",                          # tampered
    lambda: jwt.encode({"sub": "u-anna", "aud": APP, "iss": ISS, "exp": int(time.time()) + 60},
                       "secret", algorithm="HS256"),                     # symmetric alg
    lambda: jwt.encode({"sub": "u-anna", "aud": APP, "iss": ISS, "exp": int(time.time()) + 60},
                       None, algorithm="none"),                          # unsigned
    lambda: "not-a-token",
])
def test_bad_tokens_are_nobody_even_from_loopback(bad):
    for peer in ("100.64.0.5", "127.0.0.1"):
        with pytest.raises(ident.Unauthenticated):
            who({"X-Llama-Serve-Identity": bad()}, peer=peer)


def test_header_without_keys_is_nobody():
    with pytest.raises(ident.Unauthenticated):
        ident.identify({"X-Llama-Serve-Identity": token()}, "100.64.0.5", app_id=APP, jwks=None,
                       jwks_url=None)


def test_keys_from_url(monkeypatch):
    seen = []

    def fetch(url):
        seen.append(url)
        return JWKS
    monkeypatch.setattr(ident, "_fetch_jwks", fetch)
    ident._JWKS_CACHE.clear()
    p = ident.identify({"X-Llama-Serve-Identity": token()}, "100.64.0.5", app_id=APP,
                       jwks_url="https://door.local/jwks.json", issuer=ISS)
    ident.identify({"X-Llama-Serve-Identity": token()}, "100.64.0.5", app_id=APP,
                   jwks_url="https://door.local/jwks.json", issuer=ISS)
    assert p.sub == "u-anna" and seen == ["https://door.local/jwks.json"]   # cached


class TestThroughTailscaleServe:
    """Tailscale Serve and the dashboard's owner gate reach the app from 127.0.0.1 -- such a
    request is not "someone at this machine". Serve sets Tailscale-User-Login for tailnet
    peers (and strips client-supplied copies): it is the owner only if it is the owner's
    login (LLAMA_SERVE_OWNER_TAILSCALE_LOGIN); Funnel requests are never the owner."""

    def test_owner_through_serve(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", "owner@github")
        p = who({"Tailscale-User-Login": "owner@github"}, peer="127.0.0.1")
        assert p.is_owner and p.sub == "u-owner"

    def test_other_tailnet_user_is_not_the_owner(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", "owner@github")
        with pytest.raises(ident.Unauthenticated):
            who({"Tailscale-User-Login": "friend@example.com"}, peer="127.0.0.1")

    def test_unknown_owner_login_means_nobody(self, monkeypatch):
        monkeypatch.delenv("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", raising=False)
        with pytest.raises(ident.Unauthenticated):
            who({"Tailscale-User-Login": "owner@github"}, peer="127.0.0.1")

    def test_funnel_is_never_the_owner(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", "owner@github")
        with pytest.raises(ident.Unauthenticated):
            who({"Tailscale-Funnel-Request": "?1"}, peer="127.0.0.1")


class TestPomeriumDoor:
    """The door (Pomerium) sends its signed assertion as X-Pomerium-Jwt-Assertion, with the
    app's public host as audience (LLAMA_SERVE_IDENTITY_AUDIENCE). Accepting that header name
    is safe: only a signature by the door's key counts."""

    def test_pomerium_header_accepted(self):
        assert who({"X-Pomerium-Jwt-Assertion": token()}).sub == "u-anna"

    def test_audience_from_environment(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVE_IDENTITY_AUDIENCE", "sonne.apps.example")
        assert who({"X-Pomerium-Jwt-Assertion": token(aud="sonne.apps.example")}).sub == "u-anna"
        with pytest.raises(ident.Unauthenticated):
            who({"X-Pomerium-Jwt-Assertion": token(aud=APP)})   # now only the public host counts

    def test_forged_pomerium_header_is_nobody_even_from_loopback(self):
        with pytest.raises(ident.Unauthenticated):
            who({"X-Pomerium-Jwt-Assertion": token(key=OTHER)}, peer="127.0.0.1")


class TestRebindingAndForeignPages:
    """2026-10-23 (DNS rebinding / cross-site requests): "no identity header from this machine = the owner" made every
    web page the owner – one that points its own name at 127.0.0.1 (Host = its name), or one posting to the app
    (foreign Origin). Without a door identity, Host must name this machine (an IP, localhost, LLAMA_SERVE_LOCAL_HOSTS
    – ai-serve passes this machine's names and addresses – or the app's public name) and an Origin must be one of
    those too. Door identities are signed and unaffected."""

    def test_this_machine_by_ip_or_localhost(self):
        for host in ("127.0.0.1:8037", "localhost:8037", "[::1]:8037"):
            assert who({"Host": host}, peer="127.0.0.1").is_owner

    def test_a_rebinding_host_is_nobody(self):
        with pytest.raises(ident.Unauthenticated, match="host"):
            who({"Host": "evil.example:8037"}, peer="127.0.0.1")

    def test_a_foreign_page_is_nobody(self):
        for origin in ("https://evil.example", "null", "http://192.168.1.50"):
            with pytest.raises(ident.Unauthenticated, match="page"):
                who({"Host": "localhost:8037", "Origin": origin}, peer="127.0.0.1")

    def test_names_from_ai_serve_and_the_door(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVE_LOCAL_HOSTS", "my-pc,my-pc.example-tailnet.ts.net,100.101.102.103")
        monkeypatch.setenv("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", "owner@github")
        p = who({"Host": "my-pc.example-tailnet.ts.net:18037", "Origin": "https://my-pc.example-tailnet.ts.net:18037",
                 "Tailscale-User-Login": "owner@github"}, peer="127.0.0.1")
        assert p.via == "tailscale"
        assert who({"Host": "localhost:8037", "Origin": "http://100.101.102.103:8037"}, peer="127.0.0.1").is_owner
        monkeypatch.setenv("LLAMA_SERVE_IDENTITY_AUDIENCE", "spielwiese.apps.example.org")
        assert who({"Host": "localhost", "Origin": "https://spielwiese.apps.example.org"}, peer="127.0.0.1").is_owner

    def test_door_identities_are_not_affected(self):
        p = who({"X-Llama-Serve-Identity": token(), "Host": "spielwiese.apps.example.org",
                 "Origin": "https://spielwiese.apps.example.org"}, peer="127.0.0.1")
        assert p.via == "door"
