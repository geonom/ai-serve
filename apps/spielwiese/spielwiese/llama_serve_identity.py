"""Who is using this app? -- reference implementation of docs/identity.md (v1) in
ai-serve-ui. Copy this single file into an app; it needs `PyJWT[crypto]`.

    from llama_serve_identity import identify, Unauthenticated

    person = identify(request.headers, request.client.host, app_id="storytime")
    person.sub        # the key for everything this person owns
    person.is_owner   # the owner of this machine

Rules (security-relevant, tests in ai-serve-ui tests/unit/test_identity_sdk.py):
- `X-Llama-Serve-Identity` (or Pomerium's `X-Pomerium-Jwt-Assertion`) present -> must be an
  ES256 JWT that verifies against the door's keys (`LLAMA_SERVE_IDENTITY_JWKS` URL, or keys
  passed in), with `aud` = `LLAMA_SERVE_IDENTITY_AUDIENCE` (the app's public host) or else
  this app's id,
  unexpired, from the expected issuer (`LLAMA_SERVE_IDENTITY_ISSUER`, if set), with a `sub`.
  Anything that doesn't verify is nobody -- also on loopback: a bad header never falls back
  to the owner.
- no header from loopback -> the owner (`LLAMA_SERVE_OWNER_SUB`, set by the dashboard;
  "owner" until the owner's login is linked) -- unless Tailscale Serve brought it: with
  `Tailscale-User-Login` it is the owner only if that is `LLAMA_SERVE_OWNER_TAILSCALE_LOGIN`;
  with `Tailscale-Funnel-Request` it is nobody.
- no header from anywhere else -> nobody (`Unauthenticated`, answer 401).
"""
from __future__ import annotations

import ipaddress
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import jwt

__version__ = "1"
HEADER = "X-Llama-Serve-Identity"
# Pomerium (the door) sends its signed assertion under its own name; only the signature counts.
DOOR_HEADERS = (HEADER, "X-Pomerium-Jwt-Assertion")
_LEEWAY_S = 30
_JWKS_TTL_S = 300
_JWKS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_lock = threading.Lock()


class Unauthenticated(Exception):
    """Nobody we know -- answer 401."""


@dataclass(frozen=True)
class Person:
    sub: str
    name: str
    email: Optional[str]
    picture: Optional[str]
    groups: Tuple[str, ...]
    via: str          # "door" (logged in) | "local" (at this machine) | "tailscale" (owner via Serve)
    is_owner: bool


def _loopback(peer: Optional[str]) -> bool:
    try:
        ip = ipaddress.ip_address((peer or "").strip("[]"))
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _fetch_jwks(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 -- URL from the dashboard's env
        return json.load(r)


def _keys(jwks: Optional[Mapping[str, Any]], jwks_url: Optional[str]) -> Mapping[str, Any]:
    if jwks is not None:
        return jwks
    if not jwks_url:
        raise Unauthenticated("no identity keys configured (LLAMA_SERVE_IDENTITY_JWKS)")
    with _lock:
        hit = _JWKS_CACHE.get(jwks_url)
        if hit and time.monotonic() - hit[0] < _JWKS_TTL_S:
            return hit[1]
    try:
        keys = _fetch_jwks(jwks_url)
    except Exception as e:  # unreachable door -> nobody, never "open"
        raise Unauthenticated(f"identity keys not reachable: {e}") from e
    with _lock:
        _JWKS_CACHE[jwks_url] = (time.monotonic(), keys)
    return keys


def _get(headers: Mapping[str, str], name: str) -> Optional[str]:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _header(headers: Mapping[str, str]) -> Optional[str]:
    for name in DOOR_HEADERS:
        value = _get(headers, name)
        if value is not None:
            return value
    return None


def identify(headers: Mapping[str, str], peer: Optional[str], *, app_id: str,
             jwks: Optional[Mapping[str, Any]] = None, jwks_url: Optional[str] = None,
             issuer: Optional[str] = None, owner_sub: Optional[str] = None) -> Person:
    owner = owner_sub or os.environ.get("LLAMA_SERVE_OWNER_SUB") or "owner"
    raw = _header(headers)
    if raw is None:
        if not _loopback(peer):
            raise Unauthenticated("no identity")
        # Tailscale Serve (and the dashboard's owner gate) also connect from loopback: such a
        # request is someone on the tailnet, not someone at this machine. Serve sets these
        # headers itself and strips client-supplied copies.
        if _get(headers, "Tailscale-Funnel-Request") is not None:
            raise Unauthenticated("from the internet without identity")
        ts_login = _get(headers, "Tailscale-User-Login")
        if ts_login is not None:
            owner_login = os.environ.get("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN")
            if not owner_login or ts_login != owner_login:
                raise Unauthenticated("tailnet user who is not the owner")
            return Person(sub=owner, name="", email=None, picture=None, groups=(), via="tailscale", is_owner=True)
        return Person(sub=owner, name="", email=None, picture=None, groups=(), via="local", is_owner=True)

    keyset = _keys(jwks, jwks_url or os.environ.get("LLAMA_SERVE_IDENTITY_JWKS"))
    issuer = issuer or os.environ.get("LLAMA_SERVE_IDENTITY_ISSUER") or None
    # the door puts the app's public host into `aud` -- set via the environment
    audience = os.environ.get("LLAMA_SERVE_IDENTITY_AUDIENCE") or app_id
    try:
        kid = jwt.get_unverified_header(raw).get("kid")
        candidates = [k for k in keyset.get("keys", []) if kid is None or k.get("kid") == kid]
        if not candidates:
            raise Unauthenticated("unknown signing key")
        last: Optional[Exception] = None
        for jwk in candidates:
            try:
                key = jwt.PyJWK(jwk, algorithm="ES256").key
                claims = jwt.decode(raw, key, algorithms=["ES256"], audience=audience, issuer=issuer,
                                    leeway=_LEEWAY_S, options={"require": ["exp", "aud", "sub"]})
                break
            except jwt.PyJWTError as e:
                last = e
        else:
            raise Unauthenticated(f"identity does not verify: {last}")
    except Unauthenticated:
        raise
    except Exception as e:  # malformed token, bad key material, ...
        raise Unauthenticated(f"identity does not verify: {e}") from e

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise Unauthenticated("identity without sub")
    groups = claims.get("groups") or []
    return Person(sub=sub, name=str(claims.get("name") or ""), email=claims.get("email"),
                  picture=claims.get("picture"),
                  groups=tuple(str(g) for g in groups) if isinstance(groups, list) else (),
                  via="door", is_owner=sub == owner)
