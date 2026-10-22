"""Who may use the dashboard and the local APIs. Pure -- no I/O; the adapter is
backend/access/middleware.py (which looks up `tailscale whois` and the owner).

Why (architecture review 2026-10-16): the backend listens on 0.0.0.0:8030 and had no
key, so every device on the LAN or tailnet could load/stop models, start/stop
services and pause/cancel/delete training jobs.

Vocabulary:
- Peer: the address the request came from. uvicorn rewrites it from
  X-Forwarded-For only for loopback peers (forwarded_allow_ips="127.0.0.1", set
  explicitly in main.py), so a rewritten peer was loopback -- the most trusted class
  -- and the rewrite can only make it look *less* trusted, never more.
- Network: the private overlay network comes from backend/network (Tailscale by
  default; static/WireGuard/Nebula/... or none). "tailnet" below means that network.
- Owner: the Tailscale login owning this machine. Its devices are trusted like this
  machine; other users' shared-in devices are not.
- Bypass paths: /public/ (internet API with its own key + ExposureGuard) and
  /api/training/ (own fail-closed token). /api/training-monitor/ is NOT a bypass.
- ApiKey: LLAMA_SERVE_API_KEY, an alternative credential for clients that are neither
  this machine nor the owner's tailnet devices (e.g. the LAN). Shorter than 32 chars
  = disabled. Never an extra requirement for loopback or the owner's devices.
- Tunnel-marked requests (Funnel / Cloudflare headers) get nothing but /public/,
  whatever else they carry."""
import enum
import hmac
import ipaddress
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from backend.public_api.domain import ExposureGuard

_TRAINING_PREFIX = "/api/training/"


class PeerKind(enum.Enum):
    LOOPBACK = "loopback"
    TAILNET = "tailnet"
    OTHER = "other"


def _tailscale_ranges(peer: str) -> bool:
    from backend.network import TailscaleProvider, in_networks
    return in_networks(TailscaleProvider.networks, peer)


def classify_peer(peer: Optional[str], is_private: Callable[[str], bool] = None) -> PeerKind:
    """TAILNET = the private overlay network of the configured provider
    (backend/network) -- Tailscale's ranges unless told otherwise."""
    try:
        ip = ipaddress.ip_address(peer or "")
    except ValueError:
        return PeerKind.OTHER
    if ip.is_loopback:
        return PeerKind.LOOPBACK
    if (is_private or _tailscale_ranges)(str(ip)):
        return PeerKind.TAILNET
    return PeerKind.OTHER


class ApiKey:
    MIN_LENGTH = 32
    __slots__ = ("_secret",)

    def __init__(self, raw: Optional[str]):
        raw = (raw or "").strip()
        self._secret = raw.encode() if len(raw) >= self.MIN_LENGTH else None

    @property
    def enabled(self) -> bool:
        return self._secret is not None

    def matches(self, authorization: Optional[str]) -> bool:
        if self._secret is None or not (authorization or "").startswith("Bearer "):
            return False
        return hmac.compare_digest(authorization[7:].encode(), self._secret)

    def __repr__(self) -> str:
        return f"ApiKey({'***' if self.enabled else 'disabled'})"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _same_login(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


def _is_training_path(path: str) -> bool:
    return (path.startswith(_TRAINING_PREFIX) and ".." not in path.split("/")
            and "%" not in path and "\\" not in path)


class AccessPolicy:
    def __init__(self, api_key: ApiKey, guard: Optional[ExposureGuard] = None,
                 is_private: Optional[Callable[[str], bool]] = None):
        self.api_key = api_key
        self.guard = guard or ExposureGuard()
        self.is_private = is_private  # the network provider's overlay ranges

    def needs_whois(self, peer: Optional[str]) -> bool:
        return classify_peer(peer, self.is_private) is PeerKind.TAILNET

    def decide(self, peer: Optional[str], path: str, headers: Mapping[str, str],
               owner: Optional[str], peer_login: Optional[str]) -> Decision:
        h = {k.lower(): v for k, v in headers.items()}
        if ExposureGuard.is_public_path(path):
            return Decision(True, "public API (own key)")
        if self.guard.is_external(h):
            return Decision(False, "internet requests only reach /public/")
        if _is_training_path(path):
            return Decision(True, "training API (own token)")
        if self.api_key.matches(h.get("authorization")):
            return Decision(True, "API key")
        kind = classify_peer(peer, self.is_private)
        if kind is PeerKind.LOOPBACK:
            serve_login = h.get("tailscale-user-login")
            if serve_login is None:
                return Decision(True, "this machine")
            if _same_login(serve_login, owner):
                return Decision(True, "owner via Tailscale Serve")
            return Decision(False, f"Tailscale user {serve_login} is not the owner")
        if kind is PeerKind.TAILNET:
            if not owner:
                return Decision(False, "Tailscale owner unknown -- tailnet access refused")
            if _same_login(peer_login, owner):
                return Decision(True, "owner's tailnet device")
            return Decision(False, f"tailnet device of {peer_login or 'unknown user'} is not the owner's")
        return Decision(False, "only this machine and the owner's Tailscale devices; "
                               "other networks need the API key (LLAMA_SERVE_API_KEY)")
