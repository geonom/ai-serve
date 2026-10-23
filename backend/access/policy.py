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
  whatever else they carry.
- Host and Origin (2026-10-23, DNS rebinding and cross-site requests): requests trusted
  for where they come from (this machine, the owner's tailnet devices) must name this
  machine in Host -- an IP address, localhost or one of its names -- and, if they carry
  an Origin, come from a page of this machine (its names or its own addresses). A page
  that re-points its name at 127.0.0.1 sends Host = its own name; a page anywhere else
  sends a foreign Origin. Credential holders (public key, training token, API key) are
  not affected -- a page can't forge those. Switched on by `known_hosts` (main.py)."""
import enum
import hmac
import ipaddress
from dataclasses import dataclass
from typing import Callable, FrozenSet, Mapping, Optional, Tuple
from urllib.parse import urlsplit

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


KnownHosts = Tuple[FrozenSet[str], FrozenSet[str]]   # (this machine's names, its own IP addresses)


def _ip(value: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return None


def _host_name(host: str) -> str:
    """'my-pc:8030' -> 'my-pc', '[::1]:8030' -> '::1'."""
    host = host.strip()
    if host.startswith("["):
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def host_problem(headers: Mapping[str, str], known: KnownHosts) -> Optional[str]:
    """Why Host/Origin don't name this machine -- None if they do (or are absent)."""
    names, own_ips = {n.lower() for n in known[0]}, known[1]
    host = headers.get("host")
    if host:
        name = _host_name(host).lower()
        if _ip(name) is None and name not in names:
            return (f"unknown host name {name!r} (a page pointing its own name at this machine?) -- "
                    "add it to AI_SERVE_ALLOWED_HOSTS if it is yours")
    origin = headers.get("origin")
    if origin is not None:
        oh = (urlsplit(origin).hostname or "").lower() if origin != "null" else ""
        ip = _ip(oh) if oh else None
        if not (oh in names or (ip is not None and (ip.is_loopback or str(ip) in own_ips))):
            return f"request from a foreign page (Origin {origin!r})"
    return None


def _same_login(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


def _is_training_path(path: str) -> bool:
    return (path.startswith(_TRAINING_PREFIX) and ".." not in path.split("/")
            and "%" not in path and "\\" not in path)


class AccessPolicy:
    def __init__(self, api_key: ApiKey, guard: Optional[ExposureGuard] = None,
                 is_private: Optional[Callable[[str], bool]] = None,
                 known_hosts: Optional[Callable[[], KnownHosts]] = None):
        self.api_key = api_key
        self.guard = guard or ExposureGuard()
        self.is_private = is_private  # the network provider's overlay ranges
        self.known_hosts = known_hosts  # this machine's names and addresses; None = no Host/Origin check

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
        if self.known_hosts is not None and kind is not PeerKind.OTHER:
            problem = host_problem(h, self.known_hosts())
            if problem:
                return Decision(False, problem)
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
