"""The private overlay network behind one small interface, so Tailscale is one
provider, not a hard dependency (user decision 2026-10-16: keep Tailscale for now,
make open-source alternatives possible -- the Funnel may then be unavailable).

    is_private(ip)   address belongs to the overlay network
    owner()          identity that owns this machine (None = unknown)
    identify(ip)     identity of a private peer (None = unknown)
    hostname()       this machine's name/address in the overlay (None = unknown)
    capabilities     optional features: "serve" (share a port in the overlay),
                     "funnel" (public tunnel) -- callers must check them

Providers: tailscale (default), static (WireGuard / Nebula / NetBird / ZeroTier /
Headscale without whois: networks + address->identity map from a file; on
WireGuard an overlay address is bound to one device key), none (loopback + API key
only). Rules and reasons: tests/unit/test_network_provider.py."""
import ipaddress
import json
import os
from typing import Callable, Dict, FrozenSet, List, Optional

_TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))


class ProviderError(ValueError):
    pass


def _ip(value):
    try:
        return ipaddress.ip_address(value or "")
    except ValueError:
        return None


def in_networks(nets, value) -> bool:
    ip = _ip(value)
    return bool(ip and any(ip.version == n.version and ip in n for n in nets))


class TailscaleProvider:
    name = "tailscale"
    capabilities: FrozenSet[str] = frozenset({"serve", "funnel"})
    networks = _TAILNET

    def __init__(self, info: Optional[Callable] = None, whois: Optional[Callable[[str], Optional[str]]] = None):
        # looked up per call (not bound once) so the CLI adapter stays replaceable
        from backend.services import tailscale as ts
        self._info = info or (lambda: ts.info())
        self._whois = whois or (lambda ip: ts.whois(ip))

    def is_private(self, ip: Optional[str]) -> bool:
        return in_networks(self.networks, ip)

    def owner(self) -> Optional[str]:
        info = self._info()
        return info.owner if info else None

    def identify(self, ip: str) -> Optional[str]:
        return self._whois(ip) if self.is_private(ip) else None

    def hostname(self) -> Optional[str]:
        info = self._info()
        return getattr(info, "dns_name", None) if info else None


class StaticProvider:
    name = "static"
    capabilities: FrozenSet[str] = frozenset()

    def __init__(self, networks: List, owner: str, peers: Dict[str, str], hostname: Optional[str] = None):
        self.networks = tuple(networks)
        self._owner = owner
        self._peers = peers
        self._hostname = hostname

    @classmethod
    def from_dict(cls, d: Dict) -> "StaticProvider":
        unknown = set(d) - {"networks", "owner", "peers", "hostname", "_comment"}
        if unknown:
            raise ProviderError(f"unknown fields in the network config: {sorted(unknown)}")
        nets = []
        for raw in d.get("networks") or []:
            try:
                n = ipaddress.ip_network(raw, strict=False)
            except ValueError:
                raise ProviderError(f"not a network: {raw!r}")
            # must be a private/overlay range -- a "private network" covering the
            # internet would trust everybody
            if not (n.is_private or any(n.subnet_of(t) for t in _TAILNET if t.version == n.version)) \
                    or n.prefixlen < (8 if n.version == 4 else 32):
                raise ProviderError(f"{raw} is not a private overlay range")
            nets.append(n)
        if not nets:
            raise ProviderError("at least one private network is required")
        owner = str(d.get("owner") or "").strip()
        if not owner:
            raise ProviderError("owner (the identity that owns this machine) is required")
        peers = {}
        for ip, ident in (d.get("peers") or {}).items():
            if not in_networks(nets, ip):
                raise ProviderError(f"peer {ip} is outside the private networks")
            peers[str(_ip(ip))] = str(ident)
        return cls(nets, owner, peers, hostname=d.get("hostname") or None)

    def is_private(self, ip: Optional[str]) -> bool:
        return in_networks(self.networks, ip)

    def owner(self) -> Optional[str]:
        return self._owner

    def identify(self, ip: str) -> Optional[str]:
        a = _ip(ip)
        return self._peers.get(str(a)) if a and self.is_private(ip) else None

    def hostname(self) -> Optional[str]:
        return self._hostname


class NoneProvider:
    name = "none"
    capabilities: FrozenSet[str] = frozenset()

    def is_private(self, ip: Optional[str]) -> bool:
        return False

    def owner(self) -> Optional[str]:
        return None

    def identify(self, ip: str) -> Optional[str]:
        return None

    def hostname(self) -> Optional[str]:
        return None


def load_provider(name: str, config_path: str):
    name = (name or "tailscale").strip().lower()
    if name == "tailscale":
        return TailscaleProvider()
    if name == "none":
        return NoneProvider()
    if name == "static":
        try:
            with open(os.path.expanduser(config_path), encoding="utf-8") as f:
                return StaticProvider.from_dict(json.load(f))
        except (OSError, ValueError) as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"static network config {config_path} unreadable: {e}")
    raise ProviderError(f"unknown network provider {name!r} (tailscale, static, none)")


_ACTIVE = None


def set_active(provider) -> None:
    """main.py sets the configured provider once at startup."""
    global _ACTIVE
    _ACTIVE = provider


def active():
    return _ACTIVE if _ACTIVE is not None else NoneProvider()
