"""Exposure of services via Tailscale -- pure rules (no CLI calls, no sockets).

Levels (ubiquitous language):
- local    nothing beyond this machine (default).
- tailnet  `tailscale serve` (tailnet only, never Funnel) on the service's tailnet
           port -> every tailnet member the ACLs let reach this node.
- owner    same route, but through an owner gate: a loopback proxy that only passes
           requests carrying the owner's Tailscale-User-Login. Tailscale Serve sets
           that header for tailnet peers and deletes client-supplied copies
           (ipn/ipnlocal/serve.go, addTailscaleIdentityHeaders) -- so it can be trusted
           exactly when the request reaches the gate from loopback, i.e. via Serve.

There is deliberately no public level. Tailnet ports never use 443/8443/10000, the
only ports Funnel can publish, so a service can't become public by a later Funnel.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from backend.services.domain import ServiceManifest

FUNNEL_PORTS = frozenset({443, 8443, 10000})
TAILNET_PORT_OFFSET = 10000
GATE_PORT_OFFSET = 20000  # added to the tailnet port
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


class ExposureError(ValueError):
    pass


class Exposure(str, Enum):
    LOCAL = "local"
    TAILNET = "tailnet"
    OWNER = "owner"


@dataclass(frozen=True)
class ServeRoute:
    https_port: int
    target: str


@dataclass(frozen=True)
class GateSpec:
    listen_port: int
    upstream_port: int
    owner: str


@dataclass(frozen=True)
class ExposurePlan:
    level: Exposure
    route: Optional[ServeRoute]
    gate: Optional[GateSpec]
    open_path: str
    warning: Optional[str] = None

    def url(self, dns_name: Optional[str]) -> Optional[str]:
        if not self.route or not dns_name:
            return None
        return f"https://{dns_name}:{self.route.https_port}{self.open_path}"


def tailnet_port(m: ServiceManifest) -> int:
    port = m.tailnet_port if m.tailnet_port is not None else m.port + TAILNET_PORT_OFFSET
    if port in FUNNEL_PORTS:
        raise ExposureError(f"tailnet port {port} is one Funnel can publish -- choose another tailnet_port")
    if not 1024 <= port <= 65535:
        raise ExposureError(f"tailnet port {port} is out of range -- set tailnet_port in the manifest")
    return port


def plan_exposure(m: ServiceManifest, level: Exposure, owner: Optional[str]) -> ExposurePlan:
    warning = None  # manifests are loopback-only (domain.py), so the plan is the whole truth
    if level == Exposure.LOCAL:
        return ExposurePlan(level, None, None, m.open_path, warning)
    if not m.ui:
        # an API (tts-serve, ...) has no page to open -- it must not land in the tailnet
        raise ExposureError(f"{m.name} has no page of its own (\"ui\": false) -- it stays on this machine")
    if level == Exposure.TAILNET and m.identity == "single":
        # docs/identity.md 4: one person's data -- the whole tailnet includes other users'
        # shared-in devices
        raise ExposureError(f"{m.name} speichert Daten nur für eine Person (\"identity\": \"single\") -- "
                            f"freigeben nur mit „Nur ich (Tailscale)“, bis die App Daten pro Person trennt")
    tport = tailnet_port(m)
    if level == Exposure.TAILNET:
        return ExposurePlan(level, ServeRoute(tport, f"http://127.0.0.1:{m.port}"), None, m.open_path, warning)
    if not owner:
        raise ExposureError("The Tailscale owner of this machine is unknown -- is Tailscale running?")
    gport = tport + GATE_PORT_OFFSET
    if gport > 65535:
        raise ExposureError(f"gate port {gport} is out of range -- choose a lower tailnet_port")
    return ExposurePlan(level, ServeRoute(tport, f"http://127.0.0.1:{gport}"),
                        GateSpec(listen_port=gport, upstream_port=m.port, owner=owner), m.open_path, None)


# --------------------------------------------------------------------------- desired state

@dataclass(frozen=True)
class ExposureSettings:
    levels: Mapping[str, Exposure] = field(default_factory=dict)
    errors: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any) -> "ExposureSettings":
        raw = (data or {}).get("services") if isinstance(data, dict) else None
        levels: Dict[str, Exposure] = {}
        errors: List[str] = []
        for sid, value in (raw or {}).items() if isinstance(raw, dict) else []:
            try:
                levels[str(sid)] = Exposure(value)
            except ValueError:
                errors.append(f"{sid}: invalid exposure {value!r}")
        return cls(levels=levels, errors=tuple(errors))

    def level(self, service_id: str) -> Exposure:
        return self.levels.get(service_id, Exposure.LOCAL)

    def with_level(self, service_id: str, level: Exposure) -> "ExposureSettings":
        levels = dict(self.levels)
        if level == Exposure.LOCAL:
            levels.pop(service_id, None)
        else:
            levels[service_id] = level
        return ExposureSettings(levels=levels)

    def to_dict(self) -> Dict[str, Any]:
        return {"services": {k: v.value for k, v in sorted(self.levels.items())}}


# --------------------------------------------------------------------------- serve config

def parse_serve_status(status: Any) -> Dict[int, str]:
    """`tailscale serve status --json` -> {https port: proxy target} for root ("/")
    proxy mounts. Anything else (path mounts like /public, files, text) isn't
    something this module creates, so it isn't reported -- and never touched."""
    out: Dict[int, str] = {}
    web = status.get("Web") if isinstance(status, dict) else None
    for hostport, conf in (web or {}).items():
        try:
            port = int(str(hostport).rsplit(":", 1)[1])
        except (IndexError, ValueError):
            continue
        handler = ((conf or {}).get("Handlers") or {}).get("/")
        if isinstance(handler, dict) and handler.get("Proxy"):
            out[port] = handler["Proxy"]
    return out


def reconcile_serve(desired: Mapping[int, str], actual: Mapping[int, str], managed_ports: Set[int],
                    service_ports: FrozenSet[int] = frozenset()) -> List[Tuple[str, int, Optional[str]]]:
    """Actions to make Serve match `desired`. Routes are only created on the
    services' own tailnet ports (never a Funnel port). The dashboard is the only
    authority over a service's reachability, so a root route set up by hand that
    points at a service's port is removed too -- on any port. Path mounts (the
    public API's /public on 8443) and routes to ports that belong to no service
    are left alone. Every "off" removes the root ("/") handler only."""
    allowed = set(managed_ports) - FUNNEL_PORTS
    actions: List[Tuple[str, int, Optional[str]]] = []
    for port, target in desired.items():
        if port in allowed and actual.get(port) != target:
            actions.append(("set", port, target))
    for port, target in actual.items():
        if port in desired:
            continue
        if port in allowed or _target_port(target) in service_ports:
            actions.append(("off", port, None))
    return actions


def _target_port(target: str) -> Optional[int]:
    from urllib.parse import urlsplit
    try:
        u = urlsplit(target)
        return u.port if u.hostname in ("127.0.0.1", "localhost", "::1") else None
    except ValueError:
        return None


# --------------------------------------------------------------------------- gate / authority

def _ip(addr: Optional[str]):
    try:
        return ipaddress.ip_address((addr or "").split("%")[0])
    except ValueError:
        return None


def _is_loopback(addr: Optional[str]) -> bool:
    ip = _ip(addr)
    return bool(ip and ip.is_loopback)


def _is_tailnet(addr: Optional[str]) -> bool:
    ip = _ip(addr)
    return bool(ip and (ip in _TAILNET_V4 if ip.version == 4 else ip in _TAILNET_V6))


def gate_allows(peer: Optional[str], headers: Mapping[str, str], owner: Optional[str]) -> Tuple[bool, str]:
    h = {k.lower(): v for k, v in headers.items()}
    if not owner:
        return False, "owner unknown"
    if not _is_loopback(peer):
        return False, "not via Tailscale Serve"
    if "tailscale-funnel-request" in h:
        return False, "internet (Funnel) request"
    login = h.get("tailscale-user-login")
    if not login:
        return False, "no Tailscale identity"
    if login.strip().lower() != owner.strip().lower():
        return False, f"user {login} is not the owner"
    return True, "owner"


def may_change_exposure(peer: Optional[str], peer_login: Optional[str], owner: Optional[str]) -> bool:
    """This machine itself, or a tailnet device of the owner (login via `tailscale
    whois` of the peer address). Not the LAN, not other tailnet users."""
    if _is_loopback(peer):
        return True
    return bool(owner and peer_login and _is_tailnet(peer)
                and peer_login.strip().lower() == owner.strip().lower())


def foreign_routes(m: ServiceManifest, actual: Mapping[int, str]) -> List[int]:
    """Serve ports that proxy straight to this service but aren't the dashboard's own
    route -- set up by hand, so the service is reachable in the tailnet regardless
    of its level here."""
    from urllib.parse import urlsplit
    try:
        own = tailnet_port(m)
    except ExposureError:
        own = None
    out = []
    for port, target in actual.items():
        if port == own:
            continue
        try:
            u = urlsplit(target)
            hit = u.hostname in ("127.0.0.1", "localhost", "::1") and u.port == m.port
        except ValueError:
            hit = False
        if hit:
            out.append(port)
    return sorted(out)
