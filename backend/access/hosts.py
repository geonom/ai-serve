"""This machine's names and addresses, for the Host/Origin check in policy.py (DNS rebinding, cross-site requests).
Names: localhost, the hostname (+ .local), the private network's name for this machine (Tailscale's DNS name) and
AI_SERVE_ALLOWED_HOSTS (comma-separated, for names of your own: a reverse proxy, another DNS name). Addresses: every
address of this machine's interfaces. Looked up at most once a minute; a failing lookup never removes localhost."""
import os
import socket
import time
from typing import FrozenSet, Set, Tuple

_TTL_S = 60.0
_cache: dict = {}


def _network_hostname() -> str:
    from backend.network import active
    return active().hostname() or ""


def _addresses() -> Set[str]:
    import psutil
    return {a.address.split("%")[0] for addrs in psutil.net_if_addrs().values() for a in addrs
            if a.family in (socket.AF_INET, socket.AF_INET6)}


def known_hosts() -> Tuple[FrozenSet[str], FrozenSet[str]]:
    hit = _cache.get("v")
    if hit and time.monotonic() < hit[0]:
        return hit[1]
    names = {"localhost"}
    try:
        h = socket.gethostname().lower()
        names |= {h, h + ".local"}
    except OSError:
        pass
    try:
        n = _network_hostname().lower().rstrip(".")
        if n:
            names |= {n, n.split(".")[0]}
    except Exception:
        pass
    names |= {x.strip().lower() for x in os.environ.get("AI_SERVE_ALLOWED_HOSTS", "").split(",") if x.strip()}
    ips = {"127.0.0.1", "::1"}
    try:
        ips |= _addresses()
    except Exception:
        pass
    value = (frozenset(names), frozenset(ips))
    _cache["v"] = (time.monotonic() + _TTL_S, value)
    return value
