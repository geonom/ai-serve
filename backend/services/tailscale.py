"""`tailscale` CLI adapter for service exposure. Only `serve` (tailnet-only) is ever
used here -- never `funnel`."""
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

_TIMEOUT = 15


@dataclass(frozen=True)
class TailscaleInfo:
    owner: str      # login of the user owning this machine, e.g. "owner@github"
    dns_name: str   # e.g. "my-pc.example-tailnet.ts.net"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tailscale", *args], capture_output=True, text=True, timeout=_TIMEOUT)


def _json(*args: str) -> Optional[Any]:
    try:
        r = _run(*args)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout or "{}")
    except ValueError:
        return None


def info() -> Optional[TailscaleInfo]:
    status = _json("status", "--json")
    if not isinstance(status, dict) or status.get("BackendState") != "Running":
        return None
    me = status.get("Self") or {}
    user = (status.get("User") or {}).get(str(me.get("UserID"))) or {}
    owner = user.get("LoginName")
    dns = (me.get("DNSName") or "").rstrip(".")
    if not owner or not dns or me.get("Tags"):
        return None  # a tagged node has no owning user
    return TailscaleInfo(owner=owner, dns_name=dns)


def serve_status() -> Dict[str, Any]:
    return _json("serve", "status", "--json") or {}


def serve_set(port: int, target: str) -> None:
    r = _run("serve", "--bg", "--yes", f"--https={int(port)}", target)
    if r.returncode != 0:
        raise RuntimeError(f"tailscale serve --https={port} failed: {(r.stderr or r.stdout).strip()[-300:]}")


def serve_off(port: int) -> None:
    """Remove only the root ("/") handler -- other path mounts on the same port
    (e.g. the public API's /public on 8443) stay."""
    r = _run("serve", f"--https={int(port)}", "--set-path=/", "off")
    if r.returncode != 0:
        raise RuntimeError(f"tailscale serve --https={port} off failed: {(r.stderr or r.stdout).strip()[-300:]}")


def whois(addr: str) -> Optional[str]:
    data = _json("whois", "--json", addr)
    if not isinstance(data, dict):
        return None
    return (data.get("UserProfile") or {}).get("LoginName")
