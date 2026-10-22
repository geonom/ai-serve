"""Enrolling nodes -- rules (one-time codes, request validation) and the node
registry file. Reasons and rules: tests/unit/test_cluster_enrollment.py."""
import hmac
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
MIN_TOKEN = 32


class EnrollError(ValueError):
    pass


@dataclass(frozen=True)
class Enrollment:
    code: str
    expires_at: float


class EnrollmentBook:
    """Outstanding one-time codes (in memory: a backend restart voids them, which is
    the safe direction)."""

    def __init__(self, clock: Callable[[], float] = time.time, ttl_s: float = 900):
        self._clock = clock
        self._ttl = ttl_s
        self._codes: Dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> Enrollment:
        code = secrets.token_urlsafe(18)
        exp = self._clock() + self._ttl
        with self._lock:
            self._codes[code] = exp
        return Enrollment(code, exp)

    def redeem(self, code: Any) -> bool:
        if not isinstance(code, str):
            return False
        now = self._clock()
        with self._lock:
            match = None
            for known in list(self._codes):
                if self._codes[known] < now:
                    del self._codes[known]
                elif hmac.compare_digest(known.encode(), code.encode()):
                    match = known
            if match is None:
                return False
            del self._codes[match]
            return True


@dataclass(frozen=True)
class NodeRecord:
    id: str
    url: str
    token: str
    enrolled_at: float
    coordinator_url: str = ""   # how the agent reaches us -- model download sources must match it


def validate_enrollment(body: Any, peer: Optional[str], is_private: Callable[[str], bool], now: float) -> NodeRecord:
    if not isinstance(body, dict):
        raise EnrollError("JSON object expected")
    unknown = set(body) - {"node_name", "agent_url", "token", "code"}
    if unknown:
        raise EnrollError(f"unknown fields: {sorted(unknown)}")
    name = body.get("node_name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise EnrollError("node_name: letters, digits, . _ - (max 63)")
    token = body.get("token")
    if not isinstance(token, str) or len(token) < MIN_TOKEN:
        raise EnrollError(f"token must have at least {MIN_TOKEN} characters")
    url = body.get("agent_url")
    if not isinstance(url, str):
        raise EnrollError("agent_url required")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = ipaddress.ip_address(parts.hostname or "")
    except ValueError:
        raise EnrollError("agent_url must be http://<ip>:<port>")
    if parts.scheme != "http" or parts.username or parts.password or parts.path not in ("", "/") \
            or parts.query or parts.fragment or port is None or not 1024 <= port <= 65535:
        raise EnrollError("agent_url must be http://<ip>:<port> (port 1024..65535)")
    try:
        peer_ip = ipaddress.ip_address(peer or "")
    except ValueError:
        raise EnrollError("enrollment from an unknown address")
    if host != peer_ip:
        raise EnrollError("agent_url must point at the address the enrollment comes from")
    if not (peer_ip.is_loopback or is_private(str(peer_ip))):
        raise EnrollError("nodes must enroll from the private overlay network")
    netloc = f"[{host}]" if host.version == 6 else str(host)
    return NodeRecord(id=name, url=f"http://{netloc}:{port}", token=token, enrolled_at=now)


class NodeRegistry:
    """cluster_nodes.json -- holds the node tokens, so 0600 and never listed."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, NodeRecord]:
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return {}
        return {k: NodeRecord(**v) for k, v in (raw.get("nodes") or {}).items()}

    def _save(self, nodes: Dict[str, NodeRecord]) -> None:
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"nodes": {k: asdict(v) for k, v in sorted(nodes.items())}}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def add(self, rec: NodeRecord) -> None:
        with self._lock:
            nodes = self._load()
            nodes[rec.id] = rec
            self._save(nodes)

    def remove(self, node_id: str) -> bool:
        with self._lock:
            nodes = self._load()
            if node_id not in nodes:
                return False
            del nodes[node_id]
            self._save(nodes)
            return True

    def get(self, node_id: str) -> Optional[NodeRecord]:
        return self._load().get(node_id)

    def all(self) -> List[NodeRecord]:
        return list(self._load().values())

    def listing(self) -> List[Dict[str, Any]]:
        return [{"id": r.id, "url": r.url, "enrolled_at": r.enrolled_at} for r in self.all()]
