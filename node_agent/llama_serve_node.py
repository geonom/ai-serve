#!/usr/bin/env python3
"""llama-serve node agent -- lets a ai-serve-ui coordinator use this device's
accelerators (NVIDIA/AMD GPU, Apple Silicon, CPU). One file, Python 3.9+ standard
library only. Needs: Tailscale, a llama.cpp build (llama-server), GGUF models.

    python3 llama_serve_node.py --config ~/.config/llama-serve-node/config.json
    python3 llama_serve_node.py --init   # writes a config template + a fresh token

Security (it starts processes on this computer):
- Listens only on loopback or this device's address in the private overlay network
  (Tailscale by default; WireGuard/Nebula/NetBird/ZeroTier via "network": static)
  -- never 0.0.0.0/LAN.
- Every request needs the node token (Bearer, >= 32 chars) AND must come from
  loopback or from an overlay peer whose identity (`tailscale whois`, or the static
  peers map) is `coordinator_login`. A stolen token alone, or tailnet identity alone, gets nothing.
- Only the configured llama-server binary is ever started, with a model resolved
  inside `model_dirs` (.gguf, symlinks resolved, no traversal) and only the knobs
  ctx_size / parallel / gpu. No free-form arguments.
- llama-server binds 127.0.0.1; the coordinator reaches it only through this agent's
  authenticated proxy, and only on inference paths (no /slots save/restore, which
  writes files)."""
import argparse
import hmac
import ipaddress
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socketserver
from typing import Any, Dict, List, Optional


class QuickHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the socket.getfqdn() in server_bind: on a Mac whose name resolution hangs it took
    ~35 s before the server accepted connections (found in CI on macOS, 2026-10-22). server_name isn't used here."""
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

VERSION = "0.1.0"
MIN_TOKEN = 32
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_CONFIG_FIELDS = {"node_name", "listen", "port", "token", "coordinator_login", "llama_server",
                  "model_dirs", "log_dir", "max_instances", "budget_mb", "tailscale_cli", "network", "data_dir",
                  "coordinator_url", "model_quota_mb"}
# Runtimes come only from llama.cpp's own GitHub release downloads (+ the sha256 the
# coordinator read from GitHub's API).
DOWNLOAD_PREFIX = "https://github.com/ggml-org/llama.cpp/releases/download/"
MAX_DOWNLOAD = 2 * 1024 ** 3
# Models: from this node's own coordinator or from Hugging Face, sha256 always given.
HF_PREFIX = "https://huggingface.co/"
DOWNLOAD_SUBDIR = "llama-serve-downloads"   # inside the first model dir; deletions only here, only on request
PROBE_BYTES = 8 * 1024 * 1024
MAX_MODEL = 500 * 1024 ** 3
_PROXY_RE = re.compile(r"^(v1/[A-Za-z0-9_/.-]+|health|props|tokenize|detokenize|completion|embedding|rerank)$")


class ConfigError(ValueError):
    pass


class RequestError(ValueError):
    status = 400


# --------------------------------------------------------------------------- rules

def _is_tailnet(ip) -> bool:
    return (ip.version == 4 and ip in _TAILNET_V4) or (ip.version == 6 and ip in _TAILNET_V6)


def _ip(value):
    try:
        return ipaddress.ip_address(value or "")
    except ValueError:
        return None


def _tailnet_private(value: str) -> bool:
    ip = _ip(value)
    return bool(ip and _is_tailnet(ip))


def _peer_kind(peer: Optional[str], is_private=None) -> str:
    ip = _ip(peer)
    if ip is None:
        return "other"
    if ip.is_loopback:
        return "loopback"
    return "tailnet" if (is_private or _tailnet_private)(str(ip)) else "other"


class Network:
    """The private overlay network. "tailscale" (default): Tailscale's ranges,
    identity via `tailscale whois`. "static": for WireGuard / Nebula / NetBird /
    ZeroTier / Headscale without whois -- `networks` (private ranges) and `peers`
    (overlay address -> identity; on WireGuard an address is bound to one key)."""

    def __init__(self, provider: str, networks=(), peers=None, whois=None):
        self.provider = provider
        self.networks = tuple(networks)
        self.peers = peers or {}
        self._whois = whois

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Network":
        d = d or {"provider": "tailscale"}
        if not isinstance(d, dict):
            raise ConfigError("network must be an object")
        provider = d.get("provider", "tailscale")
        if provider == "tailscale":
            if set(d) - {"provider"}:
                raise ConfigError("network: tailscale takes no further fields")
            return cls("tailscale", (_TAILNET_V4, _TAILNET_V6))
        if provider != "static":
            raise ConfigError("network.provider must be tailscale or static")
        unknown = set(d) - {"provider", "networks", "peers"}
        if unknown:
            raise ConfigError(f"network: unknown fields {sorted(unknown)}")
        nets = []
        for raw in d.get("networks") or []:
            try:
                n = ipaddress.ip_network(raw, strict=False)
            except ValueError:
                raise ConfigError(f"network: not a network: {raw!r}")
            if not n.is_private or n.prefixlen < (8 if n.version == 4 else 32):
                raise ConfigError(f"network: {raw} is not a private overlay range")
            nets.append(n)
        if not nets:
            raise ConfigError("network: at least one private network")
        peers = {}
        for addr, ident in (d.get("peers") or {}).items():
            ip = _ip(addr)
            if ip is None or not any(ip.version == n.version and ip in n for n in nets):
                raise ConfigError(f"network: peer {addr} is outside the private networks")
            peers[str(ip)] = str(ident)
        return cls("static", nets, peers)

    def is_private(self, value: str) -> bool:
        ip = _ip(value)
        return bool(ip and any(ip.version == n.version and ip in n for n in self.networks))

    def identify(self, value: str) -> Optional[str]:
        if not self.is_private(value):
            return None
        if self.provider == "static":
            return self.peers.get(str(_ip(value)))
        return self._whois(value) if self._whois else None


@dataclass(frozen=True)
class AgentConfig:
    node_name: str
    listen: str
    port: int
    token: str
    coordinator_login: str
    llama_server: Optional[str]
    model_dirs: List[str]
    log_dir: str
    max_instances: int = 4
    budget_mb: Optional[Dict[str, int]] = None
    tailscale_cli: Optional[str] = None
    network: Network = None
    data_dir: str = ""
    coordinator_url: Optional[str] = None
    model_quota_mb: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentConfig":
        unknown = set(d) - _CONFIG_FIELDS
        if unknown:
            raise ConfigError(f"unknown config fields: {sorted(unknown)}")
        name = str(d.get("node_name") or "")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$", name):
            raise ConfigError("node_name: letters, digits, . _ - (max 63)")
        listen = str(d.get("listen") or "")
        try:
            ip = ipaddress.ip_address(listen)
        except ValueError:
            raise ConfigError("listen must be an IP address (loopback or this device's Tailscale address)")
        network = Network.from_dict(d.get("network"))
        if not (ip.is_loopback or network.is_private(listen)):
            raise ConfigError("listen must be loopback or an address of the private overlay network "
                              "-- never 0.0.0.0 or the LAN")
        port = d.get("port", 8040)
        if not isinstance(port, int) or not (port == 0 or 1024 <= port <= 65535):
            raise ConfigError("port must be 1024..65535")
        token = str(d.get("token") or "")
        if len(token) < MIN_TOKEN:
            raise ConfigError(f"token must have at least {MIN_TOKEN} characters")
        login = str(d.get("coordinator_login") or "")
        if "@" not in login:
            raise ConfigError("coordinator_login must be the coordinator's Tailscale login, e.g. you@github")
        binary = d.get("llama_server")
        real = None
        if binary is not None:  # None = no runtime yet (provisioning installs one)
            if not isinstance(binary, str) or not os.path.isabs(binary):
                raise ConfigError("llama_server must be an absolute path")
            real = os.path.realpath(binary)
            if not os.path.basename(real).startswith("llama-server") or not os.path.isfile(real) \
                    or not os.access(real, os.X_OK):
                raise ConfigError("llama_server must be an executable llama-server binary")
        dirs = d.get("model_dirs") or []
        if not isinstance(dirs, list) or not dirs:
            raise ConfigError("model_dirs must be a non-empty list")
        for m in dirs:
            if not isinstance(m, str) or not os.path.isabs(m) or not os.path.isdir(m):
                raise ConfigError(f"model dir must be an existing absolute path: {m!r}")
        max_inst = d.get("max_instances", 4)
        if not isinstance(max_inst, int) or not 1 <= max_inst <= 16:
            raise ConfigError("max_instances must be 1..16")
        budget = d.get("budget_mb")
        if budget is not None and (not isinstance(budget, dict) or not all(
                isinstance(v, int) and v >= 0 for v in budget.values())):
            raise ConfigError('budget_mb must map accelerator index to MB, e.g. {"0": 12000}')
        log_dir = str(d.get("log_dir") or os.path.expanduser("~/.local/state/llama-serve-node"))
        data_dir = str(d.get("data_dir") or os.path.expanduser("~/.local/share/llama-serve-node"))
        if not os.path.isabs(data_dir):
            raise ConfigError("data_dir must be an absolute path")
        coord_url = d.get("coordinator_url")
        if coord_url is not None:
            from urllib.parse import urlsplit
            u = urlsplit(str(coord_url))
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password \
                    or u.query or u.fragment or u.path not in ("", "/"):
                raise ConfigError("coordinator_url must be http(s)://<host>:<port>")
            coord_url = str(coord_url).rstrip("/")
        quota = d.get("model_quota_mb")
        if quota is not None and (not isinstance(quota, int) or isinstance(quota, bool) or quota < 100):
            raise ConfigError("model_quota_mb must be at least 100")
        return cls(node_name=name, listen=listen, port=port, token=token, coordinator_login=login,
                   llama_server=real, model_dirs=[os.path.realpath(m) for m in dirs], log_dir=log_dir,
                   max_instances=max_inst, budget_mb={str(k): v for k, v in budget.items()} if budget else None,
                   tailscale_cli=d.get("tailscale_cli"), network=network, data_dir=data_dir,
                   coordinator_url=coord_url, model_quota_mb=quota)


def refusal_reason(peer: Optional[str], authorization: Optional[str], peer_login: Optional[str],
                   token: str, coordinator_login: str, is_private=None,
                   coordinator_ip: Optional[str] = None) -> Optional[str]:
    """None = allowed; otherwise why not (logged locally -- never sent to the caller)."""
    auth = authorization or ""
    if not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:].encode(), token.encode()):
        return "missing or wrong token"
    kind = _peer_kind(peer, is_private)
    if kind == "loopback":
        return None
    if kind != "tailnet":
        return f"{peer} is not in the private overlay network"
    # The coordinator's overlay address, pinned in the config at enrollment (where its
    # identity was checked by whois): on Tailscale/WireGuard an overlay address is
    # bound to one device key, so it identifies the coordinator without a runtime
    # whois -- which failed live under macOS.
    if coordinator_ip and _ip(peer) == _ip(coordinator_ip):
        return None
    if not peer_login:
        return f"identity: whois gave nothing for {peer}"
    if peer_login.strip().lower() != coordinator_login.strip().lower():
        return f"identity: {peer_login} is not the coordinator's {coordinator_login}"
    return None


def authorize(peer: Optional[str], authorization: Optional[str], peer_login: Optional[str],
              token: str, coordinator_login: str, is_private=None, coordinator_ip: Optional[str] = None) -> bool:
    return refusal_reason(peer, authorization, peer_login, token, coordinator_login, is_private,
                          coordinator_ip) is None


def resolve_model(cfg: AgentConfig, name: Any) -> str:
    if not isinstance(name, str) or not name or os.path.isabs(name) or "\\" in name:
        raise RequestError("model must be a relative .gguf path inside a model dir")
    if not name.lower().endswith(".gguf"):
        raise RequestError("only .gguf models")
    for root in cfg.model_dirs:
        real = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath([real, root]) == root and os.path.isfile(real):
            return real
    raise RequestError(f"model not found in the model dirs: {name}")


def list_models(cfg: AgentConfig) -> List[Dict[str, Any]]:
    out = {}
    for root in cfg.model_dirs:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(".gguf"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                try:
                    path = resolve_model(cfg, rel)
                except RequestError:
                    continue
                out.setdefault(rel, {"name": rel, "size_mb": round(os.path.getsize(path) / 1048576)})
    return [out[k] for k in sorted(out)]


_POOLING = ("none", "mean", "cls", "last", "rank")


@dataclass(frozen=True)
class StartRequest:
    model: str
    ctx_size: int = 8192
    parallel: int = 1
    gpu: Optional[int] = None
    mode: str = "chat"              # chat / embedding / reranking -> fixed flags, see flags()
    pooling: Optional[str] = None   # only for embedding/reranking, llama.cpp's own list

    def flags(self) -> List[str]:
        if self.mode == "embedding":
            return ["--embedding"] + (["--pooling", self.pooling] if self.pooling else [])
        if self.mode == "reranking":  # implies --embedding and pooling "rank" in llama-server
            return ["--reranking"]
        return []

    @classmethod
    def from_dict(cls, d: Any, accelerators: List[Dict[str, Any]]) -> "StartRequest":
        if not isinstance(d, dict):
            raise RequestError("JSON object expected")
        unknown = set(d) - {"model", "ctx_size", "parallel", "gpu", "mode", "pooling"}
        if unknown:
            raise RequestError(f"unknown fields: {sorted(unknown)}")
        model = d.get("model")
        if not isinstance(model, str) or not model:
            raise RequestError("model is required")
        ctx = d.get("ctx_size", 8192)
        if not isinstance(ctx, int) or isinstance(ctx, bool) or not 512 <= ctx <= 262144:
            raise RequestError("ctx_size must be 512..262144")
        par = d.get("parallel", 1)
        if not isinstance(par, int) or isinstance(par, bool) or not 1 <= par <= 16:
            raise RequestError("parallel must be 1..16")
        gpu = d.get("gpu")
        if gpu is not None:
            gpus = {a["index"] for a in accelerators if a["kind"] in ("cuda", "rocm", "vulkan", "metal")}
            if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu not in gpus:
                raise RequestError(f"gpu must be one of {sorted(gpus)}")
        mode = d.get("mode", "chat")
        if mode not in ("chat", "embedding", "reranking"):
            raise RequestError("mode must be chat, embedding or reranking")
        pooling = d.get("pooling")
        if pooling is not None and (mode == "chat" or pooling not in _POOLING):
            raise RequestError(f"pooling ({', '.join(_POOLING)}) only for embedding/reranking")
        return cls(model=model, ctx_size=ctx, parallel=par, gpu=gpu, mode=mode, pooling=pooling)


def proxy_path_allowed(path: str) -> bool:
    path = path.split("?", 1)[0]
    return bool(_PROXY_RE.match(path)) and ".." not in path.split("/")


_FILE_RE = re.compile(r"^(cudart-)?llama-b\d{1,7}-bin-[a-z0-9.-]{1,60}\.(tar\.gz|zip)$")


@dataclass(frozen=True)
class InstallFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class InstallRequest:
    tag: str
    variant: str
    files: tuple

    @classmethod
    def from_dict(cls, d: Any) -> "InstallRequest":
        if not isinstance(d, dict) or set(d) - {"tag", "variant", "files"}:
            raise RequestError("expected {tag, variant, files}")
        tag = d.get("tag")
        if not isinstance(tag, str) or not re.fullmatch(r"b\d{1,7}", tag):
            raise RequestError("tag must look like b11681")
        variant = d.get("variant")
        if not isinstance(variant, str) or not re.fullmatch(r"[a-z0-9.-]{1,32}", variant):
            raise RequestError("invalid variant")
        files = d.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 2:
            raise RequestError("one or two files")
        out = []
        for f in files:
            if not isinstance(f, dict):
                raise RequestError("file entries must be objects")
            name, url, sha, size = f.get("name"), f.get("url"), f.get("sha256"), f.get("size")
            if not isinstance(name, str) or not _FILE_RE.match(name) or f"-{tag}-" not in name:
                raise RequestError(f"not a llama.cpp release archive of {tag}: {name!r}")
            if url != f"{DOWNLOAD_PREFIX}{tag}/{name}":
                raise RequestError("downloads only from llama.cpp's GitHub releases")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise RequestError("sha256 must be 64 hex characters")
            if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_DOWNLOAD:
                raise RequestError("size out of range")
            out.append(InstallFile(name, url, sha, size))
        return cls(tag=tag, variant=variant, files=tuple(out))


_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.gguf$")
_HF_PATH_RE = re.compile(r"^/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*/resolve/[A-Za-z0-9._-]+/"
                         r"([A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class DownloadRequest:
    name: str
    sha256: str
    size: int
    sources: tuple

    @classmethod
    def from_dict(cls, d: Any, coordinator_url: Optional[str]) -> "DownloadRequest":
        from urllib.parse import urlsplit
        if not isinstance(d, dict) or set(d) - {"name", "sha256", "size", "sources"}:
            raise RequestError("expected {name, sha256, size, sources}")
        name = d.get("name")
        if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
            raise RequestError("name must be a plain .gguf file name")
        sha = d.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise RequestError("sha256 must be 64 hex characters")
        size = d.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_MODEL:
            raise RequestError("size out of range")
        sources = d.get("sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 3:
            raise RequestError("one to three sources")
        for src in sources:
            ok = False
            if isinstance(src, str) and coordinator_url and src == f"{coordinator_url}/api/cluster/models/{name}":
                ok = True
            elif isinstance(src, str) and src.startswith(HF_PREFIX):
                u = urlsplit(src)
                ok = (u.scheme == "https" and u.hostname == "huggingface.co" and not u.query
                      and bool(_HF_PATH_RE.match(u.path)) and u.path.rsplit("/", 1)[-1] == name
                      and ".." not in u.path.split("/"))
            if not ok:
                raise RequestError(f"source not allowed (own coordinator or huggingface.co only): {src!r}")
        return cls(name=name, sha256=sha, size=size, sources=tuple(sources))


def check_room(directory: str, need_bytes: int, quota_bytes: Optional[int]) -> None:
    """Refuse a download that doesn't fit the quota. NEVER deletes anything (user
    decision 2026-10-16: freeing space needs the user) -- the error lists what's
    there so they can delete explicitly (DELETE /v1/downloads/<name>)."""
    if quota_bytes is None:
        return
    if need_bytes > quota_bytes:
        raise RequestError(f"model ({need_bytes >> 20} MB) is larger than the quota ({quota_bytes >> 20} MB)")
    files = []
    for f in os.listdir(directory) if os.path.isdir(directory) else []:
        p = os.path.join(directory, f)
        if f.endswith(".gguf") and os.path.isfile(p):
            files.append((os.path.getmtime(p), f, os.path.getsize(p)))
    used = sum(sz for _, _, sz in files)
    if used + need_bytes <= quota_bytes:
        return
    listing = ", ".join(f"{name} ({sz >> 20} MB, last used {time.strftime('%Y-%m-%d', time.localtime(mt))})"
                        for mt, name, sz in sorted(files))
    raise RequestError(f"not enough room in the model quota: need {need_bytes >> 20} MB, "
                       f"{(quota_bytes - used) >> 20} MB free. Delete downloads you no longer need "
                       f"first: {listing}")


def _inside(root: str, path: str) -> bool:
    root = os.path.realpath(root)
    return os.path.commonpath([root, os.path.realpath(path)]) == root


def safe_extract(archive: str, dest: str) -> None:
    """Extract only regular files, dirs and symlinks that stay inside dest. Every
    member is checked before anything is written."""
    import tarfile
    import zipfile
    dest = os.path.realpath(dest)

    def check_name(name: str) -> str:
        if not name or name.startswith(("/", "\\")) or ".." in re.split(r"[/\\]", name) or ":" in name:
            raise RequestError(f"unsafe path in archive: {name!r}")
        target = os.path.join(dest, name)
        if os.path.commonpath([dest, os.path.normpath(target)]) != dest:
            raise RequestError(f"unsafe path in archive: {name!r}")
        return target

    os.makedirs(dest, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            for i in infos:
                check_name(i.filename)
            for i in infos:
                target = check_name(i.filename)
                if i.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(i) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(target, 0o755)
        return
    with tarfile.open(archive, "r:*") as t:
        members = t.getmembers()
        for m in members:
            target = check_name(m.name)
            if m.issym():
                link = m.linkname
                if os.path.isabs(link) or not os.path.normpath(os.path.join(os.path.dirname(target), link)).startswith(dest + os.sep):
                    raise RequestError(f"symlink leaves the archive: {m.name} -> {link}")
            elif not (m.isfile() or m.isdir()):
                raise RequestError(f"unsupported archive member: {m.name}")
        for m in members:
            target = check_name(m.name)
            if m.isdir():
                os.makedirs(target, exist_ok=True)
            elif m.issym():
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if os.path.lexists(target):
                    os.remove(target)
                os.symlink(m.linkname, target)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = t.extractfile(m)
                with open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(target, 0o755 if m.mode & 0o111 else 0o644)


# --------------------------------------------------------------------------- hardware

def _run(cmd: List[str], timeout: float = 10) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_accelerators() -> List[Dict[str, Any]]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits"]) if shutil.which("nvidia-smi") else None
    if out:
        accels = []
        for line in out.strip().splitlines():
            idx, name, total, free = [p.strip() for p in line.split(",")]
            accels.append({"index": int(idx), "kind": "cuda", "name": name,
                           "total_mb": int(float(total)), "free_mb": int(float(free))})
        return accels
    if shutil.which("rocm-smi"):
        info = _run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"])
        if info:
            try:
                data = json.loads(info)
                accels = []
                for key, v in sorted(data.items()):
                    if not key.startswith("card"):
                        continue
                    total = int(v.get("VRAM Total Memory (B)", 0)) // 1048576
                    used = int(v.get("VRAM Total Used Memory (B)", 0)) // 1048576
                    accels.append({"index": int(key[4:]), "kind": "rocm", "name": v.get("Card series", key),
                                   "total_mb": total, "free_mb": max(0, total - used)})
                if accels:
                    return accels
            except (ValueError, TypeError):
                pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        mem = _run(["sysctl", "-n", "hw.memsize"])
        chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon").strip()
        total = int(mem.strip()) // 1048576 if mem else 0
        # Unified memory: what's "free" for a model is mostly what the owner budgets.
        return [{"index": 0, "kind": "metal", "name": chip, "total_mb": total, "free_mb": None}]
    total = free = 0
    try:
        with open("/proc/meminfo") as f:
            info = dict(l.split(":", 1) for l in f if ":" in l)
        total = int(info["MemTotal"].split()[0]) // 1024
        free = int(info["MemAvailable"].split()[0]) // 1024
    except (OSError, KeyError, ValueError):
        pass
    return [{"index": 0, "kind": "cpu", "name": platform.processor() or platform.machine(),
             "total_mb": total, "free_mb": free}]


def detect_facts() -> Dict[str, Any]:
    """What the coordinator needs to pick a runtime build for this device."""
    plat = "android" if ("ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", "")) \
        else sys.platform
    vendors, cuda = [], None
    if shutil.which("nvidia-smi"):
        vendors.append("nvidia")
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run(["nvidia-smi"]) or "")
        cuda = [int(m.group(1)), int(m.group(2))] if m else None
    amd = False
    for card in os.listdir("/sys/class/drm") if os.path.isdir("/sys/class/drm") else []:
        try:
            with open(f"/sys/class/drm/{card}/device/vendor") as f:
                amd = amd or f.read().strip() == "0x1002"
        except OSError:
            pass
    if amd:
        vendors.append("amd")
    rocm = bool(shutil.which("rocm-smi") or os.path.isdir("/opt/rocm"))
    vulkan = any(os.path.exists(p) for p in ("/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
                                             "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
                                             "/usr/lib/libvulkan.so.1", "/system/lib64/libvulkan.so"))
    return {"platform": plat, "machine": platform.machine(), "gpu_vendors": vendors,
            "cuda_driver": cuda, "rocm": rocm, "vulkan": vulkan}


def _tailscale_cli(cfg: AgentConfig) -> Optional[str]:
    for c in (cfg.tailscale_cli, shutil.which("tailscale"),
              "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


# --------------------------------------------------------------------------- runtime

class Agent:
    def __init__(self, cfg: AgentConfig, config_path: Optional[str] = None):
        self.cfg = cfg
        self.config_path = config_path
        self.install_state: Dict[str, Any] = {"state": "idle"}
        self.downloads: Dict[str, Dict[str, Any]] = {}
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._whois: Dict[str, Any] = {}
        self._runtime: Optional[str] = None
        os.makedirs(cfg.log_dir, exist_ok=True)

    def coordinator_ip(self) -> Optional[str]:
        """The coordinator's overlay address from the enrolled coordinator_url (only
        if it is an IP literal or resolves to one inside the overlay network)."""
        if not self.cfg.coordinator_url:
            return None
        from urllib.parse import urlsplit
        host = urlsplit(self.cfg.coordinator_url).hostname or ""
        ip = _ip(host)
        if ip is None:
            try:
                ip = _ip(socket.gethostbyname(host))
            except OSError:
                return None
        return str(ip) if ip and self.cfg.network.is_private(str(ip)) else None

    def identify(self, peer: str) -> Optional[str]:
        if self.cfg.network.provider == "static":
            return self.cfg.network.identify(peer)
        return self.whois(peer) if self.cfg.network.is_private(peer) else None

    def whois(self, peer: str) -> Optional[str]:
        hit = self._whois.get(peer)
        if hit and time.monotonic() < hit[0]:
            return hit[1]
        cli = _tailscale_cli(self.cfg)
        login = None
        if cli:
            out = _run([cli, "whois", "--json", peer])
            try:
                login = (json.loads(out or "{}").get("UserProfile") or {}).get("LoginName")
            except ValueError:
                login = None
        self._whois[peer] = (time.monotonic() + (300 if login else 30), login)
        return login

    def runtime_version(self) -> Optional[str]:
        if self.cfg.llama_server is None:
            return None
        if self._runtime is None:
            out = subprocess.run([self.cfg.llama_server, "--version"], capture_output=True, text=True, timeout=15)
            self._runtime = ((out.stdout or "") + (out.stderr or "")).strip().splitlines()[0:1]
            self._runtime = self._runtime[0] if self._runtime else "unknown"
        return self._runtime

    def runtime_variant(self) -> Optional[str]:
        """cpu / vulkan / cuda-12.8 / rocm-10.0 / metal for a runtime this agent
        installed; None for no runtime or one the owner configured."""
        if not self.cfg.llama_server:
            return None
        root = os.path.join(os.path.realpath(self.cfg.data_dir), "runtimes") + os.sep
        if not self.cfg.llama_server.startswith(root):
            return None
        return os.path.relpath(self.cfg.llama_server, root).split(os.sep)[0].split("-", 1)[-1]

    def backend_device(self, gpu: int) -> str:
        """The device number the installed runtime uses for our accelerator `gpu`.
        Vulkan numbers GPUs on its own (found live: 4060 Ti = Vulkan0, 3090 = Vulkan1
        while nvidia-smi says 0 = 3090), so map by name via --list-devices; GPUs
        with the same name keep their relative order. Refuses instead of guessing."""
        if self.runtime_variant() != "vulkan":
            return str(gpu)
        accels = detect_accelerators()
        me = next((a for a in accels if a["index"] == gpu), None)
        if me is None:
            raise RequestError(f"no GPU {gpu}")
        if getattr(self, "_vk_devices", None) is None:
            env = dict(os.environ)
            libs = self._lib_dirs()
            if libs:
                env["LD_LIBRARY_PATH"] = os.pathsep.join(libs)
            try:
                out = subprocess.run([self.cfg.llama_server, "--list-devices"], capture_output=True, text=True,
                                     timeout=60, env=env)
                text = (out.stdout or "") + (out.stderr or "")
            except (OSError, subprocess.TimeoutExpired):
                text = ""
            self._vk_devices = [(int(m.group(1)), m.group(2).strip())
                                for m in re.finditer(r"Vulkan(\d+):\s*(.+?)\s*\(", text)]
        same_name_before = [a for a in accels if a.get("name") == me.get("name") and a["index"] < gpu]
        matches = [idx for idx, name in self._vk_devices if name == me.get("name")]
        if len(matches) <= len(same_name_before):
            raise RequestError(f"can't tell which Vulkan device GPU {gpu} ({me.get('name')}) is -- not starting "
                               f"rather than using the wrong one")
        return str(matches[len(same_name_before)])

    def runtime_kinds(self) -> List[str]:
        """Accelerator kinds the installed llama-server can drive."""
        if not self.cfg.llama_server:
            return []
        v = self.runtime_variant()
        if v is None:  # a binary the owner configured: assume it was built for this hardware
            return sorted({a["kind"] for a in detect_accelerators()} | {"cpu"})
        if v.startswith("cuda"):
            return ["cuda", "cpu"]
        if v.startswith("rocm"):
            return ["rocm", "cpu"]
        return {"cpu": ["cpu"], "vulkan": ["vulkan", "cuda", "rocm", "cpu"], "metal": ["metal", "cpu"]}.get(v, ["cpu"])

    @staticmethod
    def default_budget(a: Dict[str, Any]) -> int:
        """Memory shared with the system (Apple Silicon, CPU-only) -> half the RAM,
        so the device stays usable; dedicated GPU memory -> all of it."""
        total = int(a.get("total_mb") or 0)
        return total // 2 if a.get("kind") in ("metal", "cpu") else total

    def accelerators(self) -> List[Dict[str, Any]]:
        accels = detect_accelerators()
        for a in accels:
            budget = (self.cfg.budget_mb or {}).get(str(a["index"]))
            a["budget_default_mb"] = self.default_budget(a)
            a["budget_mb"] = min(budget, a["total_mb"]) if budget is not None else a["budget_default_mb"]
        return accels

    def set_budgets(self, changes: Any) -> Dict[str, int]:
        """{"<index>": MB or null (= back to the default)} -- validated against the
        detected hardware, persisted to the config file."""
        if not isinstance(changes, dict) or not changes:
            raise RequestError('expected {"<accelerator index>": MB or null}')
        totals = {str(a["index"]): int(a.get("total_mb") or 0) for a in detect_accelerators()}
        budgets = dict(self.cfg.budget_mb or {})
        for key, mb in changes.items():
            if str(key) not in totals:
                raise RequestError(f"no accelerator {key}")
            if mb is None:
                budgets.pop(str(key), None)
            elif not isinstance(mb, int) or isinstance(mb, bool) or not 0 < mb <= totals[str(key)]:
                raise RequestError(f"budget for {key} must be 1..{totals[str(key)]} MB")
            else:
                budgets[str(key)] = mb
        from dataclasses import replace
        self.cfg = replace(self.cfg, budget_mb=budgets or None)
        self._persist({"budget_mb": budgets or None})
        return budgets

    def _persist(self, fields: Dict[str, Any]) -> None:
        if not self.config_path or not os.path.exists(self.config_path):
            return
        with open(self.config_path) as f:
            data = json.load(f)
        for k, v in fields.items():
            if v is None:
                data.pop(k, None)
            else:
                data[k] = v
        tmp = self.config_path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.config_path)

    def node_info(self) -> Dict[str, Any]:
        return {"name": self.cfg.node_name, "agent_version": VERSION, "platform": sys.platform,
                "machine": platform.machine(), "facts": detect_facts(), "accelerators": self.accelerators(),
                "runtime": {"llama_server": self.runtime_version()}, "runtime_kinds": self.runtime_kinds(),
                "instances": [self.public(i) for i in self.instances.values()]}

    @staticmethod
    def public(inst: Dict[str, Any]) -> Dict[str, Any]:
        proc = inst["proc"]
        if inst["state"] in ("loading", "ready") and proc.poll() is not None:
            inst["state"] = "exited"
        return {k: inst.get(k) for k in ("id", "model", "ctx_size", "parallel", "gpu", "mode", "pooling",
                                         "state", "started_at")}

    def start(self, req: StartRequest) -> Dict[str, Any]:
        if self.cfg.llama_server is None:
            err = RequestError("no runtime installed on this node yet")
            err.status = 409
            raise err
        path = resolve_model(self.cfg, req.model)
        with self.lock:
            live = [i for i in self.instances.values() if i["proc"].poll() is None]
            if len(live) >= self.cfg.max_instances:
                raise RequestError(f"at most {self.cfg.max_instances} instances on this node")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            iid = uuid.uuid4().hex[:12]
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR", "LD_LIBRARY_PATH")}
            libs = self._lib_dirs()
            if libs:
                env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
            kinds = {a["index"]: a["kind"] for a in detect_accelerators()}
            if req.gpu is not None:
                # Each backend has its own device filter -- found live: a Vulkan build
                # ignores CUDA_VISIBLE_DEVICES and spreads the model over every GPU.
                # Vulkan numbers devices on its own -- see backend_device().
                if self.runtime_variant() == "vulkan":
                    env["GGML_VK_VISIBLE_DEVICES"] = self.backend_device(req.gpu)
                elif kinds.get(req.gpu) == "cuda":
                    env["CUDA_VISIBLE_DEVICES"] = str(req.gpu)
                elif kinds.get(req.gpu) == "rocm":
                    env["HIP_VISIBLE_DEVICES"] = str(req.gpu)
            argv = [self.cfg.llama_server, "-m", path, "--host", "127.0.0.1", "--port", str(port),
                    "-c", str(req.ctx_size), "-np", str(req.parallel), "-ngl", "999"] + req.flags()
            log = open(os.path.join(self.cfg.log_dir, f"llama-server-{iid}.log"), "ab")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=env, start_new_session=True)
            inst = {"id": iid, "model": req.model, "ctx_size": req.ctx_size, "parallel": req.parallel,
                    "gpu": req.gpu, "mode": req.mode, "pooling": req.pooling, "state": "loading", "started_at": time.time(), "port": port,
                    "proc": proc, "log": log}
            self.instances[iid] = inst
        threading.Thread(target=self._wait_ready, args=(inst,), daemon=True).start()
        return self.public(inst)

    def _wait_ready(self, inst: Dict[str, Any], timeout: float = 600) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and inst["proc"].poll() is None and inst["state"] == "loading":
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{inst['port']}/health", timeout=2) as r:
                    if r.status == 200:
                        inst["state"] = "ready"
                        return
            except OSError:
                pass
            time.sleep(0.5)
        if inst["state"] == "loading":
            inst["state"] = "failed"

    def install_runtime(self, req: InstallRequest, wait: bool = False) -> Dict[str, Any]:
        with self.lock:
            if self.install_state.get("state") == "installing":
                raise RequestError("an installation is already running")
            self.install_state = {"state": "installing", "tag": req.tag, "variant": req.variant,
                                  "done_bytes": 0, "total_bytes": sum(f.size for f in req.files)}
        t = threading.Thread(target=self._install, args=(req,), daemon=True)
        t.start()
        if wait:
            t.join()
        return dict(self.install_state)

    def _install(self, req: InstallRequest) -> None:
        import hashlib
        root = os.path.join(self.cfg.data_dir, "runtimes")
        final = os.path.join(root, f"{req.tag}-{req.variant}")
        work = final + ".part"
        try:
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work)
            for f in req.files:
                archive = os.path.join(work, f.name)
                h = hashlib.sha256()
                got = 0
                with urllib.request.urlopen(f.url, timeout=60) as r, open(archive, "wb") as out:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        got += len(chunk)
                        if got > f.size + (1 << 20):
                            raise RequestError(f"{f.name} is larger than announced")
                        h.update(chunk)
                        out.write(chunk)
                        self.install_state["done_bytes"] += len(chunk)
                if h.hexdigest() != f.sha256:
                    raise RequestError(f"sha256 mismatch for {f.name} -- not installed")
                safe_extract(archive, os.path.join(work, "files"))
                os.remove(archive)
            binary = None
            for dirpath, _, files in os.walk(os.path.join(work, "files")):
                for name in files:
                    if name in ("llama-server", "llama-server.exe"):
                        binary = os.path.join(dirpath, name)
            if not binary:
                raise RequestError("no llama-server in the archive")
            shutil.rmtree(final, ignore_errors=True)
            os.rename(work, final)
            binary = os.path.join(final, os.path.relpath(binary, work))
            os.chmod(binary, 0o755)
            self._activate(binary)
            self.install_state.update(state="installed", llama_server=binary, runtime=self.runtime_version())
        except Exception as e:  # reported via GET /v1/runtimes
            shutil.rmtree(work, ignore_errors=True)
            self.install_state.update(state="failed", error=f"{type(e).__name__}: {e}"[:400])

    def _activate(self, binary: str) -> None:
        from dataclasses import replace
        self.cfg = replace(self.cfg, llama_server=os.path.realpath(binary))
        self._runtime = None
        self._vk_devices = None
        self._persist({"llama_server": self.cfg.llama_server})

    def download_model(self, req: DownloadRequest, wait: bool = False) -> Dict[str, Any]:
        with self.lock:
            cur = self.downloads.get(req.name)
            if cur and cur.get("state") == "downloading":
                raise RequestError(f"{req.name} is already downloading")
            self.downloads[req.name] = {"state": "downloading", "done_bytes": 0, "total_bytes": req.size,
                                        "tried": [], "errors": []}
        t = threading.Thread(target=self._download, args=(req,), daemon=True)
        t.start()
        if wait:
            t.join()
        return dict(self.downloads[req.name])

    def _download_dir(self) -> str:
        return os.path.join(self.cfg.model_dirs[0], DOWNLOAD_SUBDIR)

    def list_downloads(self) -> List[Dict[str, Any]]:
        d = self._download_dir()
        out = []
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, f)
            if f.endswith(".gguf") and os.path.isfile(p):
                out.append({"name": f, "size_bytes": os.path.getsize(p), "last_used": os.path.getmtime(p)})
        return out

    def delete_download(self, name: str) -> None:
        """Only on the user's explicit request; only our own downloads; never a
        model a running instance uses."""
        if not isinstance(name, str) or not _MODEL_NAME_RE.match(name):
            raise RequestError("not a downloaded model name")
        path = os.path.join(self._download_dir(), name)
        if not os.path.isfile(path):
            raise RequestError(f"{name} is not a download of this node")
        real = os.path.realpath(path)
        for i in self.instances.values():
            if i["proc"].poll() is None:
                try:
                    if resolve_model(self.cfg, i["model"]) == real:
                        err = RequestError(f"{name} is in use by a running instance -- stop it first")
                        err.status = 409
                        raise err
                except RequestError as e:
                    if getattr(e, "status", 400) == 409:
                        raise
        os.remove(path)

    @staticmethod
    def _probe(url: str) -> float:
        """Bytes/s for the first PROBE_BYTES (0 = unusable)."""
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes=0-{PROBE_BYTES - 1}"})
            t0 = time.monotonic()
            with urllib.request.urlopen(r, timeout=30) as resp:
                n = len(resp.read(PROBE_BYTES))
            return n / max(time.monotonic() - t0, 1e-6)
        except OSError:
            return 0.0

    def _download(self, req: DownloadRequest) -> None:
        import hashlib
        st = self.downloads[req.name]
        try:
            directory = os.path.join(self.cfg.model_dirs[0], DOWNLOAD_SUBDIR)
            os.makedirs(directory, exist_ok=True)
            target = os.path.join(directory, req.name)
            quota = self.cfg.model_quota_mb * 1024 * 1024 if self.cfg.model_quota_mb else None
            check_room(directory, req.size, quota)
            if shutil.disk_usage(directory).free < req.size + (1 << 30):
                raise RequestError("not enough free disk space")
            speeds = {u: self._probe(u) for u in req.sources}
            order = [u for u in sorted(req.sources, key=lambda u: -speeds[u]) if speeds[u] > 0]
            st["speeds_mb_s"] = {u: round(v / 1048576, 1) for u, v in speeds.items()}
            if not order:
                raise RequestError("no source reachable")
            part = target + ".part"
            for url in order:
                st["tried"].append(url)
                st["done_bytes"] = 0
                h = hashlib.sha256()
                try:
                    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as out:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            st["done_bytes"] += len(chunk)
                            if st["done_bytes"] > req.size:
                                raise RequestError("larger than announced")
                            h.update(chunk)
                            out.write(chunk)
                except (OSError, RequestError) as e:
                    st["errors"].append(f"{url}: {e}")
                    continue
                if h.hexdigest() != req.sha256:
                    st["errors"].append(f"{url}: sha256 mismatch")
                    continue
                os.replace(part, target)
                st.update(state="done", source=url, path=target)
                return
            raise RequestError("all sources failed")
        except Exception as e:
            try:
                os.remove(os.path.join(self.cfg.model_dirs[0], DOWNLOAD_SUBDIR, req.name + ".part"))
            except OSError:
                pass
            st.update(state="failed", error=f"{type(e).__name__}: {e}"[:400])

    def _lib_dirs(self) -> List[str]:
        """Prebuilt runtimes ship their .so files next to the binary (and the CUDA
        runtime archive beside it) -- put those dirs on the library path."""
        root = os.path.join(os.path.realpath(self.cfg.data_dir), "runtimes")
        binary = self.cfg.llama_server or ""
        if not binary.startswith(root + os.sep):
            return []
        top = os.path.join(root, os.path.relpath(binary, root).split(os.sep)[0])
        return sorted({d for d, _, fs in os.walk(top) if any(".so" in f or f.endswith(".dylib") for f in fs)})

    def stop(self, iid: str) -> bool:
        with self.lock:
            inst = self.instances.pop(iid, None)
        if inst is None:
            return False
        proc = inst["proc"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        inst["log"].close()
        return True


def stop_all(agent: Agent) -> None:
    for iid in list(agent.instances):
        agent.stop(iid)


# --------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = f"llama-serve-node/{VERSION}"
    agent: Agent

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.client_address[0]} {fmt % args}\n")

    def _json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 64 * 1024 * 1024:
            raise RequestError("request body too large")
        return self.rfile.read(n) if n else b""

    def _authorized(self) -> bool:
        peer = self.client_address[0]
        net = self.agent.cfg.network
        coord_ip = self.agent.coordinator_ip()
        login = None
        if _peer_kind(peer, net.is_private) == "tailnet" and not (coord_ip and _ip(peer) == _ip(coord_ip)):
            login = self.agent.identify(peer)
        reason = refusal_reason(peer, self.headers.get("Authorization"), login, self.agent.cfg.token,
                                self.agent.cfg.coordinator_login, net.is_private, coord_ip)
        if reason:
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} refused {self.command} {self.path} from {peer}: {reason}\n")
            self._json(401, {"error": "unauthorized"})
        return reason is None

    def _route(self, method: str) -> None:
        if not self._authorized():
            return
        a = self.agent
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        try:
            if method == "GET" and parts == ["v1", "node"]:
                return self._json(200, a.node_info())
            if method == "GET" and parts == ["v1", "models"]:
                return self._json(200, {"models": list_models(a.cfg)})
            if parts == ["v1", "budgets"] and method == "PUT":
                try:
                    body = json.loads(self._body() or b"{}")
                except ValueError:
                    raise RequestError("invalid JSON")
                return self._json(200, {"budget_mb": a.set_budgets(body)})
            if parts == ["v1", "downloads"] and method == "GET":
                return self._json(200, {"downloads": a.downloads, "files": a.list_downloads()})
            if len(parts) == 3 and parts[:2] == ["v1", "downloads"] and method == "DELETE":
                a.delete_download(parts[2])
                return self._json(200, {"deleted": parts[2]})
            if parts == ["v1", "models", "download"] and method == "POST":
                try:
                    body = json.loads(self._body() or b"{}")
                except ValueError:
                    raise RequestError("invalid JSON")
                return self._json(202, a.download_model(DownloadRequest.from_dict(body, a.cfg.coordinator_url)))
            if parts == ["v1", "runtimes"] and method == "GET":
                return self._json(200, {"active": a.cfg.llama_server, "install": a.install_state})
            if parts == ["v1", "runtimes"] and method == "POST":
                try:
                    body = json.loads(self._body() or b"{}")
                except ValueError:
                    raise RequestError("invalid JSON")
                return self._json(202, a.install_runtime(InstallRequest.from_dict(body)))
            if parts[:2] == ["v1", "instances"]:
                if method == "GET" and len(parts) == 2:
                    return self._json(200, {"instances": [a.public(i) for i in a.instances.values()]})
                if method == "POST" and len(parts) == 2:
                    try:
                        body = json.loads(self._body() or b"{}")
                    except ValueError:
                        raise RequestError("invalid JSON")
                    req = StartRequest.from_dict(body, detect_accelerators())
                    return self._json(201, a.start(req))
                inst = a.instances.get(parts[2]) if len(parts) >= 3 else None
                if inst is None:
                    return self._json(404, {"error": "no such instance"})
                if len(parts) == 3 and method == "GET":
                    return self._json(200, a.public(inst))
                if len(parts) == 3 and method == "DELETE":
                    a.stop(parts[2])
                    return self._json(200, {"stopped": parts[2]})
                if len(parts) >= 4 and parts[3] == "proxy":
                    rest = self.path.split(f"/v1/instances/{parts[2]}/proxy/", 1)[-1]
                    return self._proxy(method, inst, rest)
            return self._json(404, {"error": "not found"})
        except RequestError as e:
            return self._json(e.status, {"error": str(e)})

    def _proxy(self, method: str, inst: Dict[str, Any], path: str) -> None:
        if not proxy_path_allowed(path):
            return self._json(403, {"error": "path not allowed through the agent"})
        if inst["state"] != "ready":
            return self._json(503, {"error": f"instance is {inst['state']}"})
        body = self._body() if method in ("POST", "PUT") else None
        conn = HTTPConnection("127.0.0.1", inst["port"], timeout=3600)
        try:
            conn.request(method, "/" + path, body=body,
                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k in ("Content-Type", "Content-Length"):
                if resp.getheader(k):
                    self.send_header(k, resp.getheader(k))
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError as e:
            try:
                self._json(502, {"error": f"llama-server unreachable: {e}"})
            except OSError:
                pass
        finally:
            conn.close()
            self.close_connection = True

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    def do_PUT(self):
        self._route("PUT")


def make_server(cfg: AgentConfig, config_path: Optional[str] = None) -> ThreadingHTTPServer:
    agent = Agent(cfg, config_path)
    handler = type("BoundHandler", (Handler,), {"agent": agent})
    server = QuickHTTPServer((cfg.listen, cfg.port), handler)
    server.daemon_threads = True
    server.agent = agent
    return server


def _tailscale_ip() -> Optional[str]:
    cli = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    out = _run([cli, "ip", "-4"]) if os.path.exists(cli) or shutil.which("tailscale") else None
    return out.strip().splitlines()[0] if out and out.strip() else None


def _post_json(url: str, body: Dict[str, Any]):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def enroll(coordinator: str, code: str, listen: str, name: str, model_dirs: List[str], config_path: str,
           port: int = 8040, whois=None, post=_post_json) -> Dict[str, Any]:
    """Register this device with a coordinator using a one-time code from its
    dashboard. The token is made here and never typed by anyone; the coordinator's
    identity is checked via whois of its address, independently of its answer."""
    if os.path.exists(config_path):
        raise ConfigError(f"{config_path} exists -- remove it to enroll again")
    from urllib.parse import urlsplit
    parts = urlsplit(coordinator)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError("coordinator must be http(s)://<host>:<port>")
    coord_ip = parts.hostname
    if _ip(coord_ip) is None:
        coord_ip = socket.gethostbyname(parts.hostname)
    token = secrets.token_urlsafe(32)
    status, answer = post(coordinator.rstrip("/") + "/api/cluster/enroll",
                          {"code": code, "node_name": name, "token": token,
                           "agent_url": f"http://{listen}:{port}"})
    if status != 201:
        raise ConfigError(f"enrollment refused ({status}): {answer.get('detail') or answer}")
    claimed = answer.get("coordinator_login")
    actual = whois(coord_ip) if whois else None
    if not claimed or not actual or claimed.strip().lower() != actual.strip().lower():
        raise ConfigError(f"coordinator identity mismatch: it claims {claimed!r}, whois says {actual!r}")
    for d in model_dirs:
        os.makedirs(d, exist_ok=True)
    cfg = {"node_name": name, "listen": listen, "port": port, "token": token,
           "coordinator_login": actual, "coordinator_url": coordinator.rstrip("/"),
           "llama_server": None, "model_dirs": model_dirs}
    AgentConfig.from_dict(cfg)  # refuse to write something the agent wouldn't start with
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    return {"node_id": answer.get("node_id"), "config": config_path, "coordinator_login": actual}


SERVICE_LABEL = "net.llama-serve.node"
SYSTEMD_UNIT = "llama-serve-node.service"


def _xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def service_plan(platform_: str, python: str, script: str, config: str, home: str, uid: int) -> Dict[str, Any]:
    """Files to write and commands to run so the agent starts with the device and
    restarts if it dies -- per user, never as root."""
    if platform_ == "darwin":
        path = os.path.join(home, "Library", "LaunchAgents", f"{SERVICE_LABEL}.plist")
        logs = os.path.join(home, "Library", "Logs", "llama-serve-node.log")
        args = "".join(f"    <string>{_xml(a)}</string>\n" for a in (python, script, "--config", config))
        content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{SERVICE_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>{_xml(logs)}</string>
  <key>StandardErrorPath</key><string>{_xml(logs)}</string>
</dict>
</plist>
"""
        target = f"gui/{uid}"
        return {"files": [(path, content)],
                # A loaded copy is unloaded first and waited for: found live, re-installing
                # over a loaded agent failed with "Bootstrap failed: 5" (old script kept
                # running), and bootout is asynchronous (bootstrapping right after it
                # failed the same way and left nothing running).
                "unload": ["launchctl", "bootout", f"{target}/{SERVICE_LABEL}"],
                "loaded": ["launchctl", "print", f"{target}/{SERVICE_LABEL}"],
                "start": [["launchctl", "bootstrap", target, path]],
                "stop": [["launchctl", "bootout", f"{target}/{SERVICE_LABEL}"]],
                "hint": f"Starts at login; log: {logs}"}
    if platform_ == "android":
        path = os.path.join(home, ".termux", "boot", "llama-serve-node")
        content = (f'#!/data/data/com.termux/files/usr/bin/sh\ntermux-wake-lock\n'
                   f'exec "{python}" "{script}" --config "{config}" >> "{home}/llama-serve-node.log" 2>&1\n')
        return {"files": [(path, content)], "start": [], "stop": [],
                "hint": "Install the Termux:Boot app and open it once; the agent then starts after every boot. "
                        "Keep the phone on power and Wi-Fi."}
    path = os.path.join(home, ".config", "systemd", "user", SYSTEMD_UNIT)
    content = f"""[Unit]
Description=llama-serve node agent
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
ExecStart={python} "{script}" --config "{config}"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""
    return {"files": [(path, content)],
            "start": [["systemctl", "--user", "daemon-reload"],
                      ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT]],
            "stop": [["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT]],
            "hint": "Runs while you're logged in. To keep it running without a login session "
                    "(e.g. a headless Pi): loginctl enable-linger $USER"}


def install_service(plan: Dict[str, Any], run=None, sleep=time.sleep) -> None:
    quiet = run or (lambda cmd: subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode)
    run = run or (lambda cmd: subprocess.run(cmd).returncode)
    if plan.get("unload"):
        quiet(plan["unload"])  # fails harmlessly when nothing is loaded
        for _ in range(20):
            if quiet(plan["loaded"]) != 0:
                break
            sleep(0.5)
    for path, content in plan["files"]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755 if path.endswith("llama-serve-node") else 0o644)
    for cmd in plan["start"]:
        run(cmd)


def uninstall_service(plan: Dict[str, Any], run=None) -> None:
    run = run or (lambda cmd: subprocess.run(cmd).returncode)
    for cmd in plan["stop"]:
        run(cmd)
    for path, _ in plan["files"]:
        if os.path.exists(path):
            os.remove(path)


def _init_config(path: str) -> None:
    if os.path.exists(path):
        sys.exit(f"{path} exists -- not overwriting")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    template = {
        "node_name": socket.gethostname().split(".")[0],
        "listen": "<this device's Tailscale IP, see: tailscale ip -4>",
        "port": 8040,
        "token": secrets.token_urlsafe(32),
        "coordinator_login": "<the coordinator's Tailscale login, e.g. you@github>",
        "llama_server": "<absolute path to llama-server>",
        "model_dirs": [os.path.expanduser("~/models")],
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(template, f, indent=2)
    print(f"Wrote {path} -- fill in the <...> fields. Token (give it to the coordinator):\n{template['token']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default = os.path.expanduser("~/.config/llama-serve-node/config.json")
    p.add_argument("--config", default=default)
    p.add_argument("--init", action="store_true", help="write a config template with a fresh token")
    p.add_argument("--enroll", metavar="COORDINATOR", help="register with a coordinator, e.g. http://my-pc:8030")
    p.add_argument("--code", help="one-time code from the coordinator's dashboard")
    p.add_argument("--listen", help="this device's overlay address (default: tailscale ip -4)")
    p.add_argument("--name", default=socket.gethostname().split(".")[0])
    p.add_argument("--port", type=int, default=8040)
    p.add_argument("--model-dir", action="append",
                   help="where models live / get downloaded (default ~/llama-serve-node/models)")
    p.add_argument("--install-service", action="store_true",
                   help="start the agent with this device and restart it if it stops (per user, no root)")
    p.add_argument("--uninstall-service", action="store_true", help="undo --install-service")
    args = p.parse_args()
    if args.install_service or args.uninstall_service:
        plat = detect_facts()["platform"]
        plan = service_plan(plat, sys.executable, os.path.abspath(sys.argv[0]), os.path.abspath(args.config),
                            os.path.expanduser("~"), os.getuid())
        if args.uninstall_service:
            uninstall_service(plan)
            print("Service removed.")
            return
        try:  # refuse to install a setup that wouldn't start
            with open(args.config) as f:
                AgentConfig.from_dict(json.load(f))
        except (OSError, ValueError) as e:
            sys.exit(f"Not installing the service -- the config doesn't work: {e}")
        install_service(plan)
        print(f"Installed: {plan['files'][0][0]}\n{plan['hint']}")
        return
    if args.init:
        return _init_config(args.config)
    if args.enroll:
        if not args.code:
            sys.exit("--code is required")
        listen = args.listen or _tailscale_ip()
        if not listen:
            sys.exit("could not find this device's Tailscale address -- pass --listen")
        dirs = [os.path.abspath(os.path.expanduser(d)) for d in (args.model_dir or ["~/llama-serve-node/models"])]
        holder = Agent.__new__(Agent)
        holder.cfg = type("C", (), {"tailscale_cli": None})()
        holder._whois = {}
        try:
            r = enroll(args.enroll, args.code, listen, args.name, dirs, args.config, args.port,
                       whois=lambda ip: Agent.whois(holder, ip))
        except ConfigError as e:
            sys.exit(f"Enrollment failed: {e}")
        me = os.path.abspath(sys.argv[0])
        print(f"Enrolled as '{r['node_id']}' with {r['coordinator_login']}. Config: {r['config']}\n"
              f"Start it with this device (recommended):  python3 {me} --install-service\n"
              f"Or run it in this terminal:               python3 {me}")
        return
    if os.stat(args.config).st_mode & 0o077:
        sys.exit(f"{args.config} contains the token -- chmod 600 it first")
    with open(args.config) as f:
        cfg = AgentConfig.from_dict(json.load(f))
    try:
        server = make_server(cfg, args.config)
    except OSError as e:
        if e.errno in (48, 98):  # EADDRINUSE on macOS / Linux
            sys.exit(f"Port {cfg.port} on {cfg.listen} is already in use -- the agent is probably already "
                     f"running (as a service?). Restart that one instead: macOS "
                     f"'launchctl kickstart -k gui/$(id -u)/{SERVICE_LABEL}', Linux "
                     f"'systemctl --user restart {SYSTEMD_UNIT}'.")
        raise
    signal.signal(signal.SIGTERM, lambda *_: (stop_all(server.agent), sys.exit(0)))
    print(f"llama-serve node '{cfg.node_name}' on {cfg.listen}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    finally:
        stop_all(server.agent)


if __name__ == "__main__":
    main()
