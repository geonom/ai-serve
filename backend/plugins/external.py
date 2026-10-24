"""External model servers: a standard OpenAI-compatible server the owner already runs -- Ollama, LM Studio,
llama-server, vLLM ... -- added to ai-serve by port, without an adapter app (owner, 2026-10-21: the Playground and
every extension should work with standard Ollama or LM Studio). They become `llm` plugins (backend/plugins/llm.py):
their models are listed and routed like any plugin's.

Rules (pure part; tests tests/unit/test_external_servers.py): loopback only -- an entry has a port, never a host, so
the gateway can't be pointed elsewhere; not this gateway's port; plain ids; the OpenAI path ends in /v1. Never
started or stopped from here (the owner runs them; Ollama and LM Studio manage their own memory) and no GPU of
their own in the GPU manager's books -- their VRAM counts as "other programs". Stored in MODEL_SERVERS_FILE
(gitignored)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from backend.config import MODEL_SERVERS_FILE as FILE
from backend.plugins.contracts import ContractError, Provision, _path

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_FIELDS = {"id", "name", "port", "openai"}
from backend.config import GATEWAY_PORT  # noqa: E402
_RESERVED_PORTS = {8030, GATEWAY_PORT}


@dataclass(frozen=True)
class ExternalServer:
    id: str
    name: str
    port: int
    openai: str = "/v1"
    external: bool = field(default=True, repr=False)

    @property
    def provision(self) -> Provision:
        return Provision("llm", {"openai": self.openai}, "none")

    # what the plugin registry reads from a manifest
    @property
    def provides(self) -> Dict[str, Provision]:
        return {"llm": self.provision}

    health_path = None
    match: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "port": self.port, "openai": self.openai}


def parse_server(raw: Any) -> ExternalServer:
    if not isinstance(raw, dict):
        raise ContractError("a server is an object {id, name?, port, openai?}")
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ContractError(f"unknown field(s) {', '.join(unknown)} -- only loopback servers, by port")
    sid, port = raw.get("id"), raw.get("port")
    if not isinstance(sid, str) or not _ID.match(sid):
        raise ContractError("id: lowercase letters, digits and '-'")
    if type(port) is not int or not 1024 <= port <= 65535 or port in _RESERVED_PORTS:
        raise ContractError("port: 1024-65535, not this gateway's")
    name = raw.get("name", sid)
    if not isinstance(name, str) or not 0 < len(name.strip()) <= 60:
        raise ContractError("name: up to 60 characters")
    openai = _path(raw.get("openai", "/v1"), "openai")
    if not openai.endswith("/v1"):
        raise ContractError("openai: the path of the OpenAI-compatible API, ending in /v1")
    return ExternalServer(sid, name.strip(), port, openai)


def parse_file(raw: Any) -> Tuple[List[ExternalServer], List[str]]:
    items = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return [], ["model servers file: {\"servers\": [...]}"] if raw is not None else []
    out, errors, ids = [], [], set()
    for i, item in enumerate(items):
        try:
            s = parse_server(item)
            if s.id in ids:
                raise ContractError(f"duplicate id {s.id!r}")
            ids.add(s.id)
            out.append(s)
        except ContractError as e:
            errors.append(f"server {i}: {e}")
    return out, errors


# Standard ports of well-known servers, to offer them in the dashboard when something answers there.
KNOWN = (
    ExternalServer("ollama", "Ollama", 11434),
    ExternalServer("lmstudio", "LM Studio", 1234),
)


def load(path: str) -> Tuple[List[ExternalServer], List[str]]:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return [], []
    except (OSError, ValueError) as e:
        return [], [f"{path}: {e}"]
    return parse_file(raw)


def save(path: str, servers: List[ExternalServer]) -> None:
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"servers": [s.to_dict() for s in servers]}, f, indent=2)
    os.replace(tmp, path)


def merged(servers: List[ExternalServer], new: Mapping[str, Any]) -> List[ExternalServer]:
    s = parse_server(dict(new))
    if any(x.port == s.port and x.id != s.id for x in servers):
        raise ContractError(f"port {s.port} is already {next(x.name for x in servers if x.port == s.port)}")
    return [x for x in servers if x.id != s.id] + [s]
