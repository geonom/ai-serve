"""Pure rules of the Strata adapter (no I/O): what the gateway learns about Strata without loading it, and how
requests are adjusted for it. Tests: tests/test_rules.py."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

MODEL_ID = "strata"
_ALIAS = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def max_context(config: Mapping[str, Any]) -> Optional[int]:
    """--max-context from the engine config's args (Strata answers no more than that)."""
    args = config.get("args") or []
    for i, a in enumerate(args):
        if a == "--max-context" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except (TypeError, ValueError):
                return None
    return None


def catalog(config: Mapping[str, Any], extra_aliases: List[str]) -> Dict[str, Any]:
    """The plugin contract's llm catalog (GET /v1/models), from the config alone -- loading Strata takes minutes."""
    name = config.get("model_name") if isinstance(config.get("model_name"), str) else None
    aliases = [a for a in dict.fromkeys(([name] if name else []) + extra_aliases) if a and _ALIAS.match(a)]
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "label": "Strata" + (f" ({name})" if name else ""),
        "description": "Large MoE model spread over both graphics cards; loading takes 4–10 minutes.",
        "kind": "chat", "context_length": max_context(config), "thinking": True, "aliases": aliases[:10]}]}


def upstream_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """A chat request as Strata wants it: its own model name, and no repeat_penalty for structured output --
    Strata validates json_schema after generation, and repeat_penalty pushes it away from repeating the schema's
    own keys/quotes (StoryTime's experience)."""
    out = dict(body)
    out.pop("model", None)
    if out.get("response_format"):
        out.pop("repeat_penalty", None)
    return out


def state_of(alive: bool, health: Optional[Mapping[str, Any]]) -> str:
    if not alive and health is None:
        return "stopped"
    return "ready" if health and health.get("loaded") else "loading"
