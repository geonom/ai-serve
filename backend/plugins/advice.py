"""Model advice from `advice` plugins (docs/plugins.md; 2026-10-25, owner: "eine Schnittstelle über ai-serve, über die
du angefragt werden kannst, welche Modelle nach Parametern wie Geschwindigkeit, Zieldomäne, etc. besonders geeignet
wären (auch RAM-Verbrauch)"; first plugin: llm-benchy, which ranks from its own measurements). Three ways in, one
answer underneath:
- apps:   GET /v1/llm/advice/domains, POST /v1/llm/advice/recommend (structured; routers/advice.py)
- people: POST /v1/llm/advice/ask -- a question in words; a chat model turns it into a recommend request and words
          the answer from the ranked rows only (the rows come back too, so every claim can be checked)
- agents: /v1/llm/advice/mcp -- an MCP server (streamable HTTP, JSON answers) with read-only tools

What the gateway adds to a plugin's ranking: which models this machine can actually serve (`where`, `loaded` from
/v1/llm/models) and, by default, only those (`installed_only`). Nothing here knows a plugin by name.
Tests: tests/unit/test_advice_plugins.py."""
from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from backend.plugins import registry
from backend.plugins.contracts import ContractError, merge_catalogs, normalize_catalog, normalize_recommendation

CATALOG_TTL_S = 30.0
RECOMMEND_TIMEOUT_S = 120.0          # the first ranking after a plugin start pools the whole benchmark DB (~15 s)
MCP_PROTOCOL = "2025-06-18"
_cache: Tuple[float, List[Dict[str, Any]], List[str]] = (0.0, [], [])

# what a recommend request may carry (llm-benchy's advice.parse_request has the exact rules) -- also the MCP tool's
# input schema and the shape the ask planner fills
REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domain": {"type": "string", "description": "a domain id from the domains list (use case or dimension)"},
        "weights": {"type": "object", "additionalProperties": {"type": "number"},
                    "description": "instead of domain: {dimension: weight > 0}"},
        "language": {"type": "string", "enum": ["de", "en", "fr"],
                     "description": "score only trials in this prompt language"},
        "ctx": {"type": "integer", "description": "context size the model must run at (default 8192)"},
        "gpus": {"type": "array", "items": {"type": "string"},
                 "description": "GPU names or parts of them (e.g. \"4060 Ti\") speed and memory should refer to"},
        "min_gen_tps": {"type": "number", "description": "minimum generation speed, tokens/s"},
        "min_prefill_tps": {"type": "number", "description": "minimum prompt processing speed, tokens/s"},
        "max_vram_mb": {"type": "integer", "description": "most GPU memory the model may need, MB"},
        "models": {"type": "array", "items": {"type": "string"}, "description": "only these model ids"},
        "include_thin": {"type": "boolean", "description": "also models with too little benchmark data"},
        "installed_only": {"type": "boolean", "description": "only models this machine can serve (default true)"},
        "limit": {"type": "integer", "description": "how many models (default 10, max 100)"},
    },
    "additionalProperties": False,
}


class AdviceError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ catalog and ranking

async def _ask_catalog(p: registry.Plugin) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not await registry.ensure_running(p, 60.0):
        return [], [f"{p.id}: not reachable"]
    try:
        r = await registry.call(p, "catalog", timeout=RECOMMEND_TIMEOUT_S)
        if r.status_code != 200:
            return [], [f"{p.id}: catalog answered {r.status_code}"]
        return normalize_catalog("advice", p.id, r.json())
    except (httpx.HTTPError, ValueError) as e:
        return [], [f"{p.id}: {type(e).__name__}"]


async def catalog(fresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    global _cache
    plugins = registry.providers("advice")
    if not plugins:
        return [], []
    at, entries, errors = _cache
    if not fresh and entries and time.monotonic() - at < CATALOG_TTL_S:
        return entries, errors
    parts, errors = [], []
    for p in plugins:
        e, errs = await _ask_catalog(p)
        parts.append((p.id, e))
        errors += errs
    entries, clashes = merge_catalogs(parts)
    errors += [f"{c['ignored']}: domain {c['id']!r} already belongs to {c['kept']}" for c in clashes]
    _cache = (time.monotonic(), entries, errors)
    return entries, errors


def choose(plugins: Sequence[registry.Plugin], entries: Sequence[Dict[str, Any]],
           domain: Optional[str]) -> registry.Plugin:
    """The plugin to ask: the one whose catalog lists the domain, else (custom weights) the first."""
    if not plugins:
        raise AdviceError("no advice plugin -- install one (e.g. llm-benchy) under Apps & Services", 503)
    if domain:
        owner = next((e["plugin"] for e in entries if e["id"] == domain), None)
        if owner is None:
            raise AdviceError(f"unknown domain {domain!r} -- GET /v1/llm/advice/domains lists them")
        return next(p for p in plugins if p.id == owner)
    return plugins[0]


def servable(llm_catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """/v1/llm/models (`models`: [{id, kind, where: [{place, loaded, ...}], loaded}]) -> {id lowercased: {where,
    loaded}} for chat models something here can serve."""
    out = {}
    for m in llm_catalog.get("models") or []:
        if m.get("kind", "chat") == "chat" and m.get("where"):
            out[str(m["id"]).lower()] = {"where": list(m["where"]), "loaded": bool(m.get("loaded"))}
    return out


def plugin_request(body: Dict[str, Any], served: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """The caller's request as the plugin gets it: `installed_only` (the gateway's own option) becomes a `models`
    list of what this machine serves -- intersected with the caller's own list when it gave one."""
    req = {k: v for k, v in body.items() if k != "installed_only"}
    if served is not None:
        if isinstance(req.get("models"), list):
            req["models"] = [m for m in req["models"] if isinstance(m, str) and m.lower() in served]
        else:
            req["models"] = sorted(served)
    return req


def annotate(rec: Dict[str, Any], served: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for m in rec["models"]:
        here = served.get(m["id"].lower())
        m["where"] = here["where"] if here else []
        m["loaded"] = bool(here and here["loaded"])
    return rec


async def recommend(body: Any, llm_catalog: Callable[[], Awaitable[Dict[str, Any]]]) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise AdviceError("the request is a JSON object")
    unknown = sorted(set(body) - set(REQUEST_SCHEMA["properties"]))
    if unknown:
        raise AdviceError(f"unknown field(s) {', '.join(unknown)} (known: {', '.join(REQUEST_SCHEMA['properties'])})")
    entries, _ = await catalog()
    p = choose(registry.providers("advice"), entries, body.get("domain"))
    served = servable(await llm_catalog())
    req = plugin_request(body, served if body.get("installed_only", True) is not False else None)
    if not await registry.ensure_running(p, 60.0):
        raise AdviceError(f"{p.name} is not reachable -- see Apps & Services, logs/service-{p.id}.log", 503)
    try:
        r = await registry.call(p, "recommend", "POST", timeout=RECOMMEND_TIMEOUT_S, json=req)
    except httpx.HTTPError as e:
        raise AdviceError(f"{p.name}: {type(e).__name__}", 503)
    try:
        data = r.json()
    except ValueError:
        data = None
    if r.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        raise AdviceError(detail if isinstance(detail, str) else f"{p.name} answered {r.status_code}",
                          400 if r.status_code in (400, 422) else 502)
    try:
        rec, errors = normalize_recommendation(p.id, data)
    except ContractError as e:
        raise AdviceError(str(e), 502)
    if errors:
        rec["errors"] = errors
    return annotate(rec, served)


# ------------------------------------------------------------------ a question in words

def plan_messages(question: str, domains: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    listing = "\n".join(f"- {d['id']}: {d['label']} (dimensions: {', '.join(d.get('dimensions') or [])}; measured "
                        f"in: {', '.join(d.get('languages') or []) or '?'})" for d in domains)
    return [{"role": "system", "content":
             "You turn a question about which local language model to use into a JSON request for a model "
             "ranking. Fill only what the question says or clearly implies; leave everything else null.\n"
             f"Domains (pick the closest id, or null when none fits):\n{listing}\n"
             "Prefer a domain measured in the language the task needs; a domain without it ranks nothing. "
             "language: the language the model must work in (de, en, fr), not the language of the question "
             "unless the task is about it. ctx: context tokens the task needs. gpus: GPU names the question "
             "names. min_gen_tps: tokens/s for writing. max_vram_mb: GPU memory limit in MB (16 GB = 16000)."},
            {"role": "user", "content": question}]


def plan_schema(domain_ids: Sequence[str]) -> Dict[str, Any]:
    def nullable(schema):
        return {"anyOf": [schema, {"type": "null"}]}
    fields = {"domain": nullable({"type": "string", "enum": list(domain_ids)}),
              "language": nullable({"type": "string", "enum": ["de", "en", "fr"]}),
              "ctx": nullable({"type": "integer"}), "gpus": nullable({"type": "array", "items": {"type": "string"}}),
              "min_gen_tps": nullable({"type": "number"}), "max_vram_mb": nullable({"type": "integer"})}
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def plan_request(raw: Any, domain_ids: Sequence[str], limit: int = 5) -> Dict[str, Any]:
    """The planner's JSON -> a recommend request (null fields dropped; nonsense dropped rather than sent on)."""
    if not isinstance(raw, dict):
        raise AdviceError("the model didn't answer with a request", 502)
    req: Dict[str, Any] = {"limit": limit}
    if raw.get("domain") in domain_ids:
        req["domain"] = raw["domain"]
    if raw.get("language") in ("de", "en", "fr"):
        req["language"] = raw["language"]
    for key, lo, hi in (("ctx", 512, 2_000_000), ("min_gen_tps", 0.1, 20_000), ("max_vram_mb", 256, 10_000_000)):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
            req[key] = int(v) if key != "min_gen_tps" else float(v)
    gpus = raw.get("gpus")
    if isinstance(gpus, list) and gpus and all(isinstance(g, str) and 0 < len(g) <= 60 for g in gpus):
        req["gpus"] = gpus[:4]
    if "domain" not in req:
        raise AdviceError("the question names no task the ranking knows -- GET /v1/llm/advice/domains lists them")
    return req


def facts(rec: Dict[str, Any], top: int = 5, excluded: int = 5) -> Dict[str, Any]:
    """The rows the answer may use -- smaller than the full ranking so a small model reads all of it."""
    def speed(m):
        return [{k: s[k] for k in ("gpu", "ctx", "gen_tps", "prefill_tps")} for s in m["speed"][:3]]
    return {"domain": rec["domain"]["label"] or rec["domain"]["id"], "language": rec["language"], "ctx": rec["ctx"],
            "models": [{"id": m["id"], "rank": m["rank"], "tier": m["tier"], "score": m["score"],
                        "speed": speed(m), "vram_mb": (m["memory"] or {}).get("vram_mb"),
                        "vram_gpu": (m["memory"] or {}).get("gpu"), "warnings": m["warnings"],
                        "loaded": m.get("loaded", False)} for m in rec["models"][:top]],
            "left_out": [{"id": e["id"], "reason": e["reason"]} for e in rec["excluded"][:excluded]],
            "notes": rec["notes"]}


def answer_messages(question: str, rec_facts: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "system", "content":
             "Answer the question with the benchmark ranking below. Use only these facts -- no other models, no "
             "numbers that aren't there. Reply in the language of the question, 80-200 words, plain sentences: "
             "the best fit first with its score, speed (tokens/s on which GPU) and GPU memory, then alternatives. "
             "Models in the same tier are about equally good. Mention a warning when it matters, and say so when "
             "nothing fits.\n\nRanking (JSON):\n" + json.dumps(rec_facts, ensure_ascii=False)},
            {"role": "user", "content": question}]


# ------------------------------------------------------------------ MCP (agents)

def mcp_tools() -> List[Dict[str, Any]]:
    read_only = {"readOnlyHint": True, "openWorldHint": False}
    return [
        {"name": "list_model_domains", "title": "Domains the model advice can rank for",
         "description": "Use cases (agentic, creative_writing, translation, ...) and benchmark dimensions with the "
                        "prompt languages measured. Call this first to pick a domain id.",
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
         "annotations": read_only, "risk": "read"},
        {"name": "recommend_models", "title": "Which local models suit a task",
         "description": "Ranks the language models this machine can serve for a domain from benchmark "
                        "measurements: score (0..1, same tier = about equally good), measured speed per GPU, "
                        "measured peak GPU memory, and why other models were left out.",
         "inputSchema": REQUEST_SCHEMA, "annotations": read_only, "risk": "read"},
    ]


def _rpc_result(rid: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _rpc_error(rid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _tool_result(data: Any, is_error: bool = False) -> Dict[str, Any]:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    out: Dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if isinstance(data, dict):
        out["structuredContent"] = data
    return out


async def mcp_handle(msg: Any, llm_catalog: Callable[[], Awaitable[Dict[str, Any]]],
                     version: str) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message -> its response (None for a notification). Tool failures are tool results with
    isError (the agent reads why), protocol mistakes are JSON-RPC errors."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _rpc_error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    method, rid = msg["method"], msg.get("id")
    if "id" not in msg:
        return None                              # notifications/initialized, cancelled, ...
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    if method == "initialize":
        wanted = params.get("protocolVersion")
        return _rpc_result(rid, {"protocolVersion": wanted if isinstance(wanted, str) else MCP_PROTOCOL,
                                 "capabilities": {"tools": {"listChanged": False}},
                                 "serverInfo": {"name": "ai-serve-model-advice", "version": version},
                                 "instructions": "Ask which local language model fits a task: list_model_domains, "
                                                 "then recommend_models."})
    if method == "ping":
        return _rpc_result(rid, {})
    if method == "tools/list":
        return _rpc_result(rid, {"tools": mcp_tools()})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        try:
            if name == "list_model_domains":
                entries, errors = await catalog()
                if not entries:
                    raise AdviceError("no advice plugin answered" + (f": {'; '.join(errors)}" if errors else ""), 503)
                return _rpc_result(rid, _tool_result({"domains": [
                    {k: e[k] for k in ("id", "kind", "label", "description", "dimensions", "languages")}
                    for e in entries]}))
            if name == "recommend_models":
                return _rpc_result(rid, _tool_result(await recommend(args, llm_catalog)))
        except AdviceError as e:
            return _rpc_result(rid, _tool_result(str(e), is_error=True))
        return _rpc_error(rid, -32602, f"unknown tool {name!r}")
    return _rpc_error(rid, -32601, f"method not found: {method}")
