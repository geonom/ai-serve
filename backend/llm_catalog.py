"""GET /v1/llm/models: every language model the gateway can reach, for apps to pick from (owner, 2026-10-20:
the Spielwiese should offer everything that runs through the gateway; it listed only local files, without
Strata or device-only models, and put embedding models into the chat list). Pure: the gateway passes in
what it found (local files with their header traits, running instances, device snapshots, llm plugins' models).

Vocabulary:
- kind   chat | embedding | rerank | base (a model without a chat template: completion-only, e.g. YuE music)
         -- from the GGUF header (pooling type, classifier head, encoder architecture, chat template); for a
         file that only lives on a device, from its name (`kind_source: "name"`).
- where  the places a model can run: local (this machine), device (an enrolled node, via the gateway's
         routing), plugin (an llm plugin's own server, e.g. Strata; `exclusive` when it owns every GPU while it
         runs).
Tests: tests/unit/test_llm_catalog.py."""
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_ENCODER_ARCHS = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "xlm-roberta",
                  "modern-bert", "neo-bert", "t5encoder"}
_QUANT = re.compile(r"[-._]((?:I?Q\d+(?:_[A-Z0-9]+)*)|(?:[BF]F?16)|F32|MXFP4(?:_MOE)?)$", re.IGNORECASE)


def classify(metadata: Optional[Mapping[str, Any]], name: str) -> str:
    lname = name.lower()
    if metadata is None:
        if "rerank" in lname:
            return "rerank"
        if "embed" in lname or re.search(r"(^|[-_.])(bge|e5|gte)([-_.]|$)", lname):
            return "embedding"
        return "chat"
    arch = metadata.get("general.architecture") or ""
    pooling = metadata.get(f"{arch}.pooling_type")
    if pooling == 4 or f"{arch}.classifier.output_labels" in metadata or "rerank" in lname:
        return "rerank"
    if pooling in (1, 2, 3) or arch in _ENCODER_ARCHS:
        return "embedding"
    if "tokenizer.chat_template" not in metadata:
        return "base"
    return "chat"


def traits(metadata: Mapping[str, Any], name: str) -> Dict[str, Any]:
    template = metadata.get("tokenizer.chat_template") or ""
    template = template if isinstance(template, str) else ""
    return {"kind": classify(metadata, name), "arch": metadata.get("general.architecture"),
            "thinking": "enable_thinking" in template or "<think>" in template or "reasoning" in template}


def pretty_label(name: str) -> Tuple[str, Optional[str]]:
    stem = name[:-5] if name.lower().endswith(".gguf") else name
    m = _QUANT.search(stem)
    if not m:
        return stem, None
    return stem[:m.start()], m.group(1).upper()


def _row(name: str) -> Dict[str, Any]:
    label, quant = pretty_label(name)
    return {"id": name, "label": label, "quant": quant, "kind": "chat", "kind_source": "name", "arch": None,
            "size_mb": None, "context_length": None, "vision": False, "mtp": False, "thinking": False,
            "exclusive": False, "where": [], "loaded": False}


def build_catalog(local: Iterable[Mapping[str, Any]], loaded_local: Mapping[str, Mapping[str, Any]],
                  devices: Iterable[Mapping[str, Any]], plugins: Iterable[Mapping[str, Any]] = (),
                  plugin_states: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for m in local:
        r = rows.setdefault(m["name"], _row(m["name"]))
        t = m.get("traits") or {}
        r.update(kind=t.get("kind") or classify(None, m["name"]), kind_source="header" if t else "name",
                 arch=t.get("arch"), thinking=bool(t.get("thinking")), size_mb=m.get("size_mb"),
                 context_length=m.get("context_length"), vision=bool(m.get("vision")), mtp=bool(m.get("mtp")))
        inst = loaded_local.get(m["name"])
        place = {"place": "local", "loaded": inst is not None}
        if inst is not None:
            place.update(gpu_id=inst.get("gpu_id"), ctx_size=inst.get("ctx_size"))
        r["where"].append(place)
    for d in devices:
        ready = {i.get("model") for i in d.get("instances") or [] if i.get("state") == "ready"}
        for name, size in (d.get("models") or {}).items():
            r = rows.get(name)
            if r is None:
                r = rows[name] = _row(name)
                r.update(kind=classify(None, name), size_mb=size or None)
            r["where"].append({"place": "device", "node": d["id"], "online": bool(d.get("online")),
                               "loaded": name in ready})
    for e in plugins:   # a plugin's id never takes over a local file's row (those are .gguf names)
        if e["id"] in rows:
            continue
        ps = (plugin_states or {}).get(e["plugin"]) or {}
        state = ps.get("state") or "stopped"
        r = rows[e["id"]] = _row(e["id"])
        r.update(label=e.get("label") or e["id"], quant=None, kind=e.get("kind") or "chat", kind_source="plugin",
                 thinking=bool(e.get("thinking")), vision=bool(e.get("vision")), context_length=e.get("context_length"),
                 exclusive=bool(ps.get("exclusive")), aliases=list(e.get("aliases") or []),
                 description=e.get("description") or "", license=e.get("license") or None,
                 commercial_use=e.get("commercial_use"),
                 where=[{"place": "plugin", "plugin": e["plugin"], "name": ps.get("name") or e["plugin"], "state": state,
                         "loaded": state == "ready"}])
    for r in rows.values():
        r["loaded"] = any(p.get("loaded") for p in r["where"])
    order = {"chat": 0, "base": 1, "embedding": 2, "rerank": 3}
    models = sorted(rows.values(), key=lambda r: (order.get(r["kind"], 9), not r["exclusive"], r["label"].lower(), r["id"]))
    return {"models": models}
