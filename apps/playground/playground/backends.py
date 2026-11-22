"""What the Playground talks to -- pure rules, no I/O (tests tests/test_backends.py).

Normally the ai-serve gateway; but the owner wants it to work with a *standard* OpenAI-compatible server too
(Ollama, LM Studio, llama-server -- 2026-10-21: "dass z.B. die Playground auch mit standard Ollama oder LM Studio
funktionieren würde"). Such a server has only the standard API (/v1/models, /v1/chat/completions, /v1/embeddings),
so the gateway's catalogs are rebuilt from /v1/models here, and what needs ai-serve (images, speech, documents)
reports itself unavailable with a reason instead of failing.

Vocabulary: kind ai-serve | openai | down."""
from __future__ import annotations

import re
from typing import Any, Dict, List

REASON = ("The connected server is a standard OpenAI server (e.g. Ollama or LM Studio) – it doesn't offer this; "
          "that needs ai-serve.")
EMPTY: Dict[str, Any] = {
    "/v1/audio/models": {"tts": {"available": False, "reason": REASON, "engines": []},
                         "stt": {"available": False, "reason": REASON, "engines": []}},
    "/v1/images/models": {"object": "list", "data": []},
}
FEATURES = {
    "ai-serve": {"chat": True, "texts": True, "images": True, "speech": True, "documents": True},
    "openai": {"chat": True, "texts": True, "images": False, "speech": False, "documents": False},
    "down": {"chat": False, "texts": False, "images": False, "speech": False, "documents": False},
}
def upstream_kind(previous, catalog_status: int, models_status) -> str:
    """What the upstream is, from the status of GET /v1/llm/models (ai-serve's catalog) and, if asked, of GET
    /v1/models; 0 = no answer in time. A known ai-serve that doesn't answer in time is still ai-serve -- slow is not
    a different server (2026-11-22: after a restart and while the backend stalled, the Playground said "gateway not
    running" or switched images and speech off)."""
    if catalog_status == 200:
        return "ai-serve"
    if catalog_status == 0 and previous == "ai-serve":
        return "ai-serve"
    return "openai" if models_status == 200 else "down"


def cache_seconds(catalog_status: int) -> int:
    """A sure answer is kept 30 s; a guess after no answer only 5 s."""
    return 30 if catalog_status != 0 else 5


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,159}$")


def _kind(name: str) -> str:
    n = name.lower()
    return "rerank" if "rerank" in n else "embedding" if "embed" in n else "chat"


def llm_catalog(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """A standard /v1/models answer -> the gateway's /v1/llm/models shape (only what the Playground reads)."""
    items: List[Any] = []
    if isinstance(raw, dict):
        items = raw.get("data") if isinstance(raw.get("data"), list) else raw.get("models") if isinstance(raw.get("models"), list) else []
    rows, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = it.get("id") or it.get("model") or it.get("name")
        if not isinstance(mid, str) or not _ID.match(mid) or mid in seen:
            continue
        seen.add(mid)
        ctx = (it.get("meta") or {}).get("n_ctx_train") if isinstance(it.get("meta"), dict) else None
        rows.append({"id": mid, "label": mid, "quant": None, "kind": _kind(mid), "kind_source": "name",
                     "context_length": ctx if type(ctx) is int else None, "vision": False, "mtp": False,
                     "thinking": False, "exclusive": False, "where": [{"place": "local", "loaded": False}],
                     "loaded": False})
    return {"models": rows}
