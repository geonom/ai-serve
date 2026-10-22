"""What the Spielwiese talks to -- pure rules, no I/O (tests tests/test_backends.py).

Normally the ai-serve gateway; but the owner wants it to work with a *standard* OpenAI-compatible server too
(Ollama, LM Studio, llama-server -- 2026-10-21: "dass z.B. die Spielwiese auch mit standard Ollama oder LM Studio
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
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,159}$")


def _kind(name: str) -> str:
    n = name.lower()
    return "rerank" if "rerank" in n else "embedding" if "embed" in n else "chat"


def llm_catalog(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """A standard /v1/models answer -> the gateway's /v1/llm/models shape (only what the Spielwiese reads)."""
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
