"""What the Playground forwards to the ai-serve-ui gateway -- pure rules, no I/O.

Threat model: this server listens on 127.0.0.1 and forwards the browser's calls to the
gateway on 127.0.0.1:8030, which trusts loopback. Without rules it would hand anyone
who reaches the Playground (a tailnet user, once the dashboard exposes it) the
gateway's whole management API. So:
- only what a playground needs is forwarded: chat, images, speech, OCR and the
  read-only lists they use -- never start/stop, services, cluster, training, keys;
- the real client goes along (X-Forwarded-For = the peer seen here; the gateway honours
  it only from 127.0.0.1), so the gateway's own access policy still decides who may;
- client-supplied forwarding headers are dropped; Tailscale identity headers pass only
  when the request came over loopback (Tailscale Serve sets them there and strips
  them from clients);
- paths with traversal, encoded separators or odd characters are refused; bodies are
  capped (MAX_BODY_BYTES).

Vocabulary: route(method, path) -> the path to forward, or Refused;
forward_headers(headers, peer) -> the headers to send upstream.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping

MAX_BODY_BYTES = 200 * 1024 * 1024  # STT accepts 100 MB of audio, sent as base64

_SEG = r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}"   # one path segment, no dots-only, no '/'
_RULES = [
    ("GET", r"/api/models"),
    ("GET", r"/api/voices"),
    ("GET", r"/api/status"),
    ("GET", r"/api/comfyui/loras"),
    ("GET", r"/api/comfyui/models"),
    ("GET", r"/api/comfyui/workflows"),
    ("GET", rf"/api/comfyui/task/{_SEG}"),
    ("GET", r"/api/comfyui/batch/status"),
    ("GET", rf"/api/comfyui/images/{_SEG}"),
    ("POST", r"/api/chat/completions"),
    ("POST", r"/v1/chat/completions"),
    ("POST", r"/api/tokenize"),
    ("POST", r"/api/comfyui/generate"),
    ("POST", r"/api/comfyui/edit"),
    ("POST", r"/api/comfyui/style-transfer"),
    ("POST", r"/api/comfyui/batch"),
    ("POST", r"/api/comfyui/batch/(?:pause|resume|cancel)"),
    ("POST", r"/api/tts/generate"),
    ("POST", r"/api/stt/transcribe"),
    ("POST", r"/api/ocr"),
    ("POST", r"/v1/images/generations"),   # MOSS-Image (also with positions, layout.js), Iris, catalog ids
    ("GET", r"/v1/images/models"),         # the gateway's image model catalog (Bild tab's model list)
    ("POST", r"/v1/embeddings"),           # "Texte vergleichen": similarity of texts
    ("POST", r"/v1/rerank"),               # "Texte vergleichen": rank documents for a question
    ("GET", r"/v1/models"),                # the standard OpenAI model list (a plain Ollama / LM Studio upstream)
    ("GET", r"/v1/llm/models"),            # the gateway's language models: here, on devices, Strata (Chat tab)
    ("GET", r"/v1/audio/models"),          # the gateway's speech engines with their modes and options (Sprache tab)
    ("POST", r"/v1/images/cancel"),        # stop one running image by its request id (Vergleich's cancel)
    ("GET", r"/api/plugins/installs"),     # what can be downloaded (missing models / engines) -- shown to everyone
    ("POST", rf"/api/plugins/{_SEG}/install"),   # start a download: the owner only (quota.classify -> owner_only)
    ("POST", rf"/api/plugins/{_SEG}/start"),     # start a plugin a page needs (only plugins, not apps): owner only
]
_COMPILED = [(m, re.compile(p + r"\Z")) for m, p in _RULES]
_QUERY = re.compile(r"\A[A-Za-z0-9._~=&%+-]*\Z")

# Hop-by-hop and anything that would let a client pose as someone else upstream.
_DROP = frozenset({
    "host", "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer",
    "upgrade", "proxy-authorization", "cookie", "forwarded", "x-forwarded-for", "x-forwarded-host",
    "x-forwarded-proto", "x-forwarded-port", "x-real-ip", "content-length",
    "origin", "referer",   # the gateway refuses foreign pages; this is a server-side call (see test_policy.py)
})
_LOOPBACK = frozenset({"127.0.0.1", "::1"})


class Refused(Exception):
    status = 403


def route(method: str, path: str) -> str:
    """The path to forward (query kept), or Refused."""
    target, _, query = path.partition("?")
    if query and not _QUERY.match(query):
        raise Refused("query not allowed")
    if ".." in target or "%" in target or "\\" in target:
        raise Refused("path not allowed")
    for m, rx in _COMPILED:
        if m == method and rx.match(target):
            return path
    raise Refused(f"{method} {target} is not part of the Playground")


def forward_headers(headers: Mapping[str, str], peer: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        low = k.lower()
        if low in _DROP:
            continue
        if low.startswith("tailscale-") and peer not in _LOOPBACK:
            continue
        out[k] = v
    out["X-Forwarded-For"] = peer
    return out
