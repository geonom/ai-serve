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

import json
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
    ("POST", r"/v1/images/upscale"),       # enlarge a picture 2x / 4x (Real-ESRGAN, JSON base64)
    ("POST", r"/v1/images/edits"),         # multipart: task=remove_background (cut-out); quota "image"
    ("POST", r"/v1/images/cancel"),        # stop one running image by its request id (Vergleich's cancel)
    ("GET", r"/api/plugins/installs"),     # what can be downloaded (missing models / engines) -- shown to everyone
    ("POST", rf"/api/plugins/{_SEG}/install"),   # start a download: the owner only (quota.classify -> owner_only)
    ("POST", rf"/api/plugins/{_SEG}/start"),     # start a plugin a page needs (only plugins, not apps): owner only
    ("GET", r"/v1/realtime"),              # voice conversation (Omni): a WebSocket, see wants_websocket()
    ("POST", r"/v1/audio/speech"),         # Talk's short spoken confirmations in the speech model's voice
    ("GET", r"/v1/video/models"),          # video models (Video tab, 2026-11-13)
    ("POST", r"/v1/video/jobs"),           # start a video (quota: charged VIDEO_JOB_ESTIMATE_S per job)
    ("GET", rf"/v1/video/jobs/{_SEG}"),    # its status (job ids are <plugin>.<job>)
    ("POST", rf"/v1/video/jobs/{_SEG}/cancel"),
    ("GET", rf"/v1/video/jobs/{_SEG}/files/{_SEG}"),   # the finished MP4
    ("GET", r"/v1/video/prompt/options"),  # camera/look settings of the prompt writer
    ("POST", r"/v1/video/prompt"),         # write a detailed prompt from an idea (a chat call: quota "chat")
]
# the only paths a WebSocket upgrade may go to -- after it the connection is a byte tunnel to the gateway
_WEBSOCKET_PATHS = frozenset({"/v1/realtime"})
_WEBSOCKET_KEEP = ("sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions", "sec-websocket-protocol")
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


def wants_websocket(path: str, headers: Mapping[str, str]) -> bool:
    """True for a WebSocket upgrade to an allowed path; an upgrade anywhere else is Refused (it would become a raw
    tunnel past the per-request rules)."""
    low = {k.lower(): v for k, v in headers.items()}
    if "websocket" not in low.get("upgrade", "").lower():
        return False
    if path.partition("?")[0] not in _WEBSOCKET_PATHS:
        raise Refused("WebSocket not allowed here")
    return True


def websocket_headers(headers: Mapping[str, str], peer: str) -> Dict[str, str]:
    """forward_headers() plus the handshake fields the gateway needs for the upgrade."""
    out = forward_headers(headers, peer)
    out = {k: v for k, v in out.items() if k.lower() not in _WEBSOCKET_KEEP}
    for k, v in headers.items():
        if k.lower() in _WEBSOCKET_KEEP:
            out[k] = v
    out["Upgrade"] = "websocket"
    out["Connection"] = "Upgrade"
    return out


# fields only the owner may send, per path: files saved by name share one folder and are read back by name
# (2026-11-26); the voices/ folder holds recordings of real people, and a transcript is written next to the clip
# (2026-11-27)
_OWNER_FIELDS = (
    (re.compile(r"^/api/comfyui/(generate|edit|style-transfer)$"), ("filename",)),
    (re.compile(r"^/api/tts/generate$"), ("ref_voice_path",)),
    (re.compile(r"^/api/stt/transcribe$"), ("audio_path",)),
)


def guest_body(method: str, path: str, body):
    """A guest's request body as forwarded, without the fields only the owner may send (_OWNER_FIELDS). Anything
    else, or a body that isn't a JSON object, goes unchanged."""
    if method != "POST" or not body:
        return body
    fields = next((f for rx, f in _OWNER_FIELDS if rx.match(path)), None)
    if not fields:
        return body
    try:
        data = json.loads(body)
    except ValueError:
        return body
    if not isinstance(data, dict) or not any(k in data for k in fields):
        return body
    for k in fields:
        data.pop(k, None)
    return json.dumps(data).encode()
