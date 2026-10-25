"""Security domain of the public API (/public/v1) -- pure rules, no FastAPI, no I/O.

Threat model: a client on the internet that reaches this backend only through a tunnel
(Tailscale Funnel / Cloudflare Tunnel) and may or may not hold the public key. Assets:
the internal API (/api, /v1 -- start/stop/train/unload, arbitrary model paths), the
GPUs (time + VRAM), local files and paths, the reference voice clips in voices/, and
other machines reachable from here (SSRF via media URLs).

Ubiquitous language:
- PublicApiKey     the one secret a remote client presents (Bearer); too weak = disabled.
- ExposureGuard    decides whether a request came through a tunnel ("external") and,
                   if so, whether its path is inside the public surface.
- ModelCatalog     the only models that exist publicly: ModelAlias id -> target.
- *Policy          turns an untrusted request body into a safe command for one kind of
                   work (chat / speech / image / transcription) or raises PolicyViolation.
- AdmissionControl how much GPU work remote clients may have in flight at once.
- ErrorPolicy      what an external client may learn from a failure (no paths, no
                   internals -- a reference id points to the server log instead).
"""
from __future__ import annotations

import hmac
import posixpath
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple


# --------------------------------------------------------------------------- errors

class PolicyViolation(Exception):
    """A request the public API refuses. `message` is safe to show the caller."""
    status = 400
    error_type = "invalid_request_error"

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status


class Unauthorized(PolicyViolation):
    status = 401
    error_type = "authentication_error"


class ApiDisabled(PolicyViolation):
    status = 503
    error_type = "service_unavailable"


class UnknownModel(PolicyViolation):
    # 400, not 404: the endpoint exists, the request names a model it doesn't
    # serve. Nextcloud's OpenAI app probes endpoints with model "invalid-model"
    # and reads 404 as "endpoint missing", which would disable images/TTS there.
    status = 400
    error_type = "model_not_found"


class PayloadTooLarge(PolicyViolation):
    status = 413


class TooManyRequests(PolicyViolation):
    status = 429
    error_type = "rate_limit_error"


# --------------------------------------------------------------------------- key

class PublicApiKey:
    """The shared secret. A secret shorter than MIN_LENGTH is treated as no secret at
    all (API disabled) -- failing closed beats protecting the GPUs with a weak key."""
    MIN_LENGTH = 32

    __slots__ = ("_secret",)

    def __init__(self, secret: str):
        self._secret = secret.encode()

    @classmethod
    def from_secret(cls, raw: Optional[str]) -> Optional["PublicApiKey"]:
        raw = (raw or "").strip()
        if len(raw) < cls.MIN_LENGTH:
            return None
        return cls(raw)

    def verify(self, authorization_header: Optional[str]) -> bool:
        header = authorization_header or ""
        if not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:].encode(), self._secret)

    def __repr__(self) -> str:
        return "PublicApiKey(***)"

    __str__ = __repr__


def authenticate(key: Optional[PublicApiKey], authorization_header: Optional[str]) -> None:
    if key is None:
        raise ApiDisabled("The public API is disabled on this server.")
    if not key.verify(authorization_header):
        raise Unauthorized("Invalid or missing API key.")


# --------------------------------------------------------------------------- exposure

class ExposureGuard:
    """Tunnels mark the requests they forward: Tailscale Funnel sets
    `Tailscale-Funnel-Request: ?1` and strips any client-supplied copy (ipn/ipnlocal/
    serve.go), Cloudflare Tunnel sets `Cf-Connecting-Ip`. A marked request may only
    reach the public surface -- even if the tunnel itself is misconfigured to forward
    more. Unmarked (local / LAN / tailnet) requests are not this guard's business."""

    DEFAULT_MARKERS = ("tailscale-funnel-request", "cf-connecting-ip")
    PUBLIC_PREFIX = "/public/"

    def __init__(self, markers: Iterable[str] = DEFAULT_MARKERS):
        self._markers = frozenset(m.lower() for m in markers)

    def is_external(self, headers: Mapping[str, str]) -> bool:
        return any(k.lower() in self._markers for k in headers)

    def permits(self, path: str, headers: Mapping[str, str]) -> bool:
        if not self.is_external(headers):
            return True
        return self.is_public_path(path)

    @classmethod
    def is_public_path(cls, path: str) -> bool:
        if "\\" in path or "%" in path:
            return False
        segments = path.split("/")
        if any(s in (".", "..") for s in segments):
            return False
        normalized = posixpath.normpath(path)
        return path.startswith(cls.PUBLIC_PREFIX) and (normalized + "/").startswith(cls.PUBLIC_PREFIX) \
            and normalized != cls.PUBLIC_PREFIX.rstrip("/")


# --------------------------------------------------------------------------- catalog

class ModelKind(str, Enum):
    CHAT = "chat"
    STT = "stt"
    TTS = "tts"
    IMAGE = "image"


_ALIAS_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ModelAlias:
    id: str
    kind: ModelKind
    target: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    voices: FrozenSet[str] = frozenset()

    @classmethod
    def from_dict(cls, d: Any) -> "ModelAlias":
        if not isinstance(d, dict):
            raise ValueError(f"model entry must be an object, got {type(d).__name__}")
        alias_id = d.get("id")
        if not isinstance(alias_id, str) or not _ALIAS_ID.match(alias_id):
            raise ValueError(f"invalid alias id {alias_id!r}")
        try:
            kind = ModelKind(d.get("kind"))
        except ValueError:
            raise ValueError(f"{alias_id}: kind must be one of {[k.value for k in ModelKind]}")
        target = d.get("target", alias_id)
        if (not isinstance(target, str) or not target or "/" in target or "\\" in target
                or target.startswith(".") or ".." in target):
            raise ValueError(f"{alias_id}: target must be a bare model/engine name, got {target!r}")
        defaults = d.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise ValueError(f"{alias_id}: defaults must be an object")
        voices = d.get("voices") or []
        if not isinstance(voices, list) or not all(isinstance(v, str) and v for v in voices):
            raise ValueError(f"{alias_id}: voices must be a list of names")
        return cls(id=alias_id, kind=kind, target=target, defaults=dict(defaults), voices=frozenset(voices))


class ModelCatalog:
    def __init__(self, aliases: List[ModelAlias]):
        self.aliases = list(aliases)

    @classmethod
    def from_dict(cls, data: Any) -> "ModelCatalog":
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ValueError("catalog needs a 'models' list")
        aliases = [ModelAlias.from_dict(e) for e in entries]
        ids = [a.id for a in aliases]
        if len(ids) != len(set(ids)):
            raise ValueError("alias ids must be unique")
        return cls(aliases)

    def resolve(self, model: Optional[str], kind: ModelKind) -> ModelAlias:
        candidates = [a for a in self.aliases if a.kind == kind]
        if not candidates:
            raise UnknownModel(f"No {kind.value} model is available.")
        if not model:
            return candidates[0]
        for a in candidates:
            if a.id == model:
                return a
        shown = model if isinstance(model, str) and _ALIAS_ID.match(model) else "(invalid name)"
        raise UnknownModel(f"Model '{shown}' is not available for {kind.value}. "
                           f"Available: {', '.join(a.id for a in candidates)}")


# --------------------------------------------------------------------------- chat

class ChatPolicy:
    """Only plain OpenAI chat fields get through. Everything else -- placement/lifetime
    knobs (gpu_id, ctx_size, keep_alive, ...), llama-server internals (id_slot, lora,
    n_predict, grammar, samplers, ...) and audio output (would clone voices) -- is
    dropped; an alias's `defaults` are the only way to set them. Media must be inline
    (data: URLs): a remote URL would make this machine fetch it (SSRF)."""

    MAX_TOKENS = 8192
    MAX_MESSAGES = 500
    ALLOWED_FIELDS = frozenset({
        "messages", "temperature", "top_p", "max_tokens", "stream", "stream_options", "stop", "seed",
        "presence_penalty", "frequency_penalty", "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "user", "thinking",
    })
    ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
    PART_TYPES = frozenset({"text", "image_url", "input_audio"})

    def sanitize(self, body: Mapping[str, Any], alias: ModelAlias) -> Dict[str, Any]:
        if not isinstance(body, Mapping):
            raise PolicyViolation("Request body must be a JSON object.")
        self._check_messages(body.get("messages"))
        n = body.get("n", 1)
        if n not in (None, 1):
            raise PolicyViolation("Only n=1 is supported.")

        out: Dict[str, Any] = {k: v for k, v in body.items() if k in self.ALLOWED_FIELDS}
        if "max_tokens" not in out and body.get("max_completion_tokens") is not None:
            out["max_tokens"] = body["max_completion_tokens"]
        if "max_tokens" in out:
            mt = out["max_tokens"]
            if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
                raise PolicyViolation("max_tokens must be a positive integer.")
            out["max_tokens"] = min(mt, self.MAX_TOKENS)
        ctk = body.get("chat_template_kwargs")
        if isinstance(ctk, Mapping) and isinstance(ctk.get("enable_thinking"), bool):
            out["chat_template_kwargs"] = {"enable_thinking": ctk["enable_thinking"]}

        for k, v in alias.defaults.items():
            out.setdefault(k, v)
        out["model"] = alias.target
        return out

    def _check_messages(self, messages: Any) -> None:
        if not isinstance(messages, list) or not messages:
            raise PolicyViolation("messages must be a non-empty list.")
        if len(messages) > self.MAX_MESSAGES:
            raise PolicyViolation(f"At most {self.MAX_MESSAGES} messages are allowed.")
        for m in messages:
            if not isinstance(m, Mapping) or m.get("role") not in self.ROLES:
                raise PolicyViolation(f"Each message needs a role out of {sorted(self.ROLES)}.")
            content = m.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise PolicyViolation("Message content must be a string or a list of parts.")
            for part in content:
                self._check_part(part)

    def _check_part(self, part: Any) -> None:
        if not isinstance(part, Mapping) or part.get("type") not in self.PART_TYPES:
            raise PolicyViolation(f"Content part type must be one of {sorted(self.PART_TYPES)}.")
        if part["type"] == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else image
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise PolicyViolation("Images must be sent inline as data:image/... URLs.")


# --------------------------------------------------------------------------- speech

_LANGUAGE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


@dataclass(frozen=True)
class SpeechCommand:
    engine: str
    text: str
    voice: str
    response_format: str
    speed: float
    language: Optional[str]


class SpeechPolicy:
    """voices/ holds local clips of real people: a caller may only pick voices on the
    alias's allowlist; anything else gets the engine's (synthetic) default voice."""

    MAX_INPUT_CHARS = 4096
    DEFAULT_VOICE = "alloy"  # an OpenAI voice name = "no voice given" in the gateway
    FORMATS = frozenset({"mp3", "opus", "aac", "flac", "wav", "pcm"})

    def sanitize(self, body: Mapping[str, Any], alias: ModelAlias) -> SpeechCommand:
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise PolicyViolation("input must be a non-empty string.")
        if len(text) > self.MAX_INPUT_CHARS:
            raise PolicyViolation(f"input is limited to {self.MAX_INPUT_CHARS} characters.")
        fmt = body.get("response_format") or "mp3"
        if fmt not in self.FORMATS:
            raise PolicyViolation(f"response_format must be one of {sorted(self.FORMATS)}.")
        speed = body.get("speed", 1.0)
        if speed is None:
            speed = 1.0
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            raise PolicyViolation("speed must be a number.")
        language = body.get("language") or alias.defaults.get("language")
        if language is not None and (not isinstance(language, str) or not _LANGUAGE.match(language)):
            raise PolicyViolation("language must be a language code like 'de'.")
        voice = body.get("voice")
        voice = voice if isinstance(voice, str) and voice in alias.voices else self.DEFAULT_VOICE
        return SpeechCommand(engine=alias.target, text=text, voice=voice, response_format=fmt,
                             speed=min(max(float(speed), 0.5), 2.0), language=language)


# --------------------------------------------------------------------------- image

@dataclass(frozen=True)
class ImageCommand:
    model_name: str
    prompt: str
    n: int
    base_resolution: int
    aspect_ratio: str
    steps: Optional[int]      # None = the model's own default


class ImagePolicy:
    MAX_PROMPT_CHARS = 4000
    MAX_N = 4
    SIZES: Mapping[str, Tuple[int, str]] = {
        "auto": (1024, "1:1"), "256x256": (512, "1:1"), "512x512": (512, "1:1"), "768x768": (768, "1:1"),
        "1024x1024": (1024, "1:1"),
        "1024x768": (1024, "4:3"), "4:3": (1024, "4:3"), "768x1024": (1024, "3:4"), "3:4": (1024, "3:4"),
        "1280x720": (1024, "16:9"), "1792x1024": (1024, "16:9"), "1536x1024": (1024, "16:9"), "16:9": (1024, "16:9"),
        "720x1280": (1024, "9:16"), "1024x1792": (1024, "9:16"), "1024x1536": (1024, "9:16"), "9:16": (1024, "9:16"),
    }

    def sanitize(self, body: Mapping[str, Any], alias: ModelAlias) -> ImageCommand:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PolicyViolation("prompt must be a non-empty string.")
        if len(prompt) > self.MAX_PROMPT_CHARS:
            raise PolicyViolation(f"prompt is limited to {self.MAX_PROMPT_CHARS} characters.")
        n = body.get("n", 1)
        n = 1 if n is None else n
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= self.MAX_N:
            raise PolicyViolation(f"n must be an integer between 1 and {self.MAX_N}.")
        size = body.get("size") or "1024x1024"
        if size not in self.SIZES:
            raise PolicyViolation(f"size must be one of {', '.join(self.SIZES)}.")
        base_res, aspect = self.SIZES[size]
        steps = alias.defaults.get("steps")
        return ImageCommand(model_name=alias.target, prompt=prompt, n=n, base_resolution=base_res,
                            aspect_ratio=aspect, steps=steps)


# --------------------------------------------------------------------------- transcription

@dataclass(frozen=True)
class TranscriptionCommand:
    language: Optional[str]
    response_format: str


class TranscriptionPolicy:
    MAX_UPLOAD_BYTES = 100 * 1024 * 1024
    FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})

    def sanitize(self, language: Optional[str], response_format: Optional[str],
                 size_bytes: int) -> TranscriptionCommand:
        if size_bytes <= 0:
            raise PolicyViolation("file is empty.")
        if size_bytes > self.MAX_UPLOAD_BYTES:
            raise PayloadTooLarge(f"file is limited to {self.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        fmt = response_format or "json"
        if fmt not in self.FORMATS:
            raise PolicyViolation(f"response_format must be one of {sorted(self.FORMATS)}.")
        language = language or None
        if language is not None and not _LANGUAGE.match(language):
            raise PolicyViolation("language must be a language code like 'de'.")
        return TranscriptionCommand(language=language, response_format=fmt)


# --------------------------------------------------------------------------- limits / admission

class BodyLimits:
    DEFAULT = 25 * 1024 * 1024  # inline images/audio in chat are base64
    UPLOAD = TranscriptionPolicy.MAX_UPLOAD_BYTES + 1024 * 1024  # + multipart overhead

    def for_path(self, path: str) -> int:
        return self.UPLOAD if path.rstrip("/").endswith("/audio/transcriptions") else self.DEFAULT


class AdmissionControl:
    """Caps remote GPU work in flight. Single event loop -> a plain counter suffices."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max(1, int(max_concurrent))
        self.in_flight = 0

    def acquire(self) -> None:
        if self.in_flight >= self.max_concurrent:
            raise TooManyRequests("Too many requests in progress, try again shortly.")
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


# --------------------------------------------------------------------------- errors -> public

class ErrorPolicy:
    """Domain refusals keep their (authored, safe) message. Everything raised further
    in -- model manager, TTS workers, ComfyUI, ffmpeg -- can carry paths, PIDs and
    library dumps, so only a generic message by status class leaves, plus a reference
    the server log carries next to the real detail."""

    _GENERIC = {
        400: ("The request could not be processed.", "invalid_request_error"),
        404: ("Not found.", "not_found"),
        413: ("Request too large.", "invalid_request_error"),
        422: ("The request could not be processed.", "invalid_request_error"),
        429: ("Too many requests, try again shortly.", "rate_limit_error"),
        503: ("The server is busy, try again later.", "service_unavailable"),
        504: ("The request timed out.", "timeout"),
        500: ("Internal error.", "server_error"),
        502: ("The model backend failed to process the request.", "server_error"),
    }
    _BUSY = frozenset({409, 503, 507})

    def to_public(self, exc: BaseException, ref: str) -> Tuple[int, Dict[str, Any]]:
        if isinstance(exc, PolicyViolation):
            return exc.status, self._body(exc.message, exc.error_type, ref)
        return self.to_public_http(500, None, ref)

    def to_public_http(self, status: int, detail: Any, ref: str) -> Tuple[int, Dict[str, Any]]:
        if status in self._BUSY:
            status = 503
        elif status not in self._GENERIC:
            status = 500 if status >= 500 or status < 400 else 400
        message, error_type = self._GENERIC[status]
        return status, self._body(message, error_type, ref)

    @staticmethod
    def _body(message: str, error_type: str, ref: str) -> Dict[str, Any]:
        return {"error": {"message": message, "type": error_type, "reference": ref}}
