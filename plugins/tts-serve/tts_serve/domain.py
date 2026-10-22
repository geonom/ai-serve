"""tts-serve domain -- pure rules (no processes, no sockets, no GPU).

Threat model: tts-serve listens on 127.0.0.1 and starts worker processes described
by provider manifests in its own providers/ directory -- that directory is the trust
anchor. A request must never make it run something else, read an arbitrary file
(reference audio comes in as base64, never as a path), or hand unchecked data to a
process as arguments.

Ubiquitous language:
- ProviderManifest  providers/<id>/llama-serve.provider.json: how to start a TTS
                    engine's worker, its GPU/VRAM needs, keep-alive, variants, aliases.
- ProviderRegistry  every provider directory with a valid manifest.
- Mode              what one variant of an engine needs (reference, transcript, style text or tags,
                    preset speaker) and which numeric params it takes -- the engine's self-description,
                    so clients build their forms from /v1/engines instead of knowing every engine.
- TtsRequest        the one request shape all engines share; each provider's
                    adapter.py turns it into its worker's own payload.
- eviction_order    which idle engines to unload, least recently used first, when a
                    cold start needs room on a GPU.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MANIFEST_NAME = "llama-serve.provider.json"
VRAM_MARGIN_MB = 512
FRACTION_EXTRA_MB = 500  # a fraction-reserving engine (nano-vllm) needs a bit on top

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_LANGUAGE = re.compile(r"^[A-Za-z]{2,3}([-_][A-Za-z]{2,4})?$|^[A-Z][a-z]{2,15}$|^auto$|^Auto$")
_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_FORBIDDEN_ENV = frozenset({
    "PATH", "BASH_ENV", "ENV", "PROMPT_COMMAND", "SHELLOPTS", "BASHOPTS", "IFS", "CDPATH",
    "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME", "PERL5OPT", "PERL5LIB", "NODE_OPTIONS", "RUBYOPT",
})
_FIELDS = ("id", "name", "description", "capability", "command", "env", "gpu", "vram_mb", "vram_fraction",
           "startup_timeout_s", "keep_alive_s", "concurrent", "aliases", "variants", "variant_swaps",
           "default_reference", "license", "setup", "modes", "languages")
_SEVERITY = ["error", "starting", "loading", "generating", "idle", "unloaded"]


class ProviderError(ValueError):
    pass


class RequestError(Exception):
    status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnknownEngine(RequestError):
    status = 404


# --------------------------------------------------------------------------- manifest

LEGACY_PARAMS = frozenset({"temperature", "exaggeration", "cfg_weight"})  # engines without `modes`
INPUT_KINDS = ("reference", "ref_text", "instruct", "speaker")
_PARAM_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_LABEL_MAX = 200
_MODE_FIELDS = {"variant", "label", "description", "inputs", "speakers", "instruct", "params"}


def _text(v: Any, what: str, required: bool = False, limit: int = _LABEL_MAX) -> str:
    if v is None and not required:
        return ""
    if not isinstance(v, str) or (required and not v.strip()) or len(v) > limit:
        raise ProviderError(f"{what} must be a string (max {limit} characters)")
    return v.strip()


def _number(v: Any, what: str) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ProviderError(f"{what} must be a number")
    return float(v)


def _parse_param(p: Any) -> Dict[str, Any]:
    if not isinstance(p, dict) or not set(p) <= {"name", "label", "min", "max", "step", "default"}:
        raise ProviderError("a param is {name, label?, min, max, step?, default?}")
    name = p.get("name")
    if not isinstance(name, str) or not _PARAM_NAME.match(name):
        raise ProviderError(f"param name {name!r} must be lowercase letters, digits, _")
    lo, hi = _number(p.get("min"), f"param {name} min"), _number(p.get("max"), f"param {name} max")
    if lo >= hi:
        raise ProviderError(f"param {name}: min must be below max")
    out = {"name": name, "label": _text(p.get("label"), f"param {name} label") or name, "min": lo, "max": hi}
    if p.get("step") is not None:
        step = _number(p["step"], f"param {name} step")
        if not 0 < step <= hi - lo:
            raise ProviderError(f"param {name}: step out of range")
        out["step"] = step
    if p.get("default") is not None:
        default = _number(p["default"], f"param {name} default")
        if not lo <= default <= hi:
            raise ProviderError(f"param {name}: default outside min..max")
        out["default"] = default
    return out


def _parse_instruct(v: Any) -> Dict[str, Any]:
    if not isinstance(v, dict) or v.get("kind") not in ("text", "tags") or not set(v) <= {"kind", "hint", "groups", "examples"}:
        raise ProviderError("instruct is {kind: text|tags, hint?, groups? (tags), examples?}")
    out: Dict[str, Any] = {"kind": v["kind"], "hint": _text(v.get("hint"), "instruct hint", limit=500)}
    if v["kind"] == "tags":
        groups = v.get("groups")
        if (not isinstance(groups, list) or not groups or not all(
                isinstance(g, list) and g and all(isinstance(t, str) and 0 < len(t) <= 40 for t in g) for g in groups)):
            raise ProviderError("instruct tags need groups: a list of lists of short strings")
        out["groups"] = [list(g) for g in groups]
    examples = v.get("examples") or []
    if not isinstance(examples, list) or not all(isinstance(e, str) and len(e) <= 300 for e in examples):
        raise ProviderError("instruct examples must be a list of short strings")
    out["examples"] = list(examples)
    return out


def _parse_modes(raw: Any, variants: Sequence[str]) -> Tuple[Dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ProviderError("modes must be a non-empty list")
    out, seen = [], set()
    for d in raw:
        if not isinstance(d, dict) or not set(d) <= _MODE_FIELDS:
            raise ProviderError(f"a mode takes only {', '.join(sorted(_MODE_FIELDS))}")
        variant = d.get("variant")
        if variants and variant not in variants:
            raise ProviderError(f"mode variant {variant!r} is not one of the manifest's variants")
        if not variants and variant is not None:
            raise ProviderError("an engine without variants has one mode with variant null")
        if variant in seen:
            raise ProviderError(f"two modes for variant {variant!r}")
        seen.add(variant)
        inputs = d.get("inputs") or {}
        if not isinstance(inputs, dict) or any(k not in INPUT_KINDS or v not in ("required", "optional")
                                               for k, v in inputs.items()):
            raise ProviderError(f"mode inputs map {'/'.join(INPUT_KINDS)} to required|optional")
        mode: Dict[str, Any] = {"variant": variant, "label": _text(d.get("label"), "mode label", required=True),
                                "description": _text(d.get("description"), "mode description", limit=500),
                                "inputs": dict(inputs)}
        speakers = d.get("speakers") or []
        if not isinstance(speakers, list) or not all(isinstance(x, str) and _TOKEN.match(x) for x in speakers):
            raise ProviderError("speakers must be a list of simple names")
        if "speaker" in inputs and not speakers:
            raise ProviderError("a mode with a speaker input needs its speakers list")
        mode["speakers"] = list(speakers)
        if d.get("instruct") is not None:
            mode["instruct"] = _parse_instruct(d["instruct"])
        params = d.get("params") or []
        if not isinstance(params, list):
            raise ProviderError("params must be a list")
        mode["params"] = [_parse_param(p) for p in params]
        out.append(mode)
    return tuple(out)


def _parse_languages(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(x, str) and _LANGUAGE.match(x) for x in raw):
        raise ProviderError("languages must be a list of language codes")
    return tuple(raw)


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.normpath(path)).relative_to(os.path.normpath(root))
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ProviderManifest:
    id: str
    name: str
    dir: Path
    command: Tuple[str, ...]
    env: Mapping[str, str]
    gpu: int
    vram_mb: Optional[int]
    vram_fraction: Optional[float]
    startup_timeout_s: float
    keep_alive_s: int
    concurrent: bool
    aliases: Tuple[str, ...]
    variants: Tuple[str, ...]
    variant_swaps: bool
    default_reference: bool
    description: str = ""
    license: str = ""
    setup: str = ""
    modes: Tuple[Dict[str, Any], ...] = ()
    languages: Tuple[str, ...] = ()

    @property
    def installed(self) -> bool:
        """Its program is there (e.g. ./venv/bin/python after setup.sh) -- a provider folder alone is only an offer."""
        exe = self.command[0] if self.command else ""
        if not exe.startswith(("./", "../")):
            return True                          # a program on PATH: can't tell, count it as installed
        return (self.dir / exe).exists()

    @property
    def param_names(self) -> frozenset:
        """Numeric params this engine takes: declared by its modes, else the three older engines use."""
        if not self.modes:
            return LEGACY_PARAMS
        return frozenset(p["name"] for m in self.modes for p in m["params"])

    @classmethod
    def from_dict(cls, d: Any, provider_dir: Path) -> "ProviderManifest":
        if not isinstance(d, dict):
            raise ProviderError("manifest must be a JSON object")
        unknown = sorted(k for k in d if k not in _FIELDS and not k.startswith(("_", "$")))
        if unknown:
            raise ProviderError(f"unknown field(s): {', '.join(unknown)}")
        pid = d.get("id")
        if not isinstance(pid, str) or not _ID.match(pid):
            raise ProviderError(f"invalid id {pid!r}")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProviderError("name is required")
        if d.get("capability") != "tts":
            raise ProviderError("capability must be 'tts'")
        provider_dir = Path(provider_dir)

        command = d.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
            raise ProviderError("command must be a non-empty list of strings")
        exe = command[0]
        # Lexically inside the provider (a venv's bin/python may symlink to a shared
        # interpreter elsewhere -- that's how venvs work), no '..', executable.
        if not exe.startswith("./") or ".." in Path(exe).parts:
            raise ProviderError("command[0] must be an executable inside the provider, written ./<path>")
        exe_path = provider_dir / exe[2:]
        if not _inside(exe_path, provider_dir):
            raise ProviderError(f"command[0] {exe} is not an executable inside the provider")
        if not exe_path.exists() or not os.access(exe_path, os.X_OK):
            # the usual case on a fresh machine: the provider's environment hasn't been built
            raise ProviderError(f"not set up yet: {exe} is missing -- run this provider's setup.sh")

        env = d.get("env") or {}
        if not isinstance(env, dict):
            raise ProviderError("env must be an object")
        for k, v in env.items():
            if not isinstance(k, str) or not _ENV_KEY.match(k) or k in _FORBIDDEN_ENV or k.startswith(("LD_", "DYLD_")):
                raise ProviderError(f"env {k!r} is not allowed")
            if not isinstance(v, str):
                raise ProviderError(f"env {k} must be a string")

        gpu = d.get("gpu", 0)
        if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
            raise ProviderError("gpu must be a GPU index")
        vram_mb, vram_fraction = d.get("vram_mb"), d.get("vram_fraction")
        if (vram_mb is None) == (vram_fraction is None):
            raise ProviderError("exactly one of vram_mb / vram_fraction is required")
        if vram_mb is not None and (not isinstance(vram_mb, int) or isinstance(vram_mb, bool) or vram_mb <= 0):
            raise ProviderError("vram_mb must be a positive integer")
        if vram_fraction is not None and (not isinstance(vram_fraction, (int, float)) or not 0 < vram_fraction <= 1):
            raise ProviderError("vram_fraction must be in (0, 1]")

        def _num(key, default, lo, hi):
            v = d.get(key, default)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
                raise ProviderError(f"{key} must be a number in [{lo}, {hi}]")
            return v

        startup = float(_num("startup_timeout_s", 60, 1, 3600))
        keep_alive = int(_num("keep_alive_s", 60, 0, 86400))
        flags = {}
        for key, default in (("concurrent", False), ("variant_swaps", False), ("default_reference", False)):
            v = d.get(key, default)
            if not isinstance(v, bool):
                raise ProviderError(f"{key} must be true/false")
            flags[key] = v
        aliases = d.get("aliases") or []
        variants = d.get("variants") or []
        for key, items in (("aliases", aliases), ("variants", variants)):
            if not isinstance(items, list) or not all(isinstance(a, str) and _TOKEN.match(a) and " " not in a
                                                        for a in items):
                raise ProviderError(f"{key} must be a list of simple names")
        if any(not _ID.match(a) for a in aliases):
            raise ProviderError("aliases must be lowercase ids")
        return cls(
            id=pid, name=name.strip(), dir=provider_dir, command=tuple(command), env=dict(env), gpu=gpu,
            vram_mb=vram_mb, vram_fraction=float(vram_fraction) if vram_fraction is not None else None,
            startup_timeout_s=startup, keep_alive_s=keep_alive, aliases=tuple(aliases), variants=tuple(variants),
            description=str(d.get("description") or ""), license=str(d.get("license") or ""),
            setup=str(d.get("setup") or ""), modes=_parse_modes(d.get("modes"), variants),
            languages=_parse_languages(d.get("languages")), **flags,
        )

    def _fill(self, value: str, port: int) -> str:
        return (value.replace("{port}", str(port)).replace("{dir}", str(self.dir))
                .replace("{vram_fraction}", str(self.vram_fraction) if self.vram_fraction is not None else ""))

    def argv(self, port: int) -> List[str]:
        return [str(self.dir / self.command[0][2:])] + [self._fill(a, port) for a in self.command[1:]]

    def environment(self, port: int) -> Dict[str, str]:
        return {k: self._fill(v, port) for k, v in self.env.items()}

    @property
    def default_variant(self) -> Optional[str]:
        return self.variants[0] if self.variants else None


def vram_need_mb(m: ProviderManifest, gpu_total_mb: int) -> int:
    if m.vram_fraction is not None:
        return int(gpu_total_mb * m.vram_fraction) + FRACTION_EXTRA_MB + VRAM_MARGIN_MB
    return int(m.vram_mb) + VRAM_MARGIN_MB


# --------------------------------------------------------------------------- registry

class ProviderRegistry:
    def __init__(self, providers: List[ProviderManifest], errors: List[Dict[str, str]]):
        self.providers = providers
        self.errors = errors

    @classmethod
    def load(cls, root: Path) -> "ProviderRegistry":
        root = Path(root)
        providers: List[ProviderManifest] = []
        errors: List[Dict[str, str]] = []
        names: Dict[str, str] = {}
        if not root.is_dir():
            return cls([], [{"dir": str(root), "error": "providers directory not found"}])
        for d in sorted(root.iterdir(), key=lambda p: p.name):
            if d.name.startswith(".") or d.is_symlink() or not d.is_dir() or not (d / MANIFEST_NAME).is_file():
                continue
            try:
                with open(d / MANIFEST_NAME, encoding="utf-8") as f:
                    m = ProviderManifest.from_dict(json.load(f), d)
                clash = [n for n in (m.id, *m.aliases) if n in names]
                if clash:
                    raise ProviderError(f"name {clash[0]!r} already used by {names[clash[0]]}")
                for n in (m.id, *m.aliases):
                    names[n] = m.id
                providers.append(m)
            except (ProviderError, OSError, ValueError) as e:
                errors.append({"dir": str(d), "error": str(e)})
        return cls(sorted(providers, key=lambda p: p.id), errors)

    def resolve(self, engine: str) -> ProviderManifest:
        key = (engine or "").strip().lower()
        for p in self.providers:
            if key == p.id or key in p.aliases:
                return p
        raise UnknownEngine(f"Unknown TTS engine {engine!r}. Available: {', '.join(p.id for p in self.providers)}")


# --------------------------------------------------------------------------- request

@dataclass(frozen=True)
class TtsRequest:
    text: str
    language: str = "en"
    engine: Optional[str] = None
    variant: Optional[str] = None
    ref_audio: Optional[bytes] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    speaker: Optional[str] = None
    params: Mapping[str, float] = field(default_factory=dict)
    keep_alive: Optional[int] = None
    response_format: str = "json"

    MAX_TEXT = 20000
    MAX_REF_BYTES = 20 * 1024 * 1024
    MAX_PARAMS = 16   # which names an engine takes is its manifest's business (checked at generate)
    FORMATS = ("json", "wav")

    @classmethod
    def from_dict(cls, d: Any) -> "TtsRequest":
        if not isinstance(d, Mapping):
            raise RequestError("request must be a JSON object")
        allowed = {"text", "language", "engine", "variant", "ref_audio_base64", "ref_text", "instruct",
                   "speaker", "params", "keep_alive", "response_format"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise RequestError(f"unknown field(s): {', '.join(unknown)}")
        text = d.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("text must be a non-empty string")
        if len(text) > cls.MAX_TEXT:
            raise RequestError(f"text is limited to {cls.MAX_TEXT} characters")
        language = d.get("language") or "en"
        if not isinstance(language, str) or not _LANGUAGE.match(language):
            raise RequestError("language must be a language code like 'de' (or a name like 'German')")
        engine = d.get("engine")
        if engine is not None and (not isinstance(engine, str) or not _TOKEN.match(engine)):
            raise RequestError("invalid engine name")
        variant = d.get("variant")
        if variant is not None and (not isinstance(variant, str) or not _TOKEN.match(variant)):
            raise RequestError("invalid variant name")
        ref_audio = None
        raw = d.get("ref_audio_base64")
        if raw:
            if not isinstance(raw, str):
                raise RequestError("ref_audio_base64 must be a string")
            try:
                ref_audio = base64.b64decode(raw, validate=True)
            except (binascii.Error, ValueError):
                raise RequestError("ref_audio_base64 is not valid base64")
            if len(ref_audio) > cls.MAX_REF_BYTES:
                raise RequestError("reference audio is too large")
        for key in ("ref_text", "instruct", "speaker"):
            v = d.get(key)
            if v is not None and (not isinstance(v, str) or len(v) > 4000):
                raise RequestError(f"{key} must be a string")
        params = d.get("params") or {}
        if not isinstance(params, Mapping):
            raise RequestError("params must be an object")
        clean: Dict[str, float] = {}
        if len(params) > cls.MAX_PARAMS:
            raise RequestError(f"at most {cls.MAX_PARAMS} params")
        for k, v in params.items():
            if not isinstance(k, str) or not _PARAM_NAME.match(k):
                raise RequestError(f"invalid param name {k!r}")
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise RequestError(f"param {k} must be a number")
            clean[k] = float(v)
        keep_alive = d.get("keep_alive")
        if keep_alive is not None and (not isinstance(keep_alive, int) or isinstance(keep_alive, bool)
                                       or not 0 <= keep_alive <= 86400):
            raise RequestError("keep_alive must be 0..86400 seconds")
        fmt = d.get("response_format") or "json"
        if fmt not in cls.FORMATS:
            raise RequestError(f"response_format must be one of {', '.join(cls.FORMATS)}")
        return cls(text=text, language=language, engine=engine, variant=variant, ref_audio=ref_audio,
                   ref_text=d.get("ref_text"), instruct=d.get("instruct"), speaker=d.get("speaker"),
                   params=clean, keep_alive=keep_alive, response_format=fmt)


# --------------------------------------------------------------------------- eviction / status

def eviction_order(target: str, gpu: int, engines: Iterable[Mapping[str, Any]]) -> List[str]:
    """Idle engines on the same GPU, least recently used first. Busy ones never."""
    candidates = [e for e in engines if e["id"] != target and e["gpu"] == gpu and e["idle"]]
    return [e["id"] for e in sorted(candidates, key=lambda e: e.get("last_used", 0.0))]


def overall_status(statuses: Sequence[str]) -> str:
    present = [s for s in statuses if s in _SEVERITY]
    if not present:
        return "unloaded"
    return min(present, key=_SEVERITY.index)


def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"vor {s} s"
    if s < 3600:
        return f"vor {s // 60} min"
    if s < 86400:
        return f"vor {s // 3600} h"
    return f"vor {s // 86400} d"


def status_facts(engines: Sequence[Mapping[str, Any]], default_engine: Optional[str], now: float) -> List[Dict[str, str]]:
    """A few plain facts for ai-serve-ui's overview (its manifest's status_path points at
    /v1/status, which carries them as "facts"). The dashboard measures VRAM itself."""
    names = {e["id"]: e.get("name") or e["id"] for e in engines}
    loaded = [names[e["id"]] for e in engines if e.get("status") in ("idle", "loading", "generating")]
    working = [names[e["id"]] for e in engines if e.get("status") == "generating" or (e.get("inflight") or 0) > 0]
    failed = [names[e["id"]] for e in engines if e.get("status") == "error"]
    facts = [{"label": "Geladen", "value": ", ".join(loaded) if loaded else "keine Engine"}]
    if working:
        facts.append({"label": "Arbeitet", "value": ", ".join(working)})
    if failed:
        facts.append({"label": "Fehler", "value": ", ".join(failed)})
    used = [e for e in engines if e.get("last_used") and e.get("status") != "unloaded"]
    if used:
        last = max(used, key=lambda e: e["last_used"])
        facts.append({"label": "Zuletzt genutzt", "value": f"{names[last['id']]}, {_ago(now - last['last_used'])}"})
    if default_engine:
        facts.append({"label": "Standard", "value": names.get(default_engine, default_engine)})
    return facts
