"""Service management domain -- pure rules for the dashboard's Services tab.

Threat model: /api/services starts and stops processes, and anyone who can reach the
dashboard (LAN, tailnet; /public tunnels are blocked by ExposureGuard) can call it.
It must not become a way to run an arbitrary command or to signal a process that
doesn't belong to a registered service.

Ubiquitous language:
- ServiceRegistry  services.json (this repo) names the trusted project directories --
                   the trust anchor; nothing reaches the registry through the API.
- ServiceManifest  each project's own llama-serve.service.json (the "provider"): how
                   to start it (a script inside the project, no shell), its port, how
                   to recognise its processes (`match`), where to open it.
- ObservedProcess  one row of a process snapshot (pid, parent, cwd, cmdline, ports).
- ServiceInstance  a running copy of a service: the process tree rooted at the
                   top-most matching ancestor inside the project, with >=1 listening
                   port. Instances started outside the dashboard count too.
- LaunchPlan       argv/cwd/env for starting; plan_start refuses an occupied port.
- plan_stop        which pids may be signalled: only a whole instance tree, chosen
                   by its root pid.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

MANIFEST_NAME = "llama-serve.service.json"
BUILTIN_CONTROLLERS = frozenset({"comfyui"})

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
# Variables that change what code runs, not just how the app is configured.
_FORBIDDEN_ENV = frozenset({
    "PATH", "BASH_ENV", "ENV", "PROMPT_COMMAND", "SHELLOPTS", "BASHOPTS", "IFS", "CDPATH",
    "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME", "PERL5OPT", "PERL5LIB", "NODE_OPTIONS", "RUBYOPT",
})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "fish", "ksh"})
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
# Loopback only: reachability beyond this machine is chosen in the dashboard
# (Tailscale levels, backend/services/exposure.py), never by the app itself.
_BINDS = _LOOPBACK


# --------------------------------------------------------------------------- errors

class ManifestError(ValueError):
    pass


class ServiceError(Exception):
    status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnknownService(ServiceError):
    status = 404


class ServiceConflict(ServiceError):
    status = 409


# --------------------------------------------------------------------------- helpers

def is_inside(path: Path | str, root: Path | str) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def _url_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//") or "\\" in value:
        raise ManifestError(f"{field_name} must be a path starting with a single '/'")
    return value


# --------------------------------------------------------------------------- presentation

# How a service is shown in the dashboard (redesign 2026-10-16). Text from other repos
# must not bring markup, foreign files or remote URLs into the page: categories and
# icons are names from fixed sets (the page draws the icons itself), a screenshot is a
# raster image inside the project served by the dashboard, status facts are short
# plain strings fetched from the app's own loopback port.
CATEGORIES = ("ai", "stories", "learning", "knowledge", "tools", "integrations")
DEFAULT_CATEGORY = "tools"
ICONS = ("speaker", "mic", "image", "book", "branch", "crown", "waves", "bubbles", "magnifier", "archive",
         "bars", "cloud", "chat", "trend", "code", "music", "video", "globe", "puzzle", "game", "camera",
         "pen", "map", "heart", "star", "tool", "sun", "home")
SCREENSHOT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
MAX_SCREENSHOT_BYTES = 5 * 1024 * 1024
# How an app handles people (docs/identity.md section 4). Missing = "single": assume an app
# keeps one person's data until it says otherwise -- such an app is only for the owner.
IDENTITY_MODES = ("none", "single", "multi")
DEFAULT_IDENTITY = "single"
# How far an app is (owner 2026-10-17): alpha / beta / stable (= fully tested). Missing = not rated.
MATURITY_LEVELS = ("alpha", "beta", "stable")
MAX_STATUS_FACTS = 6
_FACT_LABEL_MAX = 40
_FACT_VALUE_MAX = 80


def _screenshot_path(value: Any, project_dir: Path) -> Path:
    if not isinstance(value, str) or not value.startswith("./") or len(value) <= 2 or "\\" in value:
        raise ManifestError("screenshot must be a file inside the project, written as ./<path>")
    path = project_dir / value[2:]
    if path.suffix.lower() not in SCREENSHOT_TYPES:
        raise ManifestError(f"screenshot must be one of {', '.join(sorted(SCREENSHOT_TYPES))}")
    if not is_inside(path, project_dir):
        raise ManifestError("screenshot resolves outside the project directory")
    if not path.is_file():
        raise ManifestError(f"screenshot {value} does not exist")
    return path


def screenshot_file(m: "ServiceManifest") -> Optional[Path]:
    """The screenshot to serve, re-checked now: the file may have changed (or become a
    symlink pointing elsewhere) since the manifest was read."""
    if m.screenshot is None:
        return None
    p = m.screenshot
    try:
        if (p.suffix.lower() not in SCREENSHOT_TYPES or not is_inside(p, m.project_dir) or not p.is_file()
                or p.stat().st_size > MAX_SCREENSHOT_BYTES):
            return None
    except OSError:
        return None
    return p


def _fact_text(value: Any, limit: int) -> Optional[str]:
    if isinstance(value, bool):
        text = "ja" if value else "nein"
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = " ".join(value.split())
    else:
        return None
    if not text:
        return None
    return text if len(text) <= limit else text[:limit - 1] + "…"


def parse_status_facts(data: Any) -> List[Dict[str, str]]:
    """A service's own status (`status_path`) -> at most six {label, value} strings.
    Anything else in the answer is ignored; malformed entries are skipped."""
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, list):
        return []
    out: List[Dict[str, str]] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        label, value = _fact_text(f.get("label"), _FACT_LABEL_MAX), _fact_text(f.get("value"), _FACT_VALUE_MAX)
        if label is None or value is None:
            continue
        out.append({"label": label, "value": value})
        if len(out) == MAX_STATUS_FACTS:
            break
    return out


# --------------------------------------------------------------------------- manifest spec

MANIFEST_FIELDS = ("id", "name", "description", "command", "env", "port", "bind", "match",
                   "open_path", "health_path", "tailnet_port", "category", "icon", "screenshot", "status_path", "ui",
                   "identity", "maturity", "provides")
MANIFEST_REQUIRED = ("id", "name", "command", "port", "match")


def manifest_schema() -> Dict[str, Any]:
    """JSON Schema of llama-serve.service.json -- published for projects that want to
    integrate themselves. ServiceManifest.from_dict is the authority (it also checks
    the file system); tests keep the two in step."""
    path = {"type": "string", "pattern": r"^/(?!/)[^\\]*$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ai-serve-ui service manifest (llama-serve.service.json)",
        "type": "object",
        "required": list(MANIFEST_REQUIRED),
        "patternProperties": {"^[_$]": {}},
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "pattern": _ID.pattern, "description": "Stable id, used in URLs and logs."},
            "name": {"type": "string", "minLength": 1, "maxLength": 60},
            "description": {"type": "string", "maxLength": 300},
            "command": {
                "type": "array", "minItems": 1, "items": {"type": "string"},
                "prefixItems": [{"type": "string", "pattern": r"^\./.+"}],
                "description": "argv, no shell. command[0] is an executable script inside the project "
                               "(./run.sh). '{port}' is replaced by `port`.",
            },
            "env": {
                "type": "object", "additionalProperties": {"type": "string"},
                "propertyNames": {"pattern": _ENV_KEY.pattern},
                "description": "Extra environment ('{port}' substituted). PATH, LD_*, BASH_ENV, "
                               "PYTHONSTARTUP, NODE_OPTIONS and similar are refused.",
            },
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535,
                     "description": "Port the app listens on when started from the dashboard."},
            "bind": {"enum": sorted(_BINDS), "default": "127.0.0.1",
                     "description": "Loopback only: who else may reach the app is chosen in the dashboard "
                                    "(Tailscale), not by the app."},
            "match": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1},
                      "description": "Command-line fragments that identify the app's own processes."},
            "open_path": {**path, "default": "/"},
            "health_path": {**path, "description": "Optional; GET must answer < 500 when healthy."},
            "tailnet_port": {"type": "integer", "minimum": 1, "maximum": 65535,
                             "description": "HTTPS port on the tailnet when exposed (default port + 10000). "
                                            "443, 8443 and 10000 are refused: Funnel could publish them."},
            "category": {"enum": list(CATEGORIES), "default": DEFAULT_CATEGORY,
                         "description": "Group in the dashboard's Apps & Services page."},
            "icon": {"enum": list(ICONS), "description": "Icon name; the dashboard draws it."},
            "screenshot": {"type": "string", "pattern": r"^\./.+\.(png|jpe?g|webp)$",
                           "description": "Image inside the project (PNG/JPEG/WebP, at most 5 MB), "
                                          "shown on the app's card and when pinned to the overview."},
            "ui": {"type": "boolean", "default": True,
                   "description": "false for services without a page of their own (APIs such as tts-serve): "
                                  "no Open button, and they can't be exposed via Tailscale."},
            "identity": {"enum": list(IDENTITY_MODES), "default": DEFAULT_IDENTITY,
                         "description": "How the app separates people's data (docs/identity.md): none = keeps "
                                        "no per-person data, single = one person's data (owner only), multi = "
                                        "separated per person."},
            "maturity": {"enum": list(MATURITY_LEVELS),
                         "description": "How far the app is: alpha (works, rough), beta (usable, being tested), "
                                        "stable (fully tested). Missing = not rated."},
            "provides": {"type": "object",
                         "description": "Optional capabilities this app provides to the gateway as a plugin: "
                                        "{capability: {endpoint: path, gpu?}} with capability llm | image | video | "
                                        "audio | tts | stt | documents | translate | moderation | model3d | vision | tools | "
                                        "realtime | access | advice -- spec in docs/plugins.md."},
            "status_path": {**path, "description": "Optional; GET answers {\"facts\": [{\"label\", \"value\"}]} "
                                                   "(at most 6, plain text) shown on the overview."},
        },
    }


# --------------------------------------------------------------------------- manifest

@dataclass(frozen=True)
class ServiceManifest:
    id: str
    name: str
    description: str
    project_dir: Path
    command: Tuple[str, ...]
    env: Mapping[str, str]
    port: int
    bind: str
    match: Tuple[str, ...]
    open_path: str = "/"
    health_path: Optional[str] = None
    tailnet_port: Optional[int] = None
    category: str = DEFAULT_CATEGORY
    icon: Optional[str] = None
    screenshot: Optional[Path] = None
    status_path: Optional[str] = None
    ui: bool = True
    identity: str = DEFAULT_IDENTITY
    maturity: Optional[str] = None
    # capability -> Provision (backend/plugins/contracts.py): what the app provides to the gateway as a plugin
    provides: Mapping[str, Any] = field(default_factory=dict)
    # Set for built-in services: the backend's own code starts/stops them (e.g.
    # ComfyUI via start_comfyui(), with GPU pinning, Strata guard and busy checks);
    # no `command` is ever run for them.
    controller: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Any, project_dir: Path, reserved_ports: FrozenSet[int] = frozenset()) -> "ServiceManifest":
        if not isinstance(d, dict):
            raise ManifestError("manifest must be a JSON object")
        unknown = sorted(k for k in d if k not in MANIFEST_FIELDS and not k.startswith(("_", "$")))
        if unknown:
            raise ManifestError(f"unknown field(s): {', '.join(unknown)} (known: {', '.join(MANIFEST_FIELDS)})")
        sid = d.get("id")
        if not isinstance(sid, str) or not _ID.match(sid):
            raise ManifestError(f"invalid id {sid!r} (lowercase letters, digits, '-')")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            raise ManifestError("name must be 1-60 characters")
        description = d.get("description") or ""
        if not isinstance(description, str) or len(description) > 300:
            raise ManifestError("description must be at most 300 characters")

        project_dir = Path(project_dir)
        command = d.get("command")
        if (not isinstance(command, list) or not command
                or not all(isinstance(c, str) for c in command)):
            raise ManifestError("command must be a non-empty list of strings")
        script = command[0]
        if not script.startswith("./") or len(script) <= 2:
            raise ManifestError("command[0] must be a script inside the project, written as ./<path>")
        script_path = project_dir / script[2:]
        if not is_inside(script_path, project_dir):
            raise ManifestError("command[0] resolves outside the project directory")
        if not script_path.is_file() or not os.access(script_path, os.X_OK):
            raise ManifestError(f"command[0] {script} is not an executable file")

        env = d.get("env") or {}
        if not isinstance(env, dict):
            raise ManifestError("env must be an object")
        for k, v in env.items():
            if not isinstance(k, str) or not _ENV_KEY.match(k):
                raise ManifestError(f"invalid env name {k!r}")
            if k in _FORBIDDEN_ENV or k.startswith("LD_") or k.startswith("DYLD_"):
                raise ManifestError(f"env {k} is not allowed")
            if not isinstance(v, str):
                raise ManifestError(f"env {k} must be a string")

        port = d.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            raise ManifestError("port must be an integer between 1024 and 65535")
        if port in reserved_ports:
            raise ManifestError(f"port {port} is reserved")
        bind = d.get("bind", "127.0.0.1")
        if bind not in _BINDS:
            raise ManifestError(f"bind must be loopback ({', '.join(sorted(_BINDS))}), got {bind!r} -- "
                                f"make the app listen on 127.0.0.1; who else may reach it is chosen in the "
                                f"dashboard (Tailscale), not by the app")
        match = d.get("match")
        if not isinstance(match, list) or not match or not all(isinstance(m, str) and m for m in match):
            raise ManifestError("match must be a non-empty list of command-line fragments")
        open_path = _url_path(d.get("open_path", "/"), "open_path")
        health = d.get("health_path")
        health = _url_path(health, "health_path") if health is not None else None

        tailnet_port = d.get("tailnet_port")
        if tailnet_port is not None and (not isinstance(tailnet_port, int) or isinstance(tailnet_port, bool)
                                         or not 1 <= tailnet_port <= 65535):
            raise ManifestError("tailnet_port must be an integer port")

        category = d.get("category", DEFAULT_CATEGORY)
        if category not in CATEGORIES:
            raise ManifestError(f"category must be one of {', '.join(CATEGORIES)}")
        icon = d.get("icon")
        if icon is not None and icon not in ICONS:
            raise ManifestError(f"icon must be one of {', '.join(ICONS)}")
        screenshot = d.get("screenshot")
        screenshot = _screenshot_path(screenshot, project_dir) if screenshot is not None else None
        status_path = d.get("status_path")
        status_path = _url_path(status_path, "status_path") if status_path is not None else None
        ui = d.get("ui", True)
        if not isinstance(ui, bool):
            raise ManifestError("ui must be true or false")
        identity = d.get("identity", DEFAULT_IDENTITY)
        if identity not in IDENTITY_MODES:
            raise ManifestError(f"identity must be one of {', '.join(IDENTITY_MODES)} (docs/identity.md)")
        maturity = d.get("maturity")
        if maturity is not None and maturity not in MATURITY_LEVELS:
            raise ManifestError(f"maturity must be one of {', '.join(MATURITY_LEVELS)}")
        provides = {}
        if d.get("provides") is not None:
            from backend.plugins.contracts import ContractError, parse_provides
            try:
                provides = parse_provides(d["provides"])
            except ContractError as e:
                raise ManifestError(f"provides: {e} (docs/plugins.md)")

        return cls(id=sid, name=name.strip(), description=description, project_dir=project_dir,
                   command=tuple(command), env=dict(env), port=port, bind=bind, match=tuple(match),
                   open_path=open_path, health_path=health, tailnet_port=tailnet_port,
                   category=category, icon=icon, screenshot=screenshot, status_path=status_path, ui=ui,
                   identity=identity, maturity=maturity, provides=provides)

    @classmethod
    def builtin(cls, *, id: str, name: str, description: str, project_dir: Path, port: int,
                match: Tuple[str, ...], controller: str, open_path: str = "/",
                category: str = DEFAULT_CATEGORY, icon: Optional[str] = None,
                maturity: Optional[str] = None) -> "ServiceManifest":
        """A service the backend itself runs. Built in code, not read from a file, so
        it may use a port that is reserved for manifests (ComfyUI's 8189)."""
        if controller not in BUILTIN_CONTROLLERS:
            raise ManifestError(f"unknown controller {controller!r}")
        if not _ID.match(id) or not match:
            raise ManifestError(f"invalid built-in service {id!r}")
        return cls(id=id, name=name, description=description, project_dir=Path(project_dir), command=(),
                   env={}, port=port, bind="127.0.0.1", match=tuple(match), open_path=open_path,
                   controller=controller, category=category, icon=icon, maturity=maturity)

    def _fill(self, value: str) -> str:
        return value.replace("{port}", str(self.port))

    def argv(self) -> List[str]:
        return [str(self.project_dir / self.command[0][2:])] + [self._fill(a) for a in self.command[1:]]

    def environment(self) -> Dict[str, str]:
        return {k: self._fill(v) for k, v in self.env.items()}

    def script_name(self) -> Optional[str]:
        return os.path.basename(self.command[0]) if self.command else None

    def runs_script(self, cmdline: Sequence[str]) -> bool:
        script = self.script_name()
        return bool(script) and any(os.path.basename(a) == script for a in cmdline[:2])

    def matches(self, cmdline: Sequence[str]) -> bool:
        """A shell only counts when it runs the start script itself: a wrapper like
        `bash -c "... python3 main.py"` merely mentions the command (it may well be
        another tool's shell) and must not become part of -- and get killed with --
        the instance."""
        if cmdline and os.path.basename(cmdline[0]) in _SHELLS:
            return self.runs_script(cmdline)
        joined = " ".join(cmdline)
        return any(m in joined for m in self.match)


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class ServicesConfig:
    """Optional services.json: `include` = project dirs outside the roots,
    `exclude` = project dirs or service ids to hide. The legacy form
    {"services": [{"dir": ...}]} counts as include."""
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any) -> "ServicesConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ManifestError("services.json must be a JSON object")
        unknown = sorted(k for k in data if k not in ("include", "exclude", "services") and not k.startswith("_"))
        if unknown:
            raise ManifestError(f"services.json: unknown field(s) {', '.join(unknown)}")
        include = list(data.get("include") or [])
        legacy = data.get("services") or []
        if not isinstance(include, list) or not isinstance(legacy, list):
            raise ManifestError("services.json: include must be a list")
        for e in legacy:
            if not isinstance(e, dict):
                raise ManifestError("services.json: services entries must be {\"dir\": ...}")
            include.append(e.get("dir"))
        exclude = data.get("exclude") or []
        if not isinstance(exclude, list):
            raise ManifestError("services.json: exclude must be a list")
        for v in include:
            if not isinstance(v, str) or not os.path.isabs(v):
                raise ManifestError(f"services.json: include entries must be absolute paths, got {v!r}")
        if not all(isinstance(v, str) and v for v in exclude):
            raise ManifestError("services.json: exclude entries must be paths or service ids")
        return cls(include=tuple(include), exclude=tuple(exclude))

    def excluded_dir(self, d: Path) -> bool:
        return any(os.path.isabs(e) and os.path.realpath(e) == os.path.realpath(d) for e in self.exclude)

    @property
    def excluded_ids(self) -> Tuple[str, ...]:
        return tuple(e for e in self.exclude if not os.path.isabs(e))


@dataclass(frozen=True)
class ProjectRef:
    dir: Path
    source: str  # "root <path>" or "services.json" -- shown in the UI


def find_project_dirs(roots: Sequence[str], config: ServicesConfig) -> Tuple[List[ProjectRef], List[Dict[str, str]]]:
    """Direct, non-hidden, non-symlinked children of each root that carry a manifest,
    plus config.include. A root is the trust anchor: whatever can write a manifest
    into a project there could already change that project's own start script."""
    refs: List[ProjectRef] = []
    errors: List[Dict[str, str]] = []
    seen = set()

    def add(d: Path, source: str):
        key = os.path.realpath(d)
        if key in seen or config.excluded_dir(d):
            return
        seen.add(key)
        refs.append(ProjectRef(dir=d, source=source))

    for root in roots:
        if not os.path.isabs(root):
            raise ManifestError(f"service root must be an absolute path, got {root!r}")
        root_path = Path(root)
        if not root_path.is_dir():
            errors.append({"dir": root, "error": "service root not found", "source": "SERVICE_ROOTS"})
            continue
        for child in sorted(root_path.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
                continue
            if (child / MANIFEST_NAME).is_file():
                add(child, f"root {root}")
    for inc in config.include:
        add(Path(inc), "services.json")
    return refs, errors


class ServiceRegistry:
    def __init__(self, services: List[ServiceManifest], errors: List[Dict[str, str]], sources: Dict[str, str]):
        self.services = services
        self.errors = errors
        self._sources = sources

    @classmethod
    def load(cls, refs: Sequence[ProjectRef], reserved_ports: FrozenSet[int] = frozenset(),
             exclude_ids: Sequence[str] = (), read_manifest: Optional[Callable[[Path], Any]] = None,
             builtins: Sequence[ServiceManifest] = ()) -> "ServiceRegistry":
        read_manifest = read_manifest or _read_manifest_file
        services: List[ServiceManifest] = [b for b in builtins if b.id not in exclude_ids]
        errors: List[Dict[str, str]] = []
        sources: Dict[str, str] = {b.id: "built-in" for b in services}
        for ref in refs:
            try:
                if not ref.dir.is_dir():
                    raise ManifestError("directory not found")
                manifest = ServiceManifest.from_dict(read_manifest(ref.dir), ref.dir, reserved_ports)
                if manifest.id in exclude_ids:
                    continue
                if any(s.id == manifest.id for s in services):
                    raise ManifestError(f"duplicate id {manifest.id!r}")
                if any(s.port == manifest.port for s in services):
                    raise ManifestError(f"port {manifest.port} already used by another service")
                services.append(manifest)
                sources[manifest.id] = ref.source
            except (ManifestError, OSError, ValueError) as e:
                errors.append({"dir": str(ref.dir), "error": str(e), "source": ref.source})
        return cls(services, errors, sources)

    def get(self, service_id: str) -> ServiceManifest:
        for s in self.services:
            if s.id == service_id:
                return s
        raise UnknownService(f"Unknown service {service_id!r}")

    def source(self, service_id: str) -> str:
        return self._sources.get(service_id, "")


def _read_manifest_file(project_dir: Path) -> Any:
    path = project_dir / MANIFEST_NAME
    if not path.is_file():
        raise ManifestError(f"{MANIFEST_NAME} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- discovery

@dataclass(frozen=True)
class ObservedProcess:
    pid: int
    ppid: int
    cwd: str
    cmdline: Tuple[str, ...]
    listening_ports: FrozenSet[int] = frozenset()
    create_time: float = 0.0
    listen_addresses: FrozenSet[Tuple[str, int]] = frozenset()


@dataclass(frozen=True)
class ServiceInstance:
    root_pid: int
    pids: FrozenSet[int]
    ports: FrozenSet[int]
    cmdline: Tuple[str, ...]
    started_at: float
    exposed: Tuple[Tuple[str, int], ...] = ()  # listening beyond loopback -> bypasses the dashboard
    dev_server: bool = False   # runs the Vite dev server (serves any project file) -- never behind the door


def is_vite_dev_server(cmdline: Sequence[str]) -> bool:
    """`vite` / `vite dev` / `vite serve` (also via node_modules/.bin or vite.js), not `vite build|preview`."""
    for i, a in enumerate(cmdline):
        base = os.path.basename(a)
        if base == "vite" or (base == "vite.js" and "/vite/bin/" in a.replace("\\", "/")):
            rest = [x for x in cmdline[i + 1:] if not x.startswith("-")]
            return not rest or rest[0] in ("dev", "serve") or not rest[0].isalpha()
    return False


def _belongs(m: ServiceManifest, p: ObservedProcess) -> bool:
    """Inside the project and recognisably this service (or its start script) -- a
    terminal shell that merely sits in the project directory is not."""
    return is_inside(p.cwd, m.project_dir) and (m.matches(p.cmdline) or m.runs_script(p.cmdline))


def _is_loopback_address(addr: str) -> bool:
    import ipaddress
    try:
        return ipaddress.ip_address(addr.split("%")[0]).is_loopback
    except ValueError:
        return addr == "localhost"


def discover_instances(m: ServiceManifest, processes: Iterable[ObservedProcess]) -> List[ServiceInstance]:
    procs = {p.pid: p for p in processes}
    children: Dict[int, List[int]] = {}
    for p in procs.values():
        children.setdefault(p.ppid, []).append(p.pid)

    def tree(root: int) -> FrozenSet[int]:
        out, stack = set(), [root]
        while stack:
            pid = stack.pop()
            if pid in out:
                continue
            out.add(pid)
            stack.extend(children.get(pid, []))
        return frozenset(out)

    roots: Dict[int, ServiceInstance] = {}
    for p in procs.values():
        if not p.listening_ports or not is_inside(p.cwd, m.project_dir):
            continue
        root = p
        while root.ppid in procs and root.ppid != root.pid and _belongs(m, procs[root.ppid]):
            root = procs[root.ppid]
        if root.pid in roots:
            continue
        pids = tree(root.pid)
        if not any(m.matches(procs[x].cmdline) for x in pids if x in procs):
            continue
        ports = frozenset().union(*(procs[x].listening_ports for x in pids if x in procs))
        exposed = sorted({a for x in pids if x in procs for a in procs[x].listen_addresses
                          if not _is_loopback_address(a[0])})
        roots[root.pid] = ServiceInstance(root_pid=root.pid, pids=pids, ports=ports,
                                          cmdline=root.cmdline, started_at=root.create_time,
                                          exposed=tuple(exposed),
                                          dev_server=any(is_vite_dev_server(procs[x].cmdline) for x in pids if x in procs))
    # A "root" inside another instance's tree is part of that instance -- e.g. a
    # `sh -c` wrapper (which deliberately stops the climb) between the start script
    # and the listening process. Keep only the outermost roots.
    outer = [i for i in roots.values()
             if not any(o.root_pid != i.root_pid and i.root_pid in o.pids for o in roots.values())]
    return sorted(outer, key=lambda i: (i.started_at, i.root_pid))


# --------------------------------------------------------------------------- start / stop / open

@dataclass(frozen=True)
class LaunchPlan:
    argv: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)


def plan_start(m: ServiceManifest, processes: Iterable[ObservedProcess]) -> LaunchPlan:
    if m.controller:
        raise ServiceConflict(f"{m.name} is started by the backend itself, not by a script.")
    processes = list(processes)
    if any(m.port in i.ports for i in discover_instances(m, processes)):
        raise ServiceConflict(f"{m.name} is already running on port {m.port}.")
    for p in processes:
        if m.port in p.listening_ports:
            raise ServiceConflict(f"Port {m.port} is already in use by pid {p.pid} "
                                  f"({' '.join(p.cmdline)[:80]}).")
    return LaunchPlan(argv=m.argv(), cwd=m.project_dir, env=m.environment())


def plan_stop(m: ServiceManifest, instances: Sequence[ServiceInstance], pid: Optional[int]) -> FrozenSet[int]:
    if not instances:
        raise UnknownService(f"{m.name} is not running.")
    if pid is None:
        if len(instances) > 1:
            raise ServiceConflict(f"{m.name} runs {len(instances)} times -- say which (pid).")
        return instances[0].pids
    for inst in instances:
        if inst.root_pid == pid:
            return inst.pids
    raise UnknownService(f"pid {pid} is not a running instance of {m.name}.")


def descends_from(pid: int, ancestor: int, procs: Iterable["ObservedProcess"]) -> bool:
    """pid is ancestor or one of its descendants (per the snapshot's parent links)."""
    parent = {p.pid: p.ppid for p in procs}
    seen = set()
    while pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        if pid not in parent:
            return False
        pid = parent[pid]
    return False


def open_target(m: ServiceManifest, port: int) -> Dict[str, Any]:
    return {"port": port, "path": m.open_path, "local_only": m.bind in _LOOPBACK}


# --------------------------------------------------------------------------- child environment

# This backend's own secrets (several come from .env.public into
# os.environ) and its Python environment must not leak into the apps it starts: the
# secrets are none of their business, and PYTHONPATH/VIRTUAL_ENV pointing at this repo
# would shadow their own packages (StoryTime has a `backend` package too).
_OWN_SECRETS = frozenset({"LLAMA_SERVE_PUBLIC_API_KEY", "LLAMA_SERVE_API_KEY", "TRAINING_SERVICE_TOKEN"})
_OWN_PYTHON = frozenset({"PYTHONPATH", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONHOME"})


def child_environment(base: Mapping[str, str], manifest_env: Mapping[str, str]) -> Dict[str, str]:
    env = {k: v for k, v in base.items() if k not in _OWN_SECRETS and k not in _OWN_PYTHON}
    venv = base.get("VIRTUAL_ENV")
    if venv and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep)
                                      if p and not is_inside(p, venv))
    # where the gateway is (it may run on another port than 8030); a manifest may point elsewhere
    env["AI_SERVE_URL"] = f"http://127.0.0.1:{base.get('AI_SERVE_PORT') or 8030}"
    env.update(manifest_env)
    # docs/identity.md 2.2: the owner's stable id, for requests without identity from loopback.
    # Set last: a manifest can't make an app treat someone else as the owner.
    env["LLAMA_SERVE_OWNER_SUB"] = base.get("LLAMA_SERVE_OWNER_SUB") or "owner"
    # ... and the owner's Tailscale login, so an app can tell the owner coming through
    # Tailscale Serve from other tailnet users (both connect from loopback).
    env.pop("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN", None)
    if base.get("LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"):
        env["LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"] = base["LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"]
    # this machine's names and addresses: without a door identity, the SDK accepts only requests whose Host/Origin
    # name this machine (DNS rebinding, foreign pages). From ai-serve only -- a manifest can't widen them.
    env.pop("LLAMA_SERVE_LOCAL_HOSTS", None)
    if base.get("LLAMA_SERVE_LOCAL_HOSTS"):
        env["LLAMA_SERVE_LOCAL_HOSTS"] = base["LLAMA_SERVE_LOCAL_HOSTS"]
    return env
