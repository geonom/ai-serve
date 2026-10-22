import os
import glob
import sys

_BACKEND_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)


def _load_dotenv(filename: str) -> None:
    """Load a gitignored dotenv file (.env.public) into os.environ, if present -- read
    directly rather than relying on launcher.sh (or whatever started this
    process) having sourced it, so a plain restart.flag touch is enough to
    pick up changes regardless of how the backend was launched."""
    path = os.path.join(_PROJECT_DIR, filename)
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# This machine's own settings (paths, network provider, optional components) -- gitignored; see
# ai-serve.example.env. Loaded first so a fresh checkout runs with portable defaults and nothing of this machine.
_load_dotenv("ai-serve.local.env")
_load_dotenv(".env.public")
# who the owner is at the door (LLAMA_SERVE_OWNER_SUB = the owner's Nextcloud account id)
_load_dotenv(".env.identity")

FRONTEND_DIR = os.path.join(_PROJECT_DIR, "frontend")
LOGS_DIR = os.path.abspath(os.path.join(_BACKEND_DIR, "..", "logs"))

os.makedirs(LOGS_DIR, exist_ok=True)

def project_path(p: str) -> str:
    """A path from the settings: ~ expanded, relative ones against this project (not the caller's working dir)."""
    p = os.path.expanduser(p.strip())
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_PROJECT_DIR, p))


# The gateway's own port (dashboard, /v1, /api) and listen address. Plugins and apps learn it as AI_SERVE_URL.
GATEWAY_PORT = int(os.getenv("AI_SERVE_PORT", "8030"))
GATEWAY_HOST = os.getenv("AI_SERVE_HOST", "0.0.0.0")   # who may use it is decided by backend/access, not the bind

# Where models live: the project's own models/ (the installer puts the starter model there) and ~/gguf_models.
MODEL_DIRS_RAW = os.getenv("LLAMA_MODEL_DIRS", os.path.join(_PROJECT_DIR, "models") + "," + os.path.expanduser("~/gguf_models"))
MODEL_DIRS = [project_path(d) for d in MODEL_DIRS_RAW.split(",") if d.strip()]

# llama-server: LLAMA_SERVER_BIN, else one the installer fetched into runtimes/, else whatever is on PATH.
RUNTIMES_DIR = os.path.join(_PROJECT_DIR, "runtimes")
_DEFAULT_SERVER_BIN = "llama-server"
for _path in sorted(glob.glob(os.path.join(RUNTIMES_DIR, "*", "llama-server"))
                    + glob.glob(os.path.join(RUNTIMES_DIR, "*", "*", "llama-server")), reverse=True):
    if os.access(_path, os.X_OK):
        _DEFAULT_SERVER_BIN = _path
        break
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN") or _DEFAULT_SERVER_BIN
if os.sep in LLAMA_SERVER_BIN:          # a path (not a name found on PATH): relative ones against this project
    LLAMA_SERVER_BIN = project_path(LLAMA_SERVER_BIN)

# llama-tokenize is a sibling CLI tool in the same llama.cpp build -- tokenizer-only,
# no VRAM/full-weight load, used by /api/tokenize's model-not-yet-running fallback
# (backend/routers/dashboard.py's tokenize_text()) so a caller can get an exact token
# count before deciding whether to load a model at all.
LLAMA_TOKENIZE_BIN = os.getenv("LLAMA_TOKENIZE_BIN") or os.path.join(
    os.path.dirname(LLAMA_SERVER_BIN), "llama-tokenize")

# Optional components: empty / missing = not installed (image generation just reports unavailable).
# Other model servers (e.g. Strata) are llm plugins -- docs/plugins.md.
# What "Auto"/"default" chats load when no suitable chat model is running (2026-10-20: the old "first Gemma 4 12B
# file" picked a 2-bit quant that thought in circles until max_tokens ran out)
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "gemma-4-12b-it-qat-q4_0.gguf")
# Image generation (ComfyUI, MOSS, Iris ...) comes from image plugins -- docs/plugins.md

# Default -np (parallel slots) for auto-loaded (get_or_load_model) llama-server
# instances; a request's `parallel` field overrides it. 0 = leave llama.cpp's own
# auto choice (4 slots).
DEFAULT_PARALLEL = int(os.getenv("LLAMA_DEFAULT_PARALLEL", "1"))


API_KEY = os.getenv("LLAMA_SERVE_API_KEY", "")

# Curated, externally exposable OpenAI surface at /public/v1 (backend/routers/
# public_gateway.py) -- e.g. for a Nextcloud elsewhere. Own key, fail-closed: unset
# -> every /public route answers 503. The model list/aliases live in a JSON file,
# re-read per request so edits need no restart.
PUBLIC_API_KEY = os.getenv("LLAMA_SERVE_PUBLIC_API_KEY", "")
PUBLIC_MODELS_FILE = os.getenv("PUBLIC_MODELS_FILE", os.path.join(_PROJECT_DIR, "public_models.json"))
# Sibling apps for the dashboard's Services tab (backend/services/domain.py): every
# direct child of a root with a llama-serve.service.json shows up; the optional
# services.json adds dirs elsewhere (include) or hides some (exclude).
SERVICE_ROOTS = [project_path(r) for r in os.getenv(
    "SERVICE_ROOTS", os.path.join(_PROJECT_DIR, "apps") + os.pathsep + os.path.join(_PROJECT_DIR, "plugins")
).split(os.pathsep) if r.strip()]
SERVICES_FILE = os.getenv("SERVICES_FILE", os.path.join(_PROJECT_DIR, "services.json"))
# Standard OpenAI-compatible servers the owner runs (Ollama, LM Studio ...), added by port in the dashboard
MODEL_SERVERS_FILE = os.getenv("MODEL_SERVERS_FILE", os.path.join(_PROJECT_DIR, "model_servers.json"))
# Who may reach each service via Tailscale (local/tailnet/owner); written by the dashboard.
SERVICE_EXPOSURE_FILE = os.getenv("SERVICE_EXPOSURE_FILE", os.path.join(_PROJECT_DIR, "service_exposure.json"))
SERVICE_AUTOSTART_FILE = os.getenv("SERVICE_AUTOSTART_FILE", os.path.join(_PROJECT_DIR, "service_autostart.json"))
# Overview page arrangement (box order/width/hidden, pinned apps) -- owner preference, gitignored.
# Shares through the door for people outside (the optional `access` plugin, docs/plugins.md; backend/door/policy.py)
DOOR_SHARES_FILE = os.getenv("DOOR_SHARES_FILE", os.path.join(_PROJECT_DIR, "door_shares.json"))
DASHBOARD_LAYOUT_FILE = os.getenv("DASHBOARD_LAYOUT_FILE", os.path.join(_PROJECT_DIR, "dashboard_layout.json"))
# Max GPU-bound /public requests in flight at once (more -> 429).
PUBLIC_MAX_CONCURRENT = int(os.getenv("PUBLIC_MAX_CONCURRENT", "3"))
DEFAULT_PORT = 8031

# Fine-tuning runs in train-serve (an optional separate app) since 2026-10-17;
# /api/training/* is forwarded there (it checks its own token), started on demand.
TRAIN_SERVE_URL = os.getenv("TRAIN_SERVE_URL", "http://127.0.0.1:8038").rstrip("/")
TRAIN_SERVE_SERVICE_ID = os.getenv("TRAIN_SERVE_SERVICE_ID", "train-serve")

# Speech (tts), transcription (stt) and the door (access) are plugins: apps in Apps & Services that declare what
# they provide (docs/plugins.md, backend/plugins/registry.py) -- no URL or service id is configured here.

# Private overlay network (backend/network): tailscale (default), static
# (WireGuard/Nebula/NetBird/ZeroTier/Headscale: networks + address->identity map in
# NETWORK_CONFIG_FILE) or none (loopback + API key only).
# none = this machine only (+ LLAMA_SERVE_API_KEY); tailscale / static are optional extensions
NETWORK_PROVIDER = os.getenv("NETWORK_PROVIDER", "none")
NETWORK_CONFIG_FILE = os.getenv("NETWORK_CONFIG_FILE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "network.json"))
# Enrolled cluster nodes (backend/cluster/enrollment.py) -- holds node tokens, 0600.
CLUSTER_NODES_FILE = os.getenv("CLUSTER_NODES_FILE", os.path.join(_PROJECT_DIR, "cluster_nodes.json"))

# Engine used by POST /api/tts/generate and /v1/audio/speech when the caller names
# none. CosyVoice 3 is the recommended default: it's the only bundled engine that
# clones a specific voice AND takes a free-form style instruction in one call, and
# its weights are Apache-2.0 (OmniVoice's are CC-BY-NC -- non-commercial). Every
# engine runs in tts-serve.
DEFAULT_TTS_ENGINE = os.getenv("DEFAULT_TTS_ENGINE", "")   # empty: the TTS plugin's own default (its first installed engine)
DEFAULT_STT_ENGINE = os.getenv("DEFAULT_STT_ENGINE", "")   # empty: the first available default among the STT plugins
# Reference voice a clone-based engine (cosyvoice/chatterbox) uses when the caller
# supplies none: a filename under voices/ (its .txt sidecar is the transcript, which
# CosyVoice zero_shot needs). Default is "synth-default.wav" -- a SYNTHETIC voice
# generated from a written description (Qwen3-TTS VoiceDesign), NOT cloned from any
# real person. Any celebrity/actor clips that happen to sit in voices/ are local
# data belonging to whoever put them there; do not make them the default and do not
# publish anything derived from them. Override per deployment, e.g. to a
# native-language reference so unspecified-reference output isn't English-accented.
DEFAULT_TTS_VOICE = os.getenv("DEFAULT_TTS_VOICE", "synth-default.wav")

# HF_TOKEN / DIARIZE_MODEL for diarization are read by stt-serve itself now.
