"""Loading provider adapters (providers/<id>/adapter.py). An adapter is plain Python
run inside tts-serve: `build_payload(req: TtsRequest, ref_audio_path) -> dict` turns
the common request into its worker's payload and raises RequestError for invalid
combinations. The providers directory is tts-serve's own (trust anchor)."""
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Dict

PROVIDERS_DIR = Path(os.getenv("TTS_SERVE_PROVIDERS_DIR",
                               Path(__file__).resolve().parent.parent / "providers"))

_cache: Dict[str, ModuleType] = {}


def load_adapter(provider_dir: Path) -> ModuleType:
    path = Path(provider_dir) / "adapter.py"
    key = str(path.resolve())
    if key not in _cache:
        spec = importlib.util.spec_from_file_location(f"tts_provider_{Path(provider_dir).name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "build_payload", None)):
            raise ImportError(f"{path} has no build_payload(req, ref_audio_path)")
        _cache[key] = module
    return _cache[key]
