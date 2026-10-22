"""What a fresh ai-serve install offers to set up -- pure rules, no I/O (tests tests/unit/test_installer_offers.py).

Owner, 2026-10-21: the installer offers a few basic services with their models, optionally: a small LLM that also reads
images (Gemma 4 E4B, or E2B when it should be particularly small), speech output (Chatterbox Multilingual -- through MLX
on Apple Silicon), transcription (Whisper via stt-serve) and text-to-image (Iris-3B Turbo via sd-serve). Each offer
says what it downloads and roughly how much, and which plugin folder / setup step it needs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HfFile:
    repo: str
    path: str
    save_as: str          # file name under the models dir


@dataclass(frozen=True)
class Offer:
    id: str
    label: str
    what: str                                  # one line for people
    size_gb: float                             # rough download + install size
    license: str
    files: Tuple[HfFile, ...] = ()             # model files the core downloads (llama.cpp models)
    plugin: Optional[str] = None               # plugins/<name> to set up
    setup: Tuple[str, ...] = ()                # command inside the plugin folder
    chat_model: Optional[str] = None           # becomes DEFAULT_CHAT_MODEL


@dataclass(frozen=True)
class Machine:
    system: str          # Linux / Darwin
    machine: str         # x86_64 / arm64 / aarch64
    nvidia: bool = False

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"


def _gemma(size: str, model_gb: float) -> Offer:
    repo = f"unsloth/gemma-4-{size}-it-GGUF"
    main = f"gemma-4-{size}-it-Q4_K_M.gguf"
    return Offer(
        id=f"llm-gemma-{size.lower()}", label=f"Sprachmodell Gemma 4 {size} (liest auch Bilder)",
        what=f"Chat, Texte, Bilder beschreiben -- {size}, 4-bit; läuft mit llama.cpp auf GPU, Mac oder CPU",
        size_gb=round(model_gb + 0.9, 1), license="Gemma Terms of Use (kommerziell erlaubt)",
        # the vision module is saved as <model stem>-mmproj-F16.gguf so ai-serve pairs it with the model by name
        files=(HfFile(repo, main, main), HfFile(repo, "mmproj-F16.gguf", f"gemma-4-{size}-it-mmproj-F16.gguf")),
        chat_model=main)


def offers(m: Machine) -> List[Offer]:
    tts_provider = "chatterbox-mlx" if m.apple_silicon else "chatterbox"
    tts_size = 2.8 if m.apple_silicon else (6.5 if m.nvidia else 4.0)
    stt_size = 8.0 if m.nvidia else 2.0
    return [
        _gemma("E4B", 4.6),
        _gemma("E2B", 2.9),
        Offer(id="tts-chatterbox", label="Sprachausgabe (Chatterbox Multilingual)",
              what="Text vorlesen, auch Deutsch, mit geklonter Stimme" + (" -- über MLX" if m.apple_silicon else ""),
              size_gb=tts_size, license="MIT", plugin="tts-serve",
              setup=(f"providers/{tts_provider}/setup.sh", "--download")),
        Offer(id="stt-whisper", label="Spracherkennung (Whisper, mit Sprechertrennung)",
              what="Aufnahmen abschreiben (WhisperX small)" + ("" if m.nvidia else " -- auf der CPU"),
              size_gb=stt_size, license="BSD-2 / MIT", plugin="stt-serve", setup=("setup.sh", "--download")),
        Offer(id="image-iris", label="Bilder erzeugen (Iris-3B Turbo)",
              what="Bilder aus Beschreibungen in ~5-15 s; englische Schrift im Bild gelingt, deutsche nicht",
              size_gb=5.5, license="Apache-2.0", plugin="sd-serve", setup=("setup.sh", "--models")),
        Offer(id="image-comfyui", label="Bilder bearbeiten + mehr Bildmodelle (ComfyUI)",
              what="Bearbeiten per Anweisung, Stil übertragen (Qwen-Image-Edit 2511), Z-Image Turbo; "
                   "MOSS-Image (Text im Bild, Positionen) später per Download"
                   + (" -- auf dem Mac langsam, Bearbeiten dort noch ungetestet" if m.apple_silicon else ""),
              size_gb=31.0 if m.nvidia else 28.0, license="Apache-2.0 (Modelle), GPL-3.0 (ComfyUI)",
              plugin="image-serve", setup=("setup.sh", "--models", "z-image-turbo,qwen-edit-2511")),
    ]


def exclusive_groups() -> List[Tuple[str, ...]]:
    """Offers of which at most one makes sense (the two Gemma sizes)."""
    return [("llm-gemma-e4b", "llm-gemma-e2b")]


def validate_choice(chosen: List[str], m: Machine) -> List[str]:
    """Known ids, at most one per exclusive group; returns them in offer order."""
    by_id = {o.id: o for o in offers(m)}
    unknown = [c for c in chosen if c not in by_id]
    if unknown:
        raise ValueError(f"unknown offer(s): {', '.join(unknown)} (known: {', '.join(by_id)})")
    for group in exclusive_groups():
        if sum(1 for c in chosen if c in group) > 1:
            raise ValueError(f"choose one of {', '.join(group)}")
    return [o.id for o in offers(m) if o.id in chosen]


def env_updates(chosen: List[Offer], llama_server: Optional[str]) -> Dict[str, str]:
    """What the installer writes into ai-serve.local.env."""
    out: Dict[str, str] = {}
    if llama_server:
        out["LLAMA_SERVER_BIN"] = llama_server
    chat = next((o.chat_model for o in chosen if o.chat_model), None)
    if chat:
        out["DEFAULT_CHAT_MODEL"] = chat
    return out


def merge_env(text: str, updates: Dict[str, str]) -> str:
    """Set KEY=value lines in an env file's text: existing keys are replaced in place, new ones appended; comments
    and other keys stay."""
    lines = text.splitlines()
    done = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            done.add(key)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in done]
    return "\n".join(lines) + "\n"
