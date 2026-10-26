"""What a fresh ai-serve install offers to set up -- pure rules, no I/O (tests tests/unit/test_installer_offers.py).

Owner, 2026-10-21: the installer offers a few basic services with their models, optionally: a small LLM that also reads
images (Gemma 4 E4B, or E2B when it should be particularly small), speech output (Chatterbox Multilingual -- through MLX
on Apple Silicon), transcription (Whisper via stt-serve) and text-to-image (Iris-3B Turbo via sd-serve). Each offer
says what it downloads and roughly how much, and which plugin folder / setup step it needs.

Suggestions (2026-10-26, owner: proposals that follow the hardware found): `recommend()` says yes/no with a reason for
every offer from the largest NVIDIA card's memory (or, on Apple Silicon, the part of the RAM Metal lets the GPU use),
the RAM and the free disk space; the interactive setup uses them as its defaults, `--recommended` takes them as they
are. Unknown figures (detection failed) keep the defaults from before."""
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
    git: Optional[str] = None                  # an external plugin: cloned into plugins/<plugin> first (plugin add)
    setup: Tuple[str, ...] = ()                # command inside the plugin folder
    chat_model: Optional[str] = None           # becomes DEFAULT_CHAT_MODEL


@dataclass(frozen=True)
class Machine:
    system: str          # Linux / Darwin
    machine: str         # x86_64 / arm64 / aarch64
    nvidia: bool = False
    gpus_mb: Tuple[int, ...] = ()   # memory of each NVIDIA GPU; () = none or not measured
    ram_mb: int = 0                 # 0 = unknown
    disk_free_gb: float = 0         # where ai-serve lives; 0 = unknown

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"

    @property
    def gpu_gb(self) -> Optional[float]:
        """Memory a model can use on one GPU: the largest NVIDIA card; on Apple Silicon Metal's share of the RAM (~2/3
        up to 36 GB, 3/4 above -- as the gateway counts it); 0 without a GPU; None when not measured."""
        if self.nvidia:
            return max(self.gpus_mb) / 1024 if self.gpus_mb else None
        if self.apple_silicon:
            if not self.ram_mb:
                return None
            ram = self.ram_mb / 1024
            return ram * (2 / 3 if ram <= 36 else 3 / 4)
        return 0.0


DISK_RESERVE_GB = 10   # left free after the downloads


def _gemma(size: str, model_gb: float, mmproj_gb: float = 0.9) -> Offer:
    repo = f"unsloth/gemma-4-{size}-it-GGUF"
    main = f"gemma-4-{size}-it-Q4_K_M.gguf"
    return Offer(
        id=f"llm-gemma-{size.lower()}", label=f"Language model Gemma 4 {size.upper()} (also reads images)",
        what=f"Chat, texts, describing images -- {size.upper()}, 4-bit; runs with llama.cpp on a GPU, a Mac or the CPU",
        size_gb=round(model_gb + mmproj_gb, 1), license="Gemma Terms of Use (commercial use allowed)",
        # the vision module is saved as <model stem>-mmproj-F16.gguf so ai-serve pairs it with the model by name
        files=(HfFile(repo, main, main), HfFile(repo, "mmproj-F16.gguf", f"gemma-4-{size}-it-mmproj-F16.gguf")),
        chat_model=main)


def _comfyui(m: Machine) -> Offer:
    """With 8-12 GB on an NVIDIA card: Z-Image + FLUX.2 [klein] 4B only (klein shares Z-Image's text encoder; ~8.5 GB in
    use) -- Qwen-Image-Edit 2511 wants more; otherwise both editors."""
    gpu = m.gpu_gb
    light = m.nvidia and gpu is not None and gpu < 12
    if light:
        what, models, size = ("Z-Image Turbo, editing from up to four images (FLUX.2 [klein] 4B); "
                              "Qwen-Image-Edit 2511 and MOSS-Image as later downloads"), "z-image-turbo,flux2-klein-4b", 18.5
    else:
        what, models = ("Editing by instruction, style transfer (Qwen-Image-Edit 2511), FLUX.2 [klein] 4B, Z-Image Turbo; "
                        "MOSS-Image (text in images, positions) as a later download"), "z-image-turbo,qwen-edit-2511,flux2-klein-4b"
        size = 35.0 if m.nvidia else 32.0
    return Offer(id="image-comfyui", label="Image editing + more image models (ComfyUI)",
                 what=what,
                 size_gb=size, license="Apache-2.0 (Modelle), GPL-3.0 (ComfyUI)",
                 plugin="image-serve", setup=("setup.sh", "--models", models))


def offers(m: Machine) -> List[Offer]:
    tts_provider = "chatterbox-mlx" if m.apple_silicon else "chatterbox"
    tts_size = 2.8 if m.apple_silicon else (6.5 if m.nvidia else 4.0)
    stt_size = 8.0 if m.nvidia else 2.0
    return [
        _gemma("12b", 7.1, 0.2),
        _gemma("E4B", 4.6),
        _gemma("E2B", 2.9),
        Offer(id="tts-chatterbox", label="Speech output (Chatterbox Multilingual)",
              what="Reads text aloud in many languages, with a cloned voice" + (" -- through MLX" if m.apple_silicon else ""),
              size_gb=tts_size, license="MIT", plugin="tts-serve",
              setup=(f"providers/{tts_provider}/setup.sh", "--download")),
        Offer(id="tts-piper", label="Fast speech output on the CPU (Piper)",
              what="Reads text aloud quickly without a GPU (a Raspberry Pi too); fixed voices -- German Thorsten, "
                   "English LJSpeech, more in Downloads --, no cloning; a separate plugin from github.com/geonom/ai-serve-piper",
              size_gb=0.3, license="MIT (plugin, voices) / GPL-3.0 (Piper, run as its own program)",
              plugin="ai-serve-piper", git="https://github.com/geonom/ai-serve-piper",
              setup=("setup.sh", "--voices", "de_DE-thorsten-medium,en_US-ljspeech-medium")),
        Offer(id="stt-whisper", label="Speech recognition (Whisper, with speaker diarization)",
              what="Transcribes recordings (WhisperX small)" + ("" if m.nvidia else " -- on the CPU")
                   + "; telling speakers apart with pyannote needs a Hugging Face token (HF_TOKEN, see the README)"
                   + ("; without a token: MOSS-Transcribe or Nemotron from Downloads" if m.nvidia else ""),
              size_gb=stt_size, license="BSD-2 / MIT", plugin="stt-serve", setup=("setup.sh", "--download")),
        Offer(id="image-iris", label="Images from text (Iris-3B Turbo)",
              what="Images from descriptions in ~5-15 s; English lettering in images comes out clean, German does not",
              size_gb=5.5, license="Apache-2.0", plugin="sd-serve", setup=("setup.sh", "--models")),
        _comfyui(m),
    ]


def exclusive_groups() -> List[Tuple[str, ...]]:
    """Offers of which at most one makes sense (the Gemma sizes)."""
    return [("llm-gemma-12b", "llm-gemma-e4b", "llm-gemma-e2b")]


def describe(m: Machine) -> str:
    """One line: what the setup found."""
    parts = [f"{m.system} {m.machine}"]
    if m.nvidia:
        parts[0] += (", NVIDIA GPU" + ("s " if len(m.gpus_mb) > 1 else " ")
                     + " + ".join(f"{g / 1024:.1f}" for g in m.gpus_mb) + " GB") if m.gpus_mb else ", NVIDIA GPU"
    elif m.apple_silicon:
        parts[0] += f" (Apple Silicon, ~{m.gpu_gb:.0f} GB for the GPU)" if m.gpu_gb else " (Apple Silicon)"
    else:
        parts[0] += ", no NVIDIA GPU"
    if m.ram_mb:
        parts.append(f"RAM {round(m.ram_mb / 1024)} GB")
    if m.disk_free_gb:
        parts.append(f"{m.disk_free_gb:.0f} GB free")
    return ", ".join(parts)


def recommend(m: Machine) -> List[Tuple[str, bool, str]]:
    """(offer id, suggested?, why) for every offer, in offer order: exactly one language model; the rest by GPU
    memory (Piper instead of Chatterbox without a GPU); then cut to the free disk space (language model first, then
    speech recognition, speech output, image editing, Iris)."""
    gpu = m.gpu_gb
    known = gpu is not None
    on_gpu = f"{gpu:.1f} GB on the GPU" if known else ""
    cpu_only = not m.nvidia and not m.apple_silicon
    if known and gpu >= 12:
        llm, why_llm = "llm-gemma-12b", f"fits in {on_gpu} -- the most capable one here"
    elif known and gpu >= 6:
        llm, why_llm = "llm-gemma-e4b", f"fits in {on_gpu}; the 12B needs ~12 GB"
    elif known and cpu_only:
        llm, why_llm = "llm-gemma-e2b", "no GPU: on the CPU the smallest one answers fastest"
    elif known:
        llm, why_llm = "llm-gemma-e2b", f"only {on_gpu}"
    else:
        llm, why_llm = "llm-gemma-e4b", "the usual choice (GPU memory not measured)"
    want: Dict[str, Tuple[bool, str]] = {}
    if cpu_only:
        want["tts-chatterbox"] = (False, "very slow on the CPU -- better on a GPU or a Mac")
    elif not known or gpu >= 5:
        want["tts-chatterbox"] = (True, "runs on the GPU" + (f" ({on_gpu})" if known else ""))
    else:
        want["tts-chatterbox"] = (False, f"needs ~5 GB on the GPU, {on_gpu} here")
    if cpu_only:
        want["tts-piper"] = (True, "no GPU needed: fast on the CPU")
    elif known and gpu < 5:
        want["tts-piper"] = (True, f"Chatterbox needs ~5 GB on the GPU, {on_gpu} here; Piper runs on the CPU")
    else:
        want["tts-piper"] = (False, "Chatterbox is suggested here (cloned voices); Piper can be added any time")
    want["stt-whisper"] = (True, "runs on the GPU" if m.nvidia else "the small Whisper also runs well on the CPU")
    if m.nvidia and known and gpu >= 12:
        want["image-comfyui"] = (True, f"both image editors fit ({on_gpu})")
    elif m.nvidia and known and gpu >= 8:
        want["image-comfyui"] = (True, f"the smaller editor (FLUX.2 [klein]) fits in {on_gpu}; Qwen-Image-Edit wants ~12 GB")
    elif m.apple_silicon:
        want["image-comfyui"] = (False, "slow on a Mac, editing untried there")
    else:
        want["image-comfyui"] = (False, "needs an NVIDIA GPU with ~8 GB" + (f", {on_gpu} here" if known and m.nvidia else ""))
    if cpu_only:
        want["image-iris"] = (False, "too slow on the CPU")
    elif known and gpu < 6:
        want["image-iris"] = (False, f"needs ~6 GB on the GPU, {on_gpu} here")
    else:
        want["image-iris"] = (True, "fast images from text" + (f" ({on_gpu})" if known else ""))

    sizes = {o.id: o.size_gb for o in offers(m)}
    budget = m.disk_free_gb - DISK_RESERVE_GB if m.disk_free_gb else None
    used = 0.0
    if budget is not None:   # a smaller language model when even it doesn't fit
        sizes_llm = ["llm-gemma-12b", "llm-gemma-e4b", "llm-gemma-e2b"]
        while sizes[llm] > budget and llm != sizes_llm[-1]:
            llm = sizes_llm[sizes_llm.index(llm) + 1]
            why_llm = f"the smaller one: only {m.disk_free_gb:.0f} GB free on the disk"
    used += sizes[llm]
    for oid in ("stt-whisper", "tts-chatterbox", "tts-piper", "image-comfyui", "image-iris"):
        yes, why = want[oid]
        if oid == "image-iris" and yes and want["image-comfyui"][0]:
            want[oid] = (False, "Z-Image in ComfyUI (image editing) already makes images from text")
            continue
        if yes and budget is not None and used + sizes[oid] > budget:
            want[oid] = (False, f"not enough disk space: {m.disk_free_gb:.0f} GB free, ~{sizes[oid]} GB needed "
                                f"after the rest (and {DISK_RESERVE_GB} GB kept free)")
            continue
        if yes:
            used += sizes[oid]
    rows = []
    for o in offers(m):
        if o.id.startswith("llm-"):
            rows.append((o.id, o.id == llm, why_llm if o.id == llm else "another size fits this machine better"))
        else:
            rows.append((o.id, *want[o.id]))
    return rows


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
