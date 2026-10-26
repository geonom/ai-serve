# ai-serve

[![tests](https://github.com/geonom/ai-serve/actions/workflows/tests.yml/badge.svg)](https://github.com/geonom/ai-serve/actions/workflows/tests.yml)

A gateway for local AI on your own machine: language models (llama.cpp), text-to-speech, speech-to-text, image
generation and document reading – behind **one** OpenAI-compatible API (`/v1/...`) plus a dashboard. Installed
models are loaded when a request needs them (the plugin serving them is started too) and unloaded after a while unused
or when the memory is needed for something else. Apps such as the Playground are started by you, or with ai-serve
(Autostart).

> **Alpha.** Developed and tested on Linux with NVIDIA GPUs; first runs on one Apple Silicon Mac. AMD, Vulkan-only and
> CPU-only machines are meant to work (the installer picks the matching llama.cpp build) but haven't been tried.
> Windows will come later. What exactly has been tried: [Status](#status).

> **A vibe coding project.** ai-serve was built largely through AI-assisted ("vibe") coding, for my own machines first.
> It has a sizeable test suite and the security-relevant parts were written test-first, but it has not been audited.
> **It comes without any warranty or responsibility for functionality or security** – use it at your own risk
> (see sections 15 and 16 of the [license](LICENSE)).

## Install

Requirements: `git` and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/geonom/ai-serve.git && cd ai-serve
./install.sh
```

The installer fetches the right llama.cpp build for this machine and offers a basic set of services – all optional.
It looks at the GPU memory (on a Mac: the part of the RAM the GPU may use), the RAM and the free disk space and
suggests what fits; the suggestions are the defaults when it asks. `./ai-serve offers` shows them with the reasons,
`./install.sh --recommended --yes` takes them without asking.
llama.cpp, Gemma, Iris and the ComfyUI models are pinned and checked by sha256; Chatterbox and Whisper are fetched by
their own libraries from Hugging Face (current version, not pinned). Sizes are estimates:

| Offer | What | Approx. size |
|---|---|---|
| Gemma 4 12B, E4B or E2B | Language model that also understands images (12B from ~12 GB on the GPU) | 7.3 / 5.5 / 3.8 GB |
| Chatterbox Multilingual | Text-to-speech with voice cloning (through MLX on a Mac) | 3–7 GB |
| Piper ([ai-serve-piper](https://github.com/geonom/ai-serve-piper), cloned by the installer) | Fast text-to-speech on the CPU, fixed voices (German, English; more in *Downloads*) – suggested instead of Chatterbox without a GPU | 0.3 GB |
| Whisper (WhisperX small) | Speech-to-text; telling speakers apart with pyannote needs a free Hugging Face token (below) | 2–8 GB |
| Iris-3B Turbo | Images from text descriptions (stable-diffusion.cpp) | 5.5 GB |
| ComfyUI (image-serve) | Editing images by instruction, style transfer (Qwen-Image-Edit 2511), FLUX.2 [klein] 4B, Z-Image Turbo; MOSS-Image (text in images, placing things in boxes) as a later download. With 8–12 GB on the GPU only Z-Image + klein | ~35 GB (8–12 GB GPU: ~18.5 GB) |

Telling speakers apart: with Whisper it uses pyannote's model, which is gated on Hugging Face – accept its terms on
<https://huggingface.co/pyannote/speaker-diarization-community-1> and put `HF_TOKEN=<your token>` into
`ai-serve.local.env`; recognising the same speaker across recordings uses it too. On an NVIDIA GPU, MOSS-Transcribe
(`engine: "moss"`) and Nemotron diarization – both downloadable from *Downloads* – need no token. Transcription itself
works without one.

Then: `./ai-serve start` → <http://localhost:8030> (another port: `AI_SERVE_PORT=8031 ./ai-serve start`;
in the background: `./ai-serve start -d`; also `./ai-serve stop`, `restart`, `status`). The
**Playground** (under *Apps & Services*) lets you try everything the gateway offers. The dashboard and the
Playground are in English, with a switch to German at the top right.

## Extend

**More models** – a GGUF language model from Hugging Face: *Compute & models → Language model from Hugging Face*, or
`./ai-serve model add https://huggingface.co/<owner>/<repo>-GGUF`. Language models, embeddings, voices, Whisper sizes, TTS engines, image models: what it takes for
each kind, in [docs/models.md](docs/models.md).


Everything except llama.cpp is a **plugin**: a separate program with a small manifest (`llama-serve.service.json`)
that ai-serve starts when it's needed. Bundled: `plugins/tts-serve`, `plugins/stt-serve`, `plugins/sd-serve`, `plugins/image-serve`
(ComfyUI), `plugins/vision-serve` (faces in photos: YuNet + AuraFace, `/v1/vision/analyze`). Models a plugin offers but that aren't installed yet can be downloaded from the Playground or the
dashboard (*Compute & models → Downloads*). More
from any git repository:

```bash
./ai-serve plugin add https://github.com/<someone>/<plugin>
./ai-serve plugin list
```

A plugin is someone else's code: `plugin add` checks its manifest, and runs its `setup.sh` only after you confirm.
Two plugins live in repositories of their own, as examples: [ai-serve-piper](https://github.com/geonom/ai-serve-piper)
(fast speech on the CPU – and a template to copy for your own plugin, MIT) and
[ai-serve-strata](https://github.com/geonom/ai-serve-strata).

Existing OpenAI-compatible model servers such as **Ollama** or **LM Studio** can be added in the dashboard
(*Compute & models → More model servers*), on custom ports too. The other way round, the Playground can talk to such a
server directly. Both were tried with a plain llama-server standing in – not yet with Ollama or LM Studio themselves. Writing a plugin: [docs/plugins.md](docs/plugins.md),
[docs/service-manifest.md](docs/service-manifest.md). API: [docs/API.md](docs/API.md).

**Strata** (a 125B mixture-of-experts model for PCs with a dedicated graphics card and lots of RAM) can be added as
a language model with the plugin [ai-serve-strata](https://github.com/geonom/ai-serve-strata): [docs/strata.md](docs/strata.md).

## More machines

Other machines of yours – a MacBook, a second PC, a Raspberry Pi – can lend their GPU or memory. They need a private
network between them: [Tailscale](https://tailscale.com) is what has been tested (a free personal account is enough),
WireGuard, Headscale, NetBird, ZeroTier or Nebula are meant to work too but haven't been tried. With Tailscale:
install it on both machines with the same account, set `NETWORK_PROVIDER=tailscale` in `ai-serve.local.env`, then
*Compute & models → Add a machine* gives a one-time command for the new device. Step by step, and the other networks:
[docs/devices.md](docs/devices.md). Tried with a Linux PC and a MacBook; Raspberry Pi and Android not yet.

## Security

By default only this machine can use the dashboard and the API. Other devices on your local network need a key
(`LLAMA_SERVE_API_KEY`, at least 32 characters, in `ai-serve.local.env`); a private overlay network (Tailscale,
WireGuard …) can be configured. Nothing is exposed to the internet unless you set that up explicitly. ai-serve starts
and stops programs on your machine – only add plugins and apps you trust.

## Status

What has actually been run, as of this release (one NVIDIA Linux machine, one Apple Silicon MacBook):

| Part | Linux + NVIDIA | Mac (Apple Silicon) | Other |
|---|---|---|---|
| `./install.sh` on a clean copy, Gemma 4 E2B, chat with images | ✅ | installed and started; several Mac bugs found and fixed, chat not confirmed since | AMD / Vulkan / CPU: not tried |
| Chatterbox (TTS), Whisper (STT) offers in `./ai-serve setup` | engines run here; the offers themselves not run on a clean machine | not tried (Chatterbox uses MLX there) | – |
| Iris (sd-serve) | ✅ clean setup, Vulkan build | memory check fixed after a refusal; an image not confirmed | – |
| ComfyUI (image-serve): Z-Image, Qwen-Image-Edit, style transfer, MOSS | ✅ clean setup | not tried (int8/fp8 weights are made for CUDA; may be slow or fail) | – |
| Faces in photos (vision-serve: YuNet + AuraFace) | ✅ through the gateway, CUDA and CPU | not tried (CPU should work) | – |
| Downloads from the Playground / dashboard | ✅ ComfyUI model bundle; TTS/STT/Iris items unit-tested only | – | – |
| More machines (Tailscale) | ✅ | ✅ (as a device: Metal llama.cpp, model sent, chat through the coordinator) | Raspberry Pi, Android/Termux, WireGuard etc.: not tried |
| `./ai-serve plugin add <git-url>` | ✅ with [ai-serve-strata](https://github.com/geonom/ai-serve-strata) and [ai-serve-piper](https://github.com/geonom/ai-serve-piper) (cloned, manifest checked, setup.sh) | – | – |
| Piper (CPU speech) through ai-serve-piper | ✅ fresh copy: plugin add, voice downloaded from the dashboard API, speech via `/v1/audio/speech` | not tried | Raspberry Pi: not tried |
| Strata through the ai-serve-strata plugin | ✅ (two NVIDIA cards, 96 GB RAM) | not tried | – |

Everything else listed in the docs (other TTS engines, diarization, OCR, the GPU manager …) runs daily on the
developer's NVIDIA machine, but not necessarily from a fresh install. Reports of what works elsewhere are welcome --
for a Mac, [docs/testing-on-a-mac.md](docs/testing-on-a-mac.md) lists the steps.

## Licenses

ai-serve itself: **AGPL-3.0** ([LICENSE](LICENSE)). The models have their own licenses, which the installer shows for
each offer (e.g. Gemma Terms of Use, Chatterbox MIT, Iris Apache-2.0). Some optional engines (e.g. OmniVoice) are not
licensed for commercial use.
