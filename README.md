# ai-serve

A gateway for local AI on your own machine: language models (llama.cpp), text-to-speech, speech-to-text, image
generation and document reading – behind **one** OpenAI-compatible API (`/v1/...`) plus a dashboard. Models are
loaded when they're needed and unloaded when the memory is needed for something else.

> **Alpha.** Runs on Linux (NVIDIA, AMD, Vulkan, CPU) and macOS (Apple Silicon). Windows will come later.

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

The installer fetches the right llama.cpp build for this machine and offers a basic set of services – all optional,
every download is verified by sha256:

| Offer | What | Approx. size |
|---|---|---|
| Gemma 4 E4B or E2B | Language model that also understands images | 5.5 / 3.8 GB |
| Chatterbox Multilingual | Text-to-speech with voice cloning (through MLX on a Mac) | 3–7 GB |
| Whisper (WhisperX small) | Speech-to-text with speaker diarization | 2–8 GB |
| Iris-3B Turbo | Images from text descriptions (stable-diffusion.cpp) | 5.5 GB |
| ComfyUI (image-serve) | Editing images by instruction, style transfer (Qwen-Image-Edit 2511), Z-Image Turbo; MOSS-Image (text in images, placing things in boxes) as a later download | ~30 GB |

Then: `./ai-serve start` → <http://localhost:8030> (another port: `AI_SERVE_PORT=8031 ./ai-serve start`;
in the background: `./ai-serve start -d`; also `./ai-serve stop`, `restart`, `status`). The
**Spielwiese** ("playground", under *Apps & Services*) lets you try everything the gateway offers. The dashboard and the
Spielwiese are in English, with a switch to German at the top right.

## Extend

Everything except llama.cpp is a **plugin**: a separate program with a small manifest (`llama-serve.service.json`)
that ai-serve starts when it's needed. Bundled: `plugins/tts-serve`, `plugins/stt-serve`, `plugins/sd-serve`, `plugins/image-serve`
(ComfyUI). Models a plugin offers but that aren't installed yet can be downloaded from the Spielwiese or the
dashboard (*Compute & models → Downloads*). More
from any git repository:

```bash
./ai-serve plugin add https://github.com/<someone>/<plugin>
./ai-serve plugin list
```

A plugin is someone else's code: `plugin add` checks its manifest, and runs its `setup.sh` only after you confirm.

Existing model servers such as **Ollama** or **LM Studio** are detected and can be added in the dashboard
(*Rechner & Modelle → Weitere Modellserver*), on custom ports too. The other way round, the Spielwiese also works
directly against an Ollama or LM Studio server. Writing a plugin: [docs/plugins.md](docs/plugins.md),
[docs/service-manifest.md](docs/service-manifest.md). API: [docs/API.md](docs/API.md).

## Security

By default only this machine can use the dashboard and the API. Other devices on your local network need a key
(`LLAMA_SERVE_API_KEY`, at least 32 characters, in `ai-serve.local.env`); a private overlay network (Tailscale,
WireGuard …) can be configured. Nothing is exposed to the internet unless you set that up explicitly. ai-serve starts
and stops programs on your machine – only add plugins and apps you trust.

## Licenses

ai-serve itself: **AGPL-3.0** ([LICENSE](LICENSE)). The models have their own licenses, which the installer shows for
each offer (e.g. Gemma Terms of Use, Chatterbox MIT, Iris Apache-2.0). Some optional engines (e.g. OmniVoice) are not
licensed for commercial use.
