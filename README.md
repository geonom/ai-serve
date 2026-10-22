# ai-serve

**Deutsch** · [English](#english)

Ein Gateway für lokale KI auf dem eigenen Rechner: Sprachmodelle (llama.cpp), Sprachausgabe, Spracherkennung,
Bilderzeugung und Dokumente lesen – hinter **einer** OpenAI-kompatiblen Schnittstelle (`/v1/...`) und einem Dashboard.
Modelle werden geladen, wenn sie gebraucht werden, und wieder entladen, wenn der Speicher anderweitig gebraucht wird.

> **Alpha.** Läuft auf Linux (NVIDIA, AMD, Vulkan, CPU) und macOS (Apple Silicon). Windows kommt später.

## Installieren

Voraussetzungen: `git` und [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/owner/ai-serve.git && cd ai-serve
./install.sh
```

Der Installer holt das passende llama.cpp für diesen Rechner und bietet eine Grundausstattung an – alles optional,
jeder Download wird per sha256 geprüft:

| Angebot | Was | Größe ca. |
|---|---|---|
| Gemma 4 E4B oder E2B | Sprachmodell, das auch Bilder versteht | 5,5 / 3,8 GB |
| Chatterbox Multilingual | Sprachausgabe mit Stimmklonen (auf dem Mac über MLX) | 3–7 GB |
| Whisper (WhisperX small) | Spracherkennung mit Sprechertrennung | 2–8 GB |
| Iris-3B Turbo | Bilder aus Beschreibungen (stable-diffusion.cpp) | 5,5 GB |

Danach: `./ai-serve start` → <http://localhost:8030>. Die **Spielwiese** (unter *Apps & Services*) ist eine
Oberfläche zum Ausprobieren von allem, was das Gateway anbietet.

## Erweitern

Alles außer llama.cpp sind **Plugins**: eigene Programme mit einer kleinen Beschreibung
(`llama-serve.service.json`), die ai-serve bei Bedarf startet. Beiliegend: `plugins/tts-serve`, `plugins/stt-serve`,
`plugins/sd-serve`. Weitere aus einem Git-Repository:

```bash
./ai-serve plugin add https://github.com/<jemand>/<plugin>
./ai-serve plugin list
```

Vorhandene Modellserver wie **Ollama** oder **LM Studio** werden erkannt und lassen sich im Dashboard einbinden
(*Rechner & Modelle → Weitere Modellserver*), auch auf eigenen Ports. Umgekehrt funktioniert die Spielwiese auch direkt
gegen einen Ollama/LM-Studio-Server. Wie man ein Plugin schreibt: [docs/plugins.md](docs/plugins.md),
[docs/service-manifest.md](docs/service-manifest.md). API: [docs/API.md](docs/API.md).

## Sicherheit

Das Dashboard und die API sind standardmäßig nur von diesem Rechner aus erreichbar. Für andere Geräte im lokalen Netz
braucht es einen Schlüssel (`LLAMA_SERVE_API_KEY`, mindestens 32 Zeichen, in `ai-serve.local.env`); ein privates
Overlay-Netz (Tailscale, WireGuard …) lässt sich einbinden. Nichts wird ins Internet veröffentlicht, solange man es
nicht ausdrücklich einrichtet.

## Lizenzen

ai-serve selbst: **AGPL-3.0** ([LICENSE](LICENSE)). Die Modelle haben eigene Lizenzen, die der Installer jeweils
anzeigt (z. B. Gemma Terms of Use, Chatterbox MIT, Iris Apache-2.0). Manche optionalen Engines (z. B. OmniVoice) sind
nicht für kommerzielle Nutzung freigegeben.

---

## English

A gateway for local AI on your own machine: language models (llama.cpp), text-to-speech, speech-to-text, image
generation and document reading – behind **one** OpenAI-compatible API (`/v1/...`) plus a dashboard. Models load on
demand and are unloaded when the memory is needed for something else.

> **Alpha.** Runs on Linux (NVIDIA, AMD, Vulkan, CPU) and macOS (Apple Silicon). Windows later.

**Install:** needs `git` and [uv](https://docs.astral.sh/uv/). `git clone https://github.com/owner/ai-serve.git && cd ai-serve && ./install.sh` fetches the right llama.cpp build for
this machine and offers a basic set – Gemma 4 E4B/E2B (vision-capable LLM), Chatterbox Multilingual (TTS, MLX on
Macs), Whisper (STT with diarization), Iris-3B Turbo (images) – all optional, every download sha256-verified. Then
`./ai-serve start` → <http://localhost:8030>; the **Spielwiese** app is a playground for everything the gateway offers
(its UI is German for now).

**Extend:** everything but llama.cpp is a plugin – a program with a small manifest that ai-serve starts when needed.
`./ai-serve plugin add <git-url>` adds one from any git repository. Ollama and LM Studio are detected and can be added
as model servers (custom ports too); the Spielwiese also works directly against them. Writing a plugin:
[docs/plugins.md](docs/plugins.md).

**Security:** by default only this machine can use the dashboard and API; other LAN devices need
`LLAMA_SERVE_API_KEY` (≥32 chars). Nothing is exposed to the internet unless you set that up explicitly.

**License:** AGPL-3.0. Models carry their own licenses (shown by the installer); some optional engines are
non-commercial.
