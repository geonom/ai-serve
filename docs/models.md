# Using more models

What it takes to add a model, per kind – from "drop a file in a folder" to "write a few lines of code". Downloads
offered by a plugin (*Compute & models → Downloads*, or the Playground's pages) are the easiest way and come first;
this page is about everything else.

| Kind | How hard | In short |
|---|---|---|
| Language models (chat, vision) | easy | from Hugging Face in the dashboard / `./ai-serve model add`, or a `.gguf` into `models/` |
| Embeddings, reranking | easy | the same – the kind is read from the file |
| Models of Ollama, LM Studio, other servers | easy | *Compute & models → More model servers* |
| Speech output (TTS) | medium | set up one of the bundled engines; a new engine = a provider folder |
| Voices to clone | easy | a `.wav` (+ `.txt` transcript) in `voices/` |
| Speech recognition (STT) | easy for other Whisper sizes, code for other engines | `STT_WHISPER_MODEL` |
| Images (text-to-image, editing, style) | code | workflow, files (encoder, VAE …), nodes and a catalog entry in image-serve, or a plugin of your own |
| LoRAs for images | easy | a `.safetensors` in `models/loras/` |

Restart ai-serve after changing `ai-serve.local.env` (`./ai-serve restart`). New model files are found without one.

## Language models (llama.cpp)

Any GGUF that llama.cpp can run works. Two ways to get one:

**From Hugging Face** – in the dashboard (*Compute & models → Language model from Hugging Face*) or on the command
line:

```bash
./ai-serve model add https://huggingface.co/unsloth/Qwen3-4B-GGUF      # lists the files to choose from
./ai-serve model add hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M                 # or name the quantization directly
```

*Which address?* The page of a **GGUF repo** on huggingface.co – most model makers publish their weights as
safetensors, and others (unsloth, bartowski, ggml-org, lmstudio-community …) convert them; those repos' names end
in `-GGUF`. Search huggingface.co for the model's name plus "GGUF" and copy the address from the browser, e.g.
`https://huggingface.co/unsloth/Qwen3-4B-GGUF`. Also accepted: a file's page
(`https://huggingface.co/<owner>/<repo>/blob/main/<file>.gguf`), its download link (`…/resolve/main/<file>.gguf`),
`<owner>/<repo>` alone, and Ollama's form `hf.co/<owner>/<repo>:<QUANT>`. Not usable: the original repo
(`https://huggingface.co/Qwen/Qwen3-4B` – safetensors), datasets and spaces. Which file: `Q4_K_M` is a good balance
of size and quality; `Q8_0` is closer to the original but twice as big; models split into `-00001-of-0000N` parts
are downloaded with all their parts. A vision projector (`mmproj-*.gguf`) in the repo is offered too and saved under
a name that pairs it with the model. Every file is checked against Hugging Face's sha256; nothing is overwritten,
and a download that doesn't fit on the disk is refused (nothing is deleted). Gated models (Gemma, Llama …): accept
the license on the model's page, create a token at huggingface.co/settings/tokens and put `HF_TOKEN=<token>` into
`ai-serve.local.env`. Downloads go into the first of `LLAMA_MODEL_DIRS` (default: `models/`).

**By hand:** put the `.gguf` into `models/` in the ai-serve folder (or `~/gguf_models`, or any folder listed in
`LLAMA_MODEL_DIRS`).

Either way it appears in the dashboard, in `GET /v1/models` and `GET /v1/llm/models`; a request with its file name
as `model` loads it.

- **Seeing images** needs the model's vision projector next to it, named `<model>-mmproj-<anything>.gguf` or
  `mmproj-<model>.gguf` (repos often call it just `mmproj-F16.gguf` – rename it so it starts with the model's name).
- **Faster decoding with MTP** (models that ship a draft head): a file `mtp-<model>.gguf` next to it.
- **Embedding and reranking models** (bge-m3, nomic-embed, bge-reranker …) go into the same folder; their kind is
  read from the file. Ask them through `/v1/embeddings` and `/v1/rerank`.
- What an "Auto" chat uses: `DEFAULT_CHAT_MODEL` in `ai-serve.local.env`.
- Context size, parallel slots, GPU: per request (`ctx_size`, `parallel`, `gpu_id`) or in the dashboard's
  *Start a model by hand*.

**Other model servers** – Ollama, LM Studio, vLLM, a llama-server you run yourself: *Compute & models → More model
servers*, give the port. Their models then appear next to the local ones. ai-serve doesn't start or stop them.
**Strata** (one very large MoE model): [strata.md](strata.md). **Models on other machines**: [devices.md](devices.md).

Document reading (`/api/ocr`) uses NuExtract3 (`NuExtract3-Q8_0.gguf` + its mmproj). Any vision model can also read
documents through a plain chat request with an image.

## Speech output (TTS)

tts-serve (`plugins/tts-serve`) brings six engines: Chatterbox, Chatterbox through MLX (Apple Silicon), CosyVoice 3,
OmniVoice, Qwen3-TTS and VoxCPM2. Each lives in `providers/<id>/` with its own Python environment.

- **Set up an engine:** *Downloads* in the dashboard or the Playground, or `plugins/tts-serve/providers/<id>/setup.sh`.
  Some need an NVIDIA GPU (the Downloads list only offers what fits the machine). OmniVoice is for non-commercial use
  only.
- **Without a GPU** (a Mac, a Raspberry Pi, an older PC): the Piper plugin speaks fast on the CPU, one voice per
  language, no cloning – `./ai-serve plugin add https://github.com/geonom/ai-serve-piper`, then a voice from
  *Downloads*.
- **Default engine:** `DEFAULT_TTS_ENGINE` (else the first one installed).
- **Voices to clone:** put a short clean recording (5–15 s) into `voices/` as `<name>.wav`, its exact words as
  `<name>.txt`; then `"voice": "<name>"` on `/v1/audio/speech`. Only use voices you have the right to clone.
- **A new engine** is a new provider folder – no change to ai-serve or tts-serve: a `llama-serve.provider.json`
  (what it is, its modes, languages, parameters), an `adapter.py` (turns the common request into the engine's), a
  `worker.py` (a small HTTP server that loads the model) and a `setup.sh`. Copy `providers/chatterbox/` and adapt it.

## Speech recognition (STT)

stt-serve (`plugins/stt-serve`) runs WhisperX (+ pyannote for telling speakers apart), optionally Nemotron diarization
and MOSS-Transcribe (both NVIDIA only, set up from *Downloads*).

- **Another Whisper size:** `STT_WHISPER_MODEL=large-v3` (or `medium`, `turbo`, … – also a Hugging Face repo of a
  CTranslate2 conversion, or a local folder) in `ai-serve.local.env`. Bigger is more accurate and slower. Default:
  `small`.
- **Speakers:** with Whisper, pyannote's gated model – needs `HF_TOKEN` once its terms are accepted (see the README);
  recognising a speaker across recordings uses it as well. MOSS-Transcribe (one pass: text, speakers, timestamps) and
  Nemotron diarization are not gated, but need an NVIDIA GPU.
- **Another engine** (e.g. Parakeet, Canary): that is code in stt-serve today (`stt_serve/engine.py`) – there is no
  provider folder for STT yet.

## Images

image-serve (`plugins/image-serve`, ComfyUI) offers Z-Image Turbo, ERNIE Turbo, Qwen-Image 2.1, Qwen-Image-Edit 2511
(editing, style transfer) and MOSS-Image; sd-serve (`plugins/sd-serve`) offers Iris. What's downloadable is in
*Downloads*.

- **Any model, by hand:** ComfyUI's own interface runs every model and workflow ComfyUI supports – open image-serve
  in *Apps & Services* (it passes ComfyUI's interface through). Model files go into ComfyUI's folders
  (`plugins/image-serve/ComfyUI/models/<checkpoints|diffusion_models|text_encoders|vae|loras>/`, or
  `COMFYUI_MODELS_ROOT`).
- **LoRAs** for Z-Image and ERNIE: `.safetensors` into `models/loras/<subfolder>/`, then `lora_name` on
  `/api/comfyui/generate` (`GET /api/comfyui/loras` lists them). An optional `<name>.json` next to it can carry its
  trigger words.
- **A new model in the API and the Playground** (`/v1/images/generations`, `/v1/images/edits`): today that's code in
  image-serve. An image model rarely comes alone – it usually needs its own text encoder and VAE (sometimes a turbo
  LoRA), its own workflow (sampler, guidance, latent size), and often a newer ComfyUI or custom nodes (MOSS-Image
  needed ComfyUI v0.42, Qwen-Image 2.1 v0.37 plus a patched ComfyUI-GGUF). So: build and test the workflow in
  ComfyUI's interface first, export it (*Export (API)*) into `workflows/`, add an entry to `image_serve/catalog.py`
  (id, label, file, tasks, steps), a download bundle with every file it needs to `image_serve/bundles.py`, pin any
  custom node in `setup.sh`, and, for a new model family, teach `image_serve/comfy.py` which workflow it uses and
  where prompt/seed/steps/size go. Another quantization of a family that's already there (a different Z-Image GGUF,
  say) only needs the catalog entry.
- **A different image engine** (Diffusers, stable-diffusion.cpp with other models, a remote service): an `image`
  plugin of your own – [plugins.md](plugins.md); sd-serve is a small example.

## Not covered yet

Video, music, 3D, translation and realtime voice conversation have no plugin yet; the plugin contracts are there so
the first ones fit ([plugins.md](plugins.md)).
