# 🦙 AI-Serve Dashboard API Guide

> Engines named below (OmniVoice, CosyVoice, Qwen3-TTS, VoxCPM, Nemotron, MOSS …) are only there once their plugin
> provider is set up; the installer offers Chatterbox and Whisper. Everything here was exercised on the developer's
> NVIDIA Linux machine – see the README's *Status* for what has been tried elsewhere.

The AI-Serve Dashboard provides a RESTful API to manage multiple `llama-server` instances programmatically.

## Base URL
Default: `http://localhost:8030`

## Endpoints

### 1. List Available Models
**GET** `/api/models`
Returns a list of GGUF models and vision modules available in the configured storage directory. Each model entry includes `context_length` (its trained context size, read from the GGUF's `<arch>.context_length` metadata key, header-only — `null` if the key is missing for that architecture) alongside `name`, `path`, `mtp_draft_path`, `mtp_bundled`, `mmproj_path`, `vision_bundled`.

**Example:**
```bash
curl http://localhost:8030/api/models
```

---

### 2. GPU Status
**GET** `/api/gpu/status`
Returns real-time VRAM usage for all NVIDIA GPUs on the system.

**Example:**
```bash
curl http://localhost:8030/api/gpu/status
```

---

### 3. Start a Model
**POST** `/api/start`
Launches a new `llama-server` instance. If `port` names an already-running managed instance, that
instance is stopped and replaced — unless it has requests actively in flight, in which case the
call refuses with **409** (pass `force: true` to replace it anyway). This guards against two
concurrent callers that both default to the same port silently killing each other's in-flight work.

**Body:**
```json
{
  "model_path": "/path/to/model.gguf",
  "mmproj_path": "/path/to/mmproj.gguf",
  "port": 8001,
  "ctx_size": 32000,
  "threads": 8,
  "n_gpu_layers": -1,
  "gpu_id": 0,
  "force": false
}
```

**Example:**
```bash
curl -X POST http://localhost:8030/api/start \
     -H "Content-Type: application/json" \
     -d '{"model_path": "/home/me/gguf_models/Llama-3.gguf", "port": 8031}'
```

---

### 4. Stop a Model
**POST** `/api/stop?port={port}&force={bool}`
Terminates the instance running on the specified port. If it has requests actively in flight, this
refuses with **409** instead of killing it out from under another caller — pass `force=true` to
stop it anyway. (Two independent clients sharing this gateway can otherwise silently kill each
other's in-flight requests; this guard, and the same one on `/api/start`'s port-replace path and
the two endpoints below, closes that gap.)

**Example:**
```bash
curl -X POST http://localhost:8030/api/stop?port=8031
curl -X POST http://localhost:8030/api/stop?port=8031&force=true   # skip the busy check
```

---

### 4.1. Unload a Model
**POST** `/api/models/unload`
Manually unloads a model by specifying either the port or model name in the JSON body. A busy
`port` target refuses with 409 unless `force` is set; a busy `model_name` match is silently
skipped (and reported in the response message) unless `force` is set.

**Request Body:**
```json
{
  "port": 8031,
  "model_name": "Llama-3",
  "force": false
}
```
*(Provide either `port` or `model_name`)*

**Example:**
```bash
curl -X POST http://localhost:8030/api/models/unload \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Llama-3"}'
```

---

### 4.2. Unload All Models
**POST** `/api/models/unload_all?gpu_id={int}&force={bool}`
Manually unloads all running model server instances from the backend and frees up VRAM. Any
instance with requests actively in flight is skipped by default (reported in the response
message) — pass `force=true` to unload busy instances too.

**Example:**
```bash
curl -X POST http://localhost:8030/api/models/unload_all
```

---


### 5. Instance Status
**GET** `/api/status`
Returns the status of all managed instances, including whether they are `running` and `ready` (passed health check).

**Example:**
```bash
curl http://localhost:8030/api/status
```

---

### 5.1. Tokenize
**POST** `/api/tokenize`
Returns the exact token count for `content` under a given model's tokenizer. Request body: `content` (str, required), `model_name` (optional — fuzzy-matched: bare filename, stem, or substring). If a llama-server instance for `model_name` (or any instance, if `model_name` is omitted) is already running, tokenizes through that instance's own `/tokenize`. Otherwise falls back to the `llama-tokenize` CLI directly against the resolved GGUF file — tokenizer-only, does **not** load the model into VRAM (confirmed ~1s against a 27B model, dominated by file-open, not compute) — so a caller can get an exact count before deciding whether a model is worth loading at all. With no running instance and no `model_name`, the response's `error` lists real candidate model names rather than a bare "not found".
* Response: `success` (bool), `tokens_count` (int), `error` (str or null).

**Example:**
```bash
curl -X POST http://localhost:8030/api/tokenize \
     -H "Content-Type: application/json" \
     -d '{"content": "Hallo, wie geht es dir?", "model_name": "Qwen3-4B-Q6_K"}'
```

---

### 6. Comparison Chat (Broadcast)
**POST** `/api/chat/compare`
Sends a prompt to all `Ready` instances and returns combined results with performance metrics.

**Example:**
```bash
curl -X POST http://localhost:8030/api/chat/compare \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello!"}'
```

### 7. Benchmarking
**POST** `/api/benchmark/start`
Starts an automated sequential benchmark suite across multiple models.

**GET** `/api/benchmark/status`
Returns real-time progress of running benchmark.

**GET** `/api/benchmark/history`
Lists all completed benchmark result files.

**GET** `/api/benchmark/results/{filename}`
Retrieves a specific historical benchmark run.

**GET** `/api/benchmark/suites`
Lists available benchmark suite JSON files.

---

### 8. Unified Inference Gateway
**POST** `/api/chat/completions`
A unified chat completion endpoint that acts as a gateway proxy. Auto-loads the requested model, implements automatic retries on JSON validation error, and handles GPU VRAM limit routing (failover & HTTP 507).

---

### 8.1. Model servers from plugins (e.g. Strata)
Other OpenAI-compatible servers come from `llm` plugins (`docs/plugins.md`) — e.g. **Strata** through its adapter
(the separate `strata-plugin` app). The gateway knows none of them by name:

* `GET /v1/llm/models` lists their models with `where: [{"place": "plugin", "plugin", "name", "state": stopped |
  loading | ready}]` and `exclusive: true` for a plugin that owns every GPU while it runs.
* `/v1/chat/completions`, `/api/chat/completions`, `/v1/embeddings`, `/v1/rerank` with such a model (id or alias,
  case-insensitive, a path's basename counts) go to the plugin's server; the first request may wait while it loads.
* `POST /api/start` with such a model loads it through the plugin and answers once it serves — **on the plugin's
  port**: `{"port": <plugin port>, "requested_port": <yours>, "model", "max_context", "status"}`. Use the returned
  port. `ctx_size` above the model's `context_length` → **400**.
* `POST /api/stop?port=<plugin port>`, `POST /api/models/unload` (`port` or a plugin model name) and
  `POST /api/models/unload_all` (any `gpu_id` for an exclusive plugin) release it through the plugin — **409** while
  busy unless `force=true`.
* `GET /api/status` lists a loaded plugin server with `plugin`, its `port` and `unload_endpoint`
  `/api/plugins/<id>/release?worker=<worker>`.
* `GET /api/plugins` — every plugin with its capabilities, GPU use and status; `POST /api/plugins/{id}/release`
  (`worker?`, `force?`) frees what it holds (**409** busy, **404** unknown).

**GPU guard:** while an exclusive plugin's server runs or loads, loading anything else onto the GPUs is refused with
**507**. It takes the GPUs with `POST /api/coordination/gpu-exclusive {"purpose"}` → `{granted, reservation_ids,
unloaded, reason}` (idle consumers are unloaded; busy/pinned ones refuse it — fast, no waiting) and confirms each
reservation with its server's PID.

---

### 9. Asynchronous ComfyUI Task Queue & Polling
**POST** `/api/comfyui/generate`
Starts a background image generation task. Auto-launches ComfyUI if not running. Returns a `task_id`.

**POST** `/api/comfyui/edit`
Starts a background image edit task.
* Request body fields: `image_base64` (str), `prompt` (str), `model_name` (optional str, default `"Qwen_Image_Edit-Q2_K.gguf"`), `steps` (optional int), `seed` (optional int), `denoise` (optional float, 0.0 - 1.0).

**POST** `/api/comfyui/style-transfer`
Transfers the artistic style of a reference image to a content image.
* Request body fields: `content_image_base64` (str), `style_image_base64` (str), `prompt` (optional str), `model_name` (optional str, default `"qwen-image-edit-2511-Q2_K.gguf"`), `steps` (optional int), `seed` (optional int), `denoise` (optional float 0.0 - 1.0, default `1.0`; values around `0.6` to `0.8` provide balanced style transfer while preserving original composition).

**GET** `/api/comfyui/task/{task_id}`
Retrieves the execution status, node progress, and final base64 image of a queued ComfyUI task.

---

### 10. Speech Synthesis & Transcription (TTS & STT)
**GET** `/api/voices`
Returns a list of local preset voices, their absolute paths, and their stored transcriptions.

**POST** `/api/tts/generate`
Generates speech. Engine-agnostic — `engine` selects the backend; a caller that just wants speech omits it and gets the **server default** (`DEFAULT_TTS_ENGINE`; unset = the first installed engine – with the installer's offer that is Chatterbox). An unspecified-engine request with no reference clip clones `DEFAULT_TTS_VOICE`; if tts-serve isn't available the request gets 503.

| `engine` | What | Runs in | Weights license |
|---|---|---|---|
| `"cosyvoice"` | **CosyVoice 3** — clone from a reference clip, and with `variant="instruct"` a **free-form style description** ("slower, angrier, higher") *on the cloned voice*. Language is inferred from the text. | tts-serve | Apache-2.0 |
| `"omnivoice"` | Clone, **or** 23 fixed instruct-design tags (accent/age/pitch/gender/whisper) — never both. | tts-serve | **CC-BY-NC — non-commercial only** |
| `"qwen3"` | 3 checkpoints via `variant` (`base` clone / `voice_design` style / `custom_voice` preset+style). Clone and style are mutually exclusive. | tts-serve | Apache-2.0 |
| `"chatterbox"` | **Chatterbox-Multilingual** — clone + expressiveness knobs (`exaggeration` / `cfg_weight`). 23 languages incl. German. Output carries an inaudible Perth watermark. | tts-serve | MIT |
| `"voxcpm"` | **VoxCPM2** on Nano-vLLM — clone (`variant` `reference` / `prompt`), batches concurrent requests, fastest per word. | tts-serve | Apache-2.0 |

Every engine runs in **tts-serve** (`~/projects/tts-serve`, port 8050), a separate service this gateway forwards to and starts on demand; aliases (e.g. `cosy`, `chatter`, `qwen3tts`) come from its provider manifests. If tts-serve is unavailable, TTS answers **503**. Engine-specific validation errors from tts-serve come back as 400 with its message. Setting up an engine: `providers/<id>/setup.sh` in tts-serve.

* Common body fields: `text` (str), `language` (str, e.g. `"en"`, `"de"` — mapped to the full name for `engine="qwen3"`; passed as-is otherwise; **ignored by CosyVoice**, which infers it), `engine` (omit for the default), `response_format` (`"json"` or `"binary"`), `keep_alive` (int, default 300).
* `engine="cosyvoice"` `variant`: `"zero_shot"` (default, clone; uses `ref_text` — a `voices/` preset's `.txt` sidecar or `DEFAULT_TTS_VOICE`'s automatically — but **transparently switches to `cross_lingual`** when there's no substantial `ref_text`, and the worker also retries via `cross_lingual` if `inference_zero_shot` hits its short-input crash), `"instruct"` (clone **+** free-form `instruct`; the one to use for "clone this voice but slower/angrier"), `"cross_lingual"` (clone from the reference **audio alone**, no `ref_text`; robust to any input length). With no reference at all, the default voice is cloned.
* `engine="chatterbox"`: optional `ref_voice_path`/`ref_audio_base64` for cloning; `exaggeration` (~0..1), `cfg_weight` (~0..1), `temperature`. `instruct` is ignored (no free-form channel).
* **Voice cloning** (both engines): pass `ref_voice_path` (a preset under `voices/`) *or* `ref_audio_base64` (an uploaded reference clip), plus `ref_text` (its transcript). `ref_text` is optional for OmniVoice — WhisperX auto-transcribes when omitted — but **required** for `engine="qwen3"` (`variant="base"`).
* **Described voice** (no reference clip): `instruct` — a natural-language voice/style description. OmniVoice uses it directly; for `engine="qwen3"` set `variant="voice_design"`.
* `engine="qwen3"` picks a checkpoint via `variant`:
  * `"base"` (default) — zero-shot voice cloning (see above). Reproduces the reference clip's delivery **as-is**. The pinned `qwen-tts` (0.1.1) `generate_voice_clone()` has **no style channel**, so an `instruct` sent with `variant="base"` is forwarded to the worker but currently a **no-op** (a warning is logged; visible via `GET /api/logs/tts`). To steer style you need `variant="voice_design"` or `variant="custom_voice"`.
  * `"voice_design"` — voice built purely from a free-form `instruct` description (timbre / emotion / pace / prosody), no reference clip.
  * `"custom_voice"` — one of 9 preset timbres via `speaker` (required), with an optional `instruct` style layered on top. This is the only variant that combines a fixed voice **and** a style instruct.
* `POST /api/tts/qwen3/generate` still exists but is **deprecated** in favour of `engine="qwen3"` here.

**GET** `/api/tts/status`
Engine-agnostic TTS subsystem status for other processes to poll — is it up, still loading, or busy — without caring which engine serves the request.
* Returns: `status` (`error` | `starting` | `loading` | `generating` | `idle` | `unloaded` — the worst engine state), `ready` (bool), `busy` (bool), `default_engine` (from `DEFAULT_TTS_ENGINE`, else the first installed engine, e.g. `"cosyvoice"`), and `engines` (`omnivoice`, `qwen3tts`, `cosyvoice`, `chatterbox`) with per-engine `status`/`loaded`/`device` (plus `variant`/`pid` for the workers).
* The same payload is also embedded as the `tts` key of `GET /api/system/status`.

**GET** `/api/logs/tts`
Recent TTS log lines, engine-agnostic: the TTS-related tail of `backend.log` (forwarding) plus tts-serve's worker logs (current run of each).

**POST** `/api/tts/unload` (legacy path for OmniVoice) · **POST** `/api/tts/{engine}/unload` (a tts-serve engine, e.g. `/api/tts/cosyvoice/unload`; `?force=true` to unload while busy, otherwise 409) → `{"status": "unloaded" | "already unloaded"}`.

**POST** `/api/coordination/gpu-preflight` — body `{purpose, gpu, need_mb}` → `{"free_mb"}`. Asked by tts-serve and stt-serve before a model loads: 507 while Strata holds the GPUs; for a non-TTS purpose short on room, idle tts-serve engines on that GPU are unloaded first; otherwise reports the GPU's free VRAM.

**GET** `/api/stt/diarize/backends`
Reports which `diarize_backend` values are actually usable right now — `{backends: {whisperx: {available, loaded, description}, nemotron: {available, loaded, description}}}`. `nemotron.available` is `false` (with a setup hint in `description`) until `scripts/setup_nemotron_diarize.sh` has been run.

---

**POST** `/api/stt/transcribe`
Transcribes audio using WhisperX. **All STT runs in stt-serve** (`~/projects/stt-serve`, port 8060), which this gateway forwards to and starts on demand (503 if unavailable); `audio_path` is resolved here (inside `voices/` only) and sent on as base64. `audio_base64`, `audio_base64_list`, or `audio_path`, plus optional `diarize`.
* Request body: `audio_base64`, `audio_base64_list` (list of str — several independently-encoded containers, each decoded on its own then concatenated as PCM before processing; useful for assembling one clip out of several short chunks, e.g. enough combined duration for a reliable speaker embedding), or `audio_path`; `keep_alive` (int, default 300); `diarize` (bool, default `false`) — label each turn with a speaker; `diarize_backend` (`"whisperx"` default, or `"nemotron"`) — which model does the diarizing, see below; `num_speakers` (exact count, most accurate if known), `min_speakers` / `max_speakers` (bounds, ignored if `num_speakers` is set); `language` (ISO 639-1, e.g. `"de"`) — pin the transcription language instead of WhisperX's own first-30s auto-detect, which is unreliable right after a cold model load (confirmed live: same German clip translated to fluent English on the first call after load, correct German on the next). `task="transcribe"` is always forced server-side regardless, so auto-detect drift can never silently turn into a translation.
* `diarize_backend="whisperx"` (default): pyannote.audio via `whisperx.diarize`. `diarize_backend="nemotron"`: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) via an isolated-venv subprocess worker (`NEMOTRON_DIARIZE_PYTHON_BIN`, `scripts/setup_nemotron_diarize.sh`) -- needed because its transformers support is git-main-only as of integration, incompatible with pinning it alongside OmniVoice's own transformers requirement in this project's main `venv/`. Word-level alignment (for per-word speaker labels, not just per-segment) always runs via WhisperX regardless of which backend diarizes. **Does not support** `return_embeddings`/`known_speakers` (Nemotron doesn't produce comparable speaker embeddings -- a 400 if combined); `num_speakers`/`min_speakers`/`max_speakers` are honored only approximately (the model has no native "expect exactly N speakers" input -- detected speakers are ranked by total speaking duration and top-N kept, not a real clustering constraint the way pyannote's own `num_speakers` is). Verified live end-to-end (real two-speaker audio): correct speaker split at the true boundary, both cold-load (~36s, includes the ~100M-param HF download) and warm (~0.2s) calls.
* `diarize=true` speaker-identity fields: `return_embeddings` (bool, default `false`) — include a `speaker_embeddings: {label: [floats]}` dict in the response, one pyannote embedding (256 floats) per detected speaker, persistable and reusable across separate calls (diarization's own `SPEAKER_00`-style labels are only stable *within* one call); `known_speakers` (list of `{label, embedding}` — a roster to cosine-match detected speakers against, embeddings from a prior call's `return_embeddings` response); `match_threshold` (float, default `0.7`) — minimum cosine similarity to count as a match. A match **replaces** the generic label everywhere in the response (segments and `transcript`); every segment also gets explicit `similarity` (best score, populated even below threshold) and `matched`/`matched_label` fields, so a caller never has to infer a match from the label string alone.
* Response: `transcript` (str — plain text normally; `"SPEAKER_00: ...\nSPEAKER_01: ..."` per turn when `diarize=true`, using matched labels where applicable) and `segments` (list of `{start, end, text}`, plus `speaker`/`similarity`/`matched`/`matched_label` per entry when diarized). `duration_s` (float) is always included when computable. When `diarize=true`, `segments` are **speaker turns**, not raw whisper segments — WhisperX's own segmentation routinely spans a speaker change (confirmed live: two speakers back-to-back with no silence gap came back as one segment), so diarization runs WhisperX's word-alignment step first and regroups by per-word speaker; the `.txt` sidecar written for `audio_path` (used elsewhere as a TTS reference's `ref_text`) always gets the plain unlabeled transcript, never speaker prefixes.
* `warning` (str, only present) when `return_embeddings`/`known_speakers` was requested on audio under ~8s — speaker embeddings are unreliable that short (confirmed: a real speaker can be missed entirely at 4s, clean separation only from ~8s on); the transcript itself is unaffected, only speaker-identity fields are suspect.
* Diarization lazy-loads a second in-process model on first use (same venv as WhisperX — `whisperx` and `pyannote-audio` are already dependencies, no separate install) and a language-specific word-alignment model on top of that. `HF_TOKEN` env var is optional but recommended for the diarization pipeline's HF download.
* `POST /api/stt/diarize/unload` unloads just the whisperx-backend diarization + alignment models (keeps WhisperX itself loaded); `POST /api/stt/diarize/nemotron/unload` unloads the Nemotron worker subprocess; `POST /api/stt/unload` unloads everything STT-related, including the Nemotron worker if it's running.

**Live STT** (chunked, VAD-based incremental transcription over a session — REST polling, not a websocket):
* **POST** `/api/stt/live/start` — body: `language`, `diarize`, `diarize_backend` (same as `/api/stt/transcribe`, same `return_embeddings`/`known_speakers` restriction with `"nemotron"` — rejected with 400 at session-start time, not per-chunk), `silence_ms` (default `1200` — trailing silence that finalizes a segment; lower it, e.g. `400`, for lively conversation where a long threshold never flushes), `keep_alive` (default `600` — idle seconds before the session force-finalizes and is dropped), `return_embeddings`/`known_speakers`/`match_threshold` (same as above, applied per finalized segment). Returns `{session_id, silence_ms, keep_alive}`.
* **POST** `/api/stt/live/chunk` — body: `session_id`, `audio_base64` (one chunk, any container). Appended to the session's buffer; finalizes into a segment automatically whenever trailing silence crosses `silence_ms` (checked on every `/chunk` *and* every `/poll`, so a caller that never polls still gets segments finalized promptly). Returns `{status, new_segments, pending_duration_s}`.
* **GET** `/api/stt/live/poll?session_id=...&since=0` — returns newly finalized segments (`segments[since:]`), `next_since` (pass as the next call's `since`), `pending_duration_s`, `closed`.
* **POST** `/api/stt/live/stop` — body: `session_id`. Force-finalizes whatever's pending regardless of trailing silence, removes the session, returns the full `{status, transcript, segments}`.
* **PATCH** `/api/stt/live/settings` — body: `session_id`, optional `silence_ms`, `keep_alive` — adjust a running session live.
* **GET** `/api/stt/live/sessions` — lists active sessions: `session_id`, `created_at`, `idle_seconds`, `language`, `diarize`, `silence_ms`, `keep_alive`, `segment_count`, `pending_duration_s`.
* Segment finalization uses a plain energy/RMS trailing-silence check, not the neural VAD models this project uses elsewhere for batch diarization — a deliberate lighter-weight choice for a check that runs on every chunk of a live stream. Diarization within a live session is segment-level only (no word-alignment step); the finalized chunk's dominant speaker (by total diarized overlap) is used.

---

### 11. OpenAI Compatible Audio API (Dashboard Gateway)
The gateway exposes OpenAI compatible endpoints under `http://localhost:8030/v1` for easy integration with standard OpenAI clients:

**POST** `/v1/audio/speech`
Generates binary WAV audio. The `voice` field can be a filename from the `voices/` directory (matched prefix, case-insensitive, e.g. `"synth-default"`), one of OpenAI's built-in voice names (ignored — treated as "no voice"), otherwise a voice **description**. (`voices/` may also hold local clips of real people — those are the operator's own data; don't rely on them in shared examples or publish anything derived from them.)
* Engine follows `model`: `"qwen"` in it → Qwen3-TTS, `"chatter"` → Chatterbox, `"omni"` → OmniVoice; anything else (incl. `"tts-1"`) → the server default (`DEFAULT_TTS_ENGINE`, else the first installed engine).
* **CosyVoice:** a matched preset `voice` → clone it (`zero_shot`, transcript from its `.txt`); a free-form `voice` → that description drives `instruct` on `DEFAULT_TTS_VOICE`; nothing → `zero_shot` on `DEFAULT_TTS_VOICE`.
* **OmniVoice:** the description is an instruct-design tag — narrow vocabulary (accent / age / pitch / whisper); anything outside it fails upstream (HTTP 500, model's error in `detail`).
* **Qwen3:** a non-file `voice` routes to the **VoiceDesign** checkpoint (free-form, no fixed vocabulary). For reference-clip cloning or preset timbres, use `POST /api/tts/generate` with `engine="qwen3"` and `variant="base"` / `"custom_voice"`.

**POST** `/v1/audio/transcriptions`
Accepts `file` (multipart/form-data) and transcribes using WhisperX. Optional form fields: `language` (ISO 639-1, e.g. `de` — pass it whenever known, auto-detect is unreliable) and `response_format` (`json` default → `{"text"}`, `text`, `verbose_json` → `{task, text, language, duration, segments[{id,start,end,text}]}`, `srt`, `vtt`). `/v1/audio/translations` is not implemented.

---

### 12. Text Embeddings API
**POST** `/v1/embeddings` (OpenAI Compatible)
**POST** `/api/embeddings` (Unified Dashboard Gateway)

Calculates vector embeddings for a given input text using the specified model. If the model is not currently loaded, the gateway will auto-load it on an available GPU (ensuring the `--embedding` flag is enabled).

**Body:**
```json
{
  "model": "your-model-name.gguf",
  "input": "The food was delicious and the service was excellent.",
  "encoding_format": "float"
}
```

**Example:**
```bash
curl -X POST http://localhost:8030/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "gemma-2-9b-it.Q8_0.gguf", "input": "Hello world"}'
```

---

### 13. Public API (`/public/v1`) — curated surface for external clients

The **only** path meant to be reachable from outside this machine, via `scripts/public_tunnel.sh` (Tailscale, own port 8443, only `/public` mounted). Client config, e.g. Nextcloud's "OpenAI and LocalAI integration": service URL `https://<host>.ts.net:8443/public/v1`, API key = `LLAMA_SERVE_PUBLIC_API_KEY`.

* **Auth:** `Authorization: Bearer <LLAMA_SERVE_PUBLIC_API_KEY>` (env or gitignored `.env.public`, ≥32 chars). Unset/too short → 503; wrong key → 401. Checked before the body is read. Independent of `LLAMA_SERVE_API_KEY`.
* **Tunnel requests** (`Tailscale-Funnel-Request` / `Cf-Connecting-Ip` present) get 404 for every path outside `/public/` — the backend enforces this itself, not only the tunnel.
* **Models** come only from `public_models.json` (path via `PUBLIC_MODELS_FILE`, re-read per request). Entry: `id` (what clients see), `kind` (`chat` / `stt` / `tts` / `image`), `target` (bare GGUF filename / TTS engine / ComfyUI UNET), optional `defaults` (fields applied unless the caller sets them — the only way to set `gpu_id`, `ctx_size`, `keep_alive`, …), for `tts` a `voices` allowlist. No `model` → first entry of that kind; unknown → **400** (`model_not_found`, listing the valid ids — not 404: Nextcloud probes endpoints with `model: "invalid-model"` and reads 404 as "endpoint missing").
* **Limits:** body 25 MB (transcription uploads 100 MB), `PUBLIC_MAX_CONCURRENT` (3) GPU-bound requests in flight → else 429.
* **Errors** use OpenAI's shape: `{"error": {"message", "type", "reference"}}`. Refusals by the public API itself say what's wrong; failures deeper in (model load, TTS worker, ComfyUI) only say a generic message — the `reference` matches the server log line with the details. Busy states (409/507) become 503.

| Endpoint | Accepted input |
|---|---|
| **GET** `/public/v1/models` | — Aliases only, each with its `kind`. |
| **POST** `/public/v1/chat/completions` | `messages` (≤500; roles system/developer/user/assistant/tool; parts `text`, `image_url` **as `data:image/…` only**, `input_audio`), `temperature`, `top_p`, `max_tokens`/`max_completion_tokens` (capped at 8192), `stream`, `stream_options`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `user`, `thinking`, `chat_template_kwargs.enable_thinking`. Anything else is dropped; `n` must be 1. `model` in responses (streamed too) is the alias. |
| **POST** `/public/v1/audio/transcriptions` | multipart `file` (≤100 MB), `model`, `language` (`de`, `en-US`, …), `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`). |
| **POST** `/public/v1/audio/speech` | `input` (≤4096 chars), `voice` (alias allowlist, else default voice), `response_format` (`mp3` default, `opus`, `aac`, `flac`, `wav`, `pcm`), `speed` (clamped 0.5–2.0), `language`. |
| **POST** `/public/v1/images/generations` | `prompt` (≤4000 chars), `n` (1–4, sequential), `size` (`1024x1024`, `512x512`, `1792x1024`, `1024x1792`, `1536x1024`, `1024x1536`, `1024x768`, `768x1024`, `1280x720`, `720x1280`, `auto`, ratios like `16:9`). Always `b64_json`. |

```bash
curl https://<host>.ts.net:8443/public/v1/chat/completions \
  -H "Authorization: Bearer $LLAMA_SERVE_PUBLIC_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "chat", "messages": [{"role": "user", "content": "Hallo"}]}'
```

Check what a tunnel exposes before handing out its URL: `scripts/check_public_exposure.sh https://<host>.ts.net:8443`.

---

### 14. Services (sibling apps)

Dashboard-only (blocked through tunnels). Every direct child of `SERVICE_ROOTS` with a `llama-serve.service.json` is a service; optional `services.json` adds (`include`) or hides (`exclude`) projects. **How to integrate an app: [`docs/service-manifest.md`](service-manifest.md).**

* **GET** `/api/services` → `{"services": [{id, name, description, port, project_dir, source, status (running|starting|stopped|exited), exit_code, has_log, instances: [{pid, ports, started_at, managed, healthy, exposed: ["0.0.0.0:5000", …], open: {port, path, local_only}}], exposure: {level, url, warning, error, tailnet_port}}], "errors": [{dir, error, source}], "roots": [...], "tailscale": {available, owner, dns_name}}`. Instances started outside the dashboard are included (`managed: false`).
* **GET** `/api/services/manifest-schema` → JSON Schema of the manifest.
* **POST** `/api/services/{id}/start` → `{"status": "starting", pid, port}`; 409 if the port is in use.
* **POST** `/api/services/{id}/stop` body `{"pid": <instance pid>}` (optional when exactly one instance runs) → `{"status": "stopped", terminated, killed}`; 404 for a pid that isn't an instance of that service, 409 if ambiguous.
* **POST** `/api/services/{id}/exposure` body `{"level": "local" | "tailnet" | "owner"}` → who can reach it via Tailscale Serve (tailnet only, never public). 403 unless called from this machine or one of the owner's tailnet devices; 400 when the level can't apply (e.g. no valid tailnet port); 503 without Tailscale; 422 for any other level.
* **GET** `/api/services/{id}/log?tail=200` → output of copies started from the dashboard.

---

## Direct `llama-server` Connection (OpenAI API)

Once an instance is launched (e.g. on port `8031`), you can connect directly to it, bypassing the dashboard manager entirely. It provides a standard OpenAI compatible API.

### Standard Chat Completions
**POST** `http://localhost:8031/v1/chat/completions`

**Example:**
```bash
curl http://localhost:8031/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Multimodal / Vision Requests
If the instance is loaded with an `mmproj` vision module, you can send images. **Important:** The `image_url` object should be placed *before* the `text` object within the array.

**Example Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,...(base64 data)..."
          }
        },
        {
          "type": "text",
          "text": "What do you see?"
        }
      ]
    }
  ]
}
```

### Multimodal / Audio & Multimedia Requests
If the instance is loaded with an audio/multimedia projector (`mmproj` enabled), you can send audio files. The format field must be exactly `"wav"` or `"mp3"`.

**Example Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64_encoded_audio>",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": "Transcribe this audio exactly."
        }
      ]
    }
  ]
}
```

---

## Interactive Documentation
You can access the interactive Swagger UI at: [http://localhost:8030/docs](http://localhost:8030/docs)
