# Plugins: optional capabilities for the gateway

The gateway's core is small: llama.cpp model management, the GPU manager, the model catalogs, access rules and
Apps & Services. Everything else -- images, video, music, speech, transcription, documents, translation, moderation, 3D, agent tools,
realtime conversation, other model servers, access for other people -- comes from **plugins**: apps that tell the gateway what they provide. Without any plugin, ai-serve still
runs (chat, embeddings, rerank, document reading through llama.cpp); a plugin adds a capability, removing it takes
the capability away. Nothing of a plugin is built into the gateway.

Rules (checked, tested): `backend/plugins/contracts.py`, `tests/unit/test_plugin_contracts.py`.

## Declaring a plugin

A plugin is an ordinary app (`docs/service-manifest.md`) whose `llama-serve.service.json` has a `provides` block:

```json
{
  "id": "image-serve",
  "name": "Bilder",
  "command": ["./run.sh"],
  "port": 8070,
  "match": ["image_serve"],
  "ui": false,
  "provides": {
    "image": {
      "catalog": "/v1/images/models",
      "generate": "/v1/images/generations",
      "edit": "/v1/images/edits",
      "cancel": "/v1/images/cancel",
      "gpu": "shared"
    }
  }
}
```

- **capability**: `llm`, `image`, `video`, `audio`, `tts`, `stt`, `documents`, `translate`, `moderation`,
  `model3d`, `vision`, `tools`, `realtime` or `access` (below). One app may provide several.
- **endpoints**: plain paths on the app's own port. The gateway only ever calls
  `http://127.0.0.1:<port><path>` -- no hosts, queries, `..` or `//` are accepted, so a manifest can't point the
  gateway anywhere else.
- **gpu**: `none` (default), `shared` (the plugin asks the GPU manager before it loads anything, see below) or
  `exclusive` (owns every GPU while it runs, e.g. a model split across all cards; nothing else gets GPU memory then).
  `tools` and `access` never take a GPU of their own.
- The gateway starts a plugin on demand (through Apps & Services) when its capability is needed and it isn't
  running; a plugin that is down makes its capability report "not available" with a reason -- it is never fatal.

## Standard first (for apps and plugins)

Owner, 2026-10-21: the Playground -- and every extension -- should also work with a standard Ollama or LM Studio.
So:
- **Apps** talk the standard OpenAI API first (`/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/…`,
  `/v1/images/…`) and treat ai-serve's additions (`/v1/llm/models`, `/v1/audio/models`, `/v1/images/models`,
  `/api/…`) as optional: when one answers 404, fall back to the standard call or show that page as "needs ai-serve"
  instead of failing. Make the server URL configurable and send an optional Bearer key. Reference: the Playground
  (`playground/backends.py`, `/sw/backend`).
- **Plugins** speak standard APIs wherever one exists (`llm` = OpenAI; `moderation` = OpenAI moderations), so a
  standard server can be one without an adapter -- see "A standard server without an app" below.

## Catalogs: "what can you do?"

Every capability except `access` has a catalog. The gateway asks every plugin of a capability, checks each entry,
tags it with the plugin's id and merges them into its own lists. Requests name an entry's `id`; the gateway sends
them to the plugin that listed it.

Every entry has these fields (more per capability, below):

| field | meaning |
|---|---|
| `id` | lowercase letters, digits, `.`, `_`, `-` (max 64). Global: if two plugins list the same id, the plugin whose own id sorts first keeps it, the other entry is ignored and reported |
| `label`, `description` | shown to people (trimmed, length-limited) |
| `available` | installed and usable right now (`true`), else listed but greyed out |
| `license` | e.g. `Apache-2.0`, `CC-BY-NC-4.0` |
| `commercial_use` | `true` / `false` -- anything else counts as unknown. Matters when people use ai-serve for work |
| `default` | the plugin's suggestion for its capability |
| `aliases` | older names that should still resolve to this entry |

Unknown fields are dropped; at most 500 entries per plugin; a catalog that isn't JSON of the expected shape counts
as an error for that plugin only.

## Long work: jobs

One pattern for everything that takes longer than a request should wait (video, music, documents, 3D; optional
for images, speech and transcription). The capability declares `jobs: "/v1/<thing>/jobs"`:

- `POST {jobs}` with the capability's request body → **202** `{"id": "…", "status": "queued"}`
- `GET {jobs}/{id}` → `{"id", "status": queued | running | done | failed | cancelled, "progress": 0..1 | null,
  "message", "error", "result"}`
- `POST {jobs}/{id}/cancel` → the status (cancelled, or done if it was too late)
- `result`: `{"files": [{"name", "type", "url"}], "text"?, "data"?}` -- `url` is a **path on the plugin**
  (`/v1/video/jobs/<id>/files/out.mp4`); the gateway fetches it from there and hands it on, so a plugin can't send
  anyone elsewhere.

Job ids: letters, digits, `_`, `-` (max 64). A plugin keeps finished jobs long enough to be fetched (an hour or
more) and never deletes people's results on its own.

## The capabilities

### llm -- another OpenAI-compatible server (MLX, vLLM, Ollama, Strata …)

`{"llm": {"openai": "/v1", "load": "/plugin/load", "sparse": "/v1/sparse", "gpu": "exclusive"}}` (`openai` ends in
`/v1`). Catalog = `GET {openai}/models` (`{"data": [{"id", "kind"?, "label"?, "context_length"?, "thinking"?,
"vision"?, "aliases"?}]}`, `kind` chat | embedding | rerank | sparse, default chat) -- answer it **without loading
the model** (the gateway asks whenever it lists models). Requests name an id or an alias (case-insensitive; the
basename of a path counts); the gateway sends `model: <id>`. Optional `load`: `POST {"model"}` → answer once it
serves (`/api/start` uses it; without one the first request loads). The gateway lists these
next to its own llama.cpp models and forwards `/v1/chat/completions` (streamed or not), `/v1/embeddings` and
`/v1/rerank` for those ids to `{openai}/…`; errors keep their status and message. Optional `sparse`:
`POST {"model", "input": [text]}` → `{"data": [{"index", "indices": [int], "values": [float]}]}` (lexical weights,
e.g. bge-m3's, which llama.cpp doesn't return).

**A standard server without an app** (Ollama, LM Studio, llama-server, vLLM … that the owner already runs): no
manifest needed -- the dashboard ("Rechner & Modelle" → "Weitere Modellserver") adds it by **port** (loopback only,
never a host; `GET/PUT/DELETE /api/model-servers[/{id}]`, stored in gitignored `model_servers.json`, rules in
`backend/plugins/external.py`). Ollama (11434) and LM Studio (1234) are offered when something answers on their
usual ports. It becomes an `llm` plugin with `openai` only: its `/v1/models` is the catalog (ids as the server names
them -- `llama3.2:3b`, `qwen/qwen3-8b`; kind guessed from the name when not given), requests are routed to it, it is
never started, stopped or unloaded by ai-serve and has no GPU in the GPU manager's books (its memory counts as
"other programs"). An app with a manifest on the same port or id wins over an entry.

### image -- generating, editing and image tools

- `catalog` → `{"data": [entry]}` with `tasks` (`generate`, `edit`, `style`, `inpaint`, `upscale`,
  `remove_background`, `segment`), `aspect_ratios`, `steps` `{default, adjustable, max}`, `text_in_image`, `layout`,
  `max_images` (input images for `edit`; default 1, 0 for generate-only models), `media` (what its prompt enhancer
  takes as `medium`).
- `generate`: the OpenAI images body plus the gateway's extensions (`aspect_ratio`, `steps`, `seed`, `layout`, and
  `medium` + `brief` for a model whose row lists `media`: the plugin's prompt enhancer builds the layout and answers
  it as `layout` next to `data`);
  answers `{"created", "data": [{"b64_json"}]}`. Header `X-Request-Id` identifies the job for `cancel`.
- `edit`: `multipart/form-data` like OpenAI's `/v1/images/edits` (`image`, `prompt`, `model`, `seed`; a `task`
  field for upscale / remove_background / segment / inpaint with `mask`; `remove_background` takes no `prompt` and
  answers the image as a PNG with alpha -- image-serve's `birefnet`). Several images: `image` repeated (the gateway
  also takes OpenAI's `image[]`), in prompt order "picture 1, 2 …", at most the entry's `max_images`; `aspect_ratio`
  = the output's shape (one of the entry's `aspect_ratios`, else the first image's).
- `cancel`: `POST {"request_id"}`. Optional `jobs` for slow models.

### video

`catalog` entries add `tasks` (`text_to_video`, `image_to_video`, `video_to_video`), `max_seconds`, `resolutions`
(`"1280x720"`), `fps`. `jobs` body: `{"model", "prompt", "image_base64"?, "seconds"?, "resolution"?, "seed"?}`;
result files are videos (`video/mp4`).

### audio -- music and sound effects

`catalog` entries add `tasks` (`music`, `sound`) and `max_seconds`. `jobs` body: `{"model", "prompt", "lyrics"?,
"seconds"?, "seed"?}`; result files are audio.

### tts -- speech

- `catalog` → `{"engines": [entry]}` with `languages`, `params` and `modes` (what each mode needs -- the format
  tts-serve uses, see its provider manifests).
- `generate`: tts-serve's request (`engine, text, language, variant, instruct, speaker, ref_audio_base64, ref_text,
  params, keep_alive, response_format`); answers audio. Optional `unload`, `status`, `jobs` (long texts).
- Several `tts` plugins at once: their engines are merged into one list (a duplicate engine id stays with the plugin
  whose id sorts first, the other is reported); every request goes to the plugin that lists its engine. The default
  engine is `DEFAULT_TTS_ENGINE`, else the first installed plugin default in plugin-id order.
- A complete, small example to copy (MIT): [ai-serve-piper](https://github.com/geonom/ai-serve-piper) -- manifest,
  catalog, generate, status, downloads with sha256, tests, CI.

### stt -- transcription

- `catalog` → `{"engines": [entry]}` with `outputs` and `fields` -- the options as plain form fields (`name`,
  `type` bool | choice | int | language | list, `show_if` …).
- Several `stt` plugins at once, like `tts`: engines merged (a duplicate id stays with the plugin that sorts first), a
  transcription goes to the plugin that lists its `engine`, a live session stays with the plugin that started it
  (`live/start` answers its `session_id`). Default: `DEFAULT_STT_ENGINE`, else the first available plugin default.
- `transcribe`: stt-serve's request (audio as base64, the fields above); answers transcript + segments. Optional
  `live` (`{live}/start|chunk|poll|stop|settings|sessions`, stt-serve's live sessions), `jobs` (long recordings).

### documents -- reading documents (OCR, extraction)

`catalog` entries add `modes` (e.g. `markdown`, `structured`) and `inputs` (`image`, `pdf`, `epub`, `docx`,
`html`, `text`). `jobs` body: `{"model", "files_base64": [..], "mode", "template"?, "instructions"?}`; result
`text` (markdown/plain) and/or `data` (structured) and per-page details in `data.pages`.

### translate

`catalog` entries add `languages`. `translate`: `POST {"model", "text": str | [str], "source"?, "target"}` →
`{"translations": [str]}`.

### moderation -- safety checks

`catalog` entries add `categories`. `moderate`: the OpenAI moderations shape (`{"model", "input"}` →
`{"results": [{"flagged", "categories", "category_scores"}]}`).

### model3d -- 3D models

`catalog` entries add `tasks` (`text_to_3d`, `image_to_3d`) and `formats` (`glb`, `gltf`, `obj`, `ply`, `stl`,
`usdz`). `jobs` body: `{"model", "prompt"?, "image_base64"?, "format"?, "seed"?}`; result files are models.

### vision -- faces and objects in images

`{"vision": {"catalog": "/v1/vision/models", "analyze": "/v1/vision/analyze", ...}}`. `catalog` entries add `tasks`
(`faces`, `objects`), `max_images` per call, `open_vocabulary` (objects by free-text labels) and, for face models,
`embedding` `{model, dim, metric, thresholds: {maybe, likely}}` -- `model` names the vector space: vectors of
different models or versions are never comparable, so apps store it next to each vector and re-embed when it
changes. `analyze`: multipart `image` (repeated), `model`, `tasks` (comma-separated, only ones the model lists),
`embed`, `labels`; answer `{"images": [{"width", "height", "faces"?: [...], "objects"?: [...], "error"?}]}`, one row
per image in order. Faces: `{"box": [x, y, w, h], "landmarks": [[x, y] x 5], "score", "quality", "embedding"?}`
in pixels of the image after its EXIF rotation; objects: `{"label", "box", "score"}`. The gateway's
`/v1/vision/analyze` takes `tasks` = faces | objects | both and sends each task to its model (`face_model`,
`object_model`, else the default for it), one call when one model does both. **Images and embeddings are
biometric data:** a vision plugin never stores or logs them.

### tools -- tools for agents (MCP)

`{"tools": {"mcp": "/mcp"}}`: an MCP server over streamable HTTP. Its `tools/list` answer is the catalog
(`name` → id, `title`, `description`, `inputSchema`) plus a field ai-serve adds: **`risk`** -- `read` (looks
things up), `write` (changes the plugin's own data), `external` (reaches the internet or other services),
`execute` (runs code / changes the machine). A tool without a risk counts as `execute`. The owner decides which
tools models may call; nothing is enabled just because a plugin offers it.

### realtime -- talking with a model

`catalog` entries add `voices` and `languages`. `realtime`: a WebSocket path speaking an OpenAI-Realtime-like
protocol (audio in, audio + text out, interruptions). The gateway passes the connection through after its access
check.

### access -- letting other people in

`{"access": {"shares": "/api/routes", "status": "/status", "security": "/api/security", "people": "/api/people"}}`.
The owner shares apps in the dashboard; the gateway pushes the resulting routes to `shares` (GET returns the current
ones, PUT replaces them) and shows `status`, `security` and `people`. The plugin verifies who someone is and hands
apps a signed identity (`docs/identity.md`). Without an access plugin, ai-serve is reachable only from this machine
(and, with the network extension, the owner's own devices).

## GPU: asking before loading

A plugin with `gpu: shared` asks before it loads a model:
`POST /api/coordination/gpu-preflight {"purpose", "need_mb", "gpu"?, "current_gpu"?, "movable"?}` → `{gpu, mode,
granted, free_mb, reservation_id, hold_s, reason}`. Rules (2026-10-24, after Whisper ran out of memory because it
announced too little, always asked for GPU 0 and gave its reservation back before its first transcription):
- **Leave out `gpu`**: the manager picks one with room (`gpu` in the answer, `mode` start / stay / move) and may unload
  idle models of any kind for it. Start the worker on that GPU (e.g. `CUDA_VISIBLE_DEVICES`).
- **A no is 507** with the reason as `detail` -- never load anyway. Either fall back (e.g. the CPU) or answer 507
  yourself, so the caller retries; the gateway never queues.
- **`purpose` names what is loaded** (`stt:whisperx:medium`, `tts:cosyvoice`, `llm:<file>@<ctx>`): the manager learns
  each purpose's real peak (what the confirmed process grew by) and reserves at least that + 10 % next time, so
  `need_mb` only has to be in the right range (learned values: `gpu_needs.json`).
- A grant is held a few seconds: confirm it with the process that will use the memory
  (`POST /api/coordination/reservations/{id}/confirm {"pid"}`; what that process already holds is its baseline)
  and **release it after the first use** -- the first transcription, image, utterance -- not right after loading
  (`DELETE /api/coordination/reservations/{id}`). Unreleased, it ends when the process exits or after 2 minutes.
- **Out of memory while working**: free what you hold, ask again (the manager may now unload an idle model or pick the
  other GPU), retry once; out of memory again -> 507 ("try again"), never a bare 500.
- Report `last_used` per worker in `status`: the least recently used idle model is unloaded first. A `gpu: exclusive` plugin asks `POST /api/coordination/gpu-exclusive {"purpose"}` → `{granted,
reservation_ids, unloaded, reason}` before it starts its server and confirms every id with that server's PID: it then
has every GPU for as long as that process lives. Its `status.workers` name that process (and only while it runs --
an exclusive plugin without workers holds nothing); it is never unloaded automatically, only by `release` (from the
dashboard, `/api/stop` on its port, `unload_all`).

## Status, release, logs (every capability)

- `status` → `{"busy": bool, "ready": bool (serving now without loading first; missing = true), "facts": [{"label", "value"} | "text"], "workers": [{"id", "label", "pid", "loaded",
  "busy", "last_used", "gpu"}], "domain"?, "overview"?}`. `workers` are the processes holding models: the GPU manager
  attributes memory by their `pid` (incl. children) and lists them as `<plugin>:<worker>`. A status that can't be
  read counts as busy -- nothing of that plugin is touched. `domain`/`overview` are for `access` plugins (the public
  names the door answers on; the gateway remembers the domain so apps started while the door is down still get it).
- `release`: `POST {"worker"?, "gpu"?, "force"?}` → `{"released": [worker ids]}`; one worker, or every idle one
  (on that GPU). **409** while busy unless `force` -- the GPU manager never forces.
- `logs`: `GET ?tail=N` → `{"logs": "text"}`.

## Downloads: `install` (every capability, optional)

What a plugin can still download for this machine -- models, engines, their environments. Owner, 2026-10-22: a
missing model should be downloadable from the Playground and the dashboard instead of leaving an empty list.

- `GET {install}` → `{"items": [{"id", "label", "description", "size_mb", "state", "done_mb", "error", "license"}]}`;
  `state` is `missing | downloading | installed | failed`; `size_mb`/`done_mb` may be null (unknown). List only what
  can run here (tts-serve leaves out engines whose `requires` this machine lacks, e.g. `cuda`, `darwin-arm64`).
- `POST {install} {"id"}` → `202` with the item; starts the work in the background, once (a second request while it
  runs changes nothing). `404` for an id it doesn't offer.
- The request carries **only the id** -- which files, from where and with which checksum is the plugin's own
  (shipped) business: sd-serve checks each file against GitHub's / Hugging Face's sha256, tts-serve and stt-serve
  run the provider's own `setup.sh`. Never take URLs, paths or arguments from the request.
- The gateway: `GET /api/plugins/installs` (every plugin's items, tagged `plugin`, `plugin_name`, `capability`;
  fields checked by `normalize_install`; plugins that couldn't be started are listed under `down` with the reason
  from their log, and `POST /api/plugins/{plugin}/start` starts one -- only plugins, never other apps), `POST /api/plugins/{plugin}/install {"id"}` (only `{"id"}`; dashboard
  access = this machine and the owner's devices). The Playground forwards the start only for its owner.

## Which plugins exist

Bundled with this release: tts-serve (`tts`), stt-serve (`stt`), sd-serve and image-serve (`image`; image-serve also
carries MOSS-Image). In repositories of their own, added with `./ai-serve plugin add <url>`:
[ai-serve-piper](https://github.com/geonom/ai-serve-piper) (`tts`, CPU, MIT -- also the template for new plugins) and
[ai-serve-strata](https://github.com/geonom/ai-serve-strata) (`llm`, `gpu: exclusive`). Built and used on the developer's machine but **not part of this release**: an `access` plugin
(the "door" for letting other people in, see `docs/identity.md`) and an `llm` adapter for Strata. Document reading
(NuExtract3) still lives in the gateway; it could become a `documents` plugin. No plugin exists yet for video, audio, 3D, translation, moderation,
tools or realtime -- the contracts are there so the first ones fit.
