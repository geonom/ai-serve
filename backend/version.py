"""Backend performance/capability version.

BACKEND_VERSION is a plain incrementing integer, separate from any release
or API version. Bump it whenever a change could plausibly affect inference
speed or serving capability of the model backend — e.g. a llama.cpp binary
upgrade, added/changed speculative-decoding (MTP) support, MLX support,
GPU-selection logic, KV-cache behavior. Add a matching entry to VERSION_LOG
so `GET /api/version` explains what changed and why.

Do NOT bump for UI-only, ComfyUI, documentation, or other changes that
don't touch the llama.cpp/model-serving path.
"""

BACKEND_VERSION = 22

VERSION_LOG = [
    {
        "version": 22,
        "date": "2026-11-27",
        "change": (
            'gpu_id "all" (gateway and /api/start): one model spread over every GPU with llama.cpp\'s layer split '
            "(--split-mode layer --tensor-split by what each card can give, the card giving most leads), each card's "
            "share reserved with the GPU manager and bound to the one process; only when asked. A model used in the "
            "last 20 s (was 60) counts as busy for the GPU manager."
        ),
    },
    {
        "version": 21,
        "date": "2026-11-03",
        "change": (
            "Realtime: the conversation limit comes from llama-server's /slots (an adopted instance has no recorded "
            "count); a sentence whose speech fails is tried once more and the error names the reason; /tts gives up "
            "after 30 s without bytes; nothing from <tool_call> on is spoken. Chats with any llama.cpp model (and Strata) "
            "default to enable_thinking=false when the call says nothing (was: Qwen and Strata only)."
        ),
    },
    {
        "version": 20,
        "date": "2026-11-03",
        "change": (
            "Speech models load with 3 slots and take at most 2 realtime conversations (one slot stays free for /tts); "
            "a further conversation is refused at once instead of leaving spoken answers waiting for a slot."
        ),
    },
    {
        "version": 19,
        "date": "2026-11-03",
        "change": (
            "Speech models load with 2 slots; /v1/realtime with session.audio.output.language (not English) speaks "
            "each sentence with the model's own TTS in that language; /v1/audio/speech with a speech model's file "
            "name uses its /tts (language, voice female|male, stream)."
        ),
    },
    {
        "version": 18,
        "date": "2026-11-03",
        "change": (
            "Speech-to-speech models (LFM3-Omni/Audio, recognised by their TTS template) load with --reasoning off; "
            "WebSocket /v1/realtime passes a voice conversation through to the loaded llama-server."
        ),
    },
    {
        "version": 17,
        "date": "2026-11-03",
        "change": (
            "Chats with a Qwen model or Strata get chat_template_kwargs.enable_thinking=false when the call says "
            "nothing about thinking (thinking: true or an explicit enable_thinking still wins)."
        ),
    },
    {
        "version": 16,
        "date": "2026-11-02",
        "change": (
            "/api/start takes n_cpu_moe (llama.cpp --n-cpu-moe: routed MoE experts of the first N layers in RAM), and "
            "the GPU manager's reservation for it and for a partial n_gpu_layers counts only the tensors that land in "
            "VRAM (from the GGUF's tensor table) -- an A3B-style MoE larger than the card can be started now."
        ),
    },
    {
        "version": 15,
        "date": "2026-11-01",
        "change": (
            "An idle stop needs the instance idle (gateway requests and its own /slots) for the whole keep-alive time, "
            "not just at one instant; who started an instance survives a restart (instance_origins.json), and an "
            "/api/start-ed one gets no idle timer after adoption."
        ),
    },
    {
        "version": 14,
        "date": "2026-11-01",
        "change": (
            "A running llama-server is restarted for a bigger context / more slots only when the gateway started it, "
            "it is idle by its own /slots and not loading, and the caller asked for that ctx_size explicitly (no "
            "ctx_size = any running context). An /api/start-ed or adopted instance is never restarted by the gateway; "
            "an explicit ctx_size it can't hold gets an instance of its own (or 507); a loading instance counts as busy for "
            "manual stops too."
        ),
    },
    {
        "version": 13,
        "date": "2026-10-31",
        "change": (
            "DFlash speculative decoding: a block drafter named dflash-<model stem>.gguf pairs with its main model "
            "like an mtp- head and loads with --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl all "
            "(NVIDIA's drafter for Nemotron 3.5 Bolt E4B)."
        ),
    },
    {
        "version": 12,
        "date": "2026-10-30",
        "change": (
            "llama.cpp b10820 -> b12010 (CUDA 12.9): LFM3 models and native decision models -- POST /v1/systemone "
            "(kind 'decision', loaded with -b/-ub 8192). GPUs may be named by UUID; llama-servers are pinned by UUID "
            "and every child process gets CUDA_DEVICE_ORDER=PCI_BUS_ID. The GPU manager keeps learned peaks per GPU "
            "and, since 2026-10-26, treats the learned +10 % as optional, plans again when an unload frees less than "
            "counted, and stops idle plugin processes whose CUDA context is all that is missing."
        ),
    },
    {
        "version": 11,
        "date": "2026-10-24",
        "change": (
            "GPU provisioning: a refused preflight is 507; reservations shrink only by growth past the process's "
            "baseline at confirm; the GPU manager learns each purpose's real peak and reserves/chooses GPUs by it "
            "(LLM purposes per model and context size)."
        ),
    },
    {
        "version": 10,
        "date": "2026-10-22",
        "change": (
            "Apple Silicon: the GPU manager sees who uses the memory -- without nvidia-smi it measures each known "
            "consumer's processes (physical footprint, at least RSS) instead of seeing nobody (Iris rendering showed "
            "'nothing loaded', so nothing could be unloaded to make room either)."
        ),
    },
    {
        "version": 9,
        "date": "2026-10-22",
        "change": (
            "Apple Silicon: free GPU memory = Metal's limit (iogpu.wired_limit_mb, else ~2/3 of the RAM up to 36 GB, "
            "3/4 above) minus what the GPU holds (ioreg), as Ollama/LM Studio size it -- not the RAM unused right now, "
            "which refused Iris (7 GB) on a 24 GB Mac with '3681 MB free'. Without the ioreg figure: the old way."
        ),
    },
    {
        "version": 8,
        "date": "2026-10-21",
        "change": (
            "Auto-loaded models get their vision module (mmproj next to the GGUF) attached, so image input works "
            "through /v1/chat/completions; embedding/reranking loads never get one. llama.cpp then disables "
            "context shift and cache reuse for that instance. Found with a fresh install (Gemma 4 E2B)."
        ),
    },
    {
        "version": 7,
        "date": "2026-10-21",
        "change": (
            "Other model servers are llm plugins (docs/plugins.md): the gateway lists their models in "
            "/v1/llm/models and routes chat/embeddings/rerank for them by id or alias. Strata moved out into "
            "the separate strata-plugin app (an exclusive plugin: it takes every GPU through "
            "/api/coordination/gpu-exclusive); /api/strata/* is gone. /api/start for a plugin model answers "
            "with the plugin's port."
        ),
    },
    {
        "version": 6,
        "date": "2026-10-14",
        "change": (
            "New serving backend: Strata (separate local server for Qwen3.8-Flash-Next, "
            "layer-split across both GPUs). /api/strata/start|stop|status manage it "
            "(adopting a Strata started elsewhere), and the chat gateway routes model "
            "names starting with 'strata' to it with auto-start. While Strata runs or "
            "loads, all other GPU loads (llama-server, TTS, STT, ComfyUI) are refused "
            "with 507; Strata itself won't start while other processes hold VRAM."
        ),
    },
    {
        "version": 5,
        "date": "2026-10-02",
        "change": (
            "Every llama-server instance (gateway auto-load and /api/start) now starts "
            "with -np 1 unless the caller passes `parallel` (LLAMA_DEFAULT_PARALLEL, 0 = "
            "llama.cpp's own 4-slot default). 4 slots cost single-stream speed on every "
            "model measured, up to ~2x when the model nearly fills the card: gemma-4-31B "
            "+ MTP on the 3090 17.7 -> 39.8 t/s decode; gemma-4-12B on the 3090 +7-11%; "
            "plus ~0.6-1 GB less VRAM. Concurrent requests to an instance now queue "
            "inside llama-server unless it was loaded with parallel > 1."
        ),
    },
    {
        "version": 4,
        "date": "2026-09-10",
        "change": (
            "Added explicit ubatch/batch control (/api/start's new ubatch_size and "
            "batch_size fields, -> -ub/-b). llama.cpp's own default n_ubatch (512) is a "
            "hard floor for vision input: mtmd_helper_decode_image_chunk asserts "
            "n_ubatch >= the image's token count and llama-server crashes outright "
            "(GGML_ASSERT, not a graceful error) the moment an image needs more tokens "
            "than that. Found live: gemma-4-12b/E4B and Qwen3.5-4B all crashed mid-batch "
            "on larger form/OCR-scan vision prompts while small chart images passed. No "
            "default is set automatically — callers with a real mmproj_path and non-tiny "
            "images should pass ubatch_size=1024+ explicitly."
        ),
    },
    {
        "version": 3,
        "date": "2026-08-28",
        "change": (
            "Upgraded the llama.cpp binary from build 9986 (91c631b21) to build 10665 "
            "(ca3d5a3e1) — 679 commits, rebuilt from source with the same CUDA config "
            "(archs 86;89). Fixed get_llamacpp_version()'s version-string regex, which "
            "the upgrade broke: older builds print 'version: 9986 (91c631b21)', this "
            "build prints 'version: 0.3.0-dev (build 10665, commit ca3d5a3e1)' — a "
            "format change upstream, not something either build number reflects on its "
            "own. Now tries the new '(build N, commit H)' form first and falls back to "
            "the old form, so components.llamacpp's build_number/commit stay populated "
            "across either version instead of silently going null."
        ),
    },
    {
        "version": 1,
        "date": "2026-07-23",
        "change": (
            "Baseline. Includes MTP speculative-decoding support for both "
            "external draft models and MTP tensors bundled directly into a "
            "model's own GGUF, and single-GPU pinning for /api/start."
        ),
    },
    {
        "version": 2,
        "date": "2026-08-02",
        "change": (
            "Added explicit --pooling control (/api/start's new pooling field) so an "
            "embedding model whose GGUF metadata doesn't reliably encode pooling type "
            "(e.g. some bge-m3 quants) can be forced to the correct value instead of "
            "silently resolving to 'none' and breaking /v1/embeddings. Added reranking "
            "support: /api/start's new reranking flag (--reranking, implies embedding) "
            "for loading a dedicated cross-encoder reranker model, plus new /v1/rerank "
            "and /api/rerank gateway endpoints."
        ),
    },
]
