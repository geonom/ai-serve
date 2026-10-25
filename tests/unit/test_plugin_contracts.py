"""Plugin contracts (backend/plugins/contracts.py, docs/plugins.md): optional capabilities -- image generation,
speech, transcription, extra chat servers, access for other people -- come from apps that declare `provides` in
their llama-serve.service.json, instead of being built into the gateway (owner, 2026-10-21: "es sollte tatsächlich
nichts davon hardcoded sein, sondern alles pluginable").

Threat model: a plugin is an app found under SERVICE_ROOTS (trusted like any app there), but what it *returns* is
not: catalog entries are checked, can't take another plugin's ids, and a broken plugin can't break the catalog.
The gateway only ever calls a plugin at 127.0.0.1:<its manifest port> + a path from its own manifest."""
import pytest

from backend.plugins.contracts import (CAPABILITIES, ContractError, merge_catalogs, normalize_catalog, normalize_job, normalize_status,
                                       parse_provides, plugin_url)

IMAGE = {"catalog": "/v1/images/models", "generate": "/v1/images/generations", "edit": "/v1/images/edits",
         "cancel": "/v1/images/cancel", "status": "/v1/status", "release": "/v1/release", "gpu": "shared"}


# ------------------------------------------------------------------ what an app may declare

def test_known_capabilities():
    # 2026-10-21: owner chose to plan for these now so they don't each invent their own interface later
    assert set(CAPABILITIES) == {"llm", "image", "video", "audio", "tts", "stt", "documents", "translate",
                                 "moderation", "model3d", "tools", "realtime", "access", "vision"}


def test_provides_parsed():
    p = parse_provides({"image": IMAGE, "tts": {"catalog": "/v1/engines", "generate": "/v1/generate"}})
    assert p["image"].endpoints["generate"] == "/v1/images/generations" and p["image"].gpu == "shared"
    assert p["tts"].gpu == "none" and "unload" not in p["tts"].endpoints


def test_llm_plugin_is_openai_compatible_and_may_own_all_gpus():
    p = parse_provides({"llm": {"openai": "/v1", "sparse": "/v1/sparse", "status": "/status", "gpu": "exclusive"}})
    assert p["llm"].endpoints == {"openai": "/v1", "sparse": "/v1/sparse", "status": "/status"} and p["llm"].gpu == "exclusive"


@pytest.mark.parametrize("cap, decl", [
    ("video", {"catalog": "/v1/video/models", "jobs": "/v1/video/jobs", "status": "/s", "release": "/r", "gpu": "shared"}),
    ("audio", {"catalog": "/v1/audio/models", "jobs": "/v1/audio/jobs"}),
    ("documents", {"catalog": "/v1/documents/models", "jobs": "/v1/documents/jobs"}),
    ("model3d", {"catalog": "/v1/3d/models", "jobs": "/v1/3d/jobs"}),
    ("translate", {"catalog": "/v1/translate/models", "translate": "/v1/translate"}),
    ("moderation", {"catalog": "/v1/moderation/models", "moderate": "/v1/moderations"}),
    ("tools", {"mcp": "/mcp"}),
    ("realtime", {"catalog": "/v1/realtime/models", "realtime": "/v1/realtime"}),
    ("image", {"catalog": "/v1/images/models", "generate": "/x", "jobs": "/v1/images/jobs"}),
])
def test_new_capabilities(cap, decl):
    assert parse_provides({cap: decl})[cap].endpoints


@pytest.mark.parametrize("raw", [
    {"video": {"catalog": "/v1/video/models"}},              # long work needs jobs
    {"tools": {"mcp": "/mcp", "gpu": "shared"}},             # tools never take a GPU slot of their own
    {"chat": {"openai": "/v1"}},                             # renamed to llm
    {"llm": {"openai": "/api"}},                             # the OpenAI API's path ends in /v1
])
def test_new_capabilities_rules(raw):
    with pytest.raises(ContractError):
        parse_provides(raw)


@pytest.mark.parametrize("raw", [
    "image", [], {"video": {"catalog": "/x"}},                                       # unknown capability
    {"image": {"generate": "/v1/images/generations"}},                               # catalog missing
    {"image": {**IMAGE, "upscale": "/x"}},                                            # unknown endpoint
    {"image": {**IMAGE, "generate": "http://evil.example/x"}},                        # only paths, never hosts
    {"image": {**IMAGE, "generate": "//evil.example/x"}},
    {"image": {**IMAGE, "generate": "/v1/../../admin"}},
    {"image": {**IMAGE, "generate": "v1/images"}},                                    # must start with /
    {"image": {**IMAGE, "generate": "/v1/x?y=1"}},                                    # no query strings
    {"image": {**IMAGE, "generate": "/v1/\nx"}},
    {"image": {**IMAGE, "gpu": "all"}},
    {"access": {"shares": "/api/routes", "gpu": "exclusive"}},                       # access never needs a GPU
    {"image": {**IMAGE, "jobs": "http://x/"}},
])
def test_bad_declarations(raw):
    with pytest.raises(ContractError):
        parse_provides(raw)


def test_the_gateway_only_calls_loopback_at_the_manifest_port():
    p = parse_provides({"image": IMAGE})
    assert plugin_url(8190, p["image"], "generate") == "http://127.0.0.1:8190/v1/images/generations"
    with pytest.raises(ContractError):
        plugin_url(8190, p["image"], "unload")           # not declared
    with pytest.raises(ContractError):
        plugin_url(80, p["image"], "generate")           # privileged / odd ports are not plugin ports


# ------------------------------------------------------------------ what a plugin returns (untrusted)

def test_image_catalog_entries_are_checked_and_tagged_with_their_plugin():
    entries, errors = normalize_catalog("image", "comfyui", {"data": [
        {"id": "z-image-turbo", "label": "Z-Image Turbo", "available": True, "tasks": ["generate"],
         "aspect_ratios": ["1:1", "16:9"], "steps": {"default": 8, "adjustable": True, "max": 60}},
        {"id": "Bad Id!", "label": "x"},
        {"id": "edit-model", "label": "Edit", "tasks": ["edit", "teleport"]},
        "not an object",
    ]})
    assert [e["id"] for e in entries] == ["z-image-turbo"] and entries[0]["plugin"] == "comfyui"
    assert entries[0]["available"] is True and entries[0]["tasks"] == ["generate"]
    assert len(errors) == 3


def test_image_entries_say_how_many_input_images_they_take():
    entries, _ = normalize_catalog("image", "comfyui", {"data": [
        {"id": "group", "label": "G", "tasks": ["edit"], "max_images": 5},
        {"id": "plain", "label": "P", "tasks": ["edit"]},
        {"id": "gen", "label": "Z", "tasks": ["generate"]},
        {"id": "silly", "label": "S", "tasks": ["edit"], "max_images": 999},
    ]})
    assert {e["id"]: e["max_images"] for e in entries} == {"group": 5, "plain": 1, "gen": 0, "silly": 1}


def test_image_entries_name_the_media_their_prompt_enhancer_knows():
    entries, _ = normalize_catalog("image", "comfyui", {"data": [
        {"id": "moss", "label": "M", "tasks": ["generate"], "layout": True, "media": ["cover", "title_card", "Bad!"]},
        {"id": "z", "label": "Z", "tasks": ["generate"]},
    ]})
    by = {e["id"]: e for e in entries}
    assert by["moss"]["media"] == ["cover", "title_card"] and by["z"]["media"] == []


def test_catalog_entries_get_safe_defaults_and_lose_unknown_fields():
    entries, _ = normalize_catalog("image", "iris", {"data": [{"id": "iris-turbo", "label": "Iris", "evil": "<script>"}]})
    e = entries[0]
    assert "evil" not in e and e["available"] is False and e["tasks"] == ["generate"] and e["description"] == ""


def test_tts_and_stt_catalogs_keep_their_own_shapes():
    tts, errors = normalize_catalog("tts", "tts-serve", {"engines": [
        {"id": "cosyvoice", "name": "CosyVoice 3", "modes": [{"variant": "zero_shot", "label": "Klonen", "inputs": {"reference": "required"}}],
         "languages": ["de"], "params": ["speed"]}]})
    assert tts[0]["id"] == "cosyvoice" and tts[0]["label"] == "CosyVoice 3" and tts[0]["modes"][0]["variant"] == "zero_shot"
    stt, _ = normalize_catalog("stt", "stt-serve", {"engines": [{"id": "whisperx", "label": "WhisperX", "available": True,
                                                                  "fields": [{"name": "diarize", "type": "bool", "label": "x"}]}]})
    assert stt[0]["fields"][0]["name"] == "diarize"


def test_llm_catalog_is_openai_models_with_kinds():
    llm, _ = normalize_catalog("llm", "strata", {"object": "list", "data": [
        {"id": "strata", "object": "model"}, {"id": "bge-m3", "kind": "embedding"}, {"id": "x", "kind": "magic"}]})
    assert [(e["id"], e["kind"]) for e in llm] == [("strata", "chat"), ("bge-m3", "embedding")]
    assert llm[0]["available"] is True and llm[0]["plugin"] == "strata"


def test_every_catalog_entry_says_license_and_commercial_use():
    e = normalize_catalog("tts", "tts-serve", {"engines": [
        {"id": "omnivoice", "name": "OmniVoice", "license": "CC-BY-NC-4.0", "commercial_use": False},
        {"id": "cosyvoice", "name": "CosyVoice", "license": "Apache-2.0", "commercial_use": "yes"}]})[0]
    assert (e[0]["license"], e[0]["commercial_use"]) == ("CC-BY-NC-4.0", False)
    assert e[1]["commercial_use"] is None          # anything but true/false = unknown


def test_image_tasks_include_tools_like_upscaling():
    e = normalize_catalog("image", "p", {"data": [{"id": "up", "label": "x", "tasks": ["upscale", "remove_background", "segment"]}]})[0]
    assert e[0]["tasks"] == ["upscale", "remove_background", "segment"]


def test_video_audio_documents_model3d_entries():
    v = normalize_catalog("video", "p", {"data": [{"id": "wan", "label": "Wan", "tasks": ["text_to_video", "image_to_video"],
                                                   "max_seconds": 8, "resolutions": ["1280x720"], "fps": 16}]})[0][0]
    assert v["tasks"] == ["text_to_video", "image_to_video"] and v["max_seconds"] == 8 and v["fps"] == 16
    a = normalize_catalog("audio", "p", {"data": [{"id": "yue", "label": "YuE", "tasks": ["music"], "max_seconds": 300}]})[0][0]
    assert a["tasks"] == ["music"]
    d = normalize_catalog("documents", "p", {"data": [{"id": "nuextract3", "label": "NuExtract3", "modes": ["markdown", "structured"],
                                                       "inputs": ["image", "pdf"]}]})[0][0]
    assert d["modes"] == ["markdown", "structured"] and d["inputs"] == ["image", "pdf"]
    m = normalize_catalog("model3d", "p", {"data": [{"id": "trellis", "label": "TRELLIS", "tasks": ["image_to_3d"], "formats": ["glb"]}]})[0][0]
    assert m["formats"] == ["glb"]


def test_translate_moderation_realtime_entries():
    t = normalize_catalog("translate", "p", {"data": [{"id": "madlad", "label": "MADLAD", "languages": ["de", "en", "fr"]}]})[0][0]
    assert t["languages"] == ["de", "en", "fr"]
    m = normalize_catalog("moderation", "p", {"data": [{"id": "guard", "label": "Guard", "categories": ["violence", "sexual"]}]})[0][0]
    assert m["categories"] == ["violence", "sexual"]
    r = normalize_catalog("realtime", "p", {"data": [{"id": "voice-chat", "label": "Sprechen", "voices": ["anna"], "languages": ["de"]}]})[0][0]
    assert r["voices"] == ["anna"]


def test_tools_catalog_is_the_mcp_tool_list_with_a_risk():
    tools, errors = normalize_catalog("tools", "search", {"tools": [
        {"name": "web_search", "description": "Search the web", "inputSchema": {"type": "object"}, "risk": "external"},
        {"name": "run_code", "description": "Runs Python", "risk": "execute"},
        {"name": "read_file"},                                          # no risk given -> treated as the highest
        {"name": "Bad Name"}]})
    by = {t["id"]: t for t in tools}
    assert by["web_search"]["risk"] == "external" and by["read_file"]["risk"] == "execute"
    assert by["web_search"]["input_schema"] == {"type": "object"} and len(errors) == 1


# ------------------------------------------------------------------ long jobs (one pattern for every capability)

def test_job_status_is_checked():
    j = normalize_job({"id": "abc123", "status": "running", "progress": 0.4, "message": "Schritt 4/10", "evil": 1})
    assert j == {"id": "abc123", "status": "running", "progress": 0.4, "message": "Schritt 4/10", "result": None, "error": None}
    done = normalize_job({"id": "abc123", "status": "done", "result": {"files": [{"name": "out.mp4", "type": "video/mp4", "url": "/v1/video/jobs/abc123/files/out.mp4"}]}})
    assert done["result"]["files"][0]["name"] == "out.mp4"


@pytest.mark.parametrize("raw", [
    "x", {"status": "running"}, {"id": "../x", "status": "running"}, {"id": "a", "status": "exploded"},
    {"id": "a", "status": "running", "progress": 3},
    {"id": "a", "status": "done", "result": {"files": [{"name": "x", "url": "http://evil.example/x"}]}},   # files stay on the plugin
])
def test_bad_job_status(raw):
    with pytest.raises(ContractError):
        normalize_job(raw)


def test_a_broken_catalog_is_an_error_not_a_crash():
    entries, errors = normalize_catalog("image", "broken", "<html>oops</html>")
    assert entries == [] and errors


def test_too_many_entries_are_cut():
    entries, errors = normalize_catalog("image", "big", {"data": [{"id": f"m{i}", "label": "x"} for i in range(600)]})
    assert len(entries) == 500 and errors


# ------------------------------------------------------------------ several plugins, one catalog

def test_merge_keeps_the_first_owner_of_an_id_and_reports_the_clash():
    a, _ = normalize_catalog("image", "comfyui", {"data": [{"id": "z-image-turbo", "label": "Z"}, {"id": "ernie", "label": "E"}]})
    b, _ = normalize_catalog("image", "evil", {"data": [{"id": "z-image-turbo", "label": "fake"}, {"id": "moss", "label": "M"}]})
    merged, clashes = merge_catalogs([("comfyui", a), ("evil", b)])
    owners = {e["id"]: e["plugin"] for e in merged}
    assert owners == {"z-image-turbo": "comfyui", "ernie": "comfyui", "moss": "evil"}
    assert clashes == [{"id": "z-image-turbo", "kept": "comfyui", "ignored": "evil"}]


def test_merge_order_is_stable_by_plugin_id():
    a, _ = normalize_catalog("image", "b-plugin", {"data": [{"id": "x", "label": "x"}]})
    b, _ = normalize_catalog("image", "a-plugin", {"data": [{"id": "x", "label": "x"}]})
    merged, _ = merge_catalogs([("b-plugin", a), ("a-plugin", b)])
    assert merged[0]["plugin"] == "a-plugin"


# ------------------------------------------------------------------ status / release / logs (every capability)
# The GPU manager learns from plugins themselves which processes use GPU memory and how to free it (2026-10-21) --
# no plugin is known to it by name any more.

def test_gpu_plugins_must_say_what_they_hold_and_how_to_free_it():
    ok = {"catalog": "/v1/engines", "generate": "/v1/generate", "status": "/v1/status", "release": "/v1/release",
          "logs": "/v1/logs", "gpu": "shared"}
    assert set(parse_provides({"tts": ok})["tts"].endpoints) >= {"status", "release", "logs"}
    for missing in ("status", "release"):
        with pytest.raises(ContractError, match=missing):
            parse_provides({"tts": {k: v for k, v in ok.items() if k != missing}})
    # exclusive (owns all GPUs while it runs): status is enough
    assert parse_provides({"llm": {"openai": "/v1", "status": "/status", "gpu": "exclusive"}})
    with pytest.raises(ContractError):
        parse_provides({"llm": {"openai": "/v1", "gpu": "exclusive"}})
    # no GPU: none of it needed
    assert parse_provides({"access": {"shares": "/api/routes", "status": "/status"}})


def test_status_is_checked():
    st = normalize_status({"busy": False, "facts": [{"label": "Geladen", "value": "CosyVoice"}, "Tunnel: verbunden"],
                           "workers": [{"id": "cosyvoice", "label": "CosyVoice 3", "pid": 4242, "busy": True,
                                        "loaded": True, "last_used": 1792500000.5, "gpu": 1, "evil": 1},
                                       {"id": "Bad Id", "pid": 1}, {"id": "x", "pid": -5}],
                           "domain": "apps.example.org", "overview": "https://apps.example.org/"})
    assert st["busy"] is False and st["facts"] == [{"label": "Geladen", "value": "CosyVoice"}, {"label": "", "value": "Tunnel: verbunden"}]
    assert st["workers"] == [{"id": "cosyvoice", "label": "CosyVoice 3", "pid": 4242, "busy": True, "loaded": True,
                              "gpu": 1, "last_used": 1792500000.5},
                             {"id": "x", "label": "x", "pid": None, "busy": False, "loaded": False, "gpu": None,
                              "last_used": None}]
    assert st["domain"] == "apps.example.org" and st["overview"] == "https://apps.example.org/"


@pytest.mark.parametrize("raw, field, value", [
    ({"domain": "evil.example/x"}, "domain", None), ({"domain": "localhost"}, "domain", None),
    ({"overview": "javascript:alert(1)"}, "overview", None), ({"overview": "http://apps.example.org/"}, "overview", None),
    ("<html>", "busy", True),                                    # unreadable status: assume busy, touch nothing
    ("<html>", "ready", False), ({}, "ready", True), ({"ready": False}, "ready", False), ({"ready": "no"}, "ready", True),
])
def test_status_rejects_odd_values(raw, field, value):
    assert normalize_status(raw)[field] == value


def test_llm_entries_carry_what_a_chat_picker_needs():
    entries, errors = normalize_catalog("llm", "strata", {"data": [
        {"id": "strata", "kind": "chat", "context_length": 65536, "thinking": True, "aliases": ["Qwen3.8-X-strata.gguf"]},
        {"id": "odd", "context_length": "lots", "thinking": "yes"}]})
    assert not errors
    assert (entries[0]["context_length"], entries[0]["thinking"], entries[0]["aliases"]) == (65536, True, ["Qwen3.8-X-strata.gguf"])
    assert (entries[1]["kind"], entries[1]["context_length"], entries[1]["thinking"]) == ("chat", None, False)
