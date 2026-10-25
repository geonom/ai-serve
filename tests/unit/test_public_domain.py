"""Security domain of the public API (backend/public_api/domain.py) -- pure rules,
no FastAPI. Threat model: an internet client holding (or guessing) the public key,
reaching this backend only through the tunnel."""
import pytest

from backend.public_api.domain import (
    AdmissionControl,
    ApiDisabled,
    BodyLimits,
    ChatPolicy,
    ErrorPolicy,
    ExposureGuard,
    ImagePolicy,
    ModelAlias,
    ModelCatalog,
    ModelKind,
    PayloadTooLarge,
    PolicyViolation,
    PublicApiKey,
    SpeechPolicy,
    TooManyRequests,
    TranscriptionPolicy,
    Unauthorized,
    UnknownModel,
    authenticate,
)

STRONG = "k" * 43

CATALOG = {
    "models": [
        {"id": "chat", "kind": "chat", "target": "small.gguf", "defaults": {"thinking": False, "ctx_size": 8192}},
        {"id": "chat-large", "kind": "chat", "target": "large.gguf"},
        {"id": "whisper-1", "kind": "stt"},
        {"id": "tts", "kind": "tts", "target": "cosyvoice", "voices": ["narrator"]},
        {"id": "img", "kind": "image", "target": "z_image_turbo-Q6_K.gguf"},
    ]
}


# --------------------------------------------------------------------------- key / authentication

class TestPublicApiKey:
    def test_empty_secret_means_disabled(self):
        assert PublicApiKey.from_secret("") is None
        assert PublicApiKey.from_secret(None) is None

    def test_short_secret_is_refused_not_used(self):
        # fail closed: a weak key disables the API instead of protecting it badly
        assert PublicApiKey.from_secret("short") is None
        assert PublicApiKey.from_secret("x" * 31) is None

    def test_strong_secret_verifies_only_exact_bearer(self):
        key = PublicApiKey.from_secret(STRONG)
        assert key.verify(f"Bearer {STRONG}")
        assert not key.verify(STRONG)  # no scheme
        assert not key.verify(f"Basic {STRONG}")
        assert not key.verify(f"Bearer {STRONG}x")
        assert not key.verify(f"Bearer {STRONG[:-1]}")
        assert not key.verify("")
        assert not key.verify(None)

    def test_secret_never_appears_in_repr(self):
        key = PublicApiKey.from_secret(STRONG)
        assert STRONG not in repr(key) and STRONG not in str(key)


class TestAuthenticate:
    def test_disabled_api_raises_503(self):
        with pytest.raises(ApiDisabled) as e:
            authenticate(None, f"Bearer {STRONG}")
        assert e.value.status == 503

    def test_wrong_key_raises_401(self):
        with pytest.raises(Unauthorized) as e:
            authenticate(PublicApiKey.from_secret(STRONG), "Bearer nope")
        assert e.value.status == 401

    def test_right_key_passes(self):
        authenticate(PublicApiKey.from_secret(STRONG), f"Bearer {STRONG}")


# --------------------------------------------------------------------------- exposure guard

class TestExposureGuard:
    guard = ExposureGuard()
    FUNNEL = {"tailscale-funnel-request": "?1"}

    def test_local_and_tailnet_requests_are_not_external(self):
        assert not self.guard.is_external({})
        assert not self.guard.is_external({"x-forwarded-for": "100.64.0.1", "tailscale-user-login": "me"})
        assert self.guard.permits("/api/status", {})

    @pytest.mark.parametrize("marker", ["tailscale-funnel-request", "cf-connecting-ip"])
    def test_tunnel_markers_are_external(self, marker):
        assert self.guard.is_external({marker: "1"})

    def test_marker_header_name_is_case_insensitive(self):
        assert self.guard.is_external({"Tailscale-Funnel-Request": "?1"})

    @pytest.mark.parametrize("path", ["/public/v1/models", "/public/v1/chat/completions"])
    def test_external_may_reach_public(self, path):
        assert self.guard.permits(path, self.FUNNEL)

    @pytest.mark.parametrize("path", [
        "/", "/index.html", "/app.js", "/docs", "/openapi.json",
        "/api/status", "/api/training/jobs", "/v1/models", "/v1/chat/completions",
        "/public", "/publicx/v1/models", "/public-v1/models",
        "/public/../api/status", "/public/v1/../../api/stop", "/public/./../v1/models",
        "/public/..", "/public\\..\\api", "//public/v1/models/../../../api",
    ])
    def test_external_blocked_everywhere_else(self, path):
        assert not self.guard.permits(path, self.FUNNEL)


# --------------------------------------------------------------------------- catalog

class TestModelCatalog:
    def test_resolve_by_id_and_default_per_kind(self):
        cat = ModelCatalog.from_dict(CATALOG)
        assert cat.resolve("chat-large", ModelKind.CHAT).target == "large.gguf"
        assert cat.resolve(None, ModelKind.CHAT).id == "chat"
        assert cat.resolve("", ModelKind.STT).id == "whisper-1"
        assert cat.resolve(None, ModelKind.STT).target == "whisper-1"  # target defaults to id

    def test_alias_of_other_kind_is_unknown(self):
        cat = ModelCatalog.from_dict(CATALOG)
        with pytest.raises(UnknownModel) as e:
            cat.resolve("tts", ModelKind.CHAT)
        # 400, not 404: Nextcloud's OpenAI app probes endpoints with model
        # "invalid-model" and treats 404 as "endpoint missing" (feature disabled).
        assert e.value.status == 400 and e.value.error_type == "model_not_found"
        assert "chat, chat-large" in e.value.message

    def test_real_filename_is_not_an_alias(self):
        with pytest.raises(UnknownModel):
            ModelCatalog.from_dict(CATALOG).resolve("small.gguf", ModelKind.CHAT)

    def test_unknown_message_does_not_echo_huge_or_odd_input(self):
        with pytest.raises(UnknownModel) as e:
            ModelCatalog.from_dict(CATALOG).resolve("<script>" + "a" * 500, ModelKind.CHAT)
        assert "<script>" not in e.value.message and len(e.value.message) < 200

    @pytest.mark.parametrize("bad", [
        {"models": [{"id": "x", "kind": "nope"}]},
        {"models": [{"kind": "chat", "target": "a.gguf"}]},
        {"models": [{"id": "x", "kind": "chat", "target": "../secret.gguf"}]},
        {"models": [{"id": "x", "kind": "chat", "target": "/data/x.gguf"}]},
        {"models": [{"id": "x y", "kind": "chat", "target": "a.gguf"}]},
        {"models": [{"id": "a", "kind": "chat", "target": "a.gguf"}, {"id": "a", "kind": "stt"}]},
        {"models": "nope"},
    ])
    def test_invalid_catalog_rejected(self, bad):
        with pytest.raises(ValueError):
            ModelCatalog.from_dict(bad)

    def test_aliases_listed_in_order(self):
        assert [a.id for a in ModelCatalog.from_dict(CATALOG).aliases] == ["chat", "chat-large", "whisper-1", "tts", "img"]


# --------------------------------------------------------------------------- chat

def _chat_alias():
    return ModelCatalog.from_dict(CATALOG).resolve("chat", ModelKind.CHAT)


class TestChatPolicy:
    policy = ChatPolicy()

    def _ok(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], **extra}
        return self.policy.sanitize(body, _chat_alias())

    def test_targets_alias_and_applies_defaults(self):
        out = self._ok()
        assert out["model"] == "small.gguf"
        assert out["thinking"] is False and out["ctx_size"] == 8192

    @pytest.mark.parametrize("field", ["gpu_id", "ctx_size", "parallel", "keep_alive", "mtp", "retries",
                                       "modalities", "voice", "n_predict", "id_slot", "slot_id", "lora",
                                       "n_probs", "grammar", "cache_prompt", "samplers", "audio"])
    def test_non_allowlisted_fields_are_dropped(self, field):
        out = self._ok(**{field: 123})
        assert field not in out or (field == "ctx_size" and out[field] == 8192)

    def test_caller_can_set_ordinary_sampling_fields(self):
        out = self._ok(temperature=0.2, top_p=0.9, stop=["x"], seed=1, stream=True, thinking=True,
                       tools=[{"type": "function", "function": {"name": "f"}}], tool_choice="auto",
                       response_format={"type": "json_object"})
        assert out["temperature"] == 0.2 and out["stream"] is True and out["thinking"] is True
        assert out["tools"][0]["function"]["name"] == "f"

    def test_max_tokens_capped_and_max_completion_tokens_mapped(self):
        assert self._ok(max_tokens=10**9)["max_tokens"] == ChatPolicy.MAX_TOKENS
        assert self._ok(max_completion_tokens=100)["max_tokens"] == 100
        assert "max_completion_tokens" not in self._ok(max_completion_tokens=100)

    def test_chat_template_kwargs_only_enable_thinking_bool(self):
        out = self._ok(chat_template_kwargs={"enable_thinking": True, "mode": "x", "template": "{{evil}}"})
        assert out["chat_template_kwargs"] == {"enable_thinking": True}
        assert "chat_template_kwargs" not in self._ok(chat_template_kwargs={"enable_thinking": "yes"})

    def test_n_other_than_one_rejected(self):
        with pytest.raises(PolicyViolation):
            self._ok(n=3)

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8030/api/models/unload_all", "https://example.com/a.png",
        "file:///etc/passwd", "ftp://x/y", "data:text/html;base64,PGgxPg==",
    ])
    def test_remote_or_non_image_urls_rejected(self, url):  # SSRF / local file reads
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
        with pytest.raises(PolicyViolation):
            self.policy.sanitize({"messages": msg}, _chat_alias())

    def test_inline_image_and_audio_allowed(self):
        msg = [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        ]}]
        assert self.policy.sanitize({"messages": msg}, _chat_alias())["messages"] == msg

    @pytest.mark.parametrize("messages", [
        None, "hi", [], [{"role": "root", "content": "x"}], [{"content": "x"}],
        [{"role": "user", "content": [{"type": "file", "file": {"path": "/etc/passwd"}}]}],
        [{"role": "user", "content": 5}],
    ])
    def test_malformed_messages_rejected(self, messages):
        with pytest.raises(PolicyViolation):
            self.policy.sanitize({"messages": messages}, _chat_alias())

    def test_too_many_messages_rejected(self):
        msgs = [{"role": "user", "content": "x"}] * (ChatPolicy.MAX_MESSAGES + 1)
        with pytest.raises(PolicyViolation):
            self.policy.sanitize({"messages": msgs}, _chat_alias())


# --------------------------------------------------------------------------- speech / image / stt

def _alias(kind, model=None):
    return ModelCatalog.from_dict(CATALOG).resolve(model, kind)


class TestSpeechPolicy:
    policy = SpeechPolicy()

    def test_defaults(self):
        cmd = self.policy.sanitize({"input": "Hallo"}, _alias(ModelKind.TTS))
        assert (cmd.text, cmd.voice, cmd.response_format, cmd.speed, cmd.engine) == \
            ("Hallo", "alloy", "mp3", 1.0, "cosyvoice")

    def test_voice_allowlist(self):
        a = _alias(ModelKind.TTS)
        assert self.policy.sanitize({"input": "x", "voice": "narrator"}, a).voice == "narrator"
        for v in ("some-actor-clip", "../voices/x", "a", ""):
            assert self.policy.sanitize({"input": "x", "voice": v}, a).voice == "alloy"

    @pytest.mark.parametrize("body", [
        {}, {"input": ""}, {"input": "   "}, {"input": 5},
        {"input": "x" * (SpeechPolicy.MAX_INPUT_CHARS + 1)},
        {"input": "x", "response_format": "exe"},
        {"input": "x", "speed": "fast"},
        {"input": "x", "language": "de; rm -rf /"},
    ])
    def test_rejected(self, body):
        with pytest.raises(PolicyViolation):
            self.policy.sanitize(body, _alias(ModelKind.TTS))

    def test_speed_clamped(self):
        a = _alias(ModelKind.TTS)
        assert self.policy.sanitize({"input": "x", "speed": 4}, a).speed == 2.0
        assert self.policy.sanitize({"input": "x", "speed": 0.1}, a).speed == 0.5

    def test_language(self):
        a = _alias(ModelKind.TTS)
        assert self.policy.sanitize({"input": "x", "language": "de"}, a).language == "de"


class TestImagePolicy:
    policy = ImagePolicy()

    def test_defaults_and_size_mapping(self):
        cmd = self.policy.sanitize({"prompt": "a cat"}, _alias(ModelKind.IMAGE))
        assert (cmd.n, cmd.base_resolution, cmd.aspect_ratio, cmd.model_name) == \
            (1, 1024, "1:1", "z_image_turbo-Q6_K.gguf")
        assert self.policy.sanitize({"prompt": "a", "size": "1792x1024"}, _alias(ModelKind.IMAGE)).aspect_ratio == "16:9"

    def test_steps_default_to_the_models_own(self):
        # A fixed 4 under-sampled Z-Image (its default is 8) and Qwen-Image (40).
        assert self.policy.sanitize({"prompt": "a"}, _alias(ModelKind.IMAGE)).steps is None
        alias = ModelAlias.from_dict({"id": "img", "kind": "image", "target": "z-image-turbo",
                                      "defaults": {"steps": 6}})
        assert self.policy.sanitize({"prompt": "a"}, alias).steps == 6

    def test_nextcloud_default_size(self):
        # integration_openai's admin setting "default_image_size" ships as 768x768;
        # the Assistant's sticker generation uses it -- refusing it broke stickers.
        cmd = self.policy.sanitize({"prompt": "a", "size": "768x768"}, _alias(ModelKind.IMAGE))
        assert (cmd.base_resolution, cmd.aspect_ratio) == (768, "1:1")

    @pytest.mark.parametrize("body", [
        {}, {"prompt": ""}, {"prompt": "x" * (ImagePolicy.MAX_PROMPT_CHARS + 1)},
        {"prompt": "x", "n": 0}, {"prompt": "x", "n": ImagePolicy.MAX_N + 1}, {"prompt": "x", "n": "many"},
        {"prompt": "x", "size": "99999x99999"},
    ])
    def test_rejected(self, body):
        with pytest.raises(PolicyViolation):
            self.policy.sanitize(body, _alias(ModelKind.IMAGE))


class TestTranscriptionPolicy:
    policy = TranscriptionPolicy()

    def test_accepts_normal(self):
        cmd = self.policy.sanitize(language="de", response_format="verbose_json", size_bytes=10_000)
        assert (cmd.language, cmd.response_format) == ("de", "verbose_json")
        assert self.policy.sanitize(language=None, response_format=None, size_bytes=1).response_format == "json"

    @pytest.mark.parametrize("kw", [
        {"language": "deutsch!", "response_format": "json", "size_bytes": 1},
        {"language": None, "response_format": "html", "size_bytes": 1},
        {"language": None, "response_format": "json", "size_bytes": 0},
    ])
    def test_rejected(self, kw):
        with pytest.raises(PolicyViolation):
            self.policy.sanitize(**kw)

    def test_too_large(self):
        with pytest.raises(PayloadTooLarge):
            self.policy.sanitize(language=None, response_format="json",
                                 size_bytes=TranscriptionPolicy.MAX_UPLOAD_BYTES + 1)


# --------------------------------------------------------------------------- limits / admission / errors

class TestBodyLimits:
    def test_upload_route_gets_more(self):
        limits = BodyLimits()
        assert limits.for_path("/public/v1/audio/transcriptions") > limits.for_path("/public/v1/chat/completions")
        assert limits.for_path("/public/v1/audio/transcriptions") > TranscriptionPolicy.MAX_UPLOAD_BYTES


class TestAdmissionControl:
    def test_limits_concurrency_and_releases(self):
        ac = AdmissionControl(max_concurrent=2)
        ac.acquire()
        ac.acquire()
        with pytest.raises(TooManyRequests) as e:
            ac.acquire()
        assert e.value.status == 429
        ac.release()
        ac.acquire()

    def test_release_never_goes_negative(self):
        ac = AdmissionControl(max_concurrent=1)
        ac.release()
        ac.acquire()
        with pytest.raises(TooManyRequests):
            ac.acquire()


class TestErrorPolicy:
    policy = ErrorPolicy()

    def test_policy_violations_pass_through(self):
        status, body = self.policy.to_public(UnknownModel("Model 'x' is not available."), ref="r1")
        assert status == 400 and body["error"]["message"] == "Model 'x' is not available."

    @pytest.mark.parametrize("status,detail", [
        (400, "Model file not found: '~/gguf_models/x.gguf' -> /media/owner/Extreme SSD/x.gguf"),
        (500, "Failed to load audio: ffmpeg ... Error opening input file /tmp/tmpabc.wav"),
        (502, "CosyVoice worker error: CUDA out of memory"),
        (507, "Strata belegt die GPUs: pid 1234 /home/me/Documents/Programming/Strata"),
    ])
    def test_internal_details_never_leak(self, status, detail):
        out_status, body = self.policy.to_public_http(status, detail, ref="abc123")
        msg = body["error"]["message"]
        assert "/" not in msg and "alex" not in msg and "CUDA" not in msg
        assert body["error"]["reference"] == "abc123"
        assert 400 <= out_status < 600

    def test_busy_states_become_retryable_503(self):
        for status in (409, 503, 507):
            assert self.policy.to_public_http(status, "x", ref="r")[0] == 503

    def test_unexpected_exception_is_500_generic(self):
        status, body = self.policy.to_public(RuntimeError("secret /mnt/path"), ref="r")
        assert status == 500 and "/mnt" not in body["error"]["message"]
