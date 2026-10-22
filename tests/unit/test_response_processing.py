"""Unit tests for backend/response_processing.py — no network, no subprocesses."""
import json
import struct

import pytest

from backend.response_processing import (
    estimate_kv_overhead_mb,
    extract_thinking,
    gguf_has_bundled_mtp,
    is_loadable_gguf,
    is_mtp_draft,
    process_completion_response,
    strip_thinking_tags,
)


def _build_gguf(kv: dict) -> bytes:
    """Minimal synthetic GGUF header + KV section (no tensors) for parser tests."""
    def _encode_value(value):
        if isinstance(value, bool):
            return struct.pack("<I", 7) + struct.pack("<B", int(value))
        if isinstance(value, int):
            return struct.pack("<I", 4) + struct.pack("<I", value)
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            return struct.pack("<I", 8) + struct.pack("<Q", len(encoded)) + encoded
        raise TypeError(f"unsupported test value type: {type(value)}")

    def _encode_string(s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    body = b""
    for key, value in kv.items():
        body += _encode_string(key) + _encode_value(value)

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    return header + body


# ---------------------------------------------------------------------------
# is_loadable_gguf
# ---------------------------------------------------------------------------

class TestIsLoadableGguf:
    def test_standalone_model(self):
        assert is_loadable_gguf("mistral-7b-instruct-v0.2.Q4_K_M.gguf")

    def test_first_shard_is_loadable(self):
        assert is_loadable_gguf("llama-3-70b-00001-of-00005.gguf")

    def test_second_shard_not_loadable(self):
        assert not is_loadable_gguf("llama-3-70b-00002-of-00005.gguf")

    def test_last_shard_not_loadable(self):
        assert not is_loadable_gguf("llama-3-70b-00005-of-00005.gguf")

    def test_mmproj_filename_still_loadable_by_this_fn(self):
        # is_loadable_gguf only checks shard numbering; mmproj filtering is caller's job
        assert is_loadable_gguf("gemma-4-mmproj.gguf")

    def test_uppercase_extension(self):
        assert is_loadable_gguf("MODEL-00001-OF-00003.GGUF")

    def test_second_shard_uppercase(self):
        assert not is_loadable_gguf("MODEL-00002-OF-00003.GGUF")


# ---------------------------------------------------------------------------
# is_mtp_draft
# ---------------------------------------------------------------------------

class TestIsMtpDraft:
    def test_mtp_prefixed_file(self):
        assert is_mtp_draft("mtp-gemma-4-12b-it.gguf")

    def test_case_insensitive(self):
        assert is_mtp_draft("MTP-Gemma-4-12B-it.gguf")

    def test_regular_model_not_mtp(self):
        assert not is_mtp_draft("gemma-4-12b-it-UD-Q4_K_XL.gguf")

    def test_mtp_substring_not_at_start_not_mtp(self):
        assert not is_mtp_draft("gemma-4-12b-mtp-it.gguf")


# ---------------------------------------------------------------------------
# gguf_has_bundled_mtp
# ---------------------------------------------------------------------------

class TestGgufHasBundledMtp:
    def test_bundled_mtp_detected(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(_build_gguf({
            "general.architecture": "qwen3",
            "qwen3.nextn_predict_layers": 1,
        }))
        assert gguf_has_bundled_mtp(str(path))

    def test_no_nextn_key_not_bundled(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(_build_gguf({
            "general.architecture": "qwen3",
            "qwen3.block_count": 32,
        }))
        assert not gguf_has_bundled_mtp(str(path))

    def test_zero_nextn_layers_not_bundled(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(_build_gguf({
            "general.architecture": "qwen3",
            "qwen3.nextn_predict_layers": 0,
        }))
        assert not gguf_has_bundled_mtp(str(path))

    def test_not_a_gguf_file(self, tmp_path):
        path = tmp_path / "model.gguf"
        path.write_bytes(b"not a gguf file at all")
        assert not gguf_has_bundled_mtp(str(path))

    def test_missing_file(self):
        assert not gguf_has_bundled_mtp("/nonexistent/path/model.gguf")


# ---------------------------------------------------------------------------
# estimate_kv_overhead_mb
# ---------------------------------------------------------------------------

class TestEstimateKvOverheadMb:
    def test_2k_ctx(self):
        assert estimate_kv_overhead_mb(2048) == 1024

    def test_4k_ctx(self):
        assert estimate_kv_overhead_mb(4096) == 1024

    def test_8k_ctx(self):
        assert estimate_kv_overhead_mb(8192) == 2048

    def test_16k_ctx(self):
        assert estimate_kv_overhead_mb(16384) == 3072

    def test_32k_ctx(self):
        assert estimate_kv_overhead_mb(32768) == 5120

    def test_64k_ctx(self):
        assert estimate_kv_overhead_mb(65536) == 9216

    def test_128k_ctx(self):
        assert estimate_kv_overhead_mb(131072) == 14336

    def test_strictly_between_4k_and_8k(self):
        assert estimate_kv_overhead_mb(5000) == 2048


# ---------------------------------------------------------------------------
# extract_thinking / strip_thinking_tags
# ---------------------------------------------------------------------------

class TestExtractThinking:
    def test_no_tags(self):
        thinking, text = extract_thinking("Hello world")
        assert thinking == ""
        assert text == "Hello world"

    def test_think_tag(self):
        thinking, text = extract_thinking("<think>some thoughts</think>The answer is 42.")
        assert thinking == "some thoughts"
        assert text == "The answer is 42."

    def test_thinking_tag(self):
        thinking, text = extract_thinking("<thinking>deliberation</thinking>Result")
        assert thinking == "deliberation"
        assert text == "Result"

    def test_multiline_thinking(self):
        thinking, text = extract_thinking("<think>\nLine 1\nLine 2\n</think>\nFinal answer.")
        assert "Line 1" in thinking
        assert "Final answer." in text

    def test_strip_thinking_tags(self):
        result = strip_thinking_tags("<think>private</think>public content")
        assert result == "public content"
        assert "private" not in result

    def test_strip_multiple_tags(self):
        result = strip_thinking_tags("<think>a</think>text<thinking>b</thinking>more")
        assert result == "textmore"


# ---------------------------------------------------------------------------
# process_completion_response
# ---------------------------------------------------------------------------

class TestProcessCompletionResponse:
    def _msg(self, content, reasoning_content=None):
        m = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            m["reasoning_content"] = reasoning_content
        return m

    def test_plain_text(self):
        raw, reasoning, data = process_completion_response(self._msg("Hello"), None)
        assert raw == "Hello"
        assert reasoning == ""
        assert data is None

    def test_thinking_false_strips_tags(self):
        raw, reasoning, data = process_completion_response(
            self._msg("<think>internal</think>visible"), thinking=False
        )
        assert "internal" not in raw
        assert raw == "visible"

    def test_thinking_true_extracts(self):
        raw, reasoning, data = process_completion_response(
            self._msg("<think>reason</think>answer"), thinking=True
        )
        assert reasoning == "reason"
        assert raw == "answer"

    def test_reasoning_content_field(self):
        raw, reasoning, data = process_completion_response(
            self._msg("final", reasoning_content="thought"), thinking=None
        )
        assert reasoning == "thought"
        assert raw == "final"

    def test_empty_content_falls_back_to_reasoning_on_clean_stop(self):
        # Some models route their entire real answer through reasoning_content by design,
        # leaving content genuinely empty -- safe to promote only when generation actually
        # finished (finish_reason == "stop").
        raw, reasoning, data = process_completion_response(
            self._msg("", reasoning_content="full output"), thinking=None, finish_reason="stop"
        )
        assert raw == "full output"

    def test_empty_content_from_truncated_reasoning_not_promoted(self):
        # Regression test: confirmed live against gemma-4-12b-it (any quant) via this
        # gateway -- a response cut short by max_tokens mid-think (finish_reason="length")
        # has content empty because the model was still mid-thought, NOT because it routes
        # output through reasoning_content by design. Promoting that unfinished scratchpad
        # into raw_content duplicated the same truncated English planning text into BOTH
        # content and reasoning_content, silently handing the caller a garbled "answer".
        raw, reasoning, data = process_completion_response(
            self._msg("", reasoning_content="unfinished scratchpad..."), thinking=None, finish_reason="length"
        )
        assert raw == ""
        assert reasoning == "unfinished scratchpad..."

    def test_empty_content_fallback_requires_finish_reason(self):
        # No finish_reason passed at all (e.g. an older caller) must not promote either --
        # only an explicit "stop" opts in.
        raw, reasoning, data = process_completion_response(
            self._msg("", reasoning_content="full output"), thinking=None
        )
        assert raw == ""

    def test_json_schema_validation_passes(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        raw, reasoning, data = process_completion_response(
            self._msg('{"x": 1}'), thinking=None, response_format=schema
        )
        assert data == {"x": 1}

    def test_json_schema_validation_fails(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            process_completion_response(
                self._msg('{"x": "not_an_int"}'), thinking=None, response_format=schema
            )

    def test_invalid_json_raises(self):
        schema = {"type": "object"}
        with pytest.raises((json.JSONDecodeError, ValueError)):
            process_completion_response(
                self._msg("not json"), thinking=None, response_format=schema
            )

    def test_json_in_code_fence_parsed(self):
        schema = {"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]}
        raw, reasoning, data = process_completion_response(
            self._msg('```json\n{"y": "hi"}\n```'), thinking=None, response_format=schema
        )
        assert data == {"y": "hi"}
