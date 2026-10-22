"""Unit tests for backend/routers/gateway.py's _build_retry_nudge() and
_is_tool_format_error() -- no network, no subprocesses."""
from backend.routers.gateway import _build_retry_nudge, _is_tool_format_error


def test_peg_format_rejection_gets_tool_call_specific_nudge():
    # Exact upstream message shape from llama.cpp's chat.cpp (common_chat_parse):
    # "The model produced output that does not match the expected {format} format".
    err = "Upstream error: The model produced output that does not match the expected peg-gemma4 format"
    nudge = _build_retry_nudge(err)
    assert "exactly ONE tool per turn" in nudge
    assert "valid JSON" not in nudge
    assert err in nudge


def test_peg_format_match_is_substring_not_exact_so_other_architectures_match_too():
    # The format name varies per architecture (peg-gemma4, peg-llama3, ...) -- the
    # detection must key off the fixed part of the message, not the whole string.
    err = "The model produced output that does not match the expected peg-llama3 format"
    assert "exactly ONE tool per turn" in _build_retry_nudge(err)


def test_json_validation_failure_keeps_the_original_json_nudge():
    err = "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    nudge = _build_retry_nudge(err)
    assert "Please output valid JSON" in nudge
    assert "tool per turn" not in nudge
    assert err in nudge


def test_unrelated_upstream_error_falls_back_to_json_nudge():
    # Any other upstream failure (timeout, connection reset, ...) should still get
    # *some* corrective nudge rather than crashing -- falls back to the original
    # generic wording since there's no more specific failure kind to target.
    err = "Connection reset by peer"
    nudge = _build_retry_nudge(err)
    assert "Please output valid JSON" in nudge


def test_is_tool_format_error_matches_peg_rejection():
    assert _is_tool_format_error(
        "Upstream error: The model produced output that does not match the expected peg-gemma4 format"
    )


def test_is_tool_format_error_rejects_unrelated_errors():
    assert not _is_tool_format_error("Expecting property name enclosed in double quotes")
    assert not _is_tool_format_error("Connection reset by peer")
    assert not _is_tool_format_error("Failed to parse tool call arguments as JSON")
