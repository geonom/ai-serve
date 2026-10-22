import json
import re
import struct
from typing import Any, Dict, Optional, Tuple


# Matches the secondary-shard naming convention: model-00002-of-00005.gguf etc.
_MULTIPART_SHARD_RE = re.compile(r'-(\d{5})-of-\d{5}\.gguf$', re.IGNORECASE)


def is_loadable_gguf(name: str) -> bool:
    """Return True for standalone GGUFs and for the first shard of multi-part files."""
    m = _MULTIPART_SHARD_RE.search(name)
    return (not m) or m.group(1) == "00001"


def is_mtp_draft(name: str) -> bool:
    """Return True for MTP (multi-token-prediction) draft head files, e.g. mtp-gemma-4-12b-it.gguf."""
    return name.lower().startswith("mtp-")


_GGUF_INT_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
_GGUF_FLOAT_FMT = {6: "<f", 12: "<d"}
# Byte width of every fixed-size KV/array-element type (everything except string=8 and
# array=9, which are variable-length and need their own skip logic).
_GGUF_FIXED_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_gguf_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype in _GGUF_INT_FMT:
        fmt = _GGUF_INT_FMT[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype in _GGUF_FLOAT_FMT:
        fmt = _GGUF_FLOAT_FMT[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 7:  # bool
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == 8:  # string
        return _read_gguf_string(f)
    if vtype == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def _skip_gguf_array(f, elem_type: int, count: int) -> None:
    """Advance past an array's body without materializing its elements. Fixed-size
    element arrays skip in a single seek; string-element arrays (e.g. a 260k-entry
    tokenizer vocab) still need one length-read per element — GGUF has no upfront
    total-byte-length for those — but never decode or allocate the string content.
    """
    if elem_type in _GGUF_FIXED_SIZE:
        f.seek(count * _GGUF_FIXED_SIZE[elem_type], 1)
    elif elem_type == 8:
        for _ in range(count):
            n = struct.unpack("<Q", f.read(8))[0]
            f.seek(n, 1)
    elif elem_type == 9:  # nested arrays — unused in practice, but handle defensively
        for _ in range(count):
            nested_type = struct.unpack("<I", f.read(4))[0]
            nested_count = struct.unpack("<Q", f.read(8))[0]
            _skip_gguf_array(f, nested_type, nested_count)
    else:
        raise ValueError(f"unknown GGUF array element type {elem_type}")


def _skip_gguf_value(f, vtype: int) -> None:
    """Like _read_gguf_value, but discards the value instead of materializing it —
    used by the metadata scanners below, which only care about a couple of keys and
    would otherwise waste most of their time fully decoding huge, irrelevant tokenizer
    vocab arrays just to skip past them.
    """
    if vtype in _GGUF_FIXED_SIZE:
        f.seek(_GGUF_FIXED_SIZE[vtype], 1)
    elif vtype == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(n, 1)
    elif vtype == 9:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        _skip_gguf_array(f, elem_type, count)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_has_bundled_mtp(path: str) -> bool:
    """Detect MTP/nextn tensors bundled directly into a model's own GGUF (as opposed
    to split into a separate mtp-*.gguf draft head), by checking the metadata for a
    truthy '<arch>.nextn_predict_layers' key. Only reads the header/KV section, never
    tensor data, and skips (rather than decodes) every other value — including any
    multi-hundred-thousand-entry tokenizer vocab array — so this is cheap even for
    multi-GB files with large vocabularies.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
            f.read(4)  # version, unused
            f.read(8)  # tensor_count, unused
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                key = _read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".nextn_predict_layers") and vtype in _GGUF_INT_FMT:
                    value = _read_gguf_value(f, vtype)
                    if value > 0:
                        return True
                else:
                    _skip_gguf_value(f, vtype)
    except Exception:
        return False
    return False


def gguf_has_bundled_vision(path: str) -> bool:
    """Detect a vision encoder bundled directly into a model's own GGUF (as opposed to a
    separate mmproj sidecar file), by checking the metadata for any '<arch>.vision.*' or
    'clip.vision.*' key. Newer natively-multimodal architectures can ship this way, in
    which case no --mmproj flag is needed at all. Never needs the values themselves, so
    every one is skipped rather than decoded — cheap even for multi-GB files.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
            f.read(4)  # version, unused
            f.read(8)  # tensor_count, unused
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                key = _read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if ".vision." in key.lower():
                    return True
                _skip_gguf_value(f, vtype)
    except Exception:
        return False
    return False


def gguf_context_length(path: str) -> Optional[int]:
    """Read a GGUF model's trained context length (the '<arch>.context_length' metadata
    key -- e.g. 'qwen3.context_length', 'llama.context_length') without loading tensor
    data. Same single-pass skip-everything-else approach as gguf_has_bundled_mtp/vision
    above. Returns None if the key is missing or the file isn't a valid GGUF -- some
    architectures may name it differently, and callers should treat that as 'unknown',
    not an error.
    """
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)  # version, unused
            f.read(8)  # tensor_count, unused
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                key = _read_gguf_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key.endswith(".context_length") and vtype in _GGUF_INT_FMT:
                    return _read_gguf_value(f, vtype)
                _skip_gguf_value(f, vtype)
    except Exception:
        return None
    return None


_HEADER_MAX_ARRAY_PREVIEW = 10
_HEADER_MAX_STRING_LEN = 4000


def _summarize_header_value(value: Any) -> Any:
    """Truncate huge scalar strings (e.g. a long chat template) so the response stays
    readable and small. Arrays are already source-truncated by _read_gguf_array_preview
    below — this only ever sees the (small) preview list, not the full array.
    """
    if isinstance(value, list):
        return [_summarize_header_value(v) for v in value]
    if isinstance(value, str) and len(value) > _HEADER_MAX_STRING_LEN:
        return f"{value[:_HEADER_MAX_STRING_LEN]}... [truncated, {len(value)} chars total]"
    return value


def _read_gguf_array_preview(f, elem_type: int, count: int, max_preview: int) -> Any:
    """Decode only the first max_preview elements of an array and skip the rest without
    materializing them — avoids fully decoding e.g. a 260k-entry tokenizer vocab just to
    display the first 10 of it.
    """
    n_preview = min(count, max_preview)
    preview = [_summarize_header_value(_read_gguf_value(f, elem_type)) for _ in range(n_preview)]
    if count > n_preview:
        _skip_gguf_array(f, elem_type, count - n_preview)
        return {"_type": "array", "length": count, "preview": preview}
    return preview


def read_gguf_header(path: str, full_keys: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """Read a GGUF file's full KV metadata header — architecture, context length,
    quantization params, rope/vision/MTP config, chat template, etc. Header-only read,
    never touches tensor data. Large arrays (e.g. the tokenizer vocab) are decoded only
    up to their preview length and skipped past otherwise, so this stays cheap even for
    multi-GB files with large vocabularies. `full_keys` are kept whole (e.g. the chat template, which the
    model catalog searches for a thinking switch).
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (bad magic: {magic!r})")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == 9:
                elem_type = struct.unpack("<I", f.read(4))[0]
                count = struct.unpack("<Q", f.read(8))[0]
                metadata[key] = _read_gguf_array_preview(f, elem_type, count, _HEADER_MAX_ARRAY_PREVIEW)
            else:
                value = _read_gguf_value(f, vtype)
                metadata[key] = value if key in full_keys else _summarize_header_value(value)
    return {
        "gguf_version": version,
        "tensor_count": tensor_count,
        "kv_count": n_kv,
        "metadata": metadata,
    }


def estimate_kv_overhead_mb(ctx_size: int) -> int:
    """Rough KV-cache VRAM overhead keyed to context size (assumes typical 7-34B depth)."""
    if ctx_size <= 4096:
        return 1024
    elif ctx_size <= 8192:
        return 2048
    elif ctx_size <= 16384:
        return 3072
    elif ctx_size <= 32768:
        return 5120
    elif ctx_size <= 65536:
        return 9216
    else:
        return 14336


# Matches <think>, <thinking>, <THOUGHT_PROCESS> (and closing tags) case-insensitively.
_THINK_TAG_RE = re.compile(
    r'<(think|thinking|thought_process)>(.*?)</\1>',
    re.DOTALL | re.IGNORECASE,
)


def extract_thinking(text: str) -> tuple:
    """Return (first_thinking_block_text, text_with_all_thinking_tags_removed)."""
    match = _THINK_TAG_RE.search(text)
    thinking = match.group(2).strip() if match else ""
    return thinking, _THINK_TAG_RE.sub("", text).strip()


def strip_thinking_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def process_completion_response(
    message: Dict[str, Any],
    thinking: Optional[bool],
    response_format: Optional[Any] = None,
    finish_reason: Optional[str] = None,
) -> tuple:
    """Process a llama-server chat completion message.

    Returns (raw_content, reasoning, parsed_data).
    Raises ValueError or jsonschema.ValidationError on schema validation failure
    so the caller can retry or surface the error.
    """
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning_content", "") or message.get("thought", "") or ""

    if thinking is False:
        raw_content = strip_thinking_tags(raw_content)
    else:
        embedded_thinking, raw_content = extract_thinking(raw_content)
        if embedded_thinking and not reasoning:
            reasoning = embedded_thinking

    # Some models (confirmed live: gemma-4-12b-it, any quant) route their ENTIRE output
    # through reasoning_content while the model is still inside its think block, leaving
    # content genuinely empty -- llama-server itself does this correctly. That's only safe
    # to promote into raw_content when generation actually finished cleanly (finish_reason
    # == "stop"): the model reached its own end-of-turn and, for whichever reason, never
    # populated a separate content field, so reasoning_content IS the real answer. When
    # generation was cut short by the token budget (finish_reason == "length") or anything
    # else non-terminal, content is empty because the model was still MID-THOUGHT, not
    # because it routes output through reasoning_content by design -- promoting that
    # unfinished scratchpad into raw_content silently hands the caller a garbled "answer"
    # that's actually just an incomplete internal monologue (confirmed live: gemma-4-12b-it
    # via this gateway returned the exact same truncated English fact-extraction/planning
    # text duplicated into BOTH content and reasoning_content, while the same request sent
    # directly to llama-server correctly left content empty). Only "stop" is safe here --
    # not just "not length", since other finish_reasons (tool_calls, content_filter, etc.)
    # have the same "still incomplete, don't fabricate a final answer" problem.
    if not raw_content and reasoning and finish_reason == "stop":
        raw_content = reasoning
        if thinking is False:
            reasoning = ""

    parsed_data = None
    if response_format:
        import jsonschema
        schema_data = (
            response_format.get("json_schema", {}).get("schema")
            if isinstance(response_format, dict) and "json_schema" in response_format
            else response_format
        )
        json_str = raw_content.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            json_str = "\n".join(lines).strip()
        parsed_data = json.loads(json_str)
        jsonschema.validate(instance=parsed_data, schema=schema_data)

    return raw_content, reasoning, parsed_data
