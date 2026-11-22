"""No thinking unless a call asks for it (2026-11-03, owner: "Sollte generell der default sein, also denken off").

First only Qwen and Strata (V-Noveller: a Strata scene plan took 475 s instead of ~30 s); then V-Noveller found
gemma-4-31B writing 755 characters of reasoning for a one-line translation, content empty at max_tokens 200 (339 s
per scene). Now every llama.cpp model (here or on a device) and Strata get chat_template_kwargs.enable_thinking=false
when the call says nothing; `thinking: true` or an explicit chat_template_kwargs.enable_thinking still wins. Other
model servers behind a plugin (Ollama, LM Studio ...) get nothing they didn't ask for -- they may not know the field."""
from backend.cluster.service import Upstream
from backend.routers.gateway import _build_chat_template_kwargs, _thinking_default


def up(kind, label, model=None):
    return Upstream(kind=kind, base="http://127.0.0.1:1", label=label, model=model)


def test_llama_cpp_models_and_strata_default_to_no_thinking():
    assert _thinking_default(up("local", "Qwen3.8-27B-UD-Q4_K_XL.gguf")) is False
    assert _thinking_default(up("local", "gemma-4-31B-it-Q4_K_M.gguf")) is False
    assert _thinking_default(up("node", "gemma-4-12b-it-qat-q4_0.gguf @ mac")) is False
    assert _thinking_default(up("plugin", "strata @ strata", "strata")) is False


def test_other_model_servers_get_nothing_unasked():
    assert _thinking_default(up("plugin", "llama3.2:3b @ ollama", "llama3.2:3b")) is None
    assert _thinking_default(up("plugin", "qwen/qwen3-8b @ lmstudio", "qwen/qwen3-8b")) is None


def test_the_default_applies_only_when_the_call_says_nothing():
    assert _build_chat_template_kwargs(None, None, default=False) == {"enable_thinking": False}
    assert _build_chat_template_kwargs(True, None, default=False) == {"enable_thinking": True}
    assert _build_chat_template_kwargs(None, {"enable_thinking": True}, default=False) == {"enable_thinking": True}
    assert _build_chat_template_kwargs(None, {"mode": "x"}, default=False) == {"mode": "x", "enable_thinking": False}
    assert _build_chat_template_kwargs(None, None) is None              # a plugin server: nothing sent
