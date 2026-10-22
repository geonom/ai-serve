from strata_plugin import rules

CONFIG = {"model_name": "qwen3.8-flash-next-iq3_s", "args": ["--pack", "x", "--max-context", "65536", "--kv", "int8"]}


def test_catalog_from_the_config_alone():
    cat = rules.catalog(CONFIG, ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-strata.gguf", "bad alias!"])
    [m] = cat["data"]
    assert (m["id"], m["kind"], m["context_length"], m["thinking"]) == ("strata", "chat", 65536, True)
    assert m["aliases"] == ["qwen3.8-flash-next-iq3_s", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-strata.gguf"]
    assert m["label"] == "Strata (qwen3.8-flash-next-iq3_s)"


def test_catalog_without_details():
    [m] = rules.catalog({}, [])["data"]
    assert m["context_length"] is None and m["aliases"] == [] and m["label"] == "Strata"


def test_payload_for_strata():
    out = rules.upstream_payload({"model": "strata", "messages": [], "repeat_penalty": 1.1,
                                  "response_format": {"type": "json_object"}})
    assert "model" not in out and "repeat_penalty" not in out
    assert rules.upstream_payload({"repeat_penalty": 1.1})["repeat_penalty"] == 1.1


def test_states():
    assert rules.state_of(False, None) == "stopped"
    assert rules.state_of(True, None) == "loading"
    assert rules.state_of(True, {"loaded": False}) == "loading"
    assert rules.state_of(False, {"loaded": True}) == "ready"      # started elsewhere: adopted
