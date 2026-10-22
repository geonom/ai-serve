"""GET /v1/llm/models: every language model the gateway can reach -- on this machine, on enrolled devices,
Strata -- with what it is (chat / embedding / rerank / base model without a chat template) and what it can
do (owner, 2026-10-20: the Spielwiese should offer everything that runs through the gateway; it listed only
local files, without Strata or device-only models, and put embedding models into the chat list)."""
from backend.llm_catalog import build_catalog, classify, pretty_label


class TestClassify:
    def test_kinds_from_the_gguf_header(self):
        assert classify({"general.architecture": "bert", "bert.pooling_type": 2}, "bge-m3-Q8_0.gguf") == "embedding"
        assert classify({"general.architecture": "nomic-bert"}, "nomic-embed-text.gguf") == "embedding"
        assert classify({"general.architecture": "qwen3", "qwen3.pooling_type": 4,
                         "tokenizer.chat_template": "x"}, "Qwen3-Reranker-0.6B.gguf") == "rerank"
        assert classify({"general.architecture": "bert", "bert.classifier.output_labels": ["x"]}, "m.gguf") == "rerank"
        assert classify({"general.architecture": "llama"}, "YuE-s1-7B.gguf") == "base"     # no chat template
        assert classify({"general.architecture": "gemma4", "tokenizer.chat_template": "..."}, "gemma.gguf") == "chat"

    def test_without_a_header_the_name_decides(self):   # files that only live on a device
        assert classify(None, "bge-reranker-v2-m3-Q4_K_M.gguf") == "rerank"
        assert classify(None, "nomic-embed-text-v1.5.Q4_K_M.gguf") == "embedding"
        assert classify(None, "Qwen3-0.6B-Q4_K_M.gguf") == "chat"


def test_pretty_label():
    assert pretty_label("gemma-4-12b-it-Q4_K_M.gguf") == ("gemma-4-12b-it", "Q4_K_M")
    assert pretty_label("NuExtract3-Q8_0.gguf") == ("NuExtract3", "Q8_0")
    assert pretty_label("odd.gguf") == ("odd", None)


LOCAL = [
    {"name": "gemma-4-12b-it-Q4_K_M.gguf", "size_mb": 7000, "context_length": 131072, "vision": True, "mtp": True,
     "traits": {"kind": "chat", "thinking": True, "arch": "gemma4"}},
    {"name": "bge-m3-Q8_0.gguf", "size_mb": 600, "context_length": 8192, "vision": False, "mtp": False,
     "traits": {"kind": "embedding", "thinking": False, "arch": "bert"}},
]
DEVICES = [
    {"id": "macbook", "online": True, "models": {"Qwen3.5-0.8B-Q4_K_M.gguf": 500, "gemma-4-12b-it-Q4_K_M.gguf": 7000},
     "instances": [{"model": "Qwen3.5-0.8B-Q4_K_M.gguf", "state": "ready"}]},
    {"id": "pi", "online": False, "models": {"tiny-Q4_0.gguf": 100}, "instances": []},
]


def test_catalog_joins_places():
    cat = build_catalog(LOCAL, loaded_local={"gemma-4-12b-it-Q4_K_M.gguf": {"gpu_id": 0, "ctx_size": 32000}},
                        devices=DEVICES)
    rows = {r["id"]: r for r in cat["models"]}
    g = rows["gemma-4-12b-it-Q4_K_M.gguf"]
    assert g["label"] == "gemma-4-12b-it" and g["quant"] == "Q4_K_M" and g["kind"] == "chat"
    assert g["vision"] and g["mtp"] and g["thinking"] and g["context_length"] == 131072
    assert [p["place"] for p in g["where"]] == ["local", "device"]
    assert g["where"][0]["loaded"] is True and g["loaded"] is True
    q = rows["Qwen3.5-0.8B-Q4_K_M.gguf"]           # only on the MacBook: the gateway routes it there
    assert q["where"] == [{"place": "device", "node": "macbook", "online": True, "loaded": True}]
    assert q["kind"] == "chat" and q["kind_source"] == "name" and q["loaded"] is True
    assert rows["tiny-Q4_0.gguf"]["where"][0]["online"] is False
    assert rows["bge-m3-Q8_0.gguf"]["kind"] == "embedding"
    assert "strata" not in rows


def test_plugin_rows():
    """Models of llm plugins (e.g. Strata through its adapter): listed with the plugin and its state; exclusive
    plugins first among chat models; a plugin id never takes over a local file's row."""
    plugin_models = [{"id": "strata", "label": "Strata (qwen3.8)", "kind": "chat", "context_length": 65536,
                      "thinking": True, "vision": False, "aliases": ["qwen3.8-x-strata.gguf"], "plugin": "strata"},
                     {"id": "tiny-Q4_0.gguf", "kind": "chat", "plugin": "strata"}]
    cat = build_catalog([{"name": "tiny-Q4_0.gguf"}], loaded_local={}, devices=[], plugins=plugin_models,
                        plugin_states={"strata": {"state": "loading", "exclusive": True, "name": "Strata"}})
    s = cat["models"][0]
    assert (s["id"], s["kind"], s["exclusive"], s["loaded"], s["context_length"]) == ("strata", "chat", True, False, 65536)
    assert s["where"] == [{"place": "plugin", "plugin": "strata", "name": "Strata", "state": "loading", "loaded": False}]
    assert s["aliases"] == ["qwen3.8-x-strata.gguf"]
    assert [p["place"] for p in cat["models"][1]["where"]] == ["local"]



def test_endpoint_lists_local_models_with_their_kind(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from backend import model_manager
    from backend.routers import dashboard, gateway
    for name in ("gemma-4-12b-it-Q4_K_M.gguf", "bge-m3-Q8_0.gguf", "bge-reranker-v2-m3-Q8_0.gguf"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(model_manager, "MODEL_DIRS", [str(tmp_path)])   # never this machine's model folders
    monkeypatch.setattr(gateway, "MODEL_DIRS", [str(tmp_path)])
    monkeypatch.setattr(dashboard, "MODEL_DIRS", [str(tmp_path)])
    monkeypatch.setattr(model_manager, "get_model_traits", lambda path: None)   # no full header scans here
    from backend.main import app
    r = TestClient(app, client=("127.0.0.1", 50000)).get("/v1/llm/models")
    assert r.status_code == 200, r.text
    rows = r.json()["models"]
    assert rows and all(set(x) >= {"id", "label", "kind", "where", "loaded"} for x in rows)
    assert all(x["kind"] in ("chat", "embedding", "rerank", "base") for x in rows)
    assert all(x["kind_source"] in ("name", "config") for x in rows)   # traits faked away -> names decide
