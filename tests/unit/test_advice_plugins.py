"""advice plugins (backend/plugins/advice.py, /v1/llm/advice/*): which language model suits a task -- for apps
(domains, recommend), people (ask) and agents (MCP). The plugin is faked at the HTTP level and reached only through
the plugin registry; the gateway's own model list is faked too."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.plugins import advice, registry
from backend.plugins.contracts import ContractError, normalize_catalog, normalize_recommendation, parse_provides
from backend.routers import advice as advice_router
from backend.services.domain import ServiceManifest, ServiceRegistry

client = TestClient(app, client=("127.0.0.1", 50000))
PORT = 8040

MODEL = {"id": "gemma-12b.gguf", "rank": 1, "tier": 1, "score": 0.85, "coverage": 1.0, "reliable": True,
         "dimensions": {"tools": {"score": 0.86, "ci": [0.8, 0.9], "n": 646, "templates": 20, "templates_total": 24}},
         "speed": [{"gpu": "NVIDIA GeForce RTX 4060 Ti", "ctx": 8192, "covers_ctx": True, "gen_tps": 60.7,
                    "gen_n": 270, "prefill_tps": 1444, "prefill_n": 43}],
         "memory": {"vram_mb": 12209, "gpu": "NVIDIA GeForce RTX 4060 Ti", "ctx": 8192, "covers_ctx": True, "n": 9},
         "file_mb": 7000, "warnings": []}


class FakeAdvice:
    def __init__(self):
        self.requests = []
        self.reject = None

    async def ensure_running(self, p, timeout_s=40.0):
        return True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/advice/catalog":
            return httpx.Response(200, json={"data": [
                {"id": "agentic", "kind": "use_case", "label": "Agentisch", "dimensions": ["tools", "procedure"],
                 "languages": ["de", "en"], "available": True},
                {"id": "tools", "kind": "dimension", "label": "Tools", "dimensions": ["tools"], "available": True},
                {"id": "bad", "kind": "nonsense"}]})
        if path == "/v1/advice/recommend":
            body = json.loads(request.content)
            self.requests.append(body)
            if self.reject:
                return httpx.Response(400, json={"detail": self.reject})
            other = {**MODEL, "id": "far-away.gguf", "rank": 2, "score": 0.8}
            return httpx.Response(200, json={
                "object": "advice.recommendation", "domain": {"id": body.get("domain"), "label": "Agentisch",
                                                              "weights": {"tools": 1.0, "robustness": 0.3}},
                "language": body.get("language"), "ctx": 8192, "models": [MODEL, other, {"id": "x", "score": 7}],
                "excluded": [{"id": "lucky.gguf", "score": 1.0, "reason": "too little data"}],
                "notes": ["n"], "source": "llm-benchy", "data_as_of": "2026-10-25 10:00:00"})
        return httpx.Response(404, json={"detail": path})


def _manifest():
    return ServiceManifest(id="llm-benchy", name="llm-benchy", description="d", project_dir=Path("/x"),
                           command=("./run.sh",), env={}, port=PORT, bind="127.0.0.1", match=("x",), identity="none",
                           ui=True, provides=parse_provides({"advice": {"catalog": "/v1/advice/catalog",
                                                                        "recommend": "/v1/advice/recommend"}}))


LOCAL = [{"place": "local", "loaded": True}]


async def _llm_catalog():
    # the shape /v1/llm/models really answers (backend/llm_catalog.py)
    return {"object": "list", "models": [{"id": "gemma-12b.gguf", "kind": "chat", "where": LOCAL, "loaded": True},
                                         {"id": "embed.gguf", "kind": "embedding", "where": LOCAL},
                                         {"id": "gone.gguf", "kind": "chat", "where": []}]}


@pytest.fixture
def fake(monkeypatch):
    from backend.routers import services as svc
    f = FakeAdvice()
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([_manifest()], [], {}))
    monkeypatch.setattr(registry, "_transport", httpx.MockTransport(f))
    monkeypatch.setattr(registry, "ensure_running", f.ensure_running)
    monkeypatch.setattr(advice, "_cache", (0.0, [], []))
    monkeypatch.setattr(advice_router, "_llm_catalog", _llm_catalog)
    return f


def test_contract_needs_catalog_and_recommend_and_never_a_gpu():
    with pytest.raises(ContractError):
        parse_provides({"advice": {"catalog": "/c"}})
    with pytest.raises(ContractError):
        parse_provides({"advice": {"catalog": "/c", "recommend": "/r", "gpu": "shared"}})


def test_catalog_entries_are_checked():
    entries, errors = normalize_catalog("advice", "p", {"data": [{"id": "agentic", "dimensions": ["tools", "BAD!"],
                                                                  "languages": ["de", "xx-toolong"]},
                                                                 {"id": "x", "kind": "nonsense"}]})
    assert entries[0]["kind"] == "use_case" and entries[0]["dimensions"] == ["tools"]
    assert entries[0]["languages"] == ["de"] and len(errors) == 1


def test_recommendation_rows_are_checked_and_bad_ones_dropped():
    rec, errors = normalize_recommendation("p", {"models": [MODEL, {"id": "y", "score": 2}, {"id": "../x", "score": 1}],
                                                 "excluded": [{"id": "z", "reason": "r"}, {"id": "../z"}]})
    assert [m["id"] for m in rec["models"]] == ["gemma-12b.gguf"] and len(errors) == 2
    assert rec["models"][0]["memory"]["vram_mb"] == 12209 and rec["models"][0]["speed"][0]["gen_tps"] == 60.7
    assert [e["id"] for e in rec["excluded"]] == ["z"] and rec["provider"] == "p"
    with pytest.raises(ContractError):
        normalize_recommendation("p", {"models": "nope"})


def test_domains_list_every_plugins_domains(fake):
    d = client.get("/v1/llm/advice/domains").json()
    assert [r["id"] for r in d["data"]] == ["agentic", "tools"] and d["data"][0]["provider"] == "llm-benchy"
    assert "errors" in d


def test_recommend_asks_only_for_servable_models_and_says_where_they_run(fake):
    r = client.post("/v1/llm/advice/recommend", json={"domain": "agentic", "language": "de"})
    assert r.status_code == 200, r.text
    assert fake.requests[-1] == {"domain": "agentic", "language": "de", "models": ["gemma-12b.gguf"]}
    d = r.json()
    assert [m["id"] for m in d["models"]] == ["gemma-12b.gguf", "far-away.gguf"] and len(d["errors"]) == 1
    assert d["models"][0]["where"] == LOCAL and d["models"][0]["loaded"] is True
    assert d["models"][1]["where"] == [] and d["excluded"][0]["reason"] == "too little data"


def test_installed_only_false_and_an_own_model_list(fake):
    client.post("/v1/llm/advice/recommend", json={"domain": "agentic", "installed_only": False})
    assert "models" not in fake.requests[-1]
    client.post("/v1/llm/advice/recommend", json={"domain": "agentic", "models": ["GEMMA-12B.gguf", "gone.gguf"]})
    assert fake.requests[-1]["models"] == ["GEMMA-12B.gguf"]


@pytest.mark.parametrize("body,status,why", [
    ({"domain": "nope"}, 400, "unknown domain"),
    ({"domain": "agentic", "colour": "red"}, 400, "unknown field"),
    ([1, 2], 400, "JSON object"),
])
def test_bad_requests_say_why(fake, body, status, why):
    r = client.post("/v1/llm/advice/recommend", json=body)
    assert r.status_code == status and why in r.json()["detail"]


def test_a_plugins_refusal_keeps_its_reason(fake):
    fake.reject = "ctx: a number 512..2000000"
    r = client.post("/v1/llm/advice/recommend", json={"domain": "agentic", "ctx": 1})
    assert r.status_code == 400 and r.json()["detail"] == "ctx: a number 512..2000000"


def test_no_plugin_is_503(monkeypatch):
    from backend.routers import services as svc
    monkeypatch.setattr(svc, "_registry", lambda: ServiceRegistry([], [], {}))
    monkeypatch.setattr(advice, "_cache", (0.0, [], []))
    assert client.post("/v1/llm/advice/recommend", json={"domain": "agentic"}).status_code == 503


def test_plan_request_keeps_only_sane_fields():
    req = advice.plan_request({"domain": "agentic", "language": "de", "ctx": 8192, "gpus": ["4060 Ti"],
                               "min_gen_tps": 40, "max_vram_mb": 3}, ["agentic"], 5)
    assert req == {"limit": 5, "domain": "agentic", "language": "de", "ctx": 8192, "min_gen_tps": 40.0,
                   "gpus": ["4060 Ti"]}
    with pytest.raises(advice.AdviceError):
        advice.plan_request({"domain": "invented"}, ["agentic"])
    schema = advice.plan_schema(["agentic"])
    assert schema["properties"]["domain"]["anyOf"][0]["enum"] == ["agentic"]


def test_ask_plans_ranks_and_words_from_the_rows(fake, monkeypatch):
    calls = []

    async def chat(model, messages, response_format=None, max_tokens=600):
        calls.append((model, messages, response_format))
        if response_format:
            return json.dumps({"domain": "agentic", "language": "de", "ctx": None, "gpus": None,
                               "min_gen_tps": None, "max_vram_mb": None})
        return "  gemma-12b.gguf passt am besten.  "

    monkeypatch.setattr(advice_router, "_chat", chat)
    r = client.post("/v1/llm/advice/ask", json={"question": "Welches Modell für Tool-Calls auf Deutsch?", "model": "m"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["answer"] == "gemma-12b.gguf passt am besten." and d["request"]["domain"] == "agentic"
    assert d["recommendation"]["models"][0]["id"] == "gemma-12b.gguf"
    facts = calls[1][1][0]["content"]
    assert "gemma-12b.gguf" in facts and "too little data" in facts and calls[0][0] == "m"


def rpc(method, params=None, rid=1):
    msg = {"jsonrpc": "2.0", "method": method, **({"id": rid} if rid is not None else {})}
    if params is not None:
        msg["params"] = params
    return client.post("/v1/llm/advice/mcp", json=msg)


def test_mcp_initialize_list_and_call(fake):
    init = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}).json()["result"]
    assert init["protocolVersion"] == "2025-03-26" and "tools" in init["capabilities"]
    assert rpc("notifications/initialized", rid=None).status_code == 202
    tools = rpc("tools/list").json()["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_model_domains", "recommend_models"]
    assert all(t["risk"] == "read" and t["annotations"]["readOnlyHint"] for t in tools)
    domains = rpc("tools/call", {"name": "list_model_domains", "arguments": {}}).json()["result"]
    assert [d["id"] for d in domains["structuredContent"]["domains"]] == ["agentic", "tools"]
    res = rpc("tools/call", {"name": "recommend_models", "arguments": {"domain": "agentic"}}).json()["result"]
    assert res["isError"] is False and res["structuredContent"]["models"][0]["id"] == "gemma-12b.gguf"


def test_mcp_errors(fake):
    bad = rpc("tools/call", {"name": "recommend_models", "arguments": {"domain": "nope"}}).json()["result"]
    assert bad["isError"] is True and "unknown domain" in bad["content"][0]["text"]
    assert rpc("tools/call", {"name": "nope"}).json()["error"]["code"] == -32602
    assert rpc("resources/list").json()["error"]["code"] == -32601
    assert client.get("/v1/llm/advice/mcp").status_code == 405
