"""Decision models on /v1/systemone (2026-10-30, owner: LiquidAI's LFM3 d1-2B): recognised from the GGUF header,
requests forwarded to llama-server's TypeSafe-compatible endpoint without the gateway's own fields."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.cluster.service import Upstream
from backend.llm_catalog import classify
from backend.main import app
from backend.routers import gateway


def test_a_decision_model_is_its_own_kind():
    meta = {"general.architecture": "lfm3_vl", "lfm3_vl.decision.type": "lfm3-d1",
            "tokenizer.chat_template": "{{ messages }}"}
    assert classify(meta, "d1-2B-Q8_0.gguf") == "decision"
    assert classify({"general.architecture": "lfm3_vl", "tokenizer.chat_template": "x"}, "LFM3-VL-2B.gguf") == "chat"


def upstream(status=200, answer=None):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(answer if answer is not None else {
                "model": "/mnt/x/d1-2B-Q8_0.gguf",
                "answers": {"refund": {"type": "noul", "noul": 0.97}},
                "usage": {"input_tokens": 40, "output_tokens": 0}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def fake(monkeypatch):
    made = []

    def install(**kw):
        srv, seen = upstream(**kw)
        made.append(srv)
        asked = {}

        async def resolve(model_name, **load):
            asked.update(model=model_name, **load)
            return Upstream(kind="node", base=f"http://127.0.0.1:{srv.server_port}", label="d1-2B-Q8_0.gguf")
        monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
        monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
        monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
        return seen, asked
    yield install
    for s in made:
        s.shutdown()


def client():
    return TestClient(app, client=("127.0.0.1", 50000))


def test_questions_are_forwarded_without_the_gateways_own_fields(fake):
    seen, asked = fake()
    r = client().post("/v1/systemone", json={
        "model": "d1-2B-Q8_0.gguf", "state": "I was charged twice, please refund one.", "gpu_id": 1,
        "keep_alive": 60, "images": ["data:image/png;base64,AAAA"], "client_tag": "x",
        "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}})
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["refund"]["noul"] == 0.97 and r.json()["model"] == "d1-2B-Q8_0.gguf"
    assert seen["path"] == "/v1/systemone"
    assert seen["body"] == {"state": "I was charged twice, please refund one.", "client_tag": "x",
                            "images": ["data:image/png;base64,AAAA"],
                            "questions": {"refund": {"type": "noul",
                                                     "instructions": "Is the customer asking for a refund?"}}}
    assert asked["model"] == "d1-2B-Q8_0.gguf" and asked["gpu_id"] == 1 and asked["use_mtp"] is False


def test_without_a_model_the_first_decision_model_is_taken(fake, monkeypatch):
    seen, asked = fake()
    monkeypatch.setattr(gateway, "_default_decision_model", lambda: "/m/d1-2B-Q8_0.gguf")
    r = client().post("/v1/systemone", json={"state": None, "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 200 and asked["model"] == "/m/d1-2B-Q8_0.gguf"
    assert seen["body"]["state"] is None


def test_no_decision_model_says_where_to_get_one(monkeypatch):
    monkeypatch.setattr(gateway, "_default_decision_model", lambda: None)
    r = client().post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 400 and "d1-2B-GGUF" in r.text


def test_a_model_that_cannot_decide_keeps_llama_servers_status(fake):
    fake(status=501, answer={"error": {"code": 501, "message": "This server does not support decisions"}})
    r = client().post("/v1/systemone", json={"model": "gemma.gguf", "state": "x",
                                             "questions": {"q": {"type": "noul", "instructions": "?"}}})
    assert r.status_code == 501 and "support decisions" in r.text
