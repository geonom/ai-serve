"""Structured output (response_format) retries are visible to the caller, and a reply cut off by max_tokens is not
retried (2026-10-25, V-Noveller: gemma-12b looped until max_tokens, three hidden retries of ~100 s each ended in a
422 after ~365 s, and a 200 reported only the last try's runtime_seconds)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.cluster.service import Upstream
from backend.main import app
from backend.routers import gateway

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def serve(replies):
    """An upstream answering each POST with the next (content, finish_reason)."""
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            content, finish = replies[min(len(calls), len(replies) - 1)]
            calls.append(finish)
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                            "finish_reason": finish}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls


@pytest.fixture
def upstream(monkeypatch):
    servers = []

    def make(replies):
        srv, calls = serve(replies)
        servers.append(srv)

        async def fake_resolve(model_name, **kw):
            return Upstream(kind="node", base=f"http://127.0.0.1:{srv.server_port}")
        monkeypatch.setattr(gateway, "_resolve_upstream", fake_resolve)
        monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
        monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
        return calls
    yield make
    for s in servers:
        s.shutdown()


client = TestClient(app, client=("127.0.0.1", 50000))
V1 = {"model": "x", "messages": [{"role": "user", "content": "Hi"}],
      "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}}
API = {"model": "x", "messages": [{"role": "user", "content": "Hi"}], "response_format": SCHEMA}


@pytest.mark.parametrize("path,body", [("/v1/chat/completions", V1), ("/api/chat/completions", API)])
def test_a_reply_cut_off_by_max_tokens_is_not_retried(upstream, path, body):
    calls = upstream([('{"a": 1, "a": 1, "a": 1, "a', "length")])
    r = client.post(path, json=body)
    assert r.status_code == 422
    assert calls == ["length"]
    assert "1 attempt" in r.json()["detail"] and "max_tokens" in r.json()["detail"]


@pytest.mark.parametrize("path,body", [("/v1/chat/completions", V1), ("/api/chat/completions", API)])
def test_attempts_and_total_time_are_reported(upstream, path, body):
    calls = upstream([("not json", "stop"), ('{"a": 1}', "stop")])
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert calls == ["stop", "stop"]
    assert d["attempts"] == 2 and d["total_seconds"] >= d["runtime_seconds"]


def test_failed_attempts_are_counted_in_the_422(upstream):
    calls = upstream([("not json", "stop")])
    r = client.post("/v1/chat/completions", json={**V1, "retries": 1})
    assert r.status_code == 422 and len(calls) == 2
    assert "2 attempts" in r.json()["detail"]
