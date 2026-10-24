"""A streamed chat whose upstream refuses the request must fail with that status and message -- not 200 with
the error JSON inside an event stream (2026-10-20: the Playground showed an empty answer bubble when an "Auto"
chat landed on a reranker: llama-server answered 500 "the current context does not logits computation")."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from backend.cluster.service import Upstream
from backend.main import app
from backend.routers import gateway


class Upstream500(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"error": {"code": 500, "message": "the current context does not logits computation. skipping",
                                     "type": "server_error"}}).encode()
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class UpstreamOk(Upstream500):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'data: {"choices":[{"delta":{"content":"Hallo"}}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(params=[Upstream500, UpstreamOk])
def upstream(request, monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), request.param)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ended = []

    async def fake_resolve(model_name, **kw):
        return Upstream(kind="node", base=f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr(gateway, "_resolve_upstream", fake_resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: ended.append(keep))
    yield request.param, ended
    srv.shutdown()


def test_stream_status_follows_the_upstream(upstream):
    kind, ended = upstream
    r = TestClient(app, client=("127.0.0.1", 50000)).post(
        "/v1/chat/completions", json={"model": "x", "stream": True, "messages": [{"role": "user", "content": "Hi"}]})
    if kind is Upstream500:
        assert r.status_code == 500 and "logits computation" in r.json()["detail"]
    else:
        assert r.status_code == 200 and "Hallo" in r.text
    assert len(ended) == 1   # the request is counted as finished exactly once either way
