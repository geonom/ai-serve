"""Stopping a non-streamed chat (2026-10-26, owner: "ai-serve should be able to cancel a non-streamed chat"):
V-Noveller stopped a client whose json_schema chat had looped, but llama-server kept generating until max_tokens and
held GPU 0 for minutes -- every other model got 507. Now the gateway stops the upstream call when the client goes away
or when POST /v1/chat/cancel names the request's X-Request-Id."""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import chat_jobs
from backend.cluster.service import Upstream
from backend.main import app
from backend.routers import gateway

RID = "chat-0123456789abcdef"


def never_gone():
    async def f():
        return False
    return f


class TestRun:
    def test_the_result_comes_back(self):
        async def work():
            return 42
        assert asyncio.run(chat_jobs.run(work(), RID, never_gone())) == 42
        assert not chat_jobs.running(RID)

    def test_cancel_stops_the_work_and_answers_409(self):
        stopped = []

        async def work():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                stopped.append(True)
                raise

        async def main():
            job = asyncio.ensure_future(chat_jobs.run(work(), RID, never_gone()))
            await asyncio.sleep(0.05)
            assert chat_jobs.running(RID)
            assert chat_jobs.cancel(RID) is True
            with pytest.raises(HTTPException) as e:
                await job
            return e.value
        err = asyncio.run(main())
        assert err.status_code == 409 and "cancelled" in err.detail
        assert stopped == [True] and not chat_jobs.running(RID)

    def test_a_cancel_before_the_request_refuses_it_on_arrival(self):
        ran = []

        async def work():
            ran.append(True)
        rid = "early-0123456789abcdef"
        assert chat_jobs.cancel(rid) is False                 # not running yet: remembered
        with pytest.raises(HTTPException) as e:
            asyncio.run(chat_jobs.run(work(), rid, never_gone()))
        assert e.value.status_code == 409 and ran == []
        # remembered once: the same id may be used again afterwards
        asyncio.run(chat_jobs.run(work(), rid, never_gone()))
        assert ran == [True]

    def test_a_client_that_went_away_stops_the_work(self):
        stopped = []

        async def work():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                stopped.append(True)
                raise
        gone = {"now": False}

        async def is_disconnected():
            return gone["now"]

        async def main():
            job = asyncio.ensure_future(chat_jobs.run(work(), None, is_disconnected, poll_s=0.02))
            await asyncio.sleep(0.05)
            gone["now"] = True
            with pytest.raises(HTTPException) as e:
                await job
            return e.value
        t0 = time.time()
        err = asyncio.run(main())
        assert err.status_code == 499 and stopped == [True] and time.time() - t0 < 2

    def test_ids_outside_the_rule_are_not_registered(self):
        async def work():
            return chat_jobs.running("short")
        assert asyncio.run(chat_jobs.run(work(), "short", never_gone())) is False
        assert chat_jobs.valid_id(RID) and not chat_jobs.valid_id("short") and not chat_jobs.valid_id("a b" * 10)


# --------------------------------------------------------------------------- through the gateway

def slow_upstream():
    """An upstream that answers after 20 s unless the gateway hangs up first (then it notes the closed connection)."""
    seen = {"closed": threading.Event(), "got": threading.Event()}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen["got"].set()
            self.connection.settimeout(0.1)
            for _ in range(200):
                try:
                    if self.connection.recv(1) == b"":      # peer closed
                        seen["closed"].set()
                        return
                except OSError:
                    pass
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "late"},
                                            "finish_reason": "stop"}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def slow(monkeypatch):
    srv, seen = slow_upstream()

    async def fake_resolve(model_name, **kw):
        return Upstream(kind="node", base=f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setattr(gateway, "_resolve_upstream", fake_resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
    yield seen
    srv.shutdown()


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Hi"}]}),
    ("/api/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Hi"}]}),
])
def test_cancel_by_request_id_ends_the_chat_and_hangs_up_upstream(slow, path, body):
    c = TestClient(app, client=("127.0.0.1", 50000))   # each request on its own loop: cancel crosses threads
    out = {}

    def call():
        out["r"] = c.post(path, json=body, headers={"X-Request-Id": RID})
    t = threading.Thread(target=call)
    t0 = time.time()
    t.start()
    assert slow["got"].wait(5)                                  # the chat is generating upstream
    assert chat_jobs.running(RID)
    r = c.post("/v1/chat/cancel", json={"request_id": RID})
    assert r.status_code == 200 and r.json() == {"cancelled": True}
    t.join(10)
    assert out["r"].status_code == 409 and time.time() - t0 < 8
    assert slow["closed"].wait(3)                              # llama-server stops on a closed connection


def test_cancel_checks_the_id():
    c = TestClient(app, client=("127.0.0.1", 50000))
    assert c.post("/v1/chat/cancel", json={"request_id": "x"}).status_code == 400
    assert c.post("/v1/chat/cancel", json={"request_id": "nothing-running-0123"}).json() == {"cancelled": False}
    chat_jobs.forget("nothing-running-0123")
