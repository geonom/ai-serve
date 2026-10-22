"""stt-serve asks ai-serve-ui's GPU manager before loading: a granted preflight
is a reservation, confirmed right away with this process's PID (models load here or
in a child), released when an in-process load is done."""
import os

import httpx

from stt_serve import coordination


def test_reserve_confirm_release(monkeypatch):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(("POST", url.split("/api/")[1], json))
        if url.endswith("gpu-preflight"):
            return httpx.Response(200, json={"free_mb": 9000, "granted": True, "reservation_id": "r5"},
                                  request=httpx.Request("POST", url))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    def delete(url, timeout=None):
        calls.append(("DELETE", url.split("/api/")[1], None))
        return httpx.Response(200, json={}, request=httpx.Request("DELETE", url))
    monkeypatch.setattr(coordination.httpx, "post", post)
    monkeypatch.setattr(coordination.httpx, "delete", delete)
    rid = coordination.preflight_sync("whisperx")
    coordination.release_sync(rid)
    assert rid == "r5"
    assert calls[1] == ("POST", "coordination/reservations/r5/confirm", {"pid": os.getpid()})
    assert calls[2] == ("DELETE", "coordination/reservations/r5", None)


def test_old_gateway_without_reservations(monkeypatch):
    monkeypatch.setattr(coordination.httpx, "post", lambda url, json=None, timeout=None: httpx.Response(
        200, json={"free_mb": 9000}, request=httpx.Request("POST", url)))
    assert coordination.preflight_sync("whisperx") is None
    coordination.release_sync(None)   # no-op
