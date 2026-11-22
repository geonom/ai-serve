"""Voice conversations (2026-11-03, owner: Omni in the Playground): the browser's WebSocket to /v1/realtime goes
through to the gateway as a byte tunnel after the upgrade -- same rules as every other call (only this path, the real
client as X-Forwarded-For, no cookies/Origin), counted as speech in the person's quota."""
import socket
import threading

from playground import quota
from playground.policy import Refused, route, websocket_headers, wants_websocket
from playground.server import make_server

UPGRADE = {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
           "Sec-WebSocket-Version": "13", "Sec-WebSocket-Extensions": "permessage-deflate",
           "Cookie": "s=1", "Origin": "http://localhost:8037", "X-Forwarded-For": "6.6.6.6"}


def test_only_the_realtime_path_takes_a_websocket():
    assert route("GET", "/v1/realtime?model=LFM3-Omni-2B-Q8_0.gguf") == "/v1/realtime?model=LFM3-Omni-2B-Q8_0.gguf"
    assert wants_websocket("/v1/realtime", UPGRADE)
    assert not wants_websocket("/v1/realtime", {"Connection": "keep-alive"})
    try:
        wants_websocket("/v1/chat/completions", UPGRADE)
        raise AssertionError("an upgrade elsewhere must be refused")
    except Refused:
        pass


def test_upgrade_headers_keep_the_handshake_and_drop_the_rest():
    h = websocket_headers(UPGRADE, peer="100.64.0.7")
    assert h["Upgrade"] == "websocket" and h["Connection"] == "Upgrade"
    assert h["Sec-WebSocket-Key"] == UPGRADE["Sec-WebSocket-Key"] and h["Sec-WebSocket-Version"] == "13"
    assert h["Sec-WebSocket-Extensions"] == "permessage-deflate"
    assert h["X-Forwarded-For"] == "100.64.0.7"
    assert "Cookie" not in h and "Origin" not in h


def test_a_conversation_counts_as_speech():
    assert quota.classify("GET", "/v1/realtime") == "speech"


def fake_gateway():
    seen = {}
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        conn, _ = srv.accept()
        head = b""
        while b"\r\n\r\n" not in head:
            head += conn.recv(4096)
        seen["head"] = head.decode()
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\nhello")
        while True:
            data = conn.recv(4096)
            if not data:
                break
            conn.sendall(data.upper())
        conn.close()
    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1], seen


def test_the_tunnel_carries_bytes_both_ways(tmp_path):
    port, seen = fake_gateway()
    sw = make_server(0, f"http://127.0.0.1:{port}", data_dir=str(tmp_path))
    threading.Thread(target=sw.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(("127.0.0.1", sw.server_address[1]), timeout=5)
        req = "GET /v1/realtime?model=x.gguf HTTP/1.1\r\nHost: localhost\r\n" + \
              "".join(f"{k}: {v}\r\n" for k, v in UPGRADE.items()) + "\r\n"
        c.sendall(req.encode())
        got = b""
        while b"hello" not in got:
            chunk = c.recv(4096)
            assert chunk, got
            got += chunk
        assert got.startswith(b"HTTP/1.1 101")
        c.sendall(b"ping")
        echo = b""
        while b"PING" not in echo:
            echo += c.recv(4096)
        c.close()
        import os, time
        owner = os.environ.get("LLAMA_SERVE_OWNER_SUB") or "owner"
        ledger = sw.RequestHandlerClass.ledger
        for _ in range(50):                      # the tunnel books its time once both sides closed
            if ledger.in_flight(owner) == 0:
                break
            time.sleep(0.05)
        assert ledger.in_flight(owner) == 0      # the conversation was booked and released
        head = seen["head"]
        assert head.startswith("GET /v1/realtime?model=x.gguf HTTP/1.1\r\n")
        assert "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" in head and "X-Forwarded-For: 127.0.0.1" in head
        assert "Cookie" not in head and "Origin" not in head and "6.6.6.6" not in head
    finally:
        sw.shutdown()


def test_an_upgrade_to_another_path_never_reaches_the_gateway(tmp_path):
    port, seen = fake_gateway()
    sw = make_server(0, f"http://127.0.0.1:{port}", data_dir=str(tmp_path))
    threading.Thread(target=sw.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(("127.0.0.1", sw.server_address[1]), timeout=5)
        req = "GET /api/status HTTP/1.1\r\nHost: localhost\r\n" + \
              "".join(f"{k}: {v}\r\n" for k, v in UPGRADE.items()) + "\r\n"
        c.sendall(req.encode())
        got = c.recv(4096)
        assert got.startswith(b"HTTP/1.0 403") and "head" not in seen
        c.close()
    finally:
        sw.shutdown()


def test_speech_from_the_gateway_is_forwarded_and_counted():
    """Talk speaks short confirmations with the speech model's own voice (POST /v1/audio/speech, 2026-11-03)."""
    assert route("POST", "/v1/audio/speech") == "/v1/audio/speech"
    assert quota.classify("POST", "/v1/audio/speech") == "speech"
    try:
        route("GET", "/v1/audio/speech")
        raise AssertionError("only POST")
    except Refused:
        pass
