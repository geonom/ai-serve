"""The real server against a fake gateway: forwards only what the policy allows, passes
the real client on, streams, serves static files without traversal."""
import http.client
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spielwiese.server import make_server

SEEN = []


class FakeGateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        SEEN.append((self.command, self.path, dict(self.headers.items())))
        if self.path == "/v1/stream":
            pass
        body = json.dumps({"ok": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        SEEN.append((self.command, self.path, dict(self.headers.items()), data))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f"data: {i}\n\n".encode())
            self.wfile.flush()


def serve(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # a request over loopback carrying Tailscale Serve's login is the owner only if it is the owner's
        os.environ["LLAMA_SERVE_OWNER_TAILSCALE_LOGIN"] = "me"
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory()   # never the real data/ (usage, people, galleries)
        cls.gw = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway))
        cls.sw = serve(make_server(0, f"http://127.0.0.1:{cls.gw.server_address[1]}", data_dir=cls.tmp.name))
        cls.port = cls.sw.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.sw.shutdown()
        cls.gw.shutdown()

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_allowed_get_forwarded_with_real_client(self):
        SEEN.clear()
        st, data = self.req("GET", "/api/models", headers={"X-Forwarded-For": "6.6.6.6", "Tailscale-User-Login": "me"})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(data)["path"], "/api/models")
        hdrs = {k.lower(): v for k, v in SEEN[-1][2].items()}
        self.assertEqual(hdrs["x-forwarded-for"], "127.0.0.1")
        self.assertEqual(hdrs["tailscale-user-login"], "me")  # came over loopback, as via Serve

    def test_management_refused_and_never_reaches_gateway(self):
        SEEN.clear()
        for m, p in [("POST", "/api/start"), ("GET", "/api/services"), ("POST", "/api/cluster/enrollments")]:
            st, _ = self.req(m, p, body=b"{}" if m == "POST" else None)
            self.assertEqual(st, 403, p)
        self.assertEqual(SEEN, [])

    def test_post_streams_through(self):
        st, data = self.req("POST", "/v1/chat/completions", body=b'{"stream": true}',
                            headers={"Content-Type": "application/json"})
        self.assertEqual(st, 200)
        self.assertEqual(data, b"data: 0\n\ndata: 1\n\ndata: 2\n\n")
        self.assertEqual(SEEN[-1][3], b'{"stream": true}')

    def test_static_and_traversal(self):
        st, data = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn(b"Spielwiese", data)
        for p in ["/../spielwiese/policy.py", "/%2e%2e/spielwiese/policy.py", "/../../etc/passwd"]:
            self.assertEqual(self.req("GET", p)[0], 404, p)

    def test_gateway_down_is_502(self):
        srv = serve(make_server(0, "http://127.0.0.1:9", data_dir=self.tmp.name))
        try:
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            c.request("GET", "/api/models")
            self.assertEqual(c.getresponse().status, 502)
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()


def test_the_server_does_not_resolve_its_own_name(monkeypatch, tmp_path):
    """http.server calls socket.getfqdn() while binding -- ~35 s on a Mac whose name resolution hangs (CI macOS):
    the Spielwiese would answer only after that."""
    import socket
    monkeypatch.setattr(socket, "getfqdn", lambda *a: (_ for _ in ()).throw(AssertionError("getfqdn called")))
    srv = make_server(0, "http://127.0.0.1:9", data_dir=str(tmp_path))
    srv.server_close()
