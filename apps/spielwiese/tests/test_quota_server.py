"""Quotas in the real server: guests are counted and stopped (429) before the gateway sees the call,
batch runs are the owner's, comparisons check the image budget up front, only the owner edits limits."""
import base64
import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spielwiese.llama_serve_identity import Person, Unauthenticated
from spielwiese.server import make_server

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
HITS = []
CATALOG = {"data": [{"id": "z", "label": "Z", "available": True, "aspect_ratios": ["1:1"],
                     "steps": {"default": 8, "adjustable": True, "max": 60}}]}


class FakeGateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(CATALOG if self.path == "/v1/images/models" else {"ok": True})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        HITS.append(self.path)
        if self.path == "/v1/images/generations":
            n = json.loads(body).get("n", 1)
            n = n if isinstance(n, int) else 1
            return self._json({"data": [{"b64_json": base64.b64encode(PNG).decode()}] * n})
        self._json({"choices": [{"message": {"content": "hi"}}]})


def fake_identify(headers, peer, *, app_id, **kw):
    who = {k.lower(): v for k, v in headers.items()}.get("x-test-user")
    if not who:
        raise Unauthenticated("nobody")
    return Person(sub=who, name=who.title(), email=None, picture=None, groups=(), via="door", is_owner=who == "alex")


def serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestQuotaServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with open(f"{self.tmp.name}/quotas.json", "w") as f:
            json.dump({"defaults": {"images_per_day": 3, "gpu_seconds_per_day": 1000, "concurrent": 1}}, f)
        self.gw = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway))
        self.sw = serve(make_server(0, f"http://127.0.0.1:{self.gw.server_address[1]}", data_dir=self.tmp.name,
                                    identify=fake_identify))
        HITS.clear()

    def tearDown(self):
        self.sw.shutdown()
        self.gw.shutdown()
        self.tmp.cleanup()

    def req(self, method, path, who, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.sw.server_address[1], timeout=10)
        c.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type": "application/json", "X-Test-User": who})
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, data

    def test_guest_images_are_counted_and_stopped_before_the_gateway(self):
        self.assertEqual(self.req("POST", "/v1/images/generations", "anna", {"prompt": "x", "n": 2})[0], 200)
        st, body = self.req("POST", "/v1/images/generations", "anna", {"prompt": "x", "n": 2})
        self.assertEqual(st, 429)
        self.assertIn("1 Bild", body["detail"])
        self.assertEqual(HITS, ["/v1/images/generations"])                 # the refused one never left
        st, q = self.req("GET", "/sw/quota", "anna")
        self.assertEqual((q["usage"]["images"], q["limits"]["images_per_day"], q["unlimited"]), (2, 3, False))

    def test_owner_is_never_stopped(self):
        for _ in range(3):
            self.assertEqual(self.req("POST", "/v1/images/generations", "alex", {"prompt": "x", "n": 2})[0], 200)
        self.assertTrue(self.req("GET", "/sw/quota", "alex")[1]["unlimited"])

    def test_bad_n_is_counted_as_at_least_one(self):
        self.req("POST", "/v1/images/generations", "anna", {"prompt": "x", "n": "lots"})
        self.assertGreaterEqual(self.req("GET", "/sw/quota", "anna")[1]["usage"]["images"], 1)

    def test_batch_is_the_owners(self):
        self.assertEqual(self.req("POST", "/api/comfyui/batch", "anna", {"items": []})[0], 403)
        self.assertEqual(self.req("POST", "/api/comfyui/batch", "alex", {"items": []})[0], 200)
        self.assertEqual(HITS, ["/api/comfyui/batch"])

    def test_chat_is_counted_in_seconds_not_images(self):
        self.assertEqual(self.req("POST", "/v1/chat/completions", "anna", {"messages": []})[0], 200)
        u = self.req("GET", "/sw/quota", "anna")[1]["usage"]
        self.assertEqual(u["images"], 0)
        self.assertGreaterEqual(u["gpu_seconds"], 0)

    def test_comparison_checks_the_image_budget_up_front(self):
        st, body = self.req("POST", "/sw/compare", "anna", {"prompt": "x", "models": ["z"], "steps": [4, 8, 12, 16]})
        self.assertEqual(st, 429, body)
        self.assertIn("Bild", body["detail"])
        st, job = self.req("POST", "/sw/compare", "anna", {"prompt": "x", "models": ["z"], "steps": [4, 8]})
        self.assertEqual(st, 202, job)
        for _ in range(100):
            job = self.req("GET", f"/sw/compare/{job['id']}", "anna")[1]
            if job["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(self.req("GET", "/sw/quota", "anna")[1]["usage"]["images"], 2)

    def test_only_the_owner_sees_and_edits_limits(self):
        self.req("GET", "/sw/me", "anna")                                   # anna is now known by name
        self.assertEqual(self.req("GET", "/sw/admin/quotas", "anna")[0], 403)
        self.assertEqual(self.req("PUT", "/sw/admin/quotas", "anna", {"defaults": {}})[0], 403)
        st, cfg = self.req("GET", "/sw/admin/quotas", "alex")
        self.assertEqual(st, 200)
        self.assertIn("anna", [p["sub"] for p in cfg["people"]])
        self.assertEqual(next(p for p in cfg["people"] if p["sub"] == "anna")["name"], "Anna")
        st, _ = self.req("PUT", "/sw/admin/quotas", "alex", {"defaults": {"images_per_day": 3}, "people": {"anna": {"images_per_day": 10}}})
        self.assertEqual(st, 200)
        self.assertEqual(self.req("GET", "/sw/quota", "anna")[1]["limits"]["images_per_day"], 10)
        self.assertEqual(self.req("PUT", "/sw/admin/quotas", "alex", {"defaults": {"images_per_day": -5}})[0], 400)
        self.assertEqual(self.req("GET", "/sw/quota", "anna")[1]["limits"]["images_per_day"], 10)   # unchanged
