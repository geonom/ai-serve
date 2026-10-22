"""The real server, per person: nobody gets anything (incl. the gateway proxy), galleries and comparisons
are each person's own, ids from requests never reach the file system, comparisons use the gateway's
catalog and run as the person."""
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
CATALOG = {"default": "z-image-turbo", "data": [
    {"id": "z-image-turbo", "label": "Z-Image Turbo", "provider": "comfyui", "available": True,
     "aspect_ratios": ["1:1", "16:9"], "steps": {"default": 8, "adjustable": True, "max": 60}},
    {"id": "iris-turbo", "label": "Iris-3B Turbo", "provider": "iris", "available": True,
     "aspect_ratios": ["1:1", "16:9"], "steps": {"default": 4, "adjustable": False, "max": 4}}]}
GENERATED = []


class FakeGateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(CATALOG if self.path == "/v1/images/models" else {"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        GENERATED.append((body, self.headers.get("X-Forwarded-For")))
        self._json({"data": [{"b64_json": base64.b64encode(PNG).decode()}]})


def fake_identify(headers, peer, *, app_id, **kw):
    who = {k.lower(): v for k, v in headers.items()}.get("x-test-user")
    if not who:
        raise Unauthenticated("nobody")
    return Person(sub=who, name=who, email=None, picture=None, groups=(), via="door", is_owner=who == "alex")


def serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestPeople(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.gw = serve(ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway))
        cls.sw = serve(make_server(0, f"http://127.0.0.1:{cls.gw.server_address[1]}", data_dir=cls.tmp.name,
                                   identify=fake_identify))
        cls.port = cls.sw.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.sw.shutdown()
        cls.gw.shutdown()
        cls.tmp.cleanup()

    def req(self, method, path, who=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if who:
            headers["X-Test-User"] = who
        c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, data

    def save(self, who, **meta):
        return self.req("POST", "/sw/gallery", who, {"image": base64.b64encode(PNG).decode(), "meta": meta})

    def test_nobody_gets_nothing(self):
        for m, p in [("GET", "/sw/me"), ("GET", "/sw/gallery"), ("POST", "/sw/compare"), ("GET", "/api/models"),
                     ("POST", "/v1/images/generations")]:
            self.assertEqual(self.req(m, p, body={} if m == "POST" else None)[0], 401, p)
        self.assertEqual(self.req("GET", "/")[0], 200)   # the page itself is fine

    def test_me(self):
        self.assertEqual(self.req("GET", "/sw/me", "alex")[1], {"name": "alex", "is_owner": True})

    def test_gallery_is_per_person(self):
        st, e = self.save("anna", prompt="a fox", model="iris-turbo")
        self.assertEqual(st, 201)
        st, img = self.req("GET", f"/sw/gallery/{e['id']}/image", "anna")
        self.assertEqual((st, img), (200, PNG))
        self.assertEqual(self.req("GET", f"/sw/gallery/{e['id']}/image", "ben")[0], 404)
        self.assertNotIn(e["id"], [x["id"] for x in self.req("GET", "/sw/gallery", "ben")[1]["images"]])
        self.assertEqual(self.req("DELETE", f"/sw/gallery/{e['id']}", "ben")[0], 404)
        self.assertEqual(self.req("DELETE", f"/sw/gallery/{e['id']}", "anna")[0], 200)
        self.assertEqual(self.req("GET", f"/sw/gallery/{e['id']}/image", "anna")[0], 404)

    def test_bad_input(self):
        self.assertEqual(self.req("POST", "/sw/gallery", "anna", {"image": base64.b64encode(b"<svg/>").decode(),
                                                                   "meta": {}})[0], 400)
        self.assertEqual(self.save("anna", evil=1)[0], 400)
        for p in ["/sw/gallery/..%2f..%2fetc/image", "/sw/gallery/../x/image", "/sw/compare/../../x"]:
            self.assertEqual(self.req("GET", p, "anna")[0], 404, p)

    def test_compare_runs_as_the_person_into_their_gallery(self):
        GENERATED.clear()
        st, job = self.req("POST", "/sw/compare", "anna", {"prompt": "a fox", "models": ["z-image-turbo", "iris-turbo"],
                                                           "steps": [4, 8], "seeds": [7], "aspect_ratio": "16:9"})
        self.assertEqual(st, 202, job)
        for _ in range(100):
            st, state = self.req("GET", f"/sw/compare/{job['id']}", "anna")
            if state["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(state["status"], "done")
        self.assertEqual([(c["model"], c["steps"]) for c in state["cells"]],
                         [("z-image-turbo", 4), ("z-image-turbo", 8), ("iris-turbo", 4)])
        self.assertTrue(all(c["image"] for c in state["cells"]))
        self.assertEqual({peer for _, peer in GENERATED}, {"127.0.0.1"})
        self.assertEqual(self.req("GET", f"/sw/compare/{job['id']}", "ben")[0], 404)
        self.assertEqual([j["id"] for j in self.req("GET", "/sw/compare", "anna")[1]["comparisons"]][0], job["id"])
        img = state["cells"][0]["image"]
        self.assertEqual(self.req("GET", f"/sw/gallery/{img}/image", "anna")[0], 200)

    def test_compare_refusals(self):
        st, body = self.req("POST", "/sw/compare", "anna", {"prompt": "x", "models": ["nope"]})
        self.assertEqual(st, 400)
        self.assertIn("nope", body["detail"])
