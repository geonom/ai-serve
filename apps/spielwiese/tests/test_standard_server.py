"""The real Spielwiese server in front of a *standard* OpenAI-compatible server (Ollama, LM Studio, llama-server):
chat and embeddings go through, the model list is rebuilt from /v1/models, the ai-serve-only catalogs answer
"unavailable", /sw/backend says what it is talking to, and an upstream key (SPIELWIESE_UPSTREAM_KEY) goes along."""
import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spielwiese.server import make_server

SEEN = []


class StandardServer(BaseHTTPRequestHandler):
    """Only what every OpenAI-compatible server has."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        SEEN.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": "llama3.2:3b"}, {"id": "nomic-embed-text:latest"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(("POST", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/chat/completions":
            return self._json(200, {"model": body.get("model"), "choices": [
                {"message": {"role": "assistant", "content": "Sofia"}, "finish_reason": "stop"}]})
        self._json(404, {"error": "not found"})


class TestStandardServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), StandardServer)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.sw = make_server(0, f"http://127.0.0.1:{cls.up.server_address[1]}", data_dir=cls.tmp.name,
                             upstream_key="sk-local-test")
        threading.Thread(target=cls.sw.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.sw.shutdown()
        cls.up.shutdown()

    def req(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.sw.server_address[1], timeout=5)
        c.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type": "application/json"} if body is not None else {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data) if data else None

    def test_backend_kind(self):
        status, data = self.req("GET", "/sw/backend")
        self.assertEqual((status, data["kind"]), (200, "openai"))
        self.assertEqual(data["features"], {"chat": True, "texts": True, "images": False, "speech": False,
                                            "documents": False})

    def test_model_list_rebuilt(self):
        status, data = self.req("GET", "/v1/llm/models")
        self.assertEqual(status, 200)
        self.assertEqual({m["id"]: m["kind"] for m in data["models"]},
                         {"llama3.2:3b": "chat", "nomic-embed-text:latest": "embedding"})

    def test_ai_serve_catalogs_are_unavailable(self):
        self.assertFalse(self.req("GET", "/v1/audio/models")[1]["tts"]["available"])
        self.assertEqual(self.req("GET", "/v1/images/models?task=all")[1]["data"], [])

    def test_chat_goes_through_with_the_upstream_key(self):
        status, data = self.req("POST", "/v1/chat/completions",
                                {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "?"}]})
        self.assertEqual((status, data["choices"][0]["message"]["content"]), (200, "Sofia"))
        self.assertIn(("POST", "/v1/chat/completions", "Bearer sk-local-test"), SEEN)


if __name__ == "__main__":
    unittest.main()
