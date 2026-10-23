"""Spielwiese proxy rules (spielwiese/policy.py) -- pure, no I/O.

Threat model: the Spielwiese server listens on 127.0.0.1 and forwards the browser's
calls to the ai-serve-ui gateway on 127.0.0.1:8030. The gateway trusts loopback, so
a forwarding server could hand anyone who reaches the Spielwiese (e.g. a tailnet user,
once the dashboard exposes it) the gateway's full management API. Rules:
- only the endpoints a playground needs are forwarded (chat, images, speech, OCR and
  the read-only lists they use) -- never start/stop, services, cluster, training, keys;
- the real client is passed on (X-Forwarded-For = the peer we saw; the gateway honours
  it only from 127.0.0.1), so the gateway's own access policy still decides;
- client-supplied forwarding/identity headers are dropped, Tailscale identity headers
  are passed on only when the request came over loopback (Tailscale Serve sets them
  there and strips them from clients);
- paths with traversal or encoded separators are refused; bodies are size-capped.
"""
import unittest

from spielwiese.policy import MAX_BODY_BYTES, Refused, forward_headers, route


class TestRoute(unittest.TestCase):
    def ok(self, method, path):
        self.assertEqual(route(method, path), path, f"{method} {path} should be forwarded")

    def refused(self, method, path):
        with self.assertRaises(Refused, msg=f"{method} {path} must be refused"):
            route(method, path)

    def test_playground_endpoints(self):
        for m, p in [("POST", "/api/chat/completions"), ("POST", "/v1/chat/completions"),
                     ("POST", "/api/tokenize"), ("GET", "/api/models"), ("GET", "/api/voices"),
                     ("GET", "/api/status"), ("GET", "/api/comfyui/loras"),
                     ("POST", "/api/comfyui/generate"), ("POST", "/api/comfyui/edit"),
                     ("POST", "/api/comfyui/style-transfer"), ("GET", "/api/comfyui/task/abc-123"),
                     ("POST", "/api/comfyui/batch"), ("GET", "/api/comfyui/batch/status"),
                     ("POST", "/api/comfyui/batch/pause"), ("POST", "/api/comfyui/batch/cancel"),
                     ("GET", "/api/comfyui/images/ComfyUI_00012_.png"),
                     ("POST", "/api/tts/generate"), ("POST", "/api/stt/transcribe"), ("POST", "/api/ocr"),
                     ("POST", "/v1/images/generations"),    # MOSS-Image, Iris, catalog ids
                     ("GET", "/v1/images/models"), ("POST", "/v1/images/cancel"),
                     ("GET", "/v1/audio/models"), ("GET", "/v1/llm/models"),
                     ("POST", "/v1/embeddings"), ("POST", "/v1/rerank"),        # the gateway's image and speech catalogs
                     ("POST", "/api/plugins/comfyui/start")]:   # a plugin whose downloads a page needs (owner only)
            self.ok(m, p)

    def test_only_plugins_are_started_never_apps_or_services(self):
        for m, p in [("POST", "/api/services/comfyui/start"), ("POST", "/api/services/spielwiese/stop"),
                     ("POST", "/api/plugins/comfyui/release"), ("POST", "/api/plugins/../services/x/start")]:
            self.refused(m, p)

    def test_query_string_kept_for_reads(self):
        self.assertEqual(route("GET", "/api/comfyui/task/abc?x=1"), "/api/comfyui/task/abc?x=1")

    def test_management_is_never_forwarded(self):
        for m, p in [("POST", "/api/start"), ("POST", "/api/stop"), ("POST", "/api/models/unload_all"),
                     ("POST", "/api/services/storytime/start"), ("GET", "/api/services"),
                     ("POST", "/api/cluster/enrollments"), ("GET", "/api/cluster/agent.py"),
                     ("POST", "/api/training/jobs"), ("DELETE", "/api/training-monitor/jobs/x"),
                     ("POST", "/api/gpu/consumers/llm:1/unload"), ("POST", "/api/strata/start"),
                     ("POST", "/api/comfyui/stop"), ("POST", "/api/comfyui/start"),
                     ("PUT", "/api/dashboard/layout"), ("GET", "/docs"), ("GET", "/openapi.json"),
                     ("GET", "/public/v1/models"), ("POST", "/api/coordination/gpu-preflight")]:
            self.refused(m, p)

    def test_wrong_method_refused(self):
        self.refused("GET", "/api/chat/completions")
        self.refused("DELETE", "/api/models")
        self.refused("POST", "/api/status")

    def test_traversal_and_encoding_refused(self):
        for p in ["/api/comfyui/task/../../start", "/api/comfyui/images/..%2f..%2fx", "/api/comfyui/task/a%2Fb",
                  "/api/comfyui/images/a/b.png", "//api/models", "/api/models/", "/api/comfyui/task/",
                  "/api/comfyui/batch/explode", "/api/models\\x", "api/models"]:
            self.refused("GET" if "task" in p or "images" in p or "models" in p else "POST", p)


class TestHeaders(unittest.TestCase):
    def test_real_client_passed_on_and_spoofing_dropped(self):
        h = forward_headers({"Content-Type": "application/json", "X-Forwarded-For": "100.64.0.9",
                             "Forwarded": "for=1.2.3.4", "X-Real-IP": "1.2.3.4", "Host": "evil",
                             "Connection": "keep-alive", "Cookie": "a=b", "Accept": "text/event-stream"},
                            peer="127.0.0.1")
        self.assertEqual(h["X-Forwarded-For"], "127.0.0.1")
        self.assertEqual(h["Content-Type"], "application/json")
        self.assertEqual(h["Accept"], "text/event-stream")
        for k in ("Forwarded", "X-Real-IP", "Host", "Connection", "Cookie"):
            self.assertNotIn(k, h)

    def test_tailscale_identity_only_over_loopback(self):
        ident = {"Tailscale-User-Login": "owner@github", "Tailscale-User-Name": "Owner"}
        self.assertEqual(forward_headers(ident, peer="127.0.0.1")["Tailscale-User-Login"], "owner@github")
        self.assertNotIn("Tailscale-User-Login", forward_headers(ident, peer="192.168.1.40"))
        self.assertNotIn("Tailscale-User-Login", forward_headers(ident, peer="100.71.40.2"))

    def test_header_names_case_insensitive(self):
        h = forward_headers({"x-forwarded-for": "1.2.3.4", "tailscale-user-login": "x"}, peer="10.0.0.2")
        self.assertEqual(h["X-Forwarded-For"], "10.0.0.2")
        self.assertNotIn("tailscale-user-login", {k.lower() for k in h})
        self.assertEqual([k for k in h if k.lower() == "x-forwarded-for"], ["X-Forwarded-For"])

    def test_body_cap_fits_stt_uploads(self):
        self.assertGreaterEqual(MAX_BODY_BYTES, 140 * 1024 * 1024)  # 100 MB audio as base64


if __name__ == "__main__":
    unittest.main()


def test_origin_and_referer_are_not_forwarded():
    """The gateway refuses requests from foreign pages (Origin check against DNS rebinding, 2026-10-23). The
    Spielwiese calls it server-side on the person's behalf, so the browser's Origin/Referer (e.g. the door's public
    name) must not travel along – the Spielwiese checks the person itself."""
    from spielwiese.policy import forward_headers
    out = forward_headers({"Origin": "https://spielwiese.apps.example.org", "Referer": "https://spielwiese.apps.example.org/x",
                           "Content-Type": "application/json"}, peer="127.0.0.1")
    assert "Origin" not in out and "Referer" not in out and out["Content-Type"] == "application/json"
