"""Spielwiese: try the gateway's models -- chat, images, speech, documents -- and compare image models.

Serves the page from static/ and forwards the calls it makes to the ai-serve-ui
gateway (rules in policy.py). Listens on 127.0.0.1; who else may reach it is chosen in the
ai-serve-ui dashboard (Tailscale levels, the door), never here. Every call except the page
itself needs a person (llama_serve_identity: the door's verified token, or the owner on
loopback) -- nobody gets 401. Each person's images and comparisons: gallery.py, compare.py,
under /sw/. Stdlib + PyJWT (for the identity check).

    python3 -m spielwiese.server [--port 8037] [--gateway http://127.0.0.1:8030]
"""
from __future__ import annotations

import argparse
import http.client
import json
import logging
import mimetypes
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socketserver
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

import time

from spielwiese import backends
from spielwiese import compare as compare_mod
from spielwiese import quota
from spielwiese.gallery import Gallery, GalleryError, decode_image
from spielwiese.llama_serve_identity import Unauthenticated, identify as sdk_identify
from spielwiese.policy import MAX_BODY_BYTES, Refused, forward_headers, route


class QuickHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the socket.getfqdn() in server_bind: on a Mac whose name resolution hangs it took
    ~35 s before the server accepted connections (found in CI on macOS, 2026-10-22). server_name isn't used here."""
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

STATIC_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "static"))
DATA_DIR = os.environ.get("SPIELWIESE_DATA") or os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "data"))
APP_ID = "spielwiese"
MAX_SW_BODY = 40 * 1024 * 1024   # one image as base64 + metadata
_ID = r"([0-9a-f]{25})"
COMFY_ASYNC_ESTIMATE_S = 20   # ComfyUI generate/edit return at once; the GPU time comes later -- charged per image
UPSTREAM_TIMEOUT = 1800  # Strata and big image jobs are slow
_HOP = frozenset({"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
                  "proxy-connection", "content-length", "server", "date"})
log = logging.getLogger("spielwiese")


class Handler(BaseHTTPRequestHandler):
    server_version = "Spielwiese/1"
    protocol_version = "HTTP/1.0"   # close-delimited bodies: streams pass straight through
    gateway = ("127.0.0.1", 8030)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, detail: str) -> None:
        import json
        self._send(status, json.dumps({"detail": detail}).encode())

    # ------------------------------------------------------------ static
    def _static(self) -> None:
        path = urlsplit(self.path).path
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC_DIR, name))
        if not full.startswith(STATIC_DIR + os.sep) or not os.path.isfile(full):
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    # ------------------------------------------------------------ proxy
    def _proxy(self, person) -> None:
        try:
            target = route(self.command, self.path)
        except Refused as e:
            return self._error(e.status, str(e))
        if self._catalog_without_ai_serve(urlsplit(self.path).path):
            return
        length = int(self.headers.get("Content-Length") or 0)
        if self.headers.get("Transfer-Encoding"):
            return self._error(411, "send a Content-Length")
        if length > MAX_BODY_BYTES:
            return self._error(413, "request too large")
        body = self.rfile.read(length) if length else None
        kind = quota.classify(self.command, urlsplit(self.path).path)
        if kind == "owner_only" and not person.is_owner:
            return self._error(403, "Only the owner can do this (batch runs, downloads).")
        ticket, images = None, 0
        if kind and kind != "owner_only":
            images = self._images_requested(body) if kind == "image" else 0
            try:
                ticket = self.ledger.begin(person.sub, self._limits(person), kind, images=images)
            except quota.QuotaExceeded as e:
                return self._error(429, str(e))
        started, status = time.time(), 0
        try:
            status = self._forward(target, body)
        finally:
            if ticket is not None:
                ok = 200 <= status < 300
                seconds = time.time() - started
                if ok and kind == "image" and urlsplit(self.path).path.startswith("/api/comfyui/"):
                    seconds += COMFY_ASYNC_ESTIMATE_S * images
                self.ledger.end(ticket, seconds=seconds, images=images if ok else 0)

    def _images_requested(self, body: Optional[bytes]) -> int:
        """Images a call asks for -- at least 1; an unreadable n counts as 1, a big one as asked."""
        if not urlsplit(self.path).path.startswith("/v1/images/"):
            return 1
        try:
            n = json.loads(body or b"{}").get("n", 1)
        except (ValueError, AttributeError):
            return 1
        return n if type(n) is int and 1 <= n <= 100 else 1

    upstream_key: str = ""          # Bearer key for a standard server that wants one (LM Studio, a remote vLLM ...)
    _backend: dict = {}             # {"kind", "at"} -- what the upstream is, cached briefly

    def _upstream_headers(self, headers: dict) -> dict:
        if self.upstream_key:
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {self.upstream_key}"
        return headers

    def _upstream_get(self, path: str, timeout: float = 5.0):
        """(status, parsed JSON or None) of one GET to the upstream; (0, None) when it doesn't answer."""
        conn = http.client.HTTPConnection(*self.gateway, timeout=timeout)
        try:
            conn.request("GET", path, headers=self._upstream_headers({"X-Forwarded-For": self.client_address[0]}))
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, None
        except OSError:
            return 0, None
        finally:
            conn.close()

    def _backend_kind(self) -> str:
        """ai-serve (has its catalogs) | openai (a standard server: /v1/models only) | down. Cached 30 s."""
        cached = type(self)._backend
        if cached and time.time() - cached["at"] < 30:
            return cached["kind"]
        status, _ = self._upstream_get("/v1/llm/models")
        if status == 200:
            kind = "ai-serve"
        else:
            status, _ = self._upstream_get("/v1/models")
            kind = "openai" if status == 200 else "down"
        type(self)._backend = {"kind": kind, "at": time.time()}
        return kind

    def _catalog_without_ai_serve(self, path: str) -> bool:
        """Answer an ai-serve catalog from the standard API when the upstream isn't ai-serve; False = forward."""
        if self.command != "GET" or path not in ("/v1/llm/models", "/v1/audio/models", "/v1/images/models"):
            return False
        if self._backend_kind() != "openai":
            return False
        if path == "/v1/llm/models":
            status, data = self._upstream_get("/v1/models")
            self._json_out(200 if status == 200 else 502, backends.llm_catalog(data))
        else:
            self._json_out(200, backends.EMPTY[path])
        return True

    def _forward(self, target: str, body: Optional[bytes]) -> int:
        headers = self._upstream_headers(forward_headers(dict(self.headers.items()), peer=self.client_address[0]))
        if body is not None:
            headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection(*self.gateway, timeout=UPSTREAM_TIMEOUT)
        try:
            conn.request(self.command, target, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            conn.close()
            self._error(502, f"ai-serve-ui gateway not reachable: {e}")
            return 502
        try:
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    self.send_header(k, v)
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away (e.g. stopped a stream)
        finally:
            conn.close()
        return resp.status

    # ------------------------------------------------------------ person
    identify: Callable[..., Any] = staticmethod(sdk_identify)
    gallery: Gallery
    runner: compare_mod.Runner

    def _person(self):
        try:
            return type(self).identify(dict(self.headers.items()), self.client_address[0], app_id=APP_ID)
        except Unauthenticated:
            self._error(401, "please sign in")
            return None

    ledger: quota.Ledger
    quota_file: str
    people_file: str
    _people_seen: dict = {}

    def _limits(self, person) -> quota.Limits:
        return quota.load_limits(quota.read_limits_file(self.quota_file), person.sub, person.is_owner)

    def _record(self, person) -> None:
        """Remember who used the Spielwiese (name, last seen) so the owner can set their limits."""
        now = time.time()
        if now - self._people_seen.get(person.sub, 0) < 300:
            return
        self._people_seen[person.sub] = now
        try:
            with open(self.people_file) as f:
                people = json.load(f)
        except (OSError, ValueError):
            people = {}
        people[person.sub] = {"name": (person.name or person.email or "")[:120], "owner": bool(person.is_owner),
                              "last_seen": now}
        os.makedirs(os.path.dirname(self.people_file), exist_ok=True)
        tmp = self.people_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(people, f)
        os.replace(tmp, self.people_file)

    def _quota_view(self, person) -> dict:
        lim = self._limits(person)
        return {"limits": {k: getattr(lim, k) for k in ("gpu_seconds_per_day", "images_per_day", "concurrent")},
                "usage": self.ledger.usage(person.sub), "unlimited": bool(person.is_owner)}

    def _json_out(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode())

    def _json_in(self) -> Optional[Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if self.headers.get("Transfer-Encoding"):
            self._error(411, "send a Content-Length")
            return None
        if length > MAX_SW_BODY:
            self._error(413, "request too large")
            return None
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except ValueError:
            self._error(400, "invalid JSON")
            return None

    def _catalog(self) -> dict:
        conn = http.client.HTTPConnection(*self.gateway, timeout=10)
        try:
            conn.request("GET", "/v1/images/models", headers={"X-Forwarded-For": self.client_address[0]})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            if resp.status != 200:
                raise compare_mod.CompareError("the gateway's image model list is unavailable")
            return {m["id"]: m for m in data.get("data", [])}
        except (OSError, ValueError) as e:
            raise compare_mod.CompareError(f"the gateway's image model list is unavailable ({e})")
        finally:
            conn.close()

    def _sw(self, person) -> None:
        u = urlsplit(self.path)
        path, q, sub, m = u.path, parse_qs(u.query), person.sub, self.command
        try:
            if path == "/sw/backend" and m == "GET":
                kind = self._backend_kind()
                return self._json_out(200, {"kind": kind, "features": backends.FEATURES[kind]})
            if path == "/sw/me" and m == "GET":
                return self._json_out(200, {"name": person.name or "", "is_owner": bool(person.is_owner)})
            if path == "/sw/quota" and m == "GET":
                return self._json_out(200, self._quota_view(person))
            if path == "/sw/admin/quotas":
                if not person.is_owner:
                    return self._error(403, "for the owner only")
                if m == "PUT":
                    body = self._json_in()
                    if body is None:
                        return
                    quota.write_limits_file(self.quota_file, body)
                elif m != "GET":
                    return self._error(405, "method not allowed")
                cfg = quota.read_limits_file(self.quota_file)
                try:
                    with open(self.people_file) as f:
                        seen = json.load(f)
                except (OSError, ValueError):
                    seen = {}
                people = [{"sub": sub, "name": info.get("name", ""), "owner": info.get("owner", False),
                           "last_seen": info.get("last_seen"), "usage": self.ledger.usage(sub),
                           "limits": cfg["people"].get(sub)}
                          for sub, info in sorted(seen.items(), key=lambda kv: -(kv[1].get("last_seen") or 0))]
                defaults = quota.load_limits(cfg, "", False)
                return self._json_out(200, {"defaults": {k: getattr(defaults, k) for k in
                                                         ("gpu_seconds_per_day", "images_per_day", "concurrent")},
                                            "people": people})
            if path == "/sw/gallery" and m == "GET":
                limit = int((q.get("limit") or ["60"])[0])
                return self._json_out(200, {"images": self.gallery.list(
                    sub, limit=limit, before=(q.get("before") or [None])[0],
                    comparison=(q.get("comparison") or [None])[0])})
            if path == "/sw/gallery" and m == "POST":
                body = self._json_in()
                if body is None:
                    return
                if not isinstance(body, dict) or set(body) - {"image", "meta"}:
                    raise GalleryError("expected {image, meta}")
                data, _ = decode_image(body.get("image"))
                return self._json_out(201, self.gallery.save(sub, data, body.get("meta") or {}))
            hit = re.fullmatch(rf"/sw/gallery/{_ID}/image", path)
            if hit and m in ("GET", "HEAD"):
                found = self.gallery.image(sub, hit.group(1))
                if not found:
                    return self._error(404, "not found")
                with open(found[0], "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", found[1])
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.end_headers()
                if m == "GET":
                    self.wfile.write(data)
                return
            hit = re.fullmatch(rf"/sw/gallery/{_ID}", path)
            if hit and m == "DELETE":
                if not self.gallery.delete(sub, hit.group(1)):
                    return self._error(404, "not found")
                return self._json_out(200, {"deleted": hit.group(1)})
            if path == "/sw/compare" and m == "GET":
                return self._json_out(200, {"comparisons": self.runner.list(sub)})
            if path == "/sw/compare" and m == "POST":
                body = self._json_in()
                if body is None:
                    return
                p = compare_mod.plan(body, self._catalog())
                return self._json_out(202, self.runner.start(sub, self.client_address[0], p, limits=self._limits(person)))
            hit = re.fullmatch(rf"/sw/compare/{_ID}", path)
            if hit and m == "GET":
                job = self.runner.get(sub, hit.group(1))
                return self._json_out(200, job) if job else self._error(404, "not found")
            hit = re.fullmatch(rf"/sw/compare/{_ID}/cancel", path)
            if hit and m == "POST":
                if not self.runner.cancel(sub, hit.group(1)):
                    return self._error(404, "no running comparison with this id")
                return self._json_out(200, {"cancelling": hit.group(1)})
            return self._error(404, "not found")
        except quota.QuotaExceeded as e:
            log.info("quota %s %s: %s", m, path, e)
            return self._error(429, str(e))
        except quota.QuotaError as e:
            return self._error(400, str(e))
        except (GalleryError, compare_mod.CompareError) as e:
            status = 409 if "still running" in str(e) else 400
            log.info("refused %s %s: %s", m, path, e)   # the reason, so a refusal can be traced later
            return self._error(status, str(e))
        except ValueError:
            return self._error(400, "invalid parameter")

    def _dispatch(self) -> None:
        if self.path.startswith(("/api/", "/v1/", "/sw/")):
            person = self._person()
            if person is None:
                return
            self._record(person)
            return self._sw(person) if self.path.startswith("/sw/") else self._proxy(person)
        if self.command in ("GET", "HEAD"):
            return self._static()
        return self._error(405, "method not allowed")

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _dispatch


def make_server(port: int, gateway: str = "http://127.0.0.1:8030", host: str = "127.0.0.1",
                data_dir: Optional[str] = None, identify: Optional[Callable[..., Any]] = None,
                upstream_key: str = "") -> ThreadingHTTPServer:
    u = urlsplit(gateway)
    gw = (u.hostname or "127.0.0.1", u.port or 80)
    root = data_dir or DATA_DIR
    gallery = Gallery(root)
    ledger = quota.Ledger(root)
    attrs = {"gateway": gw, "gallery": gallery, "ledger": ledger, "_people_seen": {}, "_backend": {},
             "upstream_key": upstream_key,
             "quota_file": os.path.join(root, "quotas.json"), "people_file": os.path.join(root, "people.json"),
             "runner": compare_mod.Runner(root, gallery, compare_mod.gateway_post(gw), ledger,
                                          cancel_remote=compare_mod.gateway_cancel(gw))}
    if identify is not None:
        attrs["identify"] = staticmethod(identify)
    handler = type("BoundHandler", (Handler,), attrs)
    srv = QuickHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=int(os.environ.get("SPIELWIESE_PORT", "8037")))
    p.add_argument("--gateway", default=os.environ.get("LLAMA_SERVE_URL", os.environ.get("AI_SERVE_URL") or "http://127.0.0.1:8030"),
                   help="ai-serve, or any OpenAI-compatible server: Ollama http://127.0.0.1:11434, "
                        "LM Studio http://127.0.0.1:1234")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    srv = make_server(a.port, a.gateway, upstream_key=os.environ.get("SPIELWIESE_UPSTREAM_KEY", ""))
    log.info("Spielwiese on http://127.0.0.1:%d -> %s", a.port, a.gateway)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
