"""The graphical setup: a small server on 127.0.0.1 with a setup page in the browser (owner, 2026-11-05). Started by the
double-click packages (packaging/) or by hand: `python installer/wizard.py`. Python standard library only -- it
runs before ai-serve's own environment exists and sets that up as its first step.

The page shows what this machine has and the suggested offers (installer/offers.py), takes a choice, runs the same
steps as `./ai-serve setup` (installer/steps.py) with live progress, then starts ai-serve and hands over to the
dashboard. Who may call it: installer/wizard_rules.py (host, origin, token, JSON only)."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from installer import cli  # noqa: E402
from installer import offers as O  # noqa: E402
from installer import steps as S  # noqa: E402
from installer import wizard_rules as W  # noqa: E402

SETUP_DONE = HERE / ".setup-done"   # written after a successful run
PAGE_DIR = Path(__file__).resolve().parent / "wizard_page"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


class PageReport(S.Report):
    """Keeps what the steps report for the page: the current step, the last lines, one progress bar."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lines = deque(maxlen=300)
        self.current = ""
        self.offer = ""
        self.bar = None
        self.finished = []

    def step(self, label):
        with self.lock:
            if self.current:
                self.finished.append(self.current)
            self.current, self.bar = label, None
            self.lines.append(f"== {label}")

    def log(self, text):
        with self.lock:
            self.lines.append(text)

    def progress(self, label, done, total):
        with self.lock:
            self.bar = {"label": label, "done": done, "total": total}

    def snapshot(self):
        with self.lock:
            return {"step": self.current, "offer": self.offer, "finished": list(self.finished),
                    "bar": self.bar, "log": list(self.lines)[-80:]}


class Wizard:
    def __init__(self, skip_core: bool = False):
        self.token = W.make_token()
        self.machine = cli.machine()
        self.job = W.Job()
        self.report = PageReport()
        self.skip_core = skip_core
        self.closing = threading.Event()

    def state(self) -> dict:
        m = self.machine
        rec = {oid: (yes, why) for oid, yes, why in O.recommend(m)}
        rows = []
        for o in O.offers(m):
            ok, reason = W.available(o, m)
            yes, why = rec[o.id]
            rows.append({"id": o.id, "label": o.label, "what": o.what, "size_gb": o.size_gb, "license": o.license,
                         "suggested": yes and ok, "why": why if ok else reason, "available": ok})
        return {"machine": O.describe(m), "disk_free_gb": round(m.disk_free_gb, 1),
                "reserve_gb": O.DISK_RESERVE_GB, "offers": rows, "groups": O.exclusive_groups(),
                "job": {"state": self.job.state, "error": self.job.error, **self.report.snapshot()},
                "gateway": {"running": cli.running(), "url": f"http://localhost:{cli.gateway_port()}/"}}

    def install(self, ids):
        picked = [o for o in O.offers(self.machine) if o.id in ids]
        r = self.report
        try:
            if not self.skip_core:
                S.core_environment(r)
            S.install(picked, r, before_each=lambda oid: setattr(r, "offer", oid))
            r.step("Starting ai-serve")
            r.offer = ""
            if not cli.running():
                cli.start(background=True)
            if not cli.running():
                raise RuntimeError("ai-serve didn't start -- see logs/launcher.log")
            r.step("")
            SETUP_DONE.touch()   # the packages open the dashboard instead of this page from now on
            self.job.end(ok=True)
        except BaseException as e:   # SystemExit from the download checks too
            msg = str(e) or type(e).__name__
            r.log(f"!! {msg}")
            self.job.end(ok=False, error=msg)


def handler_for(wiz: Wizard, port_box: list):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ai-serve-setup"

        def log_message(self, *a):   # no request log: the URL holds nothing secret, but keep the terminal quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, data):
            self._send(code, json.dumps(data).encode())

        def _gate(self, method) -> bool:
            path = self.path.split("?", 1)[0]
            why = W.check_request(method, path, dict(self.headers), wiz.token, port_box[0])
            if why:
                self._json(404 if why == "not found" else 403, {"error": why})
                return False
            return True

        def do_GET(self):
            if not self._gate("GET"):
                return
            path = self.path.split("?", 1)[0]
            if path in W.STATIC:
                f = PAGE_DIR / W.STATIC[path]
                self._send(200, f.read_bytes(), TYPES[f.suffix])
            else:
                self._json(200, wiz.state())

        def do_POST(self):
            if not self._gate("POST"):
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > W.MAX_BODY:
                self._json(413, {"error": "request too large"})
                return
            body = self.rfile.read(n)
            path = self.path.split("?", 1)[0]
            if path == "/api/finish":
                self._json(200, {"ok": True})
                wiz.closing.set()
                return
            try:
                ids = W.parse_choice(body, wiz.machine)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            if not wiz.job.begin():
                self._json(409, {"error": "the setup runs already" if wiz.job.state == "running" else "already done"})
                return
            threading.Thread(target=wiz.install, args=(ids,), daemon=True).start()
            self._json(202, {"ok": True})

        do_PUT = do_DELETE = do_PATCH = lambda self: self._json(405, {"error": "not allowed"})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ai-serve's graphical setup")
    ap.add_argument("--no-browser", action="store_true", help="only print the address")
    ap.add_argument("--skip-core", action="store_true", help="ai-serve's own environment exists already")
    a = ap.parse_args(argv)
    wiz = Wizard(skip_core=a.skip_core)
    port_box = [0]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(wiz, port_box))
    port_box[0] = srv.server_address[1]
    url = f"http://127.0.0.1:{port_box[0]}/#t={wiz.token}"
    print(f"ai-serve setup: {url}", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while not wiz.closing.wait(1):
            pass
        time.sleep(1)   # let the last answer reach the page
    except KeyboardInterrupt:
        pass
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
