#!/usr/bin/env python3
"""Tiny local server for voices/casting/pool60/index.html's checkbox picker.

Serves the pool60/ directory as static files (audio, index.html) and adds
one JSON API for persisting which voices got marked "zur Auswahl":

  GET  /api/selection             -> {"selected": [slug, ...]}
  POST /api/selection {slug, selected: bool} -> toggles one slug, returns
       the updated {"selected": [...]}

State lives in pool60/selections.json (a flat list of slugs), so it survives
server restarts and is trivial to read from another script later (e.g. to
pull the chosen reference clips into a final casting set). No framework --
stdlib http.server, same minimal-dependency style as the TTS worker scripts.

Usage: python3 scripts/selection_server.py [--port 8899]
"""
import argparse
import json
import os
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

POOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "casting", "pool60")
SELECTIONS_PATH = os.path.join(POOL_DIR, "selections.json")

_lock = threading.Lock()


def _load():
    if not os.path.exists(SELECTIONS_PATH):
        return set()
    with open(SELECTIONS_PATH, encoding="utf-8") as f:
        return set(json.load(f))


def _save(selected):
    with open(SELECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(selected), f, indent=2)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=POOL_DIR, **kwargs)

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/selection":
            with _lock:
                selected = sorted(_load())
            self._send_json(200, {"selected": selected})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/selection":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
            slug = req["slug"]
            want_selected = bool(req["selected"])
        except Exception as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return
        with _lock:
            selected = _load()
            if want_selected:
                selected.add(slug)
            else:
                selected.discard(slug)
            _save(selected)
        self._send_json(200, {"selected": sorted(selected)})

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet; static-file 200s aren't interesting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Selection server on http://127.0.0.1:{args.port}/index.html", flush=True)
    print(f"Serving {POOL_DIR}", flush=True)
    print(f"Selections persisted to {SELECTIONS_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
