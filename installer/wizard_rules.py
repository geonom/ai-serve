"""Rules for the graphical setup (installer/wizard.py) -- pure, no I/O (tests tests/unit/test_installer_wizard_rules.py).

Owner, 2026-11-05: a simpler graphical installer, a double-click package for Mac, Linux and Windows that opens a
setup page in the browser. That page can download and run setup scripts, so it is guarded:

- **Threat model.** The setup server listens on 127.0.0.1 only, on a random port, while the setup runs. Attackers are
  other web pages in the same browser (a form posted to 127.0.0.1, or a name re-pointed at 127.0.0.1 -- DNS
  rebinding) and other local users' programs guessing the port.
- **Host** must be exactly 127.0.0.1:<port> or localhost:<port> (a rebinding page sends its own name).
- **Origin**, when sent, must be this page.
- **Token.** Every /api/ call carries X-Setup-Token: a random token the launcher puts into the URL's fragment (never
  sent over the network, never in a log). The static page itself is no secret and needs none.
- **POST** only as application/json (a plain cross-site form can't send that) and at most 4 KB.
- **Choices** are offer ids from installer/offers.py for this machine, at most one per exclusive group; nothing
  else is accepted (no paths, URLs or commands from the page).
"""
from __future__ import annotations

import hmac
import json
import secrets
from typing import List, Mapping, Optional, Tuple

from installer import offers as O

TOKEN_HEADER = "X-Setup-Token"
MAX_BODY = 4096
STATIC = {"/": "wizard.html", "/wizard.js": "wizard.js", "/wizard.css": "wizard.css"}
API = {("GET", "/api/state"), ("POST", "/api/install"), ("POST", "/api/finish")}


def make_token() -> str:
    return secrets.token_urlsafe(32)


def _header(headers: Mapping[str, str], name: str) -> str:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v or ""
    return ""


def check_request(method: str, path: str, headers: Mapping[str, str], token: str, port: int) -> Optional[str]:
    """None when the request may be served, else why not."""
    ours = {f"127.0.0.1:{port}", f"localhost:{port}"}
    if _header(headers, "Host") not in ours:
        return "wrong host"
    origin = _header(headers, "Origin")
    if origin and origin not in {f"http://{h}" for h in ours}:
        return "foreign origin"
    if method == "GET" and path in STATIC:
        return None
    if (method, path) not in API:
        return "not found"
    if not hmac.compare_digest(_header(headers, TOKEN_HEADER).encode(), token.encode()):
        return "no or wrong token"
    if method == "POST" and _header(headers, "Content-Type").split(";")[0].strip().lower() != "application/json":
        return "only JSON"
    return None


def available(offer: O.Offer, m: O.Machine) -> Tuple[bool, str]:
    """Can this offer be set up here? On Windows not yet when its setup is a Bash script (the plugins' setup.sh)."""
    if m.system == "Windows" and offer.setup and offer.setup[0].endswith(".sh"):
        return False, "not on Windows yet"
    return True, ""


def parse_choice(body: bytes, m: O.Machine) -> List[str]:
    """The page's {"offers": [ids]} -> validated ids in offer order; ValueError on anything else."""
    if len(body) > MAX_BODY:
        raise ValueError("request too large")
    try:
        data = json.loads(body or b"null")
    except ValueError:
        raise ValueError("not JSON")
    if not isinstance(data, dict) or set(data) != {"offers"} or not isinstance(data["offers"], list) \
            or not all(isinstance(x, str) for x in data["offers"]):
        raise ValueError('expected {"offers": [offer ids]}')
    ids = O.validate_choice(data["offers"], m)
    by_id = {o.id: o for o in O.offers(m)}
    blocked = [i for i in ids if not available(by_id[i], m)[0]]
    if blocked:
        raise ValueError(f"not available on this machine: {', '.join(blocked)}")
    return ids


class Job:
    """One setup run at a time: idle -> running -> done | failed (a failed run may be started again)."""

    def __init__(self):
        self.state = "idle"
        self.error = ""

    def begin(self) -> bool:
        if self.state not in ("idle", "failed"):
            return False
        self.state, self.error = "running", ""
        return True

    def end(self, ok: bool, error: str = "") -> None:
        self.state, self.error = ("done", "") if ok else ("failed", error)
