"""httpx clients with one shared SSL context (2026-11-27). A new httpx.AsyncClient loads the CA bundle each time
(~16 ms of CPU); the dashboards poll the gateway dozens of times a second, and every poll made several -- the event
loop was busy most of the time. The shared context verifies certificates exactly like httpx's default (certifi's
bundle, hostname checks). Use `httpclient.AsyncClient(...)` wherever `httpx.AsyncClient(...)` would go; tests
(tests/unit/test_httpclient.py) refuse direct ones in backend/."""
from __future__ import annotations

import ssl
import threading
from typing import Any, Optional

import certifi
import httpx

_CTX: Optional[ssl.SSLContext] = None
_LOCK = threading.Lock()


def ssl_context() -> ssl.SSLContext:
    global _CTX
    if _CTX is None:
        with _LOCK:
            if _CTX is None:
                _CTX = ssl.create_default_context(cafile=certifi.where())
    return _CTX


def AsyncClient(**kw: Any) -> httpx.AsyncClient:   # noqa: N802 -- stands in for httpx.AsyncClient
    kw.setdefault("verify", ssl_context())
    return httpx.AsyncClient(**kw)
