"""httpx clients share one SSL context (2026-11-27: every new httpx.AsyncClient loaded the CA bundle, ~16 ms of CPU;
the dashboards' polling made dozens a second and kept the gateway's event loop busy)."""
import pathlib
import re
import ssl
import time

import httpx

from backend import httpclient


def test_clients_share_one_verifying_ssl_context():
    ctx = httpclient.ssl_context()
    assert ctx is httpclient.ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname   # still checks certificates


def test_a_client_is_cheap_to_make():
    httpclient.ssl_context()
    t = time.perf_counter()
    for _ in range(50):
        httpclient.AsyncClient(timeout=2.0)
    assert (time.perf_counter() - t) / 50 < 0.003


def test_it_is_an_httpx_client_and_keeps_every_option():
    transport = httpx.MockTransport(lambda r: httpx.Response(204))
    c = httpclient.AsyncClient(timeout=3.0, transport=transport, follow_redirects=True)
    assert isinstance(c, httpx.AsyncClient) and c.follow_redirects and c.timeout.read == 3.0
    httpclient.AsyncClient(verify=False)                 # an explicit choice still wins


def test_nothing_in_the_backend_makes_its_own():
    root = pathlib.Path(__file__).resolve().parents[2] / "backend"
    offenders = [f"{p.relative_to(root)}:{i}" for p in root.rglob("*.py") if p.name != "httpclient.py"
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if re.search(r"httpx\.(Async)?Client\(", line)]
    assert offenders == []
