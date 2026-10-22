"""The owner gate (backend/services/gate.py) as a real loopback proxy: only requests
with the owner's Tailscale identity reach the upstream app, and they reach it
unchanged (method, path, query, body, streaming)."""
import asyncio
import http.server
import socket
import threading

import aiohttp
import pytest

from backend.services.exposure import GateSpec
from backend.services.gate import GateManager

OWNER = "owner@github"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Upstream(http.server.BaseHTTPRequestHandler):
    def _reply(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        out = f"{self.command} {self.path} login={self.headers.get('tailscale-user-login')} body={body.decode()}".encode()
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _reply

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    port = _free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Upstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()


async def _call(gate_port, method="GET", path="/x?a=1", headers=None, data=None):
    async with aiohttp.ClientSession() as s:
        async with s.request(method, f"http://127.0.0.1:{gate_port}{path}", headers=headers or {}, data=data) as r:
            return r.status, await r.text()


def test_gate_passes_owner_and_blocks_others(upstream):
    async def scenario():
        gates = GateManager()
        gate_port = _free_port()
        await gates.ensure(GateSpec(listen_port=gate_port, upstream_port=upstream, owner=OWNER))
        try:
            ok = await _call(gate_port, headers={"Tailscale-User-Login": OWNER})
            post = await _call(gate_port, "POST", "/api/save", {"Tailscale-User-Login": OWNER}, data=b"hello")
            other = await _call(gate_port, headers={"Tailscale-User-Login": "someone@else"})
            anon = await _call(gate_port)
            funnel = await _call(gate_port, headers={"Tailscale-User-Login": OWNER, "Tailscale-Funnel-Request": "?1"})
            assert gates.running_ports() == {gate_port}
        finally:
            await gates.stop_all()
        return ok, post, other, anon, funnel, gates.running_ports()

    ok, post, other, anon, funnel, after = asyncio.run(scenario())
    assert ok == (200, f"GET /x?a=1 login={OWNER} body=")
    assert post == (200, f"POST /api/save login={OWNER} body=hello")
    assert other[0] == 403 and anon[0] == 403 and funnel[0] == 403
    assert "upstream" not in other[1].lower()
    assert after == set()


def test_gate_upstream_down_is_502(upstream):
    async def scenario():
        gates = GateManager()
        gate_port = _free_port()
        dead = _free_port()
        await gates.ensure(GateSpec(listen_port=gate_port, upstream_port=dead, owner=OWNER))
        try:
            return await _call(gate_port, headers={"Tailscale-User-Login": OWNER})
        finally:
            await gates.stop_all()

    assert asyncio.run(scenario())[0] == 502


def test_ensure_is_idempotent_and_replaces_changed_spec(upstream):
    async def scenario():
        gates = GateManager()
        port = _free_port()
        await gates.ensure(GateSpec(port, upstream, OWNER))
        await gates.ensure(GateSpec(port, upstream, OWNER))
        await gates.ensure(GateSpec(port, upstream, "other@user"))
        res = await _call(port, headers={"Tailscale-User-Login": OWNER})
        await gates.stop(port)
        return res, gates.running_ports()

    res, after = asyncio.run(scenario())
    assert res[0] == 403 and after == set()


def test_gate_proxies_websockets_for_the_owner_only():
    """ComfyUI's UI talks over /ws -- must pass through the gate unchanged."""
    from aiohttp import web

    async def scenario():
        async def ws_echo(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                await ws.send_str(f"echo:{msg.data}:{request.query.get('clientId')}")
            return ws

        app = web.Application()
        app.router.add_get("/ws", ws_echo)
        runner = web.AppRunner(app)
        await runner.setup()
        up_port = _free_port()
        await web.TCPSite(runner, "127.0.0.1", up_port).start()
        gates = GateManager()
        gate_port = _free_port()
        await gates.ensure(GateSpec(listen_port=gate_port, upstream_port=up_port, owner=OWNER))
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(f"http://127.0.0.1:{gate_port}/ws?clientId=abc",
                                        headers={"Tailscale-User-Login": OWNER}) as ws:
                    await ws.send_str("hi")
                    reply = (await ws.receive()).data
                denied = None
                try:
                    async with s.ws_connect(f"http://127.0.0.1:{gate_port}/ws",
                                            headers={"Tailscale-User-Login": "someone@else"}):
                        pass
                except aiohttp.WSServerHandshakeError as e:
                    denied = e.status
        finally:
            await gates.stop_all()
            await runner.cleanup()
        return reply, denied

    reply, denied = asyncio.run(scenario())
    assert reply == "echo:hi:abc" and denied == 403
