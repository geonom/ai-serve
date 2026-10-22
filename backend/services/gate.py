"""Owner gate: a loopback reverse proxy in front of a service exposed at level
"owner". Tailscale Serve forwards tailnet requests to it with the peer's
Tailscale-User-Login; only the owner's requests are passed on (decision in
exposure.gate_allows). Runs inside the backend's event loop; if the backend is
down, Serve's route points at a closed port -> 502, i.e. it fails closed."""
import asyncio
import logging
from typing import Dict, Optional, Set, Tuple

import aiohttp
from aiohttp import web

from backend.services.exposure import GateSpec, gate_allows

logger = logging.getLogger("ai-serve-ui.services.gate")

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "trailers", "transfer-encoding", "upgrade", "host",
})


class _Gate:
    def __init__(self, spec: GateSpec):
        self.spec = spec
        self.runner: Optional[web.AppRunner] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10), auto_decompress=False)
        app = web.Application(client_max_size=1024 ** 3)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.spec.listen_port).start()

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
        if self.session:
            await self.session.close()

    def _upstream(self, request: web.Request, scheme: str = "http") -> str:
        return f"{scheme}://127.0.0.1:{self.spec.upstream_port}{request.rel_url}"

    def _forward_headers(self, request: web.Request) -> Dict[str, str]:
        # Content-Length is kept: with it the body streams through unchunked, which
        # simple servers (http.server, some WSGI setups) require.
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        headers["X-Forwarded-Host"] = request.headers.get("Host", "")
        return headers

    async def handle(self, request: web.Request) -> web.StreamResponse:
        allowed, reason = gate_allows(request.remote, request.headers, self.spec.owner)
        if not allowed:
            logger.warning(f"gate :{self.spec.listen_port} refused {request.method} {request.rel_url}: {reason}")
            return web.Response(status=403, text="Forbidden: this service is only available to its owner.\n")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._websocket(request)
        try:
            async with self.session.request(
                    request.method, self._upstream(request), headers=self._forward_headers(request),
                    data=request.content if request.body_exists else None, allow_redirects=False) as up:
                resp = web.StreamResponse(status=up.status, reason=up.reason)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP_BY_HOP and k.lower() != "content-length":
                        resp.headers.add(k, v)
                if up.headers.get("Content-Length"):
                    resp.content_length = int(up.headers["Content-Length"])
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except aiohttp.ClientConnectorError:
            return web.Response(status=502, text="Service is not running.\n")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"gate :{self.spec.listen_port} upstream error: {e!r}")
            return web.Response(status=502, text="Upstream error.\n")

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        try:
            upstream = await self.session.ws_connect(self._upstream(request, "ws"),
                                                     headers=self._forward_headers(request))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return web.Response(status=502, text="Service is not running.\n")
        client = web.WebSocketResponse()
        await client.prepare(request)

        async def pipe(src, dst):
            async for msg in src:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await dst.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await dst.send_bytes(msg.data)
                else:
                    break
            await dst.close()

        await asyncio.gather(pipe(client, upstream), pipe(upstream, client), return_exceptions=True)
        return client


class GateManager:
    def __init__(self):
        self._gates: Dict[int, _Gate] = {}

    async def ensure(self, spec: GateSpec):
        current = self._gates.get(spec.listen_port)
        if current and current.spec == spec:
            return
        if current:
            await self.stop(spec.listen_port)
        gate = _Gate(spec)
        await gate.start()
        self._gates[spec.listen_port] = gate
        logger.info(f"gate :{spec.listen_port} -> :{spec.upstream_port} (owner {spec.owner})")

    async def stop(self, port: int):
        gate = self._gates.pop(port, None)
        if gate:
            await gate.stop()

    async def stop_all(self):
        for port in list(self._gates):
            await self.stop(port)

    def running_ports(self) -> Set[int]:
        return set(self._gates)

    def spec(self, port: int) -> Optional[GateSpec]:
        gate = self._gates.get(port)
        return gate.spec if gate else None
