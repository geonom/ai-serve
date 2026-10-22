"""ASGI adapter for backend/access/policy.py: reads the peer from the scope, looks up
the machine's Tailscale owner and the peer's login (`tailscale whois`, cached --
a subprocess per request would be far too slow), refuses with 403 (websockets:
close 1008). Logs each refused peer/reason at most once a minute."""
import asyncio
import json
import logging
import time
from typing import Callable, Dict, Optional, Tuple

from backend.access.policy import AccessPolicy

logger = logging.getLogger("ai-serve-ui.access")

_WHOIS_TTL_S = 300.0
_WHOIS_NEGATIVE_TTL_S = 30.0
_OWNER_TTL_S = 300.0
_LOG_EVERY_S = 60.0


class AccessMiddleware:
    def __init__(self, app, policy: AccessPolicy, owner: Callable[[], Optional[str]],
                 whois: Callable[[str], Optional[str]]):
        self.app = app
        self.policy = policy
        self._owner_fn = owner
        self._whois_fn = whois
        self._owner: Tuple[float, Optional[str]] = (0.0, None)
        self._whois: Dict[str, Tuple[float, Optional[str]]] = {}
        self._logged: Dict[Tuple[str, str], float] = {}

    async def _owner_login(self) -> Optional[str]:
        expires, value = self._owner
        if time.monotonic() < expires:
            return value
        try:
            value = await asyncio.to_thread(self._owner_fn)
        except Exception:
            value = None
        self._owner = (time.monotonic() + (_OWNER_TTL_S if value else _WHOIS_NEGATIVE_TTL_S), value)
        return value

    async def _peer_login(self, peer: str) -> Optional[str]:
        hit = self._whois.get(peer)
        if hit and time.monotonic() < hit[0]:
            return hit[1]
        try:
            login = await asyncio.to_thread(self._whois_fn, peer)
        except Exception:
            login = None
        self._whois[peer] = (time.monotonic() + (_WHOIS_TTL_S if login else _WHOIS_NEGATIVE_TTL_S), login)
        return login

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        client = scope.get("client")
        peer = client[0] if client else None
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        path = scope.get("path", "")
        login = await self._peer_login(peer) if self.policy.needs_whois(peer) else None
        owner = await self._owner_login()
        decision = self.policy.decide(peer=peer, path=path, headers=headers, owner=owner, peer_login=login)
        if decision.allowed:
            return await self.app(scope, receive, send)
        now = time.monotonic()
        if now - self._logged.get((peer or "?", decision.reason), -1e9) > _LOG_EVERY_S:
            self._logged[(peer or "?", decision.reason)] = now
            logger.warning(f"Refused {scope.get('method', 'WS')} {path!r} from {peer}: {decision.reason}")
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": f"Access denied: {decision.reason}"}).encode()
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
