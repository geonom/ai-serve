"""ASGI adapter for the exposure rules in domain.py: tunnel-marked requests outside
/public/ get a bare 404, and /public/ request bodies are capped (Content-Length up
front, chunked bodies while streaming in) before any handler parses them. An
over-limit chunked body raises PayloadTooLarge inside the handler's body read; the
public route class maps that to 413 like any other domain refusal."""
import json
import logging

from backend.public_api.domain import BodyLimits, ExposureGuard, PayloadTooLarge

logger = logging.getLogger("ai-serve-ui.public")


async def _send_json(send, status: int, body: dict):
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


class PublicExposureMiddleware:
    def __init__(self, app, guard: ExposureGuard = None, limits: BodyLimits = None):
        self.app = app
        self.guard = guard or ExposureGuard()
        self.limits = limits or BodyLimits()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        path = scope.get("path", "")
        if not self.guard.permits(path, headers):
            logger.warning(f"Blocked tunnel request outside /public/: {scope.get('method')} {path!r} "
                           f"from {headers.get('x-forwarded-for', '?')}")
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            return await _send_json(send, 404, {"detail": "Not Found"})
        if scope["type"] != "http" or not path.startswith(ExposureGuard.PUBLIC_PREFIX):
            return await self.app(scope, receive, send)

        limit = self.limits.for_path(path)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    return await _send_json(send, 413, {"error": {"message": "Request too large.",
                                                                  "type": "invalid_request_error"}})
            except ValueError:
                return await _send_json(send, 400, {"error": {"message": "Bad Content-Length.",
                                                              "type": "invalid_request_error"}})
        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise PayloadTooLarge("Request too large.")
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except PayloadTooLarge:  # only reaches here if no route mapped it
            if not started:
                await _send_json(send, 413, {"error": {"message": "Request too large.",
                                                       "type": "invalid_request_error"}})
