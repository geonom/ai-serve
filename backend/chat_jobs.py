"""Stopping a non-streamed chat completion while it runs (2026-10-26, owner: "ai-serve should be able to cancel a
non-streamed chat"). Found with V-Noveller: a json_schema chat looped, its client was stopped, but the gateway kept
waiting for llama-server, which generated until max_tokens and held GPU 0 for minutes (every other model: 507).

The handler's work runs as its own task. It ends early when
- the client goes away (polled via Request.is_disconnected) -> 499, nobody is listening any more;
- POST /v1/chat/cancel names its X-Request-Id (16-64 of [A-Za-z0-9_-], like image cancels) -> 409.
Cancelling the task closes the upstream HTTP connection; llama-server stops a generation whose connection closed.
A cancel that arrives before its request is remembered for 10 min and refuses that request on arrival (once).
Streamed chats need none of this: when the client goes away the stream relay closes upstream itself."""
import asyncio
import re
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import HTTPException

_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
EARLY_TTL_S = 600

_running: Dict[str, asyncio.Task] = {}
_early: Dict[str, float] = {}


def valid_id(rid: Optional[str]) -> bool:
    return bool(rid) and bool(_ID.match(rid))


def running(rid: str) -> bool:
    return rid in _running


def forget(rid: str) -> None:
    _early.pop(rid, None)


def cancel(rid: str) -> bool:
    """True = a running chat with this id is being stopped; False = none runs (remembered for its arrival)."""
    task = _running.get(rid)
    if task is None:
        now = time.time()
        for k in [k for k, t in _early.items() if now - t > EARLY_TTL_S]:
            del _early[k]
        _early[rid] = now
        return False
    task.get_loop().call_soon_threadsafe(task.cancel)   # the request may run on another loop (tests, threads)
    return True


async def run(work: Awaitable[Any], rid: Optional[str], is_disconnected: Callable[[], Awaitable[bool]],
              poll_s: float = 0.5) -> Any:
    rid = rid if valid_id(rid) else None
    if rid and _early.pop(rid, None) is not None:
        if asyncio.iscoroutine(work):
            work.close()
        raise HTTPException(status_code=409, detail="Chat completion cancelled")
    task = asyncio.ensure_future(work)
    if rid:
        _running[rid] = task
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_s)
            if done:
                break
            if await is_disconnected():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise HTTPException(status_code=499, detail="Client went away -- generation stopped")
        if task.cancelled():
            raise HTTPException(status_code=409, detail="Chat completion cancelled")
        return task.result()
    except asyncio.CancelledError:   # the handler itself was cancelled: take the work down with it
        task.cancel()
        raise
    finally:
        if rid:
            _running.pop(rid, None)
