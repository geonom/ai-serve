"""Smoke test: a tiny optional "does it work?" sanity check across all models.

Loads each model sequentially on one GPU, asks a simple factual question, and
checks the answer contains the expected word. Not a performance benchmark —
just confirms a model loads and produces a sane response.
"""
import asyncio
import logging
import os
from typing import Any, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.model_manager import kill_process_on_port, stop_server_by_port
from backend.routers.dashboard import StartRequest, list_models, start_server
from backend.state import state

logger = logging.getLogger("ai-serve-ui.smoketest")
router = APIRouter()

_SMOKE_TEST_PORT = 9000
_QUESTION = "What is the capital of Bulgaria?"
_EXPECTED = "sofia"


class SmokeTestRequest(BaseModel):
    model_paths: Optional[List[str]] = None  # defaults to every discovered model
    gpu_id: int = 0


async def _run_smoke_test(req: SmokeTestRequest):
    state.smoke_test_running = True
    state.smoke_test_results = []

    model_paths = req.model_paths
    if not model_paths:
        all_models = await list_models()
        model_paths = [m["path"] for m in all_models["models"]]

    state.smoke_test_progress = {"current": 0, "total": len(model_paths), "model": "", "status": "starting"}

    try:
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            state.smoke_test_progress.update({"current": i + 1, "model": model_name, "status": "loading"})

            kill_process_on_port(_SMOKE_TEST_PORT)
            try:
                await start_server(StartRequest(model_path=model_path, port=_SMOKE_TEST_PORT, gpu_id=req.gpu_id))
            except Exception as e:
                state.smoke_test_results.append({"model": model_name, "status": "error", "error": f"Failed to start: {e}"})
                continue

            ready = False
            for _ in range(30):
                await asyncio.sleep(2)
                try:
                    async with httpx.AsyncClient(timeout=1.0) as client:
                        resp = await client.get(f"http://127.0.0.1:{_SMOKE_TEST_PORT}/health")
                        if resp.status_code == 200:
                            ready = True
                            break
                except Exception:
                    pass
            if not ready:
                state.smoke_test_results.append({"model": model_name, "status": "error", "error": "Server failed to become ready"})
                await stop_server_by_port(_SMOKE_TEST_PORT)
                continue

            state.smoke_test_progress["status"] = "testing"
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"http://127.0.0.1:{_SMOKE_TEST_PORT}/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": _QUESTION}],
                            # Reasoning models (e.g. Qwen3) spend tokens on reasoning_content
                            # before emitting content — 300 leaves room for both.
                            "max_tokens": 300,
                            "temperature": 0.0,
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    message = resp.json()["choices"][0]["message"]
                    # Reasoning models split their output into content + reasoning_content;
                    # fall back to the latter if content came back empty (budget still too tight).
                    content: Any = message.get("content") or message.get("reasoning_content") or ""
                    passed = _EXPECTED in content.lower()
                    state.smoke_test_results.append({
                        "model": model_name,
                        "status": "passed" if passed else "failed",
                        "answer": content.strip(),
                    })
            except Exception as e:
                state.smoke_test_results.append({"model": model_name, "status": "error", "error": str(e)})
            finally:
                await stop_server_by_port(_SMOKE_TEST_PORT)

        state.smoke_test_progress["status"] = "completed"
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        state.smoke_test_progress["status"] = f"failed: {e}"
    finally:
        state.smoke_test_running = False


@router.get("/api/smoke-test/status", summary="Get active smoke-test progress and results")
async def get_smoke_test_status():
    return {"running": state.smoke_test_running, "progress": state.smoke_test_progress, "results": state.smoke_test_results}


@router.post("/api/smoke-test/start", summary='Ask every model "What is the capital of Bulgaria?" to confirm it loads and answers sensibly')
async def start_smoke_test(req: SmokeTestRequest):
    if state.smoke_test_running:
        raise HTTPException(status_code=400, detail="Smoke test already running")
    asyncio.create_task(_run_smoke_test(req))
    return {"status": "started"}
