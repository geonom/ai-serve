"""image-serve: ComfyUI as an ai-serve `image` plugin (ai-serve-ui docs/plugins.md). Owner, 2026-10-21: nothing of an
image backend built into the gateway. This app listens where ComfyUI used to (8189, Apps & Services id `comfyui`, so
existing reachability settings keep working) and

- speaks the plugin contract: /v1/images/models (catalog), /v1/images/generations, /v1/images/edits (task edit|style,
  style with a second `reference` image), /v1/images/cancel (by X-Request-Id), /plugin/status|release|logs;
- serves ComfyUI's job API under /api/comfyui/* (generate / edit / style-transfer / batch / task / loras / workflows /
  images -- apps from before image plugins use it; ai-serve forwards /api/comfyui/* here);
- passes everything else through to ComfyUI itself (its web UI, /prompt, /history, /view, the /ws websocket), started
  on demand on the GPU ai-serve's GPU manager picks, on the internal port COMFY_PORT (8192).
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aiohttp
import httpx
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from image_serve import catalog, comfy, config, jobs, moss
from image_serve.config import LOGS_DIR
from image_serve.state import state

logger = logging.getLogger("image-serve")

# License per catalog model (contract: every entry says whether it may be used commercially)
LICENSES: Dict[str, tuple] = {
    "z-image-turbo": ("Apache-2.0", True), "qwen-image-2.1": ("Apache-2.0", True),
    "qwen-image-2.1-int8": ("Apache-2.0", True), "qwen-edit-2511": ("Apache-2.0", True),
    "qwen-edit-2511-q3": ("Apache-2.0", True), "ernie-turbo": ("", None), "ideogram-4": ("", None),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    comfy.adopt_running()
    yield
    # ComfyUI runs in its own session and survives this app's restarts (adopted again on the way up)


app = FastAPI(title="image-serve (ComfyUI) for ai-serve", lifespan=lifespan)
app.include_router(comfy.router)


@app.get("/health")
async def health():
    return {"ok": True}


# ------------------------------------------------------------------ plugin contract

@app.get("/v1/images/models")
async def models(task: str = "all"):
    """The contract's catalog (every task; ai-serve filters). ?task=generate|edit|style narrows it."""
    try:
        listing = catalog.listing(task=task)
    except catalog.CatalogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rows = []
    for r in listing:
        lic, commercial = LICENSES.get(r["id"], ("", None))
        rows.append({**{k: r[k] for k in ("id", "label", "description", "available", "aliases", "default", "tasks",
                                          "aspect_ratios", "text_in_image", "layout", "steps")},
                     "license": lic, "commercial_use": commercial})
    if config.moss_possible() and task in ("all", "generate"):
        for r in moss.catalog(lambda v: config.comfy_ready() and moss_installed(v))["data"]:
            rows.append({**r, "default": False, "aliases": []})
    return {"object": "list", "data": rows}


# ------------------------------------------------------------------ MOSS-Image (text in images, layout)
# Moved here from the comfyui-moss plugin (2026-10-22): it needs ComfyUI v0.42+, which setup.sh now pins. Where
# image-serve runs an older ComfyUI (COMFYUI_DIR in .env), its variants aren't listed at all (config.moss_possible).

def _models_root() -> str:
    return config.COMFYUI_MODELS_ROOT


def moss_installed(variant: str) -> bool:
    root = _models_root()
    return all(os.path.isfile(os.path.join(root, rel)) for rel in moss.needed_files(variant))


async def _moss_generate(body: Dict[str, Any], rid: Optional[str]) -> Dict[str, Any]:
    if not config.moss_possible():
        v = config.comfy_version()
        raise HTTPException(status_code=400, detail="MOSS-Image needs ComfyUI v0.42 or newer"
                                                    + (f" (this one is v{'.'.join(map(str, v))})" if v else ""))
    try:
        variant, prompt, elements, width, height, n, seed, steps = moss.from_openai(body)
    except moss.InputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not (config.comfy_ready() and moss_installed(variant)):
        raise HTTPException(status_code=400, detail=f"{moss.VARIANTS[variant]['label']} is not installed on this machine")
    data = []
    for i in range(n):
        wf = moss.build_workflow(prompt, elements, width, height, seed=None if seed is None else seed + i,
                                 steps=steps, variant=variant)
        data.append({"b64_json": await _run(lambda _r, wf=wf: comfy.queue_workflow(wf, variant), None, rid)})
    return {"created": int(time.time()), "data": data}


@app.post("/v1/images/generations")
async def generations(req: comfy.ImageGenerationRequest, x_request_id: Optional[str] = Header(None)):
    rid = jobs.valid_request_id(x_request_id)   # lets the caller cancel this one job (POST /v1/images/cancel)
    if jobs.JOBS.cancelled(rid):
        raise HTTPException(status_code=409, detail="Image generation cancelled")
    if (req.model or "").strip().lower() in moss.VARIANTS:
        return await _moss_generate(req.model_dump(exclude_unset=True), rid)
    model = catalog.resolve(req.model)
    if model and "generate" not in model.tasks:
        raise HTTPException(status_code=400, detail=f"{model.label} edits images; use /v1/images/edits")
    if req.aspect_ratio is not None:
        if req.size is not None and "size" in req.model_fields_set:
            raise HTTPException(status_code=400, detail="give either size or aspect_ratio, not both")
        target = model or catalog.resolve(catalog.DEFAULT)
        if req.aspect_ratio not in target.aspects:
            raise HTTPException(status_code=400, detail=f"{target.label} supports the aspect ratios {', '.join(target.aspects)}")
    if model and not catalog.listing_row(model)["available"]:
        raise HTTPException(status_code=400, detail=f"{model.label} is not installed on this machine")
    try:
        catalog.check_steps(model, req.steps)
    except catalog.CatalogError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.layout:
        raise HTTPException(status_code=400, detail="layout needs a model that can place things (e.g. moss)")
    # Unknown names (OpenAI clients send e.g. dall-e-3) keep the default. A ComfyUI model's UNet only works with the
    # CLIP/VAE its workflow template bakes in -- the catalog pairs them.
    model_name = model.file if model else catalog.resolve(catalog.DEFAULT).file
    if not model and "ernie" in req.prompt.lower():
        model_name = "ernie"
    if req.aspect_ratio:
        base_res, aspect_ratio = 1024, req.aspect_ratio
    else:
        base_res, aspect_ratio = comfy.openai_size_to_comfy(req.size)
    data = []
    for i in range(max(1, min(int(req.n or 1), 4))):
        seed = (req.seed + i) if req.seed is not None else random.randint(1, 10_000_000)
        comfy_req = comfy.ComfyUIGenerateRequest(
            prompt=req.prompt, negative_prompt="", model_name=model_name, aspect_ratio=aspect_ratio,
            base_resolution=base_res, steps=req.steps, seed=seed)
        data.append({"b64_json": await _run(comfy.comfyui_generate, comfy_req, rid)})
    return {"created": int(time.time()), "data": data}


async def _run(submit, comfy_req, rid: Optional[str]) -> str:
    """Queue one ComfyUI job, register how to cancel exactly it, wait for the image."""
    res = await submit(comfy_req)
    pid = res["task_id"]
    if rid:
        async def stop():
            task = state.comfyui_tasks.get(pid)
            if task is not None and task.get("status") not in ("completed", "failed"):
                task.update(status="failed", error="cancelled")
            await jobs.interrupt_comfy(f"http://127.0.0.1:{state.comfyui_port}", pid)
        jobs.JOBS.register(rid, stop)
    try:
        return await comfy._await_comfyui_task(pid)
    finally:
        if rid:
            jobs.JOBS.unregister(rid)


@app.post("/v1/images/edits")
async def edits(image: UploadFile = File(...), prompt: str = Form(...), model: Optional[str] = Form(None),
                seed: Optional[int] = Form(None), task: str = Form("edit"), reference: Optional[UploadFile] = File(None),
                x_request_id: Optional[str] = Header(None)):
    """OpenAI's edits shape + the contract's `task`: edit (one image) or style (`image` = content, `reference` = style)."""
    if seed is not None and not 0 <= seed < 2 ** 50:
        raise HTTPException(status_code=400, detail="seed must be 0 .. 2^50")
    if task not in ("edit", "style"):
        raise HTTPException(status_code=400, detail="task: edit or style")
    rid = jobs.valid_request_id(x_request_id)
    content = base64.b64encode(await image.read()).decode()
    seed = seed if seed is not None else random.randint(1, 10_000_000)
    if task == "style":
        if reference is None:
            raise HTTPException(status_code=400, detail="style needs a second image: reference")
        kwargs = {"content_image_base64": content, "style_image_base64": base64.b64encode(await reference.read()).decode(),
                  "prompt": prompt, "seed": seed}
        if model:   # a catalog id (or an old file name) -> its checkpoint; 400 if it can't do the task
            kwargs["model_name"] = comfy._catalog_file(model, "style")
        b64 = await _run(comfy.comfyui_style_transfer, comfy.ComfyUIStyleTransferRequest(**kwargs), rid)
    else:
        kwargs = {"image_base64": content, "prompt": prompt, "seed": seed}
        if model:
            kwargs["model_name"] = comfy._catalog_file(model, "edit")
        b64 = await _run(comfy.comfyui_edit, comfy.ComfyUIEditRequest(**kwargs), rid)
    return {"created": int(time.time()), "data": [{"b64_json": b64}]}


@app.post("/v1/images/cancel")
async def cancel(body: Any = Body(...)):
    rid = jobs.valid_request_id(body.get("request_id") if isinstance(body, dict) else None)
    if rid is None:
        raise HTTPException(status_code=400, detail="request_id: 16-64 of A-Z, a-z, 0-9, _ and -")
    return {"cancelled": await jobs.JOBS.cancel(rid)}


def _running() -> bool:
    return state.comfyui_process is not None and state.comfyui_process.poll() is None


@app.get("/plugin/status")
async def status():
    running = _running() or comfy.find_comfyui_process() is not None
    reason = await comfy._comfyui_busy_reason() if running else None
    workers = [{"id": "comfyui", "label": "ComfyUI" + (f" ({state.comfyui_active_model})" if state.comfyui_active_model else ""),
                "pid": state.comfyui_pid or comfy.get_pid_on_port(state.comfyui_port),
                "loaded": state.comfyui_loaded_models is not None, "busy": reason is not None,
                "last_used": state.comfyui_last_used or None, "gpu": state.comfyui_gpu_id}] if running else []
    if not running:
        fact = "off (starts with the next image)"
    elif state.comfyui_active_model and state.comfyui_loaded_models is not None:
        fact = f"{state.comfyui_active_model} loaded, GPU {state.comfyui_gpu_id}"
    else:
        fact = f"ready on GPU {state.comfyui_gpu_id}, no model loaded"
    facts = [{"label": "ComfyUI", "value": fact}] + ([{"label": "Working", "value": reason}] if reason else [])
    return {"busy": reason is not None, "ready": True, "workers": workers, "facts": facts}


@app.post("/plugin/release")
async def release(body: Any = Body(default={})):
    body = body if isinstance(body, dict) else {}
    if body.get("worker") not in (None, "comfyui"):
        raise HTTPException(status_code=404, detail="no such worker")
    had = state.comfyui_loaded_models is not None
    out = await comfy.unload_comfyui(force=body.get("force") is True)   # 409 while busy (unless force)
    return {"released": ["comfyui"] if had and out.get("status") == "unloaded" else []}


@app.get("/plugin/logs")
async def logs(tail: int = 200):
    try:
        with open(os.path.join(LOGS_DIR, "comfyui.log"), encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()[-max(1, min(tail, 5000)):]
    except OSError:
        lines = []
    return {"logs": "\n".join(lines)}


# ------------------------------------------------------------------ downloads (plugin contract: install)
# The model bundles (image_serve/bundles.py): pinned files with sha256, downloaded in the background on the owner's
# request (the gateway lets only the owner start one); only the bundle id comes from the request.
_install: Dict[str, Dict[str, Any]] = {}
_install_lock = threading.Lock()


COMFYUI_ITEM = "comfyui"


def comfy_ready() -> bool:
    from image_serve import config
    return config.comfy_ready()


def _comfy_item() -> Dict[str, Any]:
    st = _install.get(COMFYUI_ITEM, {})
    state = "installed" if comfy_ready() and st.get("state") != "downloading" else (st.get("state") or "missing")
    return {"id": COMFYUI_ITEM, "label": "ComfyUI", "license": "GPL-3.0", "size_mb": 5000, "state": state,
            "description": "The engine for these image models (ComfyUI with its own Python; comes along with any model)",
            "done_mb": None, "error": st.get("error", "") if state == "failed" else ""}


def _setup_comfy() -> None:
    """setup.sh without models: ComfyUI (pinned), ComfyUI-GGUF, ComfyUI's Python. Raises when it fails."""
    import subprocess
    from image_serve.config import HERE
    log = os.path.join(LOGS_DIR, "setup.log")
    with open(log, "w", encoding="utf-8") as out:
        code = subprocess.run(["bash", str(HERE / "setup.sh")], cwd=str(HERE), stdout=out, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    if code != 0 or not comfy_ready():
        with open(log, encoding="utf-8", errors="replace") as f:
            tail = " | ".join(l.strip() for l in f.read().splitlines()[-3:] if l.strip())
        raise RuntimeError(f"setup.sh exited with {code} -- {tail[:250]}")


MOSS_BUNDLES = ("moss", "moss-bf16")


def _offered(bundle: str) -> bool:
    return bundle not in MOSS_BUNDLES or config.moss_possible()


def _install_item(bundle: str) -> Dict[str, Any]:
    from image_serve import bundles
    from image_serve.config import COMFYUI_MODELS_ROOT
    st = _install.get(bundle, {})
    there = bundles.present(bundle, COMFYUI_MODELS_ROOT) and comfy_ready()
    state = "installed" if there and st.get("state") != "downloading" else (st.get("state") or "missing")
    info = bundles.BUNDLES[bundle]
    return {"id": bundle, "label": info["label"], "description": info["description"], "license": info["license"],
            "size_mb": bundles.size_mb(bundle) + (bundles.size_mb(info["needs"])
                                                  if info.get("needs") and not bundles.present(info["needs"], COMFYUI_MODELS_ROOT) else 0),
            "state": state,
            "done_mb": (st.get("done", 0) >> 20) if state == "downloading" else None,
            "error": st.get("error", "") if state == "failed" else ""}


def _run_install(bundle: str) -> None:
    from image_serve import bundles
    from image_serve.config import COMFYUI_MODELS_ROOT

    def progress(n: int) -> None:
        _install[bundle]["done"] += n
    try:
        if not comfy_ready():                      # the engine first (a model alone can't run)
            _setup_comfy()
        needs = bundles.BUNDLES[bundle].get("needs")
        if needs and not bundles.present(needs, COMFYUI_MODELS_ROOT):   # e.g. MOSS bf16 on top of MOSS's shared files
            bundles.download(needs, COMFYUI_MODELS_ROOT, progress)
        bundles.download(bundle, COMFYUI_MODELS_ROOT, progress)
        _install[bundle] = {"state": "installed"}
        logger.info(f"{bundle}: downloaded")
    except Exception as e:
        _install[bundle] = {"state": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
        logger.error(f"{bundle}: download failed: {e}")


def _run_comfy_setup() -> None:
    try:
        _setup_comfy()
        _install[COMFYUI_ITEM] = {"state": "installed"}
    except Exception as e:
        _install[COMFYUI_ITEM] = {"state": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
        logger.error(f"ComfyUI setup failed: {e}")


@app.get("/plugin/install")
async def install_list():
    from image_serve import bundles
    return {"items": [_comfy_item()] + [_install_item(b) for b in bundles.BUNDLES if _offered(b)]}


@app.post("/plugin/install", status_code=202)
async def install_start(body: Any = Body(...)):
    from image_serve import bundles
    from image_serve.config import COMFYUI_MODELS_ROOT
    bundle = body.get("id") if isinstance(body, dict) else None
    if bundle == COMFYUI_ITEM:
        with _install_lock:
            if not comfy_ready() and _install.get(COMFYUI_ITEM, {}).get("state") != "downloading":
                _install[COMFYUI_ITEM] = {"state": "downloading"}
                threading.Thread(target=_run_comfy_setup, daemon=True).start()
        return _comfy_item()
    if not isinstance(bundle, str) or bundle not in bundles.BUNDLES:
        raise HTTPException(status_code=404, detail=f"nothing to download under that id (one of {', '.join(bundles.BUNDLES)})")
    if not _offered(bundle):
        raise HTTPException(status_code=409, detail="MOSS-Image needs ComfyUI v0.42 or newer; the ComfyUI configured here is older")
    with _install_lock:
        if not (bundles.present(bundle, COMFYUI_MODELS_ROOT) and comfy_ready()) \
                and _install.get(bundle, {}).get("state") != "downloading":
            _install[bundle] = {"state": "downloading", "done": 0}
            threading.Thread(target=_run_install, args=(bundle,), daemon=True).start()
    return _install_item(bundle)


# ------------------------------------------------------------------ ComfyUI itself (web UI, its own API)

_HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "upgrade"}


@app.websocket("/ws")
async def ws_proxy(ws: WebSocket):
    """ComfyUI's websocket (progress for its web UI), passed through both ways."""
    await ws.accept()
    query = ws.url.query
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{state.comfyui_port}/ws" + (f"?{query}" if query else "")) as up:
                async def down():
                    async for msg in up:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await ws.send_text(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await ws.send_bytes(msg.data)
                        else:
                            break

                async def upward():
                    while True:
                        m = await ws.receive()
                        if m["type"] == "websocket.disconnect":
                            break
                        if m.get("text") is not None:
                            await up.send_str(m["text"])
                        elif m.get("bytes") is not None:
                            await up.send_bytes(m["bytes"])
                tasks = [asyncio.create_task(down()), asyncio.create_task(upward())]
                _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
    except (aiohttp.ClientError, OSError) as e:
        logger.info(f"ComfyUI websocket: {e}")
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"])
async def comfy_proxy(path: str, request: Request):
    """Everything else is ComfyUI's own (web UI, /prompt, /history, /view, /object_info ...). Opening its page
    starts it, like the old built-in app did."""
    try:
        await comfy._ensure_comfyui_running()
    except HTTPException as e:
        return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    try:
        req = client.build_request(request.method, f"http://127.0.0.1:{state.comfyui_port}/{path}",
                                   params=request.query_params, headers=headers, content=await request.body())
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        return JSONResponse({"detail": f"ComfyUI: {type(e).__name__}"}, status_code=502)

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()
    out = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP}
    return StreamingResponse(body(), status_code=resp.status_code, headers=out)


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8189")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", ws="websockets")


if __name__ == "__main__":
    main()
