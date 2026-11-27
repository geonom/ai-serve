"""Video: jobs at the `video` plugin that lists the model (backend/plugins/video_gen.py). Same shape as the music /
sound jobs (audio_jobs.py). Tests: tests/unit/test_video_jobs.py."""
import logging
import time
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from backend import video_prompt
from backend.plugins import registry, video_gen
from backend.plugins.contracts import VIDEO_TASKS, ContractError, normalize_job

logger = logging.getLogger("ai-serve-ui.video_jobs")
router = APIRouter()
# The plugin answers the start only after its yes/no for GPU room, which may first start ComfyUI (up to 120 s there).
JOB_START_TIMEOUT_S = 180.0


def _row(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in entry.items() if k != "plugin"}
    out["provider"] = entry["plugin"]
    return out


@router.get("/v1/video/models", summary="Video models of every video plugin (tasks, lengths, resolutions, LoRAs)")
async def video_models(task: str = Query("all", description="text_to_video | image_to_video | all")):
    if task != "all" and task not in VIDEO_TASKS:
        raise HTTPException(status_code=400, detail=f"task: {' | '.join(VIDEO_TASKS)} | all")
    entries, errors = await video_gen.catalog()
    rows = [_row(e) for e in entries if task == "all" or task in e["tasks"]]
    out: Dict[str, Any] = {"object": "list", "data": rows}
    if errors:
        out["errors"] = errors
    return out


def _plugin_error(r: httpx.Response, what: str) -> HTTPException:
    try:
        detail = r.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    status = r.status_code if r.status_code in (400, 404, 409, 422, 429, 503, 507) else 502
    return HTTPException(status_code=status, detail=detail if isinstance(detail, str) else f"{what}: the video plugin "
                                                                                       f"answered {r.status_code}")


@router.post("/v1/video/jobs", status_code=202, summary="Make a video (a job: poll it, then fetch the file)")
async def create_video_job(body: Dict[str, Any] = Body(..., examples=[{
        "model": "ltx-2.5", "prompt": "A fox runs through fresh snow at dawn, its breath steaming", "seconds": 4}])):
    """`model` (GET /v1/video/models) or `task` (text_to_video | image_to_video), `prompt`, and as the model allows
    `image_base64` (start picture), `seconds`, `resolution`, `seed`, `negative_prompt`, `lora`, `lora_strength`,
    `sage_attention` (where the catalog row offers it; its value there is the default), `steps` (where the row has
    `steps`: min..max; more steps cost time linearly).
    Answers 202 `{id, status, model}`; poll GET /v1/video/jobs/{id}."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="the body must be a JSON object")
    task = body.get("task")
    if task is not None and task not in VIDEO_TASKS:
        raise HTTPException(status_code=400, detail=f"task: {' | '.join(VIDEO_TASKS)}")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model: a model id")
    entries, _ = await video_gen.catalog()
    try:
        entry = video_gen.pick(entries, model, task, bool(body.get("image_base64")))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    p = next((x for x in registry.providers("video") if x.id == entry["plugin"]), None)
    if p is None or not await registry.ensure_running(p, 60.0):
        raise HTTPException(status_code=503, detail=f"the video plugin {entry['plugin']} is not reachable")
    try:
        r = await registry.call(p, "jobs", method="POST", json={**body, "model": entry["id"]}, timeout=JOB_START_TIMEOUT_S)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"video plugin unreachable: {type(e).__name__}")
    if r.status_code not in (200, 202):
        raise _plugin_error(r, "job")
    jid = (r.json() or {}).get("id") if r.headers.get("content-type", "").startswith("application/json") else None
    if not isinstance(jid, str) or not video_gen._PART.match(jid):
        raise HTTPException(status_code=502, detail="the video plugin answered without a valid job id")
    return JSONResponse(status_code=202, content={"id": video_gen.job_ref(p.id, jid),
                                                  "status": r.json().get("status") or "queued", "model": entry["id"]})


async def _status(ref: str):
    found = video_gen.parse_ref(ref)
    if found is None:
        raise HTTPException(status_code=404, detail="no such video job")
    p, jid = found
    try:
        r = await registry.call(p, "jobs", suffix=f"/{jid}", timeout=30.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"video plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "job status")
    try:
        st = normalize_job(r.json())
    except (ContractError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"the video plugin's job status is not valid: {e}")
    if st["id"] != jid:
        raise HTTPException(status_code=502, detail="the video plugin answered for another job")
    return p, jid, st


@router.get("/v1/video/jobs/{ref}", summary="Status of a video job (result file when done)")
async def video_job(ref: str):
    _, _, st = await _status(ref)
    return video_gen.public_view(ref, st)


@router.post("/v1/video/jobs/{ref}/cancel", summary="Cancel a video job")
async def cancel_video_job(ref: str):
    found = video_gen.parse_ref(ref)
    if found is None:
        raise HTTPException(status_code=404, detail="no such video job")
    p, jid = found
    try:
        r = await registry.call(p, "jobs", method="POST", suffix=f"/{jid}/cancel", timeout=30.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"video plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "cancel")
    try:
        return video_gen.public_view(ref, normalize_job(r.json()))
    except (ContractError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"the video plugin's answer is not valid: {e}")


@router.get("/v1/video/jobs/{ref}/files/{name}", summary="A finished job's video file")
async def video_job_file(ref: str, name: str):
    p, jid, st = await _status(ref)
    f = next((f for f in (st.get("result") or {}).get("files") or [] if f["name"] == name), None)
    jobs_path = p.provision.endpoints["jobs"]
    if f is None or not f["url"].startswith(f"{jobs_path}/{jid}/"):
        raise HTTPException(status_code=404, detail="no such file")
    try:
        r = await registry.call(p, "jobs", suffix=f["url"][len(jobs_path):], timeout=300.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"video plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "file")
    return Response(content=r.content, media_type=f["type"],
                    headers={"Content-Disposition": f'inline; filename="{name}"'})


# ---------------------------------------------------------------- prompt writer (backend/video_prompt.py)

@router.get("/v1/video/prompt/options", summary="Camera and look settings the prompt writer knows")
async def video_prompt_options():
    return video_prompt.options()


@router.post("/v1/video/prompt", summary="Write a detailed video prompt from a short idea (a proposal to edit, no video)")
async def write_video_prompt(body: Dict[str, Any] = Body(..., examples=[{
        "model": "ltx-2.5", "prompt": "Ein Fuchs läuft durch Neuschnee",
        "settings": {"shot": "wide shot", "camera_motion": "tracking shot following the subject",
                     "lighting": "golden hour", "sound": "wind, crunching snow"}}])):
    """`prompt` (the idea, any language), `model` (video model; default: text -> video's, or picture -> video's with
    `image_base64`), `settings` (GET /v1/video/prompt/options; omitted = the chat model picks), `llm` (chat model,
    default as for chat), `seed`, `seconds` (the video's length; default the model's -- guides that time their cuts keep
    them inside it). With a start picture the caption begins from it (when the chat model can see).
    Answers `{prompt, model, task, guide, llm, used_image, runtime_seconds}`; send `prompt` to POST /v1/video/jobs."""
    from backend.routers.gateway import (_begin_request, _build_chat_template_kwargs, _end_request, _for_upstream,
                                         _resolve_upstream, _thinking_default, _upstream_client)
    try:
        r = video_prompt.parse(body)
    except video_prompt.PromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model: a model id")
    entries, _ = await video_gen.catalog()
    try:
        entry = video_gen.pick(entries, model, None, r.image is not None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    guide = entry.get("prompt_guide")
    if guide not in video_prompt.GUIDES:
        raise HTTPException(status_code=400, detail=f"no prompt guide for {entry['label']} (have: "
                                                    f"{', '.join(e['id'] for e in entries if e.get('prompt_guide') in video_prompt.GUIDES) or 'none'})")
    task = "image_to_video" if r.image else "text_to_video"
    if task not in entry["tasks"] or task not in video_prompt.GUIDES[guide]:
        need = "a start picture (image_base64)" if task == "text_to_video" else "no start picture"
        raise HTTPException(status_code=400, detail=f"{entry['label']}: the prompt writer needs {need} here")
    silent = [k for k in video_prompt.SOUND_SETTINGS if k in r.settings and not entry.get("audio")]
    if silent:
        raise HTTPException(status_code=400, detail=f"{entry['label']} makes no sound: leave out {', '.join(silent)}")
    if r.seconds is None:
        r.seconds = entry.get("default_seconds")
    lo, hi = entry.get("min_seconds") or 1, entry.get("max_seconds") or 600
    if r.seconds is not None and not lo <= r.seconds <= hi:
        raise HTTPException(status_code=400, detail=f"seconds for {entry['label']}: {lo}..{hi}")
    started = time.time()
    up = await _resolve_upstream(r.llm, keep_alive=300)
    _begin_request(up)
    try:
        used_image = r.image is not None
        async with _upstream_client(up, 300.0) as client:
            while True:
                payload: Dict[str, Any] = {"messages": video_prompt.messages(r, guide, task, used_image),
                                           "max_tokens": video_prompt.MAX_TOKENS, "temperature": 0.7, "stream": False}
                if r.seed is not None:
                    payload["seed"] = r.seed
                ctk = _build_chat_template_kwargs(None, None, default=_thinking_default(up))
                if ctk:
                    payload["chat_template_kwargs"] = ctk
                try:
                    resp = await client.post(up.url("/v1/chat/completions"), json=_for_upstream(payload, up),
                                             headers=up.headers)
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"chat model unreachable: {type(e).__name__}")
                if resp.status_code == 400 and used_image:
                    used_image = False          # a model that can't see: the caption from the words alone
                    continue
                break
        if resp.status_code != 200:
            raise HTTPException(status_code=502 if resp.status_code < 500 else resp.status_code,
                                detail=f"chat model answered {resp.status_code}: {resp.text[:300]}")
        choice = (resp.json().get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
    finally:
        _end_request(up, 300)
    return {"prompt": video_prompt.clean(text, r.prompt, paragraphs=guide in video_prompt.PARAGRAPH_GUIDES), "model": entry["id"], "task": task, "guide": guide,
            "llm": up.label, "used_image": used_image, "finish_reason": choice.get("finish_reason"),
            "runtime_seconds": round(time.time() - started, 1)}
