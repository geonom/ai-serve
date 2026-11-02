"""Music and sound effects: jobs at the `audio` plugin that lists the model (backend/plugins/audio_gen.py).
Tests: tests/unit/test_audio_jobs.py."""
import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse, Response

from backend.plugins import audio_gen, registry
from backend.plugins.contracts import ContractError, normalize_job

logger = logging.getLogger("ai-serve-ui.audio_jobs")
router = APIRouter()


def _row(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in entry.items() if k != "plugin"}
    out["provider"] = entry["plugin"]
    return out


async def generator_models(task: str) -> Dict[str, Any]:
    """GET /v1/audio/models?task=music|sound|all (the speech catalog stays at /v1/audio/models without `task`)."""
    if task not in ("music", "sound", "all"):
        raise HTTPException(status_code=400, detail="task: music | sound | all")
    entries, errors = await audio_gen.catalog()
    rows = [_row(e) for e in entries if task == "all" or task in e["tasks"]]
    default = next((r["id"] for r in rows if r.get("default") and r.get("available")), rows[0]["id"] if rows else None)
    out: Dict[str, Any] = {"object": "list", "default": default if task != "all" else None, "data": rows}
    if errors:
        out["errors"] = errors
    return out


def _plugin_error(r: httpx.Response, what: str) -> HTTPException:
    try:
        detail = r.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    status = r.status_code if r.status_code in (400, 404, 409, 422, 429, 503, 507) else 502
    return HTTPException(status_code=status, detail=detail if isinstance(detail, str) else f"{what}: the audio plugin "
                                                                                       f"answered {r.status_code}")


@router.post("/v1/audio/jobs", status_code=202, summary="Make music or a sound effect (a job: poll it, then fetch the file)")
async def create_audio_job(body: Dict[str, Any] = Body(..., examples=[{
        "task": "sound", "prompt": "Regen auf einem Blechdach, fernes Gewitter", "seconds": 15}])):
    """`model` (from GET /v1/audio/models?task=all) or `task` (music | sound; default music), `prompt`, and as the model
    allows `lyrics`, `seconds`, `seed` and its own extras. Answers 202 `{id, status}`; poll GET /v1/audio/jobs/{id}."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="the body must be a JSON object")
    task = body.get("task")
    if task is not None and task not in ("music", "sound"):
        raise HTTPException(status_code=400, detail="task: music | sound")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model: a model id")
    entries, _ = await audio_gen.catalog()
    try:
        entry = audio_gen.pick(entries, model, task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    p = next((x for x in registry.providers("audio") if x.id == entry["plugin"]), None)
    if p is None or not await registry.ensure_running(p, 60.0):
        raise HTTPException(status_code=503, detail=f"the audio plugin {entry['plugin']} is not reachable")
    try:
        r = await registry.call(p, "jobs", method="POST", json={**body, "model": entry["id"]}, timeout=60.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"audio plugin unreachable: {type(e).__name__}")
    if r.status_code not in (200, 202):
        raise _plugin_error(r, "job")
    jid = (r.json() or {}).get("id") if r.headers.get("content-type", "").startswith("application/json") else None
    if not isinstance(jid, str) or not audio_gen._PART.match(jid):
        raise HTTPException(status_code=502, detail="the audio plugin answered without a valid job id")
    return JSONResponse(status_code=202, content={"id": audio_gen.job_ref(p.id, jid), "status": r.json().get("status") or "queued",
                                                  "model": entry["id"]})


async def _status(ref: str):
    found = audio_gen.parse_ref(ref)
    if found is None:
        raise HTTPException(status_code=404, detail="no such audio job")
    p, jid = found
    try:
        r = await registry.call(p, "jobs", suffix=f"/{jid}", timeout=30.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"audio plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "job status")
    try:
        st = normalize_job(r.json())
    except (ContractError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"the audio plugin's job status is not valid: {e}")
    if st["id"] != jid:
        raise HTTPException(status_code=502, detail="the audio plugin answered for another job")
    return p, jid, st


@router.get("/v1/audio/jobs/{ref}", summary="Status of a music / sound job (result files when done)")
async def audio_job(ref: str):
    _, _, st = await _status(ref)
    return audio_gen.public_view(ref, st)


@router.post("/v1/audio/jobs/{ref}/cancel", summary="Cancel a music / sound job")
async def cancel_audio_job(ref: str):
    found = audio_gen.parse_ref(ref)
    if found is None:
        raise HTTPException(status_code=404, detail="no such audio job")
    p, jid = found
    try:
        r = await registry.call(p, "jobs", method="POST", suffix=f"/{jid}/cancel", timeout=30.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"audio plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "cancel")
    try:
        return audio_gen.public_view(ref, normalize_job(r.json()))
    except (ContractError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"the audio plugin's answer is not valid: {e}")


@router.get("/v1/audio/jobs/{ref}/files/{name}", summary="A finished job's audio file")
async def audio_job_file(ref: str, name: str):
    p, jid, st = await _status(ref)
    f = next((f for f in (st.get("result") or {}).get("files") or [] if f["name"] == name), None)
    jobs_path = p.provision.endpoints["jobs"]
    if f is None or not f["url"].startswith(f"{jobs_path}/{jid}/"):
        raise HTTPException(status_code=404, detail="no such file")
    try:
        r = await registry.call(p, "jobs", suffix=f["url"][len(jobs_path):], timeout=120.0)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"audio plugin unreachable: {type(e).__name__}")
    if r.status_code != 200:
        raise _plugin_error(r, "file")
    return Response(content=r.content, media_type=f["type"],
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
