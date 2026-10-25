"""/v1/vision/*: image analysis through `vision` plugins (backend/plugins/vision.py). The app says which tasks it
wants -- faces, objects or both -- and gets one row per image. Images and embeddings are never logged or kept here.
Tests: tests/unit/test_vision_plugins.py."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.plugins import vision

router = APIRouter()
MAX_IMAGE_BYTES = 40 * 1024 * 1024        # a large scan (TIFF) of one print


@router.get("/v1/vision/models", summary="Image analysis models (faces, objects) of the vision plugins")
async def vision_models(task: Optional[str] = None):
    entries, errors = await vision.catalog()
    if task:
        entries = [e for e in entries if task in e["tasks"]]
    rows = [{**{k: v for k, v in e.items() if k != "plugin"}, "provider": e["plugin"]} for e in entries]
    return {"object": "list", "data": rows, **({"errors": errors} if errors else {})}


@router.post("/v1/vision/analyze", summary="Faces and/or objects in one or more images")
async def vision_analyze(
    image: Optional[List[UploadFile]] = File(None),
    image_list: Optional[List[UploadFile]] = File(None, alias="image[]"),
    tasks: str = Form("faces"),                 # faces | objects | faces,objects | both
    face_model: Optional[str] = Form(None),
    object_model: Optional[str] = Form(None),
    embed: bool = Form(True),                   # faces: also the embedding (for recognition)
    labels: Optional[str] = Form(None),         # objects: what to look for (comma-separated), open-vocabulary models
):
    if image and image_list:
        raise HTTPException(status_code=400, detail="send the images either as image or as image[], not both")
    uploads = image or image_list or []
    if not uploads:
        raise HTTPException(status_code=400, detail="no image")
    try:
        wanted = vision.parse_tasks(tasks)
        entries, _ = await vision.catalog()
        steps = vision.plan(entries, wanted, {"faces": face_model, "objects": object_model})
    except vision.VisionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    top = min(e["max_images"] for e, _ in steps)
    if len(uploads) > top:
        raise HTTPException(status_code=400, detail=f"at most {top} images per call -- send an album in pages")
    files = []
    for u in uploads:
        data = await u.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"an image is larger than {MAX_IMAGE_BYTES >> 20} MB")
        files.append(("image", (u.filename or "image", data, u.content_type or "application/octet-stream")))
    answers = []
    for entry, ts in steps:
        p = vision.provider_of(entry)
        if p is None:
            raise HTTPException(status_code=503, detail=f"{entry['plugin']} is gone")
        fields = {"model": entry["id"], "tasks": ",".join(ts), "embed": "true" if embed else "false",
                  **({"labels": labels} if labels and "objects" in ts else {})}
        try:
            r = await vision.analyze(p, fields, files)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            raise HTTPException(status_code=r.status_code if r.status_code in (400, 409, 413, 422, 429, 503, 507) else 502,
                                detail=detail if isinstance(detail, str) else f"{p.name} answered {r.status_code}")
        answers.append((entry, ts, data))
    try:
        return vision.merge(len(files), answers)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
