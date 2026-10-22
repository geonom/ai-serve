"""Overview page: its saved arrangement and the recent-events feed.

GET /api/dashboard/layout   box order/width/hidden + pinned apps (default if none saved)
PUT /api/dashboard/layout   save it (rules: backend/dashboard_layout.py; 400 on garbage)
GET /api/events?limit=N     newest first (backend/events.py)
"""
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend import events
from backend.config import DASHBOARD_LAYOUT_FILE
from backend.dashboard_layout import LayoutError, parse_layout

logger = logging.getLogger("ai-serve-ui.overview")
router = APIRouter(tags=["overview"])


def _load() -> dict:
    try:
        with open(DASHBOARD_LAYOUT_FILE, encoding="utf-8") as f:
            return parse_layout(json.load(f))
    except FileNotFoundError:
        return parse_layout(None)
    except (OSError, ValueError, LayoutError) as e:
        logger.warning(f"{DASHBOARD_LAYOUT_FILE} unusable, using the default layout: {e}")
        return parse_layout(None)


@router.get("/api/dashboard/layout", summary="Arrangement of the overview page")
async def get_layout():
    return _load()


@router.put("/api/dashboard/layout", summary="Save the arrangement of the overview page")
async def put_layout(data: Any = Body(...)):
    try:
        layout = parse_layout(data)
    except LayoutError as e:
        raise HTTPException(status_code=400, detail=str(e))
    tmp = DASHBOARD_LAYOUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    os.replace(tmp, DASHBOARD_LAYOUT_FILE)
    return layout


@router.get("/api/events", summary="Recent happenings, newest first")
async def recent_events(limit: int = 20):
    return {"events": events.recent(max(1, min(limit, 200)))}
