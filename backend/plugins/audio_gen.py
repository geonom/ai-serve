"""Music and sound effects from `audio` plugins (docs/plugins.md "audio" and "Long work: jobs"; 2026-10-30, owner:
Stable Audio 3 SFX + ACE-Step in image-serve). Nothing here knows a plugin by name.

- catalog: every audio plugin's `catalog`, merged (an id belongs to the first plugin in id order), cached 10 s.
- jobs: a job lives in its plugin; the gateway's job id is `<plugin id>.<plugin's job id>` -- '.' occurs in neither
  (contracts: plugin ids, job ids are [A-Za-z0-9_-]), so the id says where to ask, and survives a gateway restart.
- result files: only paths below the plugin's own `jobs` endpoint (normalize_job + registry.call's suffix check);
  the gateway fetches them from there and hands them on.
Tests: tests/unit/test_audio_jobs.py."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from backend.plugins import registry
from backend.plugins.contracts import merge_catalogs, normalize_catalog

CATALOG_TTL_S = 10.0
_cache: Tuple[float, List[Dict[str, Any]], List[str]] = (0.0, [], [])
_PART = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


async def catalog(fresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    global _cache
    plugins = registry.providers("audio")
    if not plugins:
        return [], []
    at, entries, errors = _cache
    if not fresh and time.monotonic() - at < CATALOG_TTL_S:
        return entries, errors
    parts, errors = [], []
    for p in plugins:
        if not await registry.ensure_running(p, 30.0):
            errors.append(f"{p.id}: not reachable")
            continue
        try:
            r = await registry.call(p, "catalog", timeout=10.0)
            if r.status_code != 200:
                errors.append(f"{p.id}: catalog answered {r.status_code}")
                continue
            e, errs = normalize_catalog("audio", p.id, r.json())
            parts.append((p.id, e))
            errors += errs
        except (httpx.HTTPError, ValueError) as e:
            errors.append(f"{p.id}: {type(e).__name__}")
    entries, clashes = merge_catalogs(parts)
    errors += [f"{c['ignored']}: model id {c['id']!r} already belongs to {c['kept']}" for c in clashes]
    _cache = (time.monotonic(), entries, errors)
    return entries, errors


def invalidate() -> None:
    global _cache
    _cache = (0.0, [], [])


def pick(entries: List[Dict[str, Any]], model: Optional[str], task: Optional[str]) -> Dict[str, Any]:
    """The entry a job goes to: `model` by id or alias (case-insensitive), else the default for `task` (music when
    neither is given). ValueError says why not."""
    if model:
        key = model.strip().lower()
        e = next((e for e in entries if key == e["id"] or key in (a.lower() for a in e.get("aliases") or [])), None)
        if e is None:
            known = ", ".join(e["id"] for e in entries) or "none -- no audio plugin answers"
            raise ValueError(f"unknown audio model {model!r} (known: {known})")
        if task and task not in e["tasks"]:
            raise ValueError(f"{e['label']} makes {' / '.join(e['tasks'])}, not {task}")
        return e
    task = task or "music"
    fits = [e for e in entries if task in e["tasks"]]
    if not fits:
        raise ValueError(f"no audio model for {task}")
    return next((e for e in fits if e.get("default") and e.get("available")),
                next((e for e in fits if e.get("available")), fits[0]))


def job_ref(plugin_id: str, job_id: str) -> str:
    return f"{plugin_id}.{job_id}"


def parse_ref(ref: str) -> Optional[Tuple[registry.Plugin, str]]:
    plugin_id, sep, job_id = (ref or "").partition(".")
    if not sep or not _PART.match(plugin_id) or not _PART.match(job_id):
        return None
    p = next((x for x in registry.providers("audio") if x.id == plugin_id), None)
    return (p, job_id) if p else None


def public_view(ref: str, status: Dict[str, Any]) -> Dict[str, Any]:
    """A checked plugin job status -> what the caller sees: the gateway's id, files under the gateway's own path."""
    out = dict(status, id=ref)
    if status.get("result"):
        res = dict(status["result"])
        res["files"] = [{"name": f["name"], "type": f["type"], "url": f"/v1/audio/jobs/{ref}/files/{f['name']}"}
                        for f in res.get("files") or []]
        out["result"] = res
    return out
