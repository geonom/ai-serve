"""Talking to node agents and to GitHub's release API (I/O only)."""
from typing import Any, Dict, Optional

import httpx

from backend.cluster.enrollment import NodeRecord

GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


class NodeUnreachable(RuntimeError):
    pass


async def agent_call(node: NodeRecord, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                     timeout: float = 10.0) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.request(method, node.url + path, json=body,
                                headers={"Authorization": f"Bearer {node.token}"})
    except httpx.HTTPError as e:
        raise NodeUnreachable(f"{node.id} unreachable: {type(e).__name__}") from e
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400:
        raise NodeUnreachable(f"{node.id} answered {r.status_code}: {data.get('error') or data}")
    return data


async def fetch_release(tag: Optional[str] = None) -> Dict[str, Any]:
    """A llama.cpp release with binary assets: the given tag, or the newest one
    that has them (the plain vX.Y.Z releases carry none)."""
    async with httpx.AsyncClient(timeout=20.0, headers={"Accept": "application/vnd.github+json"}) as c:
        if tag:
            r = await c.get(f"{GITHUB_RELEASES}/tags/{tag}")
            r.raise_for_status()
            return r.json()
        r = await c.get(GITHUB_RELEASES, params={"per_page": 10})
        r.raise_for_status()
        for rel in r.json():
            if any(a.get("name", "").startswith("llama-b") and "-bin-" in a.get("name", "")
                   for a in rel.get("assets") or []):
                return rel
    raise RuntimeError("no llama.cpp release with binary assets found")


async def hf_meta(ref) -> tuple:
    """(size, sha256) of a Hugging Face file, from HF's own metadata (lfs.oid)."""
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(f"https://huggingface.co/api/models/{ref.repo}/paths-info/{ref.revision}",
                         json={"paths": [ref.file]})
        r.raise_for_status()
        items = r.json()
    for it in items if isinstance(items, list) else []:
        lfs = it.get("lfs") or {}
        if it.get("path") == ref.file and lfs.get("oid") and lfs.get("size"):
            return int(lfs["size"]), str(lfs["oid"])
    raise RuntimeError(f"{ref.repo}/{ref.file}: not found on Hugging Face (or not an LFS file)")
