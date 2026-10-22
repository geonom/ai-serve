"""Hugging Face GGUF downloads -- the I/O around backend/hf_models.py (rules there): read a repo's file tree, download
the planned files in the background with their sha256 checked while streaming, report progress. Used by the
dashboard (backend/routers/hf_download.py) and `./ai-serve model add`. HF_TOKEN (ai-serve.local.env) is sent for
gated repos (Gemma, Llama ...), and only to huggingface.co."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from backend import config as _config  # noqa: F401  -- loads ai-serve.local.env (HF_TOKEN, LLAMA_MODEL_DIRS)
from backend import hf_models

logger = logging.getLogger("ai-serve-ui.hf")
UA = "ai-serve"


def _headers(url: str) -> Dict[str, str]:
    h = {"User-Agent": UA}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and urllib.parse.urlsplit(url).hostname == "huggingface.co":
        h["Authorization"] = f"Bearer {token}"
    return h


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects file downloads to its CDN -- the token must not travel along."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != "huggingface.co":
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_NoAuthOnRedirect)


def _http_error(e: urllib.error.HTTPError, what: str) -> hf_models.HfLinkError:
    if e.code in (401, 403):
        return hf_models.HfLinkError(
            f"{what} needs a Hugging Face login: accept its license on its page, create a token "
            "(huggingface.co/settings/tokens) and put HF_TOKEN=<token> into ai-serve.local.env, then restart")
    if e.code == 404:
        return hf_models.HfLinkError(f"{what} doesn't exist on Hugging Face (or is private)")
    return hf_models.HfLinkError(f"Hugging Face answered {e.code} for {what}")


def get_tree(repo: str, revision: str) -> List[Dict[str, Any]]:
    url = (f"https://huggingface.co/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}"
           "?recursive=true&expand=false")
    try:
        with _OPENER.open(urllib.request.Request(url, headers=_headers(url)), timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise _http_error(e, repo)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise hf_models.HfLinkError(f"Hugging Face isn't reachable ({type(e).__name__})")
    return data if isinstance(data, list) else []


def model_dir() -> str:
    """Where downloads go: the first configured model folder (created if needed)."""
    from backend.config import MODEL_DIRS
    d = MODEL_DIRS[0]
    os.makedirs(d, exist_ok=True)
    return d


def inspect(text: str) -> Dict[str, Any]:
    link = hf_models.parse_link(text)
    cat = hf_models.catalog(get_tree(link.repo, link.revision))
    if not cat["models"]:
        raise hf_models.HfLinkError(f"{link.repo} has no GGUF model files (with a checksum) -- llama.cpp needs a GGUF "
                                    "repo, usually named …-GGUF")
    selected = hf_models.pick(cat, link)
    view = lambda f: {k: f[k] for k in ("path", "size", "parts", "quant") if k in f}
    return {"repo": link.repo, "revision": link.revision, "selected": selected,
            "models": [view(m) for m in cat["models"]], "mmproj": [view(p) for p in cat["mmproj"]],
            "dest": model_dir()}


def make_plan(text: str, model: Optional[str], mmproj: Optional[str]) -> Dict[str, Any]:
    """Always from a fresh listing -- sizes and hashes never come from the caller."""
    link = hf_models.parse_link(text)
    cat = hf_models.catalog(get_tree(link.repo, link.revision))
    model = model or hf_models.pick(cat, link)
    if not model:
        raise hf_models.HfLinkError("choose a file: this repo has several models")
    dest = model_dir()
    return hf_models.plan(cat, link.repo, link.revision, model, mmproj, dest, shutil.disk_usage(dest).free)


def fetch(f: Dict[str, Any], dest_dir: str, progress: Callable[[int], None]) -> None:
    """Stream one planned file to a temp file next to its target, hash while writing, move into place only when the
    hash matches (never over an existing file)."""
    target = os.path.join(dest_dir, f["save_as"])
    h = hashlib.sha256()
    tmp = None
    try:
        with _OPENER.open(urllib.request.Request(f["url"], headers=_headers(f["url"])), timeout=60) as r, \
                tempfile.NamedTemporaryFile(dir=dest_dir, delete=False, prefix=".part-") as out:
            tmp = out.name
            while chunk := r.read(1 << 20):
                out.write(chunk)
                h.update(chunk)
                progress(len(chunk))
    except urllib.error.HTTPError as e:
        if tmp:
            os.unlink(tmp)
        raise _http_error(e, f["path"])
    except BaseException:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise
    if h.hexdigest() != f["sha256"]:
        os.unlink(tmp)
        raise hf_models.HfLinkError(f"{f['save_as']}: checksum mismatch -- nothing kept")
    if os.path.exists(target):
        os.unlink(tmp)
        return
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)


# ---------------------------------------------------------------- background jobs (dashboard)

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def jobs() -> List[Dict[str, Any]]:
    with _lock:
        return [dict(j) for j in sorted(_jobs.values(), key=lambda j: -j["started"])][:20]


def start(plan: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        running = _jobs.get(plan["model"])
        if running and running["state"] == "downloading":
            return dict(running)
        job = {"id": plan["model"], "model": plan["model"], "repo": plan["repo"], "state": "downloading",
               "files": [f["save_as"] for f in plan["files"]], "already": plan["already"],
               "size": plan["size"], "done": 0, "error": "", "started": time.time()}
        _jobs[plan["model"]] = job
    if not plan["files"]:
        job["state"] = "installed"
        return dict(job)
    threading.Thread(target=_run, args=(job, plan), daemon=True).start()
    return dict(job)


def _run(job: Dict[str, Any], plan: Dict[str, Any]) -> None:
    def progress(n: int) -> None:
        job["done"] += n
    try:
        for f in plan["files"]:
            fetch(f, plan["dest"], progress)
        job["state"] = "installed"
        logger.info(f"Hugging Face: {plan['model']} downloaded from {plan['repo']}")
        try:
            from backend import events
            events.record("model", "{model} downloaded from Hugging Face", model=plan["model"])
        except Exception:
            pass
    except Exception as e:
        job["state"], job["error"] = "failed", str(e)[:300]
        logger.error(f"Hugging Face download of {plan['model']} failed: {e}")
