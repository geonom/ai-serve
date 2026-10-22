"""Downloading GGUF language models from Hugging Face -- the pure rules (tests: tests/unit/test_hf_models.py).

Owner, 2026-10-22: GGUF models should be downloadable from Hugging Face, with a clear word on which address is
meant. Threat model: the address is pasted by the owner (the dashboard is owner-only), but it is still text from
outside -- so only huggingface.co / hf.co model repos are accepted (no other host, no user@host, no datasets/spaces),
paths are checked segment by segment after decoding (no `..`), and only `.gguf` files that Hugging Face's own tree
API lists *with* a sha256 (its LFS oid) can be chosen; the download is checked against that hash. Files are saved
under their own base name in one model folder -- never outside it, never over an existing file, and nothing is ever
deleted to make room (a download that doesn't fit is refused).

Vocabulary
- link: what the owner pasted, parsed -> repo ("owner/name"), revision, path (one file) or quant (Ollama's
  `hf.co/owner/repo:Q4_K_M`).
- catalog: the repo's GGUF files grouped as models (a split model `X-00001-of-0000N.gguf` counts once, with all
  its parts) and vision projectors (mmproj).
- plan: what will be downloaded (url, sha256, size, saved as) for one chosen model (+ projector).
"""
from __future__ import annotations

import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional

_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REV_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SEG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$")
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<n>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d|Q\d+_[A-Z]+|"
                       r"BF16|F16|F32|MXFP4(?:_MOE)?))(?=$|[-_.])")
ROOM_MARGIN = 1 << 30          # keep 1 GB free besides the download


class HfLinkError(ValueError):
    pass


class Link:
    def __init__(self, repo: str, revision: str = "main", path: Optional[str] = None, quant: Optional[str] = None):
        self.repo, self.revision, self.path, self.quant = repo, revision, path, quant


def _check_path(segments: List[str]) -> str:
    if not segments or any(s in ("", ".", "..") or not _SEG_RE.match(s) for s in segments):
        raise HfLinkError("that file path isn't usable (only plain names inside the repo)")
    path = "/".join(segments)
    if not path.lower().endswith(".gguf"):
        raise HfLinkError("only .gguf files can be downloaded here (look for a repo whose name ends in -GGUF)")
    return path


def parse_link(text: Any) -> Link:
    """An address as people copy it -> Link. See the module docstring for what is accepted."""
    if not isinstance(text, str) or not text.strip():
        raise HfLinkError("paste a Hugging Face address, e.g. https://huggingface.co/unsloth/Qwen3-4B-GGUF")
    text = text.strip()
    if "://" in text:
        u = urllib.parse.urlsplit(text)
        if u.scheme not in ("https", "http") or u.hostname not in _HOSTS or u.username or u.password or u.port:
            raise HfLinkError("only addresses on huggingface.co (or hf.co) are accepted")
        raw = u.path
    elif text.split("/", 1)[0].lower() in _HOSTS:
        raw = text.split("/", 1)[1] if "/" in text else ""
    else:
        raw = text                                    # owner/repo shorthand
    raw = raw.split("?", 1)[0].split("#", 1)[0]
    segments = [urllib.parse.unquote(s) for s in raw.strip("/").split("/")] if raw.strip("/") else []
    if segments and segments[0] in ("datasets", "spaces", "collections", "papers"):
        raise HfLinkError("only model repos can be used here, not datasets or spaces")
    if len(segments) < 2:
        raise HfLinkError("the address needs a repo: owner/name, e.g. unsloth/Qwen3-4B-GGUF")
    owner, name = segments[0], segments[1]
    quant = None
    if ":" in name and len(segments) == 2:              # Ollama style: hf.co/owner/repo:Q4_K_M
        name, quant = name.split(":", 1)
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", quant):
            raise HfLinkError("the part after ':' should be a quantization like Q4_K_M")
    if not (_NAME_RE.match(owner) and _NAME_RE.match(name)):
        raise HfLinkError("the address needs a repo: owner/name, e.g. unsloth/Qwen3-4B-GGUF")
    rest = segments[2:]
    revision, path = "main", None
    if rest:
        if rest[0] not in ("blob", "resolve", "tree") or len(rest) < 2:
            raise HfLinkError("use the repo's page or a file's page (…/blob/main/<file>.gguf)")
        revision = rest[1]
        if not _REV_RE.match(revision):
            raise HfLinkError("that revision isn't usable")
        if rest[0] in ("blob", "resolve"):
            path = _check_path(rest[2:])
        elif len(rest) > 2 and any(s in ("", ".", "..") for s in rest[2:]):
            raise HfLinkError("that file path isn't usable (only plain names inside the repo)")
    return Link(f"{owner}/{name}", revision, path, quant)


def _quant(stem: str) -> Optional[str]:
    found = _QUANT_RE.findall(stem)
    return found[-1].upper() if found else None


def catalog(tree: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Hugging Face's tree listing -> {"models": [...], "mmproj": [...]}; only GGUF files with a sha256."""
    files = {}
    for e in tree or []:
        p = e.get("path") if isinstance(e, dict) else None
        lfs = e.get("lfs") if isinstance(e, dict) else None
        if e.get("type") != "file" or not isinstance(p, str) or not p.lower().endswith(".gguf"):
            continue
        oid = (lfs or {}).get("oid")
        if not (isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{64}", oid)):
            continue                                    # no checksum from Hugging Face: never offered
        try:
            _check_path(p.split("/"))
        except HfLinkError:
            continue
        files[p] = {"path": p, "size": int((lfs or {}).get("size") or e.get("size") or 0), "sha256": oid}
    models, mmproj, splits = [], [], {}
    for p, f in sorted(files.items()):
        base = p.rsplit("/", 1)[-1]
        if "mmproj" in base.lower():
            mmproj.append(f)
            continue
        if base.lower().startswith("mtp-"):
            continue
        m = _SPLIT_RE.match(base)
        if m:
            key = (p.rsplit("/", 1)[0] if "/" in p else "", m["stem"], m["total"])
            splits.setdefault(key, []).append(f)
            continue
        models.append({**f, "parts": 1, "files": [f], "quant": _quant(base[:-5])})
    for (_, stem, total), parts in splits.items():
        parts.sort(key=lambda f: f["path"])
        if len(parts) != int(total):
            continue                                    # incomplete set: can't be loaded
        models.append({"path": parts[0]["path"], "size": sum(f["size"] for f in parts), "sha256": parts[0]["sha256"],
                       "parts": len(parts), "files": parts, "quant": _quant(stem)})
    models.sort(key=lambda m: m["path"])
    return {"models": models, "mmproj": mmproj}


def pick(cat: Dict[str, Any], link: Link) -> Optional[str]:
    """The model a link names: its file, or its Ollama-style quant; None = the owner chooses from the list."""
    if link.path:
        for m in cat["models"]:
            if link.path in [f["path"] for f in m["files"]]:
                return m["path"]
        raise HfLinkError(f"{link.path} isn't a model file of this repo (or Hugging Face gives no checksum for it)")
    if link.quant:
        q = link.quant.upper()
        hits = [m["path"] for m in cat["models"] if (m["quant"] or "") == q] or \
               [m["path"] for m in cat["models"] if (m["quant"] or "").replace("UD-", "") == q.replace("UD-", "")]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise HfLinkError(f"no {link.quant} file in this repo")
        raise HfLinkError(f"several {link.quant} files -- choose one: {', '.join(hits)}")
    return None


def _mmproj_name(model_save_as: str, mmproj_path: str) -> str:
    """Saved so find_mmproj_path() pairs it: '<model stem>-mmproj-<rest>.gguf'."""
    stem = model_save_as[:-5]
    m = _SPLIT_RE.match(model_save_as)
    if m:
        stem = m["stem"]
    rest = re.sub(r"(?i)mmproj", "", mmproj_path.rsplit("/", 1)[-1][:-5]).strip("-_.") or "projector"
    return f"{stem}-mmproj-{rest}.gguf"


def plan(cat: Dict[str, Any], repo: str, revision: str, model_path: str, mmproj_path: Optional[str],
         dest_dir: str, free_bytes: int) -> Dict[str, Any]:
    """What to download for one model (+ projector) into dest_dir. Raises HfLinkError when it can't be done."""
    model = next((m for m in cat["models"] if m["path"] == model_path), None)
    if model is None:
        raise HfLinkError("choose one of the repo's model files")
    files = [(f, f["path"].rsplit("/", 1)[-1]) for f in model["files"]]
    if mmproj_path:
        proj = next((p for p in cat["mmproj"] if p["path"] == mmproj_path), None)
        if proj is None:
            raise HfLinkError("choose one of the repo's vision projector (mmproj) files")
        files.append((proj, _mmproj_name(files[0][1], proj["path"])))
    out, already = [], []
    for f, save_as in files:
        if os.path.exists(os.path.join(dest_dir, save_as)):
            already.append(save_as)
            continue
        out.append({"path": f["path"], "save_as": save_as, "size": f["size"], "sha256": f["sha256"],
                    "url": f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(f['path'])}"})
    need = sum(f["size"] for f in out)
    if out and need + ROOM_MARGIN > free_bytes:
        raise HfLinkError(f"not enough room: needs {need / 2**30:.1f} GB, {max(0, free_bytes) / 2**30:.1f} GB free in "
                          f"{dest_dir}. Nothing was deleted -- free some space yourself, or set LLAMA_MODEL_DIRS.")
    return {"repo": repo, "revision": revision, "model": files[0][1], "files": out, "already": already,
            "dest": dest_dir, "size": need}
