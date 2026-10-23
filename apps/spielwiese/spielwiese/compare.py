"""Comparisons ("Vergleich"): the same prompt and seed through several image models, optionally over
several step counts and seeds, side by side.

`plan()` (pure) turns an untrusted request into cells, checked against the gateway's catalog
(GET /v1/images/models): only installed models (one lacking the chosen aspect ratio gets its closest one), steps
only where a model can change them (Iris can't) and within its maximum, at most MAX_CELLS images.
`Runner` works a plan off in a background thread, one image after the other (one GPU), as the person
who asked: the gateway gets their client address as X-Forwarded-For, like every other Spielwiese
call, and each image lands in *their* gallery. It keeps going when the browser closes; the state is
written to the person's folder after every cell, so finished comparisons can be reopened later.
One running comparison per person. Tests: tests/test_compare.py, tests/test_compare_runner.py.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import random
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from spielwiese.gallery import ASPECTS, Gallery, GalleryError, decode_image, new_id, person_dir, valid_id
from spielwiese.quota import Ledger, Limits, QuotaExceeded

QUOTA_WAIT_S = 900   # a cell waits this long for the person's own other GPU call to finish
BUSY_WAIT_S = 900    # ... and this long for room on the GPU (the gateway's 507), trying every BUSY_RETRY_S
BUSY_RETRY_S = 10

MAX_CELLS = 32
MAX_MODELS = 8
MAX_STEPS = 6
MAX_SEEDS = 4
MAX_PROMPT = 4000
CELL_TIMEOUT_S = 1800
log = logging.getLogger("spielwiese.compare")


class CompareError(ValueError):
    status = 400


def _ints(raw: Any, name: str, lo: int, hi: int, most: int) -> List[int]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > most:
        raise CompareError(f"{name} must be a list of at most {most} numbers")
    for v in raw:
        if type(v) is not int or not lo <= v <= hi:
            raise CompareError(f"{name} must be whole numbers {lo}..{hi}")
    return sorted(set(raw))


def closest_aspect(wanted: str, offered: List[str]) -> str:
    """`wanted` if the model has it, else the offered ratio nearest to it (on a log scale, so 9:16 vs
    16:9 count as equally far from 1:1). A model missing one format shouldn't block a comparison."""
    if wanted in offered:
        return wanted
    import math

    def r(a: str) -> float:
        w, h = a.split(":")
        return math.log(int(w) / int(h))
    return min(offered, key=lambda a: abs(r(a) - r(wanted)))


def plan(body: Any, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise CompareError("expected a JSON object")
    unknown = set(body) - {"prompt", "models", "steps", "seeds", "aspect_ratio"}
    if unknown:
        raise CompareError(f"unknown fields: {', '.join(sorted(unknown))}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
        raise CompareError(f"prompt must be text, 1..{MAX_PROMPT} characters")
    models = body.get("models")
    if not isinstance(models, list) or not models or len(models) > MAX_MODELS:
        raise CompareError(f"choose 1..{MAX_MODELS} models")
    aspect = body.get("aspect_ratio") or "1:1"
    if aspect not in ASPECTS:
        raise CompareError(f"aspect_ratio must be one of {', '.join(ASPECTS)}")
    rows = []
    for m in dict.fromkeys(models):
        row = catalog.get(m) if isinstance(m, str) else None
        if row is None:
            raise CompareError(f"unknown model {m!r}")
        if not row.get("available"):
            raise CompareError(f"{row['label']} is not installed")
        if not row.get("aspect_ratios"):
            raise CompareError(f"{row['label']} lists no aspect ratio")
        rows.append(row)
    steps = _ints(body.get("steps"), "steps", 1, 60, MAX_STEPS)
    seeds = _ints(body.get("seeds"), "seeds", 0, 2**31 - 1, MAX_SEEDS) or [random.randint(0, 2**31 - 1)]
    cells = []
    for row in rows:
        st = row["steps"]
        if st["adjustable"]:
            values = [s for s in steps if s <= st["max"]] or [st["default"]]
        else:
            values = [st["default"]]
        for s in values:
            for seed in seeds:
                cells.append({"model": row["id"], "label": row["label"], "steps": s, "seed": seed,
                              "send_steps": bool(st["adjustable"]),
                              # one lane per service: different services run at the same time (on different
                              # GPUs they don't wait for each other), one service's cells one after the other
                              "lane": row.get("provider") or row["id"],
                              "aspect_ratio": closest_aspect(aspect, row["aspect_ratios"])})
    if len(cells) > MAX_CELLS:
        raise CompareError(f"{len(cells)} images -- at most {MAX_CELLS} per comparison (fewer models, steps or seeds)")
    return {"prompt": prompt.strip(), "aspect_ratio": aspect, "cells": cells}


Post = Callable[..., Tuple[int, Dict[str, Any]]]


def gateway_post(gateway: Tuple[str, int]) -> Post:
    """POST /v1/images/generations as the person (their client address goes along), tagged with the cell's
    request id so a cancel can stop exactly this image (POST /v1/images/cancel)."""
    def post(body: Dict[str, Any], peer: str, request_id: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
        conn = http.client.HTTPConnection(*gateway, timeout=CELL_TIMEOUT_S)
        try:
            data = json.dumps(body).encode()
            headers = {"Content-Type": "application/json", "Content-Length": str(len(data)), "X-Forwarded-For": peer}
            if request_id:
                headers["X-Request-Id"] = request_id
            conn.request("POST", "/v1/images/generations", body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, {"detail": raw[:300].decode("utf-8", "replace")}
        finally:
            conn.close()
    return post


def gateway_cancel(gateway: Tuple[str, int]) -> Callable[[str, str], None]:
    """POST /v1/images/cancel {request_id} as the person -- stops the image the runner is waiting for."""
    def cancel(request_id: str, peer: str) -> None:
        conn = http.client.HTTPConnection(*gateway, timeout=15)
        try:
            data = json.dumps({"request_id": request_id}).encode()
            conn.request("POST", "/v1/images/cancel", body=data,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(data)),
                                  "X-Forwarded-For": peer})
            conn.getresponse().read()
        except OSError as e:
            log.warning("cancel didn't reach the gateway: %s", e)
        finally:
            conn.close()
    return cancel


class Runner:
    def __init__(self, root: str, gallery: Gallery, post: Post, ledger: Optional[Ledger] = None,
                 cancel_remote: Optional[Callable[[str, str], None]] = None):
        self.root, self.gallery, self.post, self.ledger = root, gallery, post, ledger
        self.cancel_remote = cancel_remote
        self._inflight: Dict[str, Dict[str, str]] = {}    # comparison id -> {request id: peer} of images now
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._running: Dict[str, str] = {}       # person dir -> comparison id
        self._cancel: set = set()

    def _path(self, sub: str, cid: str) -> str:
        return os.path.join(person_dir(self.root, sub), "comparisons", f"{cid}.json")

    def _write(self, sub: str, job: Dict[str, Any]) -> None:
        path = self._path(sub, job["id"])
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        with self._write_lock:   # lanes write the same job file
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as f:
                json.dump(job, f)
            os.replace(tmp, path)

    def start(self, sub: str, peer: str, p: Dict[str, Any], wait: bool = False,
              limits: Optional[Limits] = None) -> Dict[str, Any]:
        key = person_dir(self.root, sub)
        if self.ledger is not None and limits is not None and limits.images_per_day is not None:
            left = limits.images_per_day - self.ledger.usage(sub)["images"]
            if len(p["cells"]) > left:
                raise QuotaExceeded(f"This comparison needs {len(p['cells'])} images; your quota has {max(0, left)} "
                                    f"left today (of {limits.images_per_day}).")
        with self._lock:
            if key in self._running:
                raise CompareError("a comparison is still running -- wait for it or cancel it")
            job = {"id": new_id(), "created": time.time(), "status": "running", "prompt": p["prompt"],
                   "aspect_ratio": p["aspect_ratio"],
                   "cells": [{**{k: c[k] for k in ("model", "label", "steps", "seed", "aspect_ratio")}, "status": "pending"}
                             for c in p["cells"]]}
            self._running[key] = job["id"]
        self._write(sub, job)
        t = threading.Thread(target=self._run, args=(sub, peer, p, job, key, limits), daemon=True)
        t.start()
        if wait:
            t.join()
        return job

    sleep = staticmethod(time.sleep)
    busy_wait_s = BUSY_WAIT_S

    def _post_when_free(self, body: Dict[str, Any], peer: str, cell: Dict[str, Any], sub: str,
                        job: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """The gateway answers "no room on the GPU" (507) or "not running yet" (503) at once, or isn't reachable
        for a moment (restarting), and expects the caller to retry -- a comparison runs in the background, so it waits (e.g. while an earlier, slow cell
        still holds the GPU) instead of failing every remaining cell."""
        waited = 0
        while True:
            t0 = time.time()
            rid = secrets.token_hex(16)
            with self._lock:
                self._inflight.setdefault(job["id"], {})[rid] = peer
            try:
                status, data = self.post(body, peer, request_id=rid)
            except OSError as e:
                # the gateway isn't answering (restarting, say): wait it out like a busy GPU
                status, data = 503, {"detail": f"ai-serve-ui gateway not reachable ({e})"}
            finally:
                with self._lock:
                    self._inflight.get(job["id"], {}).pop(rid, None)
            if status not in (503, 507) or waited >= self.busy_wait_s or job["id"] in self._cancel:
                if isinstance(data, dict):
                    data["_t0"] = t0
                return status, data
            cell["waiting"] = "GPU belegt – warte"
            self._write(sub, job)
            for _ in range(BUSY_RETRY_S):          # 1 s slices: a cancel takes effect at once
                if job["id"] in self._cancel:
                    break
                self.sleep(1)
            waited += BUSY_RETRY_S
            cell.pop("waiting", None)
            if job["id"] in self._cancel:          # no further try once cancelled
                if isinstance(data, dict):
                    data["_t0"] = t0
                return status, data

    def _ticket(self, sub: str, limits: Optional[Limits], job_id: str):
        """Reserve one image in the person's quota; waits while their own other GPU call runs."""
        if self.ledger is None or limits is None:
            return None
        deadline = time.time() + QUOTA_WAIT_S
        while True:
            try:
                return self.ledger.begin(sub, limits, "image", images=1)
            except QuotaExceeded as e:
                if not e.busy or time.time() > deadline or job_id in self._cancel:
                    raise
                time.sleep(2)

    def _run(self, sub: str, peer: str, p: Dict[str, Any], job: Dict[str, Any], key: str,
             limits: Optional[Limits] = None) -> None:
        """One thread per lane (service); a lane works its cells one after the other."""
        shared = {"stop": None}
        lanes: Dict[str, List[int]] = {}
        for i, spec in enumerate(p["cells"]):
            lanes.setdefault(spec.get("lane", "default"), []).append(i)
        try:
            threads = [threading.Thread(target=self._run_lane, args=(sub, peer, p, job, idx, limits, shared),
                                        daemon=True) for idx in lanes.values()]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            job["status"] = "cancelled" if job["id"] in self._cancel else "done"
        except Exception as e:   # never leave a comparison "running" forever
            log.exception("comparison failed")
            job["status"], job["error"] = "failed", str(e)[:300]
        finally:
            self._write(sub, job)
            with self._lock:
                self._running.pop(key, None)
                self._cancel.discard(job["id"])
                self._inflight.pop(job["id"], None)

    def _run_lane(self, sub: str, peer: str, p: Dict[str, Any], job: Dict[str, Any], indices: List[int],
                  limits: Optional[Limits], shared: Dict[str, Any]) -> None:
        for i in indices:
            cell, spec = job["cells"][i], p["cells"][i]
            try:
                self._run_cell(sub, peer, p, job, cell, spec, limits, shared)
            except Exception as e:   # one cell's surprise must not stop its lane
                log.exception("comparison cell failed")
                cell.update(status="failed", error=str(e)[:300] or type(e).__name__)
                self._write(sub, job)

    def _run_cell(self, sub: str, peer: str, p: Dict[str, Any], job: Dict[str, Any], cell: Dict[str, Any],
                  spec: Dict[str, Any], limits: Optional[Limits], shared: Dict[str, Any]) -> None:
        if job["id"] in self._cancel:
            cell["status"] = "cancelled"
            self._write(sub, job)
            return
        if shared["stop"]:
            cell.update(status="failed", error=shared["stop"])
            self._write(sub, job)
            return
        try:
            ticket = self._ticket(sub, limits, job["id"])
        except QuotaExceeded as e:
            shared["stop"] = str(e)
            cell.update(status="failed", error=shared["stop"])
            self._write(sub, job)
            return
        cell["status"] = "running"
        self._write(sub, job)
        body = {"model": spec["model"], "prompt": p["prompt"], "aspect_ratio": spec["aspect_ratio"],
                "seed": spec["seed"], "response_format": "b64_json"}
        if spec["send_steps"]:
            body["steps"] = spec["steps"]
        t0 = time.time()
        try:
            status, data = self._post_when_free(body, peer, cell, sub, job)
            t0 = data.pop("_t0", t0) if isinstance(data, dict) else t0   # time of the attempt that ran
            if job["id"] in self._cancel and (status in (503, 507) or (status == 409 and "cancel" in str(data).lower())):
                cell["status"] = "cancelled"
                return
            if status != 200:
                detail = data.get("detail") if isinstance(data, dict) else None
                raise CompareError(detail if isinstance(detail, str) else f"gateway answered {status}")
            image, _ = decode_image(data["data"][0]["b64_json"])
            entry = self.gallery.save(sub, image, {
                "prompt": p["prompt"], "model": spec["model"], "label": spec["label"], "steps": spec["steps"],
                "seed": spec["seed"], "aspect_ratio": spec["aspect_ratio"], "seconds": time.time() - t0,
                "source": "vergleich", "comparison": job["id"]})
            cell.update(status="done", image=entry["id"], seconds=entry.get("seconds"))
        except (CompareError, GalleryError, KeyError, IndexError, TypeError, OSError) as e:
            cell.update(status="failed", error=str(e)[:300] or type(e).__name__)
        finally:
            if ticket is not None:
                self.ledger.end(ticket, seconds=time.time() - t0, images=1 if cell["status"] == "done" else 0)
            self._write(sub, job)

    def get(self, sub: str, cid: str) -> Optional[Dict[str, Any]]:
        if not valid_id(cid):
            return None
        try:
            with open(self._path(sub, cid)) as f:
                job = json.load(f)
        except (OSError, ValueError):
            return None
        if job.get("status") == "running" and self._running.get(person_dir(self.root, sub)) != cid:
            job["status"] = "interrupted"   # the server restarted mid-way
        return job

    def running(self) -> int:
        """Comparisons generating right now (ai-serve doesn't restart the Spielwiese for an update meanwhile)."""
        return len(self._running)

    def list(self, sub: str, limit: int = 30) -> List[Dict[str, Any]]:
        d = os.path.join(person_dir(self.root, sub), "comparisons")
        try:
            ids = sorted((n[:-5] for n in os.listdir(d) if n.endswith(".json") and valid_id(n[:-5])), reverse=True)
        except OSError:
            return []
        out = []
        for cid in ids[:max(1, min(limit, 200))]:
            job = self.get(sub, cid)
            if job:
                out.append({k: job[k] for k in ("id", "created", "status", "prompt", "aspect_ratio")}
                           | {"cells": len(job["cells"]), "models": list(dict.fromkeys(c["label"] for c in job["cells"]))})
        return out

    def cancel(self, sub: str, cid: str) -> bool:
        with self._lock:
            if self._running.get(person_dir(self.root, sub)) != cid:
                return False
            self._cancel.add(cid)
            running = list(self._inflight.get(cid, {}).items())
        if self.cancel_remote is not None:
            # stop the images being generated now (one per lane), not only the cells after them -- in the
            # background: the caller (the cancel button) gets its answer at once
            for rid, peer in running:
                threading.Thread(target=self.cancel_remote, args=(rid, peer), daemon=True).start()
        return True
