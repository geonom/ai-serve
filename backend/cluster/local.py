"""Local GPU manager (I/O): measures who holds VRAM on this machine's GPUs, makes
room by unloading idle consumers across all kinds, and keeps the owner's pins.
Rules: backend/cluster/consumers.py (tests/unit/test_gpu_consumers.py).

Consumers and how they're unloaded (never deleting files):
  llm:<port>     llama-server instance   -> stop_server_by_port
  <plugin>:<worker>  a GPU plugin's worker (its status.workers) -> the plugin's release (backend/plugins/registry.py)
  training:<id>  training job            -> never
  (a gpu-exclusive plugin's workers, e.g. Strata's server) -> never automatically; it owns every GPU while it runs
"""
import asyncio
import dataclasses
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import psutil
from backend.procs import processes

from backend.cluster.consumers import (Consumer, Reservation, Reservations, attribute_vram, gpu_view, unified_procs,
                                      plan_exclusive, plan_room)

logger = logging.getLogger("ai-serve-ui.gpu")


def _nvidia_procs() -> List[Tuple[int, int, int]]:
    """(pid, gpu index, MiB) for every compute process."""
    try:
        uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10).stdout
        idx = {u.strip(): int(i) for i, u in (l.split(",") for l in uuids.strip().splitlines() if "," in l)}
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[2] in idx:
            try:
                procs.append((int(parts[0]), idx[parts[2]], int(float(parts[1]))))
            except ValueError:
                pass
    return procs


def _unified_memory() -> bool:
    import platform
    return platform.system() == "Darwin" and "arm64" in platform.machine()


def _footprint_mb(pid: int) -> Optional[int]:
    """Memory a process uses on Apple Silicon: its physical footprint (what Activity Monitor shows; Metal buffers are
    charged to it) -- at least its resident size, since llama.cpp maps the model file and Metal uses those pages
    directly. None when the process is gone. Not verified on a real Mac yet (written 2026-10-22)."""
    try:
        rss = psutil.Process(pid).memory_info().rss
    except (psutil.Error, OSError):
        return None
    footprint = 0
    try:
        import ctypes

        class _RusageV0(ctypes.Structure):    # <sys/resource.h> rusage_info_v0
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
                "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]
        info = _RusageV0()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(int(pid), 0, ctypes.byref(info)) == 0:
            footprint = int(info.phys_footprint)
    except Exception:
        pass
    return max(rss, footprint) >> 20


def _children(pid: int) -> List[int]:
    try:
        return [c.pid for c in psutil.Process(pid).children()]
    except (psutil.Error, OSError):
        return []


def _parents(pids) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for pid in pids:
        cur = pid
        for _ in range(64):
            if cur in out or cur <= 1:
                break
            try:
                parent = psutil.Process(cur).ppid()
            except (psutil.Error, OSError):
                break
            out[cur] = parent
            cur = parent
    return out


def _cmdline_pid(fragment: str, cwd: Optional[str] = None) -> Optional[int]:
    """First process whose cmdline contains `fragment` (and, if given, whose working directory is `cwd` --
    for programs started as a bare `main.py ...`, whose cmdline names nothing about the project)."""
    want = os.path.realpath(cwd) if cwd else None
    for p, pinfo in processes(["pid", "cmdline", "cwd"] if want else ["pid", "cmdline"]):
        try:
            if fragment in " ".join(pinfo["cmdline"] or []) and (
                    want is None or (pinfo.get("cwd") and os.path.realpath(pinfo["cwd"]) == want)):
                return pinfo["pid"]
        except (psutil.Error, TypeError):
            continue
    return None


class LocalGpuManager:
    def __init__(self, pins_file: str):
        self.pins_file = pins_file
        self._locks: Dict[int, asyncio.Lock] = {}
        self.reservations = Reservations(hold_s=5.0, bound_max_s=120.0)
        self._procs: List[Tuple[int, int, int]] = []
        self._parents: Dict[int, int] = {}

    # --- pins (the owner's choice, persisted)
    def pins(self) -> FrozenSet[str]:
        try:
            with open(self.pins_file, encoding="utf-8") as f:
                return frozenset(json.load(f).get("pinned") or [])
        except (FileNotFoundError, ValueError):
            return frozenset()

    def set_pin(self, consumer_id: str, pinned: bool) -> FrozenSet[str]:
        pins = set(self.pins())
        (pins.add if pinned else pins.discard)(consumer_id)
        tmp = self.pins_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pinned": sorted(pins)}, f, indent=2)
        os.replace(tmp, self.pins_file)
        return frozenset(pins)

    # --- measuring
    async def consumers(self) -> List[Consumer]:
        from backend.state import state
        out: List[Consumer] = []
        for port, info in list(state.managed_processes.items()):
            proc = info.get("process")
            if not proc or proc.poll() is not None:
                continue
            model = os.path.basename(info.get("model", "?"))
            out.append(Consumer(id=f"llm:{port}", kind="llm", label=model, pin_key=f"model:{model}",
                                root_pids={proc.pid}, last_used=info.get("last_used") or 0,
                                # still loading = busy: it has no requests yet but must
                                # not be unloaded by the next load
                                busy=info.get("active_requests", 0) > 0 or bool(info.get("loading"))))
        try:
            from backend.plugins import registry
            out += await registry.gpu_consumers()
        except Exception as e:
            logger.warning(f"GPU manager: plugin consumers unreadable: {e}")  # their VRAM counts as "other"
        try:
            from backend import train_client
            for job in await train_client.running_jobs():
                out.append(Consumer(id=f"training:{job.get('id') or job.get('job_id')}", kind="training",
                                    label=f"training {job.get('job_type', '')}", root_pids={int(job["pid"])},
                                    busy=True, evictable_kind=False))
        except Exception:
            pass
        return out

    async def snapshot(self):
        from backend.model_manager import fetch_gpu_status
        gpus = (await fetch_gpu_status()).get("gpus", [])
        if _unified_memory():   # no nvidia-smi: measure the known consumers' processes (consumers.unified_procs)
            consumers = await self.consumers()
            procs = await asyncio.to_thread(unified_procs, consumers, _children, _footprint_mb)
            parents = await asyncio.to_thread(_parents, [p for p, _, _ in procs])
            self._procs, self._parents = procs, parents
            return gpus, consumers, attribute_vram(procs, parents, consumers)
        procs = await asyncio.to_thread(_nvidia_procs)
        parents = await asyncio.to_thread(_parents, [p for p, _, _ in procs])
        self._procs, self._parents = procs, parents
        consumers = await self.consumers()
        return gpus, consumers, attribute_vram(procs, parents, consumers)

    def _mark_loading(self, consumers: List[Consumer]) -> List[Consumer]:
        """A consumer whose process holds an open reservation is still loading:
        busy, so the next request can't unload it mid-load."""
        loading = {r.pid for r in self.reservations.all() if r.pid}
        return [dataclasses.replace(c, busy=True) if c.root_pids & loading and not c.busy else c
                for c in consumers]

    def _exclusive_holder(self, gpu: int, consumers: List[Consumer]) -> Optional[str]:
        pids = {r.pid for r in self.reservations.all() if r.pid}
        alive = {p for p in pids if psutil.pid_exists(p)}
        holder = self.reservations.exclusive_holder(gpu, time.time(), alive)
        if holder:
            return holder
        # e.g. an exclusive plugin's server started elsewhere, still loading (no VRAM yet): holds every GPU
        return next((c.label for c in consumers if c.kind == "exclusive"), None)

    @staticmethod
    def _refuse_exclusive(holder: str, purpose: str):
        from fastapi import HTTPException
        raise HTTPException(status_code=507, detail=(
            f"{holder} holds the GPUs exclusively -- {purpose} can't load now (it would run both out of "
            f"memory). Stop {holder} first (Apps & Services, or POST /api/plugins/<id>/release)."))

    def _outstanding(self, gpu: int) -> int:
        """Memory promised on `gpu` but not yet allocated by the reserving processes
        (their VRAM incl. children, from the last snapshot)."""
        pids = {r.pid for r in self.reservations.all() if r.pid}
        measured: Dict[int, int] = {}
        for pid, g, mb in self._procs:
            if g != gpu:
                continue
            cur, seen = pid, set()
            while cur and cur not in seen:
                if cur in pids:
                    measured[cur] = measured.get(cur, 0) + mb
                    break
                seen.add(cur)
                cur = self._parents.get(cur)
        alive = {p for p in pids if psutil.pid_exists(p)}
        return self.reservations.outstanding(gpu, time.time(), alive, measured)

    async def view(self) -> List[Dict[str, Any]]:
        gpus, consumers, att = await self.snapshot()
        view = gpu_view(gpus, self._mark_loading(consumers), att, self.pins())
        for g in view:
            g["reserved_mb"] = self._outstanding(g["index"])
            g["reservations"] = [{"id": r.id, "mb": r.mb, "purpose": r.purpose, "pid": r.pid}
                                 for r in self.reservations.all() if r.gpu == g["index"]]
        return view

    # --- acting
    async def evict(self, consumer_id: str, consumers: Optional[List[Consumer]] = None) -> None:
        c = next((c for c in (consumers or await self.consumers()) if c.id == consumer_id), None)
        if c is None:
            raise LookupError(f"{consumer_id} is not using a GPU")
        if c.busy or not c.evictable_kind:
            raise PermissionError(f"{c.label} is busy -- not unloading it")
        if c.plugin:
            from backend.plugins import registry
            await registry.release(c.id)
        elif c.kind == "llm":
            from backend.model_manager import stop_server_by_port
            await stop_server_by_port(int(c.id.split(":", 1)[1]))
        else:
            raise PermissionError(f"{c.label} can't be unloaded from here")
        logger.info(f"GPU manager: unloaded {c.label} ({c.id})")
        from backend import events
        events.record("gpu", "{label} unloaded from GPU memory", label=c.label)

    async def room(self) -> Dict[int, int]:
        """MB that could be had on each GPU right now without unloading anyone: hardware-free minus what
        is promised to loads in progress; 0 on a GPU someone holds exclusively (an exclusive plugin)."""
        gpus, consumers, _ = await self.snapshot()
        consumers = self._mark_loading(consumers)
        out: Dict[int, int] = {}
        for g in gpus:
            i = int(g["index"])
            out[i] = 0 if self._exclusive_holder(i, consumers) else max(0, int(g.get("free") or 0) - self._outstanding(i))
        return out

    async def reserve(self, gpu: int, need_mb: int, purpose: str
                      ) -> Tuple[Optional[Reservation], List[str], str]:
        """Answer fast, yes or no (user decision: no waiting list -- callers retry).
        Yes = idle, unpinned consumers unloaded as needed (LRU, any kind) and
        `need_mb` reserved: held 5 s until confirm(pid), then bound to that process.
        Promised-but-unallocated memory of other reservations counts as used.
        Returns (reservation or None, unloaded ids, reason)."""
        lock = self._locks.setdefault(gpu, asyncio.Lock())
        async with lock:
            gpus, consumers, att = await self.snapshot()
            consumers = self._mark_loading(consumers)
            holder = self._exclusive_holder(gpu, consumers)
            if holder:
                self._refuse_exclusive(holder, purpose)
            reserved = self._outstanding(gpu)
            plan = plan_room(gpu, need_mb, gpus, consumers, att, self.pins(), reserved_mb=reserved)
            if not plan.ok:
                return None, [], plan.reason
            done = []
            for cid in plan.evict:
                try:
                    await self.evict(cid, consumers)
                    done.append(cid)
                except Exception as e:
                    logger.warning(f"GPU manager: could not unload {cid} for {purpose}: {e}")
            if plan.evict:
                # Unloaded VRAM is handed back asynchronously -- give it a moment (this
                # is about our own unloads finishing, not waiting for anyone busy)
                from backend.model_manager import fetch_gpu_status
                for _ in range(20):
                    g = next((g for g in (await fetch_gpu_status()).get("gpus", []) if int(g["index"]) == gpu), None)
                    if g and int(g.get("free") or 0) - reserved >= need_mb:
                        break
                    await asyncio.sleep(0.5)
                else:
                    return None, done, f"GPU {gpu}: unloaded {done} but still short of {need_mb} MB"
                logger.info(f"GPU manager: made {need_mb} MB room on GPU {gpu} for {purpose} by unloading {done}")
            res = self.reservations.add(gpu, need_mb, purpose, time.time())
            return res, done, ""

    async def raise_if_exclusive(self, gpu: Optional[int], purpose: str) -> None:
        """507 when a consumer (a gpu-exclusive plugin's server) holds `gpu` -- or any GPU, if None --
        exclusively. For starts that don't reserve memory themselves (the ComfyUI
        process; its models reserve per job)."""
        consumers: List[Consumer] = []
        try:
            from backend.plugins import registry
            consumers = [c for c in await registry.gpu_consumers() if c.kind == "exclusive"]
        except Exception:
            pass
        from backend.model_manager import fetch_gpu_status
        gpus = [gpu] if gpu is not None else [g["index"] for g in (await fetch_gpu_status()).get("gpus", [])]
        for g in gpus:
            holder = self._exclusive_holder(g, consumers)
            if holder:
                self._refuse_exclusive(holder, purpose)

    async def reserve_exclusive(self, gpus: List[int], purpose: str
                                ) -> Tuple[Optional[List[Reservation]], List[str], str]:
        """The whole of `gpus` for one consumer (a gpu-exclusive plugin, e.g. Strata): every idle, unpinned
        consumer on them is unloaded; busy/pinned ones (or other programs beyond a
        bare CUDA context) refuse the whole request -- then nothing is unloaded.
        Confirm each reservation with the started PID; they last as long as it lives."""
        locks = [self._locks.setdefault(g, asyncio.Lock()) for g in sorted(gpus)]
        for l in locks:
            await l.acquire()
        try:
            snap_gpus, consumers, att = await self.snapshot()
            consumers = self._mark_loading(consumers)
            plans = {}
            for g in gpus:
                holder = self._exclusive_holder(g, consumers)
                if holder:
                    return None, [], f"GPU {g} is already held exclusively by {holder}"
                if self._outstanding(g):
                    return None, [], f"GPU {g}: another load is in progress -- retry in a moment"
                plans[g] = plan_exclusive(g, snap_gpus, consumers, att, self.pins())
            refused = [p.reason for p in plans.values() if not p.ok]
            if refused:
                return None, [], "; ".join(refused)
            done: List[str] = []
            for cid in dict.fromkeys(c for p in plans.values() for c in p.evict):
                try:
                    await self.evict(cid, consumers)
                    done.append(cid)
                except Exception as e:
                    logger.warning(f"GPU manager: could not unload {cid} for {purpose}: {e}")
            if done:
                logger.info(f"GPU manager: unloaded {done} so {purpose} can take GPUs {gpus}")
                await asyncio.sleep(2)  # let the unloaded processes hand their memory back
            now = time.time()
            return [self.reservations.add(g, 0, purpose, now, exclusive=True) for g in gpus], done, ""
        finally:
            for l in reversed(locks):
                l.release()

    def confirm(self, reservation_id: str, pid: int) -> bool:
        return self.reservations.confirm(reservation_id, pid, time.time())

    def release(self, reservation_id: str) -> None:
        self.reservations.release(reservation_id)

    async def make_room(self, gpu: int, need_mb: int, purpose: str) -> Tuple[bool, List[str], str]:
        """Room without keeping a reservation (kept for callers that don't start a
        process themselves)."""
        res, done, reason = await self.reserve(gpu, need_mb, purpose)
        if res:
            self.release(res.id)
        return res is not None, done, reason


def _pins_path() -> str:
    from backend.config import _PROJECT_DIR
    return os.path.join(_PROJECT_DIR, "gpu_pins.json")


MANAGER = LocalGpuManager(_pins_path())
