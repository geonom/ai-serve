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
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import psutil
from backend.procs import processes
from backend import gpu_ids

from backend.cluster.consumers import (Consumer, NeedBook, Reservation, Reservations, attribute_vram, gpu_view, unified_procs,
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


def _spent(consumers: List[Consumer], tried: set) -> FrozenSet[str]:
    """Consumers already unloaded for this request. A plugin whose only worker was released may come back as the
    bare plugin (`<plugin>` instead of `<plugin>:<worker>`) -- that is the same process."""
    plugins = {c.plugin for c in consumers if c.id in tried and c.plugin} | {t.split(":", 1)[0] for t in tried}
    return frozenset(c.id for c in consumers if c.id in tried or (c.plugin and c.id == c.plugin and c.plugin in plugins))


def _may_exit(ids: List[str], consumers: List[Consumer]) -> bool:
    """Only idle plugin processes: every one is a plugin consumer, and nothing of that plugin is busy (on any GPU)."""
    by_id = {c.id: c for c in consumers}
    plugins = {by_id[i].plugin for i in ids if i in by_id}
    if any(i not in by_id or not by_id[i].plugin or not by_id[i].evictable_kind for i in ids):
        return False
    return not any(c.busy for c in consumers if c.plugin in plugins)


class LocalGpuManager:
    def __init__(self, pins_file: str, needs_file: Optional[str] = None):
        self.pins_file = pins_file
        self.needs_file = needs_file
        self._locks: Dict[int, asyncio.Lock] = {}
        self.reservations = Reservations(hold_s=5.0, bound_max_s=120.0)
        self._procs: List[Tuple[int, int, int]] = []
        self._parents: Dict[int, int] = {}
        self.needs = NeedBook()
        self.load_needs()

    # --- learned needs (consumers.NeedBook; persisted, so a restart keeps what was learned)
    def load_needs(self) -> None:
        if not self.needs_file:
            return
        try:
            with open(self.needs_file, encoding="utf-8") as f:
                self.needs = NeedBook.from_dict(json.load(f))
        except (OSError, ValueError):
            self.needs = NeedBook()
            return
        # figures learned per GPU number before 2026-10-30 follow their card from now on (its UUID)
        if self.needs.rekey({f"gpu{i}": u for i, u in gpu_ids.table().items()}):
            self._save_needs()

    def need(self, purpose: str, declared: int) -> int:
        return self.needs.need(purpose, int(declared))

    def _learn(self) -> None:
        finished = self.reservations.drain_finished()
        for r in finished:
            if r.peak_mb > 0:
                before = self.needs.need(r.purpose, 0)
                self.needs.observe(r.purpose, r.peak_mb, gpu_ids.key(r.gpu))
                if r.peak_mb > r.mb:
                    logger.warning(f"GPU manager: {r.purpose} reserved {r.mb} MB but grew by {r.peak_mb} MB -- "
                                   f"reserving {self.needs.need(r.purpose, 0)} MB from now on (was {before})")
        if finished:
            self._save_needs()

    def _save_needs(self) -> None:
        if self.needs_file:
            try:
                tmp = self.needs_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.needs.to_dict(), f, indent=1, sort_keys=True)
                os.replace(tmp, self.needs_file)
            except OSError as e:
                logger.warning(f"GPU manager: learned needs not saved: {e}")

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
        from backend import model_manager
        out: List[Consumer] = []
        live = [(port, info) for port, info in list(state.managed_processes.items())
                if info.get("process") and info["process"].poll() is None]
        # active_requests counts only calls through this gateway; a client on the instance's own port (llm-benchy
        # after /api/start) shows up only in its /slots -- asked in parallel, 0.5 s each at most
        working = await asyncio.gather(*(model_manager.is_port_busy(port, info) for port, info in live))
        for (port, info), generating in zip(live, working):
            model = os.path.basename(info.get("model", "?"))
            out.append(Consumer(id=f"llm:{port}", kind="llm", label=model, pin_key=f"model:{model}",
                                root_pids={info["process"].pid}, last_used=info.get("last_used") or 0,
                                # still loading = busy: it has no requests yet but must
                                # not be unloaded by the next load
                                busy=generating or bool(info.get("loading"))))
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

    def _measured(self, gpu: int, pids: Set[int]) -> Dict[int, int]:
        """VRAM on `gpu` per process in `pids`, incl. its children (from the last snapshot)."""
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
        return measured

    def _outstanding(self, gpu: int) -> int:
        """Memory promised on `gpu` but not yet allocated by the reserving processes
        (their VRAM incl. children, from the last snapshot, beyond what they held when they confirmed)."""
        pids = {r.pid for r in self.reservations.all() if r.pid}
        measured = self._measured(gpu, pids)
        alive = {p for p in pids if psutil.pid_exists(p)}
        out = self.reservations.outstanding(gpu, time.time(), alive, measured)
        self._learn()
        return out

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

    async def exit_process(self, consumer_id: str, consumers: Optional[List[Consumer]] = None) -> None:
        """Stop an idle plugin's whole process -- after an unload its CUDA context still holds memory. It starts
        again on its next request (Apps & Services). Never while the plugin or any of its workers is busy."""
        c = next((c for c in (consumers or await self.consumers()) if c.id == consumer_id), None)
        if c is None or not c.plugin:
            raise PermissionError(f"{consumer_id} is no plugin process")
        from backend.plugins import registry
        await registry.stop(c.plugin)
        logger.info(f"GPU manager: stopped {c.label}'s process ({c.plugin}) -- its CUDA context was in the way")
        from backend import events
        events.record("gpu", "{label} stopped to free its GPU memory", label=c.label)

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
        declared = int(need_mb)
        need_mb = self.need(purpose, need_mb)   # at least what this purpose grew by before (NeedBook)
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
                # the learned headroom is a margin, not a must: if the measured peak itself fits, reserve that;
                # failing that, a peak learned on another GPU gives way to what this GPU has seen (NeedBook.on_gpu)
                for low in (self.needs.minimum(purpose, declared), self.needs.on_gpu(purpose, declared, gpu_ids.key(gpu))):
                    if low >= need_mb:
                        continue
                    low_plan = plan_room(gpu, low, gpus, consumers, att, self.pins(), reserved_mb=reserved)
                    if low_plan.ok:
                        need_mb, plan = low, low_plan
                        break
                else:
                    return None, [], plan.reason
            done: List[str] = []
            tried: set = set()
            exited: set = set()
            # An unload can free less than it was counted with (2026-10-26: stt-serve's 318 MB are mostly its CUDA
            # context, which stays) -- then plan again on fresh figures without those and unload the next idle ones.
            # If what is still missing is only such contexts of idle plugin processes, stop those processes (owner,
            # 2026-10-28: Sketch-Cydonia-24B was 235 MB short on the 4060 Ti next to stt-serve's and ComfyUI's)
            for round_ in range(4):
                stop: List[str] = []
                if round_:
                    gpus, consumers, att = await self.snapshot()
                    consumers = self._mark_loading(consumers)
                    spent = _spent(consumers, tried)
                    plan = plan_room(gpu, need_mb, gpus, consumers, att, self.pins(), reserved_mb=reserved,
                                     spent=spent)
                    if not plan.ok:
                        full = plan_room(gpu, need_mb, gpus, consumers, att, self.pins(), reserved_mb=reserved,
                                         spent=frozenset(exited))
                        stop = [cid for cid in full.evict if cid in spent]
                        if not (full.ok and stop and _may_exit(stop, consumers)):
                            return None, done, f"GPU {gpu}: unloaded {done} but still short -- {plan.reason}"
                        plan = full
                if not plan.evict:
                    break
                for cid in plan.evict:
                    tried.add(cid)
                    try:
                        if cid in stop:
                            exited.add(cid)
                            await self.exit_process(cid, consumers)
                        else:
                            await self.evict(cid, consumers)
                        done.append(cid)
                    except Exception as e:
                        logger.warning(f"GPU manager: could not unload {cid} for {purpose}: {e}")
                # Unloaded VRAM is handed back asynchronously -- give it a moment (this
                # is about our own unloads finishing, not waiting for anyone busy)
                from backend.model_manager import fetch_gpu_status
                for _ in range(20):
                    g = next((g for g in (await fetch_gpu_status()).get("gpus", []) if int(g["index"]) == gpu), None)
                    if g and int(g.get("free") or 0) - reserved >= need_mb:
                        break
                    await asyncio.sleep(0.5)
                else:
                    continue
                break
            else:
                return None, done, f"GPU {gpu}: unloaded {done} but still short of {need_mb} MB"
            if done:
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
        """Bind to `pid`; what it already holds on that GPU (an already running ComfyUI or stt-serve) is the
        baseline -- only growth after this counts against the reservation."""
        r = self.reservations.get(reservation_id)
        baseline = self._measured(r.gpu, {int(pid)}).get(int(pid), 0) if r else 0
        return self.reservations.confirm(reservation_id, pid, time.time(), baseline_mb=baseline)

    def release(self, reservation_id: str) -> None:
        self.reservations.release(reservation_id)
        self._learn()

    async def release_measured(self, reservation_id: str) -> None:
        """Release after a fresh measurement, so the peak the process reached is learned (the endpoint's way)."""
        r = self.reservations.get(reservation_id)
        if r is not None and r.pid is not None:
            try:
                await self.snapshot()
                self.reservations.observe(self._measured(r.gpu, {r.pid}))
            except Exception as e:
                logger.debug(f"GPU manager: no measurement before releasing {reservation_id}: {e}")
        self.release(reservation_id)

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


MANAGER = LocalGpuManager(_pins_path(), needs_file=os.path.join(os.path.dirname(_pins_path()), "gpu_needs.json"))
