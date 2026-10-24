"""Worker lifecycle: spawn a provider's worker on demand, wait for /health, proxy
/generate, keep-alive unload, busy guard, VRAM-aware eviction between engines and
coordination with the GPU owner (ai-serve-ui). Rules live in domain.py."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from tts_serve.domain import (
    ProviderManifest,
    ProviderRegistry,
    RequestError,
    TtsRequest,
    eviction_order,
    vram_need_mb,
)
from tts_serve.providers import load_adapter

logger = logging.getLogger("tts-serve.workers")


def _out_of_memory(detail: Any) -> bool:
    text = str(detail).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text or "outofmemory" in text


class CoordinationRefused(Exception):
    """The GPU owner (ai-serve-ui) said no -- e.g. Strata holds both GPUs."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class WorkerError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- adapters to the outside

def nvidia_smi_probe() -> Dict[int, Dict[str, int]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    gpus = {}
    for line in out.strip().splitlines():
        try:
            idx, total, free = (int(x.strip()) for x in line.split(","))
            gpus[idx] = {"total": total, "free": free}
        except ValueError:
            continue
    return gpus


class HttpCoordinator:
    """Asks ai-serve-ui before a cold start. Unreachable -> None (standalone use)."""

    def __init__(self, gateway_url: Optional[str]):
        self.gateway_url = (gateway_url or "").rstrip("/")
        self.last_reservation: Optional[str] = None
        self.last_gpu: Optional[int] = None

    async def preflight(self, purpose: str, gpu: int, need_mb: int) -> Optional[int]:
        """Free MiB on the GPU (None: gateway unreachable/standalone). A granted
        preflight is a reservation in ai-serve-ui's GPU manager; its id is kept in
        `last_reservation` for confirm() with the worker's PID."""
        self.last_reservation, self.last_gpu = None, None
        if not self.gateway_url:
            return None
        body = {"purpose": purpose, "need_mb": need_mb}
        if gpu is not None:   # None: the GPU manager picks one with room (answers `gpu`)
            body["gpu"] = gpu
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{self.gateway_url}/api/coordination/gpu-preflight", json=body)
        except httpx.HTTPError:
            return None
        if r.status_code in (409, 507):
            try:
                detail = r.json().get("detail") or r.text
            except ValueError:
                detail = r.text
            raise CoordinationRefused(str(detail))
        if r.status_code != 200:
            return None
        body = r.json()
        if body.get("granted") is False:   # an older gateway said no with a 200
            raise CoordinationRefused(str(body.get("reason") or "no room on the GPUs"))
        self.last_reservation = body.get("reservation_id")
        self.last_gpu = body.get("gpu") if isinstance(body.get("gpu"), int) else None
        return body.get("free_mb")

    async def confirm(self, reservation_id: Optional[str], pid: int) -> None:
        """Bind the reservation to the started worker (within the gateway's 5 s hold);
        it then counts until the worker has allocated its memory or exits."""
        if not (self.gateway_url and reservation_id):
            return
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(f"{self.gateway_url}/api/coordination/reservations/{reservation_id}/confirm",
                             json={"pid": pid})
        except httpx.HTTPError:
            pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- pool

@dataclass
class _Slot:
    proc: Optional[subprocess.Popen] = None
    port: Optional[int] = None
    status: str = "unloaded"
    device: Optional[str] = None
    variant: Optional[str] = None
    last_used: float = 0.0
    inflight: int = 0
    gpu: Optional[int] = None   # where the running worker is (the GPU manager's pick)
    keep_alive_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class WorkerPool:
    def __init__(self, providers_dir: Path, logs_dir: Path, gpu_probe: Callable[[], Dict[int, Dict[str, int]]],
                 coordinator, default_engine: Optional[str] = None):
        self.providers_dir = Path(providers_dir)
        self.logs_dir = Path(logs_dir)
        self.gpu_probe = gpu_probe
        self.coordinator = coordinator
        self.default_engine = default_engine
        self.slots: Dict[str, _Slot] = {}
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # -- registry ------------------------------------------------------------
    def registry(self) -> ProviderRegistry:
        return ProviderRegistry.load(self.providers_dir)

    def resolve(self, engine: Optional[str]) -> ProviderManifest:
        reg = self.registry()
        if engine:
            return reg.resolve(engine)
        if self.default_engine:
            try:
                m = reg.resolve(self.default_engine)
                if m.installed:
                    return m
            except RequestError:
                pass
        installed = [m for m in reg.providers if m.installed]   # e.g. only Chatterbox set up on a fresh machine
        if installed:
            return installed[0]
        if not reg.providers:
            raise WorkerError(503, "No TTS providers are installed.")
        return reg.providers[0]

    def slot(self, pid: str) -> _Slot:
        return self.slots.setdefault(pid, _Slot())

    def log_path(self, pid: str) -> Path:
        return self.logs_dir / f"{pid}.log"

    # -- start / stop --------------------------------------------------------
    async def _make_room(self, m: ProviderManifest) -> Optional[int]:
        """The GPU to start on. With ai-serve: its GPU manager picks one with room (unloading idle models of any kind
        if needed) -- an engine that takes a fraction of its card (VoxCPM) keeps its own GPU, since its need depends
        on that card. A no is CoordinationRefused (507). Without ai-serve (standalone): the manifest's GPU, after
        unloading this service's own idle engines there."""
        gpus = await asyncio.to_thread(self.gpu_probe)
        total = gpus.get(m.gpu, {}).get("total", 0)
        need = vram_need_mb(m, total)
        free = await self.coordinator.preflight(f"tts:{m.id}", m.gpu if m.vram_fraction is not None else None, need)
        picked = getattr(self.coordinator, "last_gpu", None)
        if getattr(self.coordinator, "last_reservation", None) or picked is not None:
            return picked if picked is not None else m.gpu   # the manager made room there
        if free is None:
            free = gpus.get(m.gpu, {}).get("free")
        if free is None or free >= need:
            return None
        engines = [{"id": pid, "gpu": self._gpu_of(pid), "idle": s.running and s.status == "idle" and not s.inflight,
                    "last_used": s.last_used} for pid, s in self.slots.items()]
        for victim in eviction_order(m.id, m.gpu, engines):
            logger.info(f"Making room for {m.name} on GPU {m.gpu} (need ~{need} MiB, {free} free): unloading {victim}")
            await self.unload(victim)
            await asyncio.sleep(0.5)
            free = (await asyncio.to_thread(self.gpu_probe)).get(m.gpu, {}).get("free", free)
            if free >= need:
                return None
        logger.warning(f"{m.name} needs ~{need} MiB on GPU {m.gpu} but only {free} MiB is free after "
                       f"unloading every idle TTS engine -- it may fall back to CPU.")
        return None

    def _gpu_of(self, pid: str) -> int:
        slot = self.slots.get(pid)
        if slot is not None and slot.running and slot.gpu is not None:
            return slot.gpu
        try:
            return self.registry().resolve(pid).gpu
        except RequestError:
            return -1

    async def ensure(self, m: ProviderManifest) -> int:
        if not m.installed:
            raise WorkerError(409, f"{m.name} isn't set up on this machine yet -- download it in the Playground or the "
                                   f"dashboard, or run providers/{m.id}/setup.sh")
        slot = self.slot(m.id)
        async with slot.lock:
            if slot.running:
                return slot.port
            gpu = await self._make_room(m)
            port = _free_port()
            env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0")
            env.update(m.environment(port))
            if gpu is not None:   # the worker sees only the GPU it was given (its cuda:0)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            slot.gpu = gpu if gpu is not None else m.gpu
            log = open(self.log_path(m.id), "w", encoding="utf-8")
            logger.info(f"Starting {m.name} worker on port {port}")
            try:
                slot.proc = subprocess.Popen(m.argv(port), cwd=str(m.dir), env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
            finally:
                log.close()
            confirm = getattr(self.coordinator, "confirm", None)
            if confirm is not None:
                await confirm(getattr(self.coordinator, "last_reservation", None), slot.proc.pid)
            slot.port, slot.status, slot.device, slot.variant = port, "starting", None, None
            deadline = time.time() + m.startup_timeout_s
            async with httpx.AsyncClient(timeout=2.0) as c:
                while time.time() < deadline:
                    if slot.proc.poll() is not None:
                        slot.status = "error"
                        raise WorkerError(500, f"{m.name} worker exited during startup; see logs/{m.id}.log")
                    try:
                        r = await c.get(f"http://127.0.0.1:{port}/health")
                        if r.status_code == 200:
                            slot.status = "idle"
                            return port
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.3)
            slot.status = "error"
            await self._terminate(slot)
            raise WorkerError(500, f"{m.name} worker did not become ready in {m.startup_timeout_s:.0f}s")

    async def _terminate(self, slot: _Slot) -> None:
        proc = slot.proc
        if slot.keep_alive_task and slot.keep_alive_task is not asyncio.current_task():
            slot.keep_alive_task.cancel()
        slot.keep_alive_task = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 30)  # VoxCPM stops its nano-vllm pool on SIGTERM
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait, 5)
        slot.proc, slot.port, slot.device, slot.variant = None, None, None, None
        if slot.status != "error":
            slot.status = "unloaded"

    async def unload(self, pid: str, force: bool = False) -> str:
        slot = self.slots.get(pid)
        if slot is None or not slot.running:
            return "already unloaded"
        if not force and (slot.inflight or slot.status in ("loading", "generating", "starting")):
            raise WorkerError(409, f"{pid} is busy; try again shortly.")
        await self._terminate(slot)
        slot.status = "unloaded"
        return "unloaded"

    async def unload_idle(self, gpu: Optional[int] = None) -> List[str]:
        done = []
        for pid, slot in list(self.slots.items()):
            if slot.running and slot.status == "idle" and not slot.inflight and (gpu is None or self._gpu_of(pid) == gpu):
                await self.unload(pid)
                done.append(pid)
        return done

    async def shutdown(self) -> None:
        for slot in self.slots.values():
            await self._terminate(slot)

    def _reset_keep_alive(self, pid: str, seconds: int) -> None:
        slot = self.slot(pid)
        if slot.keep_alive_task:
            slot.keep_alive_task.cancel()

        async def expire():
            await asyncio.sleep(seconds)
            while slot.inflight or slot.status in ("loading", "generating"):
                await asyncio.sleep(1)
            logger.info(f"{pid} idle for {seconds}s -- unloading")
            await self._terminate(slot)

        slot.keep_alive_task = asyncio.create_task(expire())

    # -- generate ------------------------------------------------------------
    async def generate(self, req: TtsRequest) -> Dict[str, Any]:
        m = self.resolve(req.engine)
        unknown = sorted(set(req.params) - m.param_names)
        if unknown:
            raise RequestError(f"{m.name} takes no param(s) {', '.join(unknown)} "
                               f"(it takes: {', '.join(sorted(m.param_names)) or 'none'})")
        slot = self.slot(m.id)
        ref_path = None
        if req.ref_audio is not None:
            fd, ref_path = tempfile.mkstemp(prefix=f"tts-serve-{m.id}-", suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(req.ref_audio)
        slot.inflight += 1
        try:
            payload = load_adapter(m.dir).build_payload(req, ref_path)  # validates before any spawn
            variant = payload.get("variant") if isinstance(payload, dict) else None
            warm = slot.running and slot.device is not None and (not m.variant_swaps or slot.variant == variant)
            port = await self.ensure(m)
            keep = req.keep_alive if req.keep_alive is not None else m.keep_alive_s
            self._reset_keep_alive(m.id, keep)
            slot.status = "generating" if warm else "loading"
            slot.last_used = time.time()
            for attempt in (1, 2):
                async with httpx.AsyncClient(timeout=1800.0) as c:
                    r = await c.post(f"http://127.0.0.1:{port}/generate", json=payload)
                if r.status_code < 400:
                    break
                try:
                    detail = r.json().get("error") or r.text
                except ValueError:
                    detail = r.text
                if r.status_code < 500:
                    raise RequestError(f"{m.name}: {detail}")
                if _out_of_memory(detail):
                    # another model filled the GPU (2026-10-24): a fresh worker where the GPU manager has room
                    logger.warning(f"{m.name} ran out of GPU memory (attempt {attempt}) -- restarting it")
                    await self._terminate(slot)
                    if attempt == 1:
                        port = await self.ensure(m)
                        slot.status = "loading"
                        continue
                    slot.status = "unloaded"
                    raise WorkerError(507, f"{m.name} ran out of GPU memory -- try again in a moment")
                slot.status = "error"
                raise WorkerError(502, f"{m.name} worker error: {detail}")
            data = r.json()
            slot.device = data.get("device") or slot.device
            slot.variant = variant
            slot.last_used = time.time()
            return {"engine": m.id, "audio_base64": data["audio_base64"], "device": slot.device,
                    "variant": variant}
        except (RequestError, WorkerError, CoordinationRefused):
            raise
        except httpx.HTTPError as e:
            slot.status = "error"
            raise WorkerError(502, f"{m.name} worker unreachable: {type(e).__name__}")
        finally:
            slot.inflight = max(0, slot.inflight - 1)
            if slot.inflight == 0 and slot.status in ("generating", "loading"):
                slot.status = "idle" if slot.running else "unloaded"
            if ref_path and os.path.exists(ref_path):
                os.remove(ref_path)

    # -- views ---------------------------------------------------------------
    def engines(self) -> Dict[str, Any]:
        reg = self.registry()
        out = []
        for m in reg.providers:
            s = self.slots.get(m.id) or _Slot()
            out.append({
                "id": m.id, "name": m.name, "description": m.description, "license": m.license,
                "aliases": list(m.aliases), "variants": list(m.variants), "default_reference": m.default_reference,
                "gpu": s.gpu if s.running and s.gpu is not None else m.gpu, "vram_mb": m.vram_mb, "vram_fraction": m.vram_fraction,
                "keep_alive_s": m.keep_alive_s, "concurrent": m.concurrent,
                "modes": [dict(x) for x in m.modes], "languages": list(m.languages),
                "params": sorted(m.param_names),
                "loaded": s.running, "status": s.status if s.running or s.status == "error" else "unloaded",
                "pid": s.proc.pid if s.running else None, "device": s.device, "variant": s.variant,
                "inflight": s.inflight, "last_used": s.last_used or None, "installed": m.installed,
                "install_mb": m.install_mb,
            })
        return {"engines": out, "errors": reg.errors, "default_engine": self._default_id(reg)}

    def _default_id(self, reg: ProviderRegistry) -> Optional[str]:
        try:
            return self.resolve(None).id
        except (RequestError, WorkerError):
            return None

    def tail(self, pid: Optional[str], lines: int) -> str:
        ids = [pid] if pid else [m.id for m in self.registry().providers]
        parts = []
        for i in ids:
            p = self.log_path(i)
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            parts.append(f"==> {i} <==\n" + "\n".join(text))
        return "\n\n".join(parts)
