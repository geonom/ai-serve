"""OS adapter for the service domain: process snapshots (psutil), launching, and
signalling. Decisions about *what* to launch or signal are made in domain.py."""
import os
import subprocess
from typing import Dict, FrozenSet, List

import httpx
import psutil
from backend.procs import processes

from backend.services.domain import LaunchPlan, ObservedProcess, child_environment


def snapshot() -> List[ObservedProcess]:
    """This user's processes with cwd, cmdline and listening TCP ports. Processes we
    may not inspect are skipped -- the domain can't attribute them anyway."""
    listening: Dict[int, set] = {}
    addresses: Dict[int, set] = {}
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status == psutil.CONN_LISTEN and c.pid and c.laddr:
                listening.setdefault(c.pid, set()).add(c.laddr.port)
                addresses.setdefault(c.pid, set()).add((c.laddr.ip, c.laddr.port))
    except psutil.Error:
        pass
    me = os.getuid()
    out: List[ObservedProcess] = []
    for p, pinfo in processes(["pid", "ppid", "uids", "cmdline", "create_time"]):
        try:
            if pinfo["uids"] is None or pinfo["uids"].real != me:
                continue
            out.append(ObservedProcess(
                pid=pinfo["pid"], ppid=pinfo["ppid"] or 0, cwd=p.cwd(),
                cmdline=tuple(pinfo["cmdline"] or ()),
                listening_ports=frozenset(listening.get(pinfo["pid"], ())),
                create_time=pinfo["create_time"] or 0.0,
                listen_addresses=frozenset(addresses.get(pinfo["pid"], ())),
            ))
        except (psutil.Error, OSError):
            continue
    return out


def launch(plan: LaunchPlan, log_path: str) -> subprocess.Popen:
    """argv as given (no shell), own session so the app outlives a backend restart
    -- same convention as llama-server/ComfyUI/Strata."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "ab")
    try:
        return subprocess.Popen(
            plan.argv, cwd=str(plan.cwd), env=child_environment(os.environ, plan.env),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    finally:
        log.close()  # the child holds its own fd


def terminate(pids: FrozenSet[int], timeout: float = 15) -> Dict[str, List[int]]:
    """SIGTERM every pid of the tree, SIGKILL whatever is left after `timeout`."""
    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.Error:
            continue
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    return {"terminated": sorted(p.pid for p in procs if p not in alive), "killed": sorted(p.pid for p in alive)}


async def health(port: int, path: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            return (await client.get(f"http://127.0.0.1:{port}{path}")).status_code < 500
    except httpx.HTTPError:
        return False


def tail(path: str, lines: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            data = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(data.splitlines()[-lines:])
