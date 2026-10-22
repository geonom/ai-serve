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
    per_process = False
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.status == psutil.CONN_LISTEN and c.pid and c.laddr:
                listening.setdefault(c.pid, set()).add(c.laddr.port)
                addresses.setdefault(c.pid, set()).add((c.laddr.ip, c.laddr.port))
    except psutil.Error:
        # macOS: the system-wide list needs root. One `lsof` call lists this user's listening sockets without it;
        # without lsof, ask each of this user's processes (allowed there, but slow with many processes).
        found = _lsof_listening()
        if found is None:
            per_process = True
        else:
            for pid, ip, port in found:
                listening.setdefault(pid, set()).add(port)
                addresses.setdefault(pid, set()).add((ip, port))
    me = os.getuid()
    out: List[ObservedProcess] = []
    for p, pinfo in processes(["pid", "ppid", "uids", "cmdline", "create_time"]):
        try:
            if pinfo["uids"] is None or pinfo["uids"].real != me:
                continue
            if per_process:
                _own_listening(p, listening, addresses)
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


def parse_lsof(text: str) -> List[tuple]:
    """`lsof -Fpn` output -> [(pid, ip, port)]: 'p<pid>' starts a process, 'n<addr>:<port>' is one of its
    sockets ('*' = all interfaces, IPv6 in brackets)."""
    out, pid = [], None
    for line in text.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None and ":" in line:
            host, _, port = line[1:].rpartition(":")
            if not port.isdigit():
                continue
            host = host.strip("[]")
            out.append((pid, {"*": "0.0.0.0"}.get(host, host), int(port)))
    return out


def _lsof_listening():
    try:
        r = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-u", str(os.getuid()), "-Fpn"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    # exit 1 with no output just means "nothing listens"
    return parse_lsof(r.stdout) if r.returncode in (0, 1) else None


def _own_listening(p: psutil.Process, listening: Dict[int, set], addresses: Dict[int, set]) -> None:
    conns = getattr(p, "net_connections", None) or p.connections   # psutil >= 6 / older
    try:
        for c in conns(kind="inet"):
            if c.status == psutil.CONN_LISTEN and c.laddr:
                listening.setdefault(p.pid, set()).add(c.laddr.port)
                addresses.setdefault(p.pid, set()).add((c.laddr.ip, c.laddr.port))
    except (psutil.Error, OSError):
        pass


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
