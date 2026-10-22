"""llama.cpp for this machine: the newest release with prebuilt binaries, the build that fits (CUDA up to what the
driver supports, Metal on Apple Silicon, ROCm, Vulkan, else CPU) -- the same choice ai-serve makes for enrolled devices
(backend/cluster/runtimes.py), the same safe extraction as the device agent (node_agent: safe_extract)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from installer.download import fetch, get_json

GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def _release():
    for rel in get_json(f"{GITHUB_RELEASES}?per_page=10"):
        if any(a.get("name", "").startswith("llama-b") and "-bin-" in a.get("name", "") for a in rel.get("assets") or []):
            return rel
    raise SystemExit("no llama.cpp release with prebuilt binaries found")


def install(runtimes_dir: Path, variant: Optional[str] = None) -> Path:
    """Download + extract the fitting build into runtimes/<tag>-<variant>/; returns the llama-server path."""
    from backend.cluster.runtimes import NodeFacts, choose_runtime, parse_release
    from node_agent.llama_serve_node import detect_facts, safe_extract
    f = detect_facts()
    facts = NodeFacts(platform=f["platform"], machine=f["machine"], gpu_vendors=f["gpu_vendors"],
                      cuda_driver=tuple(f["cuda_driver"]) if f.get("cuda_driver") else None,
                      rocm=f["rocm"], vulkan=f["vulkan"])
    spec = choose_runtime(parse_release(_release()), facts, variant)
    final = runtimes_dir / f"{spec.tag}-{spec.variant}"
    existing = list(final.rglob("llama-server")) if final.is_dir() else []
    if existing:
        print(f"  llama.cpp {spec.tag} ({spec.variant}): already there")
        return existing[0]
    work = runtimes_dir / f".{spec.tag}-{spec.variant}.part"
    work.mkdir(parents=True, exist_ok=True)
    for a in spec.files:
        archive = work / a.name
        fetch(a.url, archive, a.sha256, a.name)
        safe_extract(str(archive), str(work / "files"))
        archive.unlink()
    servers = list((work / "files").rglob("llama-server"))
    if not servers:
        raise SystemExit("no llama-server in the downloaded archive")
    work.rename(final)
    binary = final / servers[0].relative_to(work)
    os.chmod(binary, 0o755)
    print(f"  llama.cpp {spec.tag} ({spec.variant}) in {final}")
    return binary
