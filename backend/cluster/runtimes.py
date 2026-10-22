"""Which llama.cpp runtime a node should install -- pure. Rules and reasons:
tests/unit/test_cluster_runtimes.py. The coordinator reads a release from GitHub's
API (with per-asset sha256 digests) and hands the agent exact URLs + hashes; the
agent itself only accepts that same URL prefix and verifies the hash."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_DOWNLOAD = "https://github.com/ggml-org/llama.cpp/releases/download/"
_TAG_RE = re.compile(r"^b\d{1,7}$")
_ASSET_RE = re.compile(
    r"^(?P<cudart>cudart-)?llama-(?P<tag>b\d+)-bin-(?P<os>ubuntu|macos|android|win)-(?P<rest>[a-z0-9.-]+)\.(tar\.gz|zip)$")


class RuntimeError_(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    size: int
    os: str
    variant: str      # cpu / vulkan / cuda-13.4 / rocm-10.0 / metal / ...
    arch: str         # x64 / arm64
    cudart: bool


@dataclass(frozen=True)
class RuntimeSpec:
    tag: str
    variant: str
    files: Tuple[Asset, ...]


@dataclass(frozen=True)
class NodeFacts:
    platform: str                  # linux / darwin / android / win32
    machine: str                   # x86_64 / arm64 / aarch64
    gpu_vendors: Sequence[str] = ()
    cuda_driver: Optional[Tuple[int, int]] = None   # "CUDA Version" the driver supports
    rocm: bool = False
    vulkan: bool = False

    @property
    def arch(self) -> str:
        return "arm64" if self.machine.lower() in ("arm64", "aarch64") else "x64"


def _split(rest: str) -> Tuple[str, str]:
    parts = rest.split("-")
    arch = parts[-1]
    variant = "-".join(parts[:-1]) or "cpu"
    return variant, arch


def parse_release(release: Dict[str, Any]) -> List[Asset]:
    tag = str(release.get("tag_name") or "")
    if not _TAG_RE.match(tag):
        raise RuntimeError_(f"unexpected release tag {tag!r}")
    out = []
    for a in release.get("assets") or []:
        name, url, digest = a.get("name", ""), a.get("browser_download_url", ""), a.get("digest") or ""
        m = _ASSET_RE.match(name)
        if not m or m.group("tag") != tag or url != f"{REPO_DOWNLOAD}{tag}/{name}":
            continue
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            continue
        variant, arch = _split(m.group("rest"))
        if m.group("os") == "macos":
            variant = "metal" if arch == "arm64" else "cpu"
        out.append(Asset(name=name, url=url, sha256=digest[7:], size=int(a.get("size") or 0), os=m.group("os"),
                         variant=variant, arch=arch, cudart=bool(m.group("cudart"))))
    return out


def _os_for(platform: str) -> Optional[str]:
    return {"linux": "ubuntu", "darwin": "macos", "android": "android", "win32": "win"}.get(platform)


def _cuda_ver(variant: str) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"cuda-(\d+)\.(\d+)", variant)
    return (int(m.group(1)), int(m.group(2))) if m else None


def choose_runtime(assets: List[Asset], f: NodeFacts, variant: Optional[str] = None) -> RuntimeSpec:
    os_ = _os_for(f.platform)
    mine = [a for a in assets if a.os == os_ and a.arch == f.arch and not a.cudart]
    if not mine:
        raise RuntimeError_(f"no llama.cpp build for {f.platform}/{f.machine}")
    by_variant = {a.variant: a for a in mine}

    def spec(a: Asset) -> RuntimeSpec:
        files = [a]
        if _cuda_ver(a.variant) and os_ == "ubuntu":
            rt = [c for c in assets if c.cudart and c.os == os_ and c.arch == a.arch and c.variant == a.variant]
            files += rt[:1]
        tag = a.name.split("-")[1]
        return RuntimeSpec(tag=tag, variant=a.variant, files=tuple(files))

    if variant:
        if variant not in by_variant:
            raise RuntimeError_(f"variant {variant!r} not available for this node: {sorted(by_variant)}")
        return spec(by_variant[variant])
    if os_ in ("macos", "android"):
        return spec(by_variant.get("metal") or by_variant.get("cpu") or mine[0])
    vendors = {v.lower() for v in f.gpu_vendors}
    if "nvidia" in vendors and f.cuda_driver:
        usable = sorted((v for v in by_variant if _cuda_ver(v) and _cuda_ver(v) <= f.cuda_driver),
                        key=_cuda_ver, reverse=True)
        if usable:
            return spec(by_variant[usable[0]])
    if "amd" in vendors and f.rocm:
        rocm = sorted(v for v in by_variant if v.startswith("rocm-"))
        if rocm:
            return spec(by_variant[rocm[-1]])
    if f.vulkan and "vulkan" in by_variant:
        return spec(by_variant["vulkan"])
    if "cpu" in by_variant:
        return spec(by_variant["cpu"])
    raise RuntimeError_(f"no suitable llama.cpp build for {f.platform}/{f.machine}")
