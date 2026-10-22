"""Choosing a llama.cpp runtime for a node (backend/cluster/runtimes.py) -- pure.

The agent installs its runtime on the coordinator's instruction (user, 2026-10-16).
llama.cpp publishes prebuilt archives per release (checked 2026-10-16, b11681):
macos-arm64/x64, ubuntu-x64/arm64 (CPU), ubuntu-vulkan-x64/arm64, ubuntu-cuda-12.8-x64,
ubuntu-cuda-13.4-x64/arm64 (+ separate cudart-* archives with the CUDA runtime
libraries), ubuntu-rocm-*, android-arm64, win-*. GitHub's API gives a sha256 per asset.

Rules under test:
- Only assets of github.com/ggml-org/llama.cpp release downloads, each with a sha256.
- Pick by the node's facts: macOS arm64 -> macos-arm64; Android -> android-arm64;
  Linux with NVIDIA -> the newest CUDA build the driver supports (driver's "CUDA
  Version"), plus the matching cudart archive; Linux with AMD+ROCm -> rocm; Linux
  with a Vulkan loader -> vulkan; otherwise CPU for the architecture.
- An explicit variant (e.g. "cpu", "vulkan") wins if that asset exists for the node.
- Nothing suitable -> a clear error, never a guess."""
import pytest

from backend.cluster.runtimes import NodeFacts, RuntimeError_, choose_runtime, parse_release

TAG = "b11681"
NAMES = [
    "cudart-llama-b11681-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11681-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-b11681-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b11681-bin-android-arm64.tar.gz",
    "llama-b11681-bin-macos-arm64.tar.gz",
    "llama-b11681-bin-ubuntu-arm64.tar.gz",
    "llama-b11681-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11681-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11681-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b11681-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b11681-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11681-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b11681-bin-ubuntu-x64.tar.gz",
    "llama-b11681-bin-win-cuda-12.4-x64.zip",
    "llama-b11681-ui.tar.gz",
]


def release(names=NAMES, tag=TAG, digest=True):
    return {"tag_name": tag, "assets": [
        {"name": n, "size": 1000,
         "browser_download_url": f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{n}",
         **({"digest": "sha256:" + "a" * 64} if digest else {})} for n in names]}


def facts(**kw):
    base = dict(platform="linux", machine="x86_64", gpu_vendors=(), cuda_driver=None, rocm=False, vulkan=False)
    base.update(kw)
    return NodeFacts(**base)


def names(spec):
    return [f.name for f in spec.files]


class TestParse:
    def test_only_github_llama_cpp(self):
        rel = release()
        rel["assets"][4]["browser_download_url"] = "https://evil.example/llama-b11681-bin-macos-arm64.tar.gz"
        assets = parse_release(rel)
        assert "llama-b11681-bin-macos-arm64.tar.gz" not in {a.name for a in assets}

    def test_assets_without_digest_dropped(self):
        assert parse_release(release(digest=False)) == []

    def test_tag_format(self):
        with pytest.raises(RuntimeError_):
            parse_release(release(tag="../../x"))


class TestChoose:
    def pick(self, f, variant=None):
        return choose_runtime(parse_release(release()), f, variant)

    def test_mac(self):
        s = self.pick(facts(platform="darwin", machine="arm64"))
        assert names(s) == ["llama-b11681-bin-macos-arm64.tar.gz"] and s.variant == "metal"

    def test_android(self):
        assert names(self.pick(facts(platform="android", machine="aarch64"))) == ["llama-b11681-bin-android-arm64.tar.gz"]

    def test_nvidia_new_driver_gets_cuda13_plus_cudart(self):
        s = self.pick(facts(gpu_vendors=("nvidia",), cuda_driver=(13, 6)))
        assert names(s) == ["llama-b11681-bin-ubuntu-cuda-13.4-x64.tar.gz",
                            "cudart-llama-b11681-bin-ubuntu-cuda-13.4-x64.tar.gz"]
        assert s.variant == "cuda-13.4"

    def test_nvidia_older_driver_gets_cuda12(self):
        s = self.pick(facts(gpu_vendors=("nvidia",), cuda_driver=(12, 9)))
        assert s.variant == "cuda-12.8"

    def test_nvidia_too_old_driver_falls_back(self):
        s = self.pick(facts(gpu_vendors=("nvidia",), cuda_driver=(11, 8), vulkan=True))
        assert s.variant == "vulkan"

    def test_amd_rocm(self):
        assert self.pick(facts(gpu_vendors=("amd",), rocm=True)).variant == "rocm-10.0"

    def test_amd_without_rocm_vulkan(self):
        assert self.pick(facts(gpu_vendors=("amd",), vulkan=True)).variant == "vulkan"

    def test_cpu_arm(self):
        assert names(self.pick(facts(machine="aarch64"))) == ["llama-b11681-bin-ubuntu-arm64.tar.gz"]

    def test_explicit_variant(self):
        assert self.pick(facts(gpu_vendors=("nvidia",), cuda_driver=(13, 6)), "cpu").variant == "cpu"

    def test_unknown_variant(self):
        with pytest.raises(RuntimeError_):
            self.pick(facts(), "cuda-99")

    def test_nothing_for_platform(self):
        with pytest.raises(RuntimeError_):
            self.pick(facts(platform="freebsd"))
