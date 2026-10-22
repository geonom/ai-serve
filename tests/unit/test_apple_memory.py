"""Apple Silicon (unified memory): how much the GPU may still take. Found on a Mac (24 GB): Iris (7 GB) was refused
with "3681 MB free" -- psutil's `available` RAM, while macOS would have made room by compressing other apps."""
from backend.model_manager import apple_gpu_free_mb, apple_gpu_limit_mb


def test_metal_limit():
    assert apple_gpu_limit_mb(24576) == 16384                 # 2/3 up to 36 GB
    assert apple_gpu_limit_mb(65536) == 49152                 # 3/4 above
    assert apple_gpu_limit_mb(24576, wired_limit_mb=20000) == 20000   # the owner's sysctl wins
    assert apple_gpu_limit_mb(24576, wired_limit_mb=99999) == 24576


def test_free_is_the_limit_minus_what_the_gpu_holds():
    assert apple_gpu_free_mb(16384, gpu_used_mb=1200, available_mb=3681) == 15184
    assert apple_gpu_free_mb(16384, gpu_used_mb=17000, available_mb=3681) == 0


def test_without_the_gpu_figure_the_cautious_old_way():
    assert apple_gpu_free_mb(16384, gpu_used_mb=None, available_mb=3681) == 3681
