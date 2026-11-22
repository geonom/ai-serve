"""How much of a model lands in VRAM when part of it stays in RAM (/api/start's `n_gpu_layers` / `n_cpu_moe`).

llama.cpp keeps the first layers on the CPU for -ngl N (it offloads the last N) and, for --n-cpu-moe N, the routed
experts (blk.<i>.ffn_{up,down,gate,gate_up}_exps) of the first N layers; shared experts (_shexp), attention and
norms stay on the GPU. Tensors outside the layers (embeddings, output) are counted as VRAM: slightly too much is the
safe side for a reservation. Tests: tests/unit/test_moe_offload.py."""
import os
import re
from typing import List, Optional, Tuple

from backend.response_processing import estimate_kv_overhead_mb, gguf_tensor_sizes

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")
_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|down|gate|gate_up)_(ch)?exps\b")


def resident_fraction(tensors: List[Tuple[str, int]], n_gpu_layers: int, n_cpu_moe: Optional[int]) -> float:
    """Share of the tensor bytes that llama.cpp puts in VRAM (1.0 = everything)."""
    total = sum(b for _, b in tensors)
    if not total:
        return 1.0
    layers = [int(m.group(1)) for m in (_LAYER_RE.match(n) for n, _ in tensors) if m]
    n_layer = max(layers) + 1 if layers else 0
    first_gpu_layer = n_layer - n_gpu_layers if 0 <= n_gpu_layers < n_layer else 0
    cpu = 0
    for name, size in tensors:
        layer = _LAYER_RE.match(name)
        if not layer:
            continue
        i = int(layer.group(1))
        if i < first_gpu_layer or (n_cpu_moe and i < n_cpu_moe and _EXPERT_RE.match(name)):
            cpu += size
    return (total - cpu) / total


def start_need(model_path: str, ctx_size: int, n_gpu_layers: int, n_cpu_moe: Optional[int]) -> Tuple[int, str]:
    """(MB to reserve, the GPU manager's purpose key). A full load keeps its old key, so learned peaks still apply;
    a partial one gets its own (`+cmoe<N>`, `+ngl<N>`) -- a full load's peak would refuse it again."""
    size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
    purpose = f"llm:{os.path.basename(model_path)}@{ctx_size}"
    partial_ngl = n_gpu_layers is not None and n_gpu_layers >= 0
    if n_cpu_moe or partial_ngl:
        tensors = gguf_tensor_sizes(model_path)
        if tensors:
            size_mb *= resident_fraction(tensors, n_gpu_layers if partial_ngl else -1, n_cpu_moe)
        if n_cpu_moe:
            purpose += f"+cmoe{n_cpu_moe}"
        if partial_ngl:
            purpose += f"+ngl{n_gpu_layers}"
    return int(size_mb + estimate_kv_overhead_mb(ctx_size)), purpose
