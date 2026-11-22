"""MoE experts in RAM and partial offload through /api/start (2026-11-02, owner: "sonst können wir MoEs ja gar nicht
so nutzen, wie sie gedacht sind").

llm-benchy wanted Qwen3.6-35B-A3B (20.9 GB) on the 16 GB card with part of it in RAM. /api/start had no way to keep
experts on the CPU (llama.cpp --n-cpu-moe), and the GPU manager's need was the whole file + KV whatever -ngl said, so
both attempts got a 507. Now the need counts only the tensors that land in VRAM, read from the GGUF's tensor table:
- `n_cpu_moe` N keeps the routed experts (blk.<i>.ffn_{up,down,gate,gate_up}_exps) of the first N layers in RAM;
  shared experts (_shexp) and attention stay on the GPU;
- `n_gpu_layers` N >= 0 keeps the first layers in RAM and offloads the last N (llama.cpp's order);
- tensors outside the layers (embeddings, output) count as VRAM -- a little too much is a safe estimate."""
import struct

import pytest
from fastapi.testclient import TestClient

from backend import gpu_share
from backend.response_processing import gguf_tensor_sizes

MB = 1024 * 1024


def write_gguf(path, tensors, alignment=None):
    """A minimal GGUF: header, an optional general.alignment, tensor infos (1-D, type 0), padded data."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    kv = b""
    n_kv = 0
    if alignment:
        kv += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", alignment)
        n_kv = 1
    align = alignment or 32
    infos, offset = b"", 0
    for name, size in tensors:
        infos += s(name) + struct.pack("<I", 1) + struct.pack("<Q", size // 4) + struct.pack("<I", 0)
        infos += struct.pack("<Q", offset)
        offset += -(-size // align) * align
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv) + kv + infos
    head += b"\0" * (-len(head) % align)
    path.write_bytes(head + b"\0" * offset)


def moe(layers=4, expert=8 * MB, attn=1 * MB, shexp=1 * MB):
    out = [("token_embd.weight", 2 * MB)]
    for i in range(layers):
        out += [(f"blk.{i}.attn_q.weight", attn), (f"blk.{i}.ffn_up_shexp.weight", shexp)]
        out += [(f"blk.{i}.ffn_{k}_exps.weight", expert) for k in ("up", "down", "gate")]
    return out + [("output.weight", 2 * MB)]


def test_tensor_sizes_come_from_the_offsets(tmp_path):
    f = tmp_path / "m.gguf"
    write_gguf(f, [("a", 4096), ("b", 100), ("c", 64)], alignment=64)
    assert gguf_tensor_sizes(str(f)) == [("a", 4096), ("b", 128), ("c", 64)]   # padded to the alignment
    write_gguf(f, [("a", 32), ("b", 40)])
    assert gguf_tensor_sizes(str(f)) == [("a", 32), ("b", 64)]                  # default alignment 32


def test_unreadable_file_has_no_tensors(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"nope")
    assert gguf_tensor_sizes(str(tmp_path / "x.gguf")) is None
    assert gguf_tensor_sizes(str(tmp_path / "missing.gguf")) is None


def test_everything_on_the_gpu_without_offload():
    assert gpu_share.resident_fraction(moe(), n_gpu_layers=-1, n_cpu_moe=None) == 1.0


def test_n_cpu_moe_keeps_the_routed_experts_of_the_first_layers_in_ram():
    t = moe()                                     # total 4 + 4*(1+1+24) = 108 MB
    total = sum(b for _, b in t)
    assert gpu_share.resident_fraction(t, -1, 2) == pytest.approx((total - 2 * 24 * MB) / total)
    assert gpu_share.resident_fraction(t, -1, 99) == pytest.approx((total - 4 * 24 * MB) / total)  # all experts
    assert gpu_share.resident_fraction(t, -1, 0) == 1.0


def test_partial_ngl_offloads_the_last_layers():
    t = moe()
    total = sum(b for _, b in t)
    # -ngl 1: layers 0-2 stay in RAM
    assert gpu_share.resident_fraction(t, 1, None) == pytest.approx((total - 3 * 26 * MB) / total)
    assert gpu_share.resident_fraction(t, 4, None) == 1.0 and gpu_share.resident_fraction(t, 50, None) == 1.0
    # both: layer 3 on the GPU, but its experts are kept back too only if n_cpu_moe covers it
    assert gpu_share.resident_fraction(t, 1, 4) == pytest.approx((total - 3 * 26 * MB - 24 * MB) / total)


def test_need_scales_the_file_size_and_names_the_split(tmp_path, monkeypatch):
    f = tmp_path / "Qwen-A3B.gguf"
    write_gguf(f, moe())
    full, purpose_full = gpu_share.start_need(str(f), 8192, -1, None)
    part, purpose_part = gpu_share.start_need(str(f), 8192, -1, 4)
    assert part < full
    assert purpose_full == "llm:Qwen-A3B.gguf@8192"            # unchanged key: learned peaks keep applying
    assert purpose_part == "llm:Qwen-A3B.gguf@8192+cmoe4"      # its own peaks, never a full load's
    assert gpu_share.start_need(str(f), 8192, 2, None)[1] == "llm:Qwen-A3B.gguf@8192+ngl2"


def test_a_file_without_a_readable_tensor_table_counts_whole(tmp_path):
    f = tmp_path / "odd.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 64)
    need, _ = gpu_share.start_need(str(f), 8192, -1, 4)
    full, _ = gpu_share.start_need(str(f), 8192, -1, None)
    assert need == full


def test_api_start_passes_n_cpu_moe_and_reserves_only_the_gpu_part(tmp_path, monkeypatch):
    from backend.main import app
    from backend.routers import dashboard
    from backend.cluster import local
    f = tmp_path / "Qwen-A3B.gguf"
    write_gguf(f, moe(expert=64 * MB))
    asked, started = [], []

    async def reserve(gpu, need, purpose):
        asked.append((gpu, need, purpose))
        return None, [], "test stops here"
    monkeypatch.setattr(local.MANAGER, "reserve", reserve)
    client = TestClient(app, client=("127.0.0.1", 50000))
    r = client.post("/api/start", json={"model_path": str(f), "gpu_id": 0, "ctx_size": 8192, "n_cpu_moe": 4,
                                        "mtp": False, "port": 18099})
    assert r.status_code == 507
    _, need, purpose = asked[-1]
    full, _ = gpu_share.start_need(str(f), 8192, -1, None)
    assert need < full and purpose.endswith("+cmoe4")
    assert dashboard.llama_offload_flags(4) == ["--n-cpu-moe", "4"]
    assert dashboard.llama_offload_flags(None) == []
    assert client.post("/api/start", json={"model_path": str(f), "n_cpu_moe": -1}).status_code == 422
