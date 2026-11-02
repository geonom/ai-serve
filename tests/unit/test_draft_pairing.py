"""Draft heads pair with their main model by name: `mtp-<stem>.gguf` (MTP heads) and `dflash-<stem>.gguf`
(DFlash block drafters, e.g. NVIDIA's for Nemotron 3.5 Bolt) -- the main model's file name starts with <stem>."""
from backend.model_manager import find_draft_model_path

PATHS = ["/m/gemma-4-12b-it-UD-Q4_K_XL.gguf", "/m/mtp-gemma-4-12b-it.gguf",
         "/m/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", "/m/dflash-Nemotron-3.5-Bolt-E4B.gguf",
         "/m/Qwen3-0.6B-Q4_K_M.gguf"]


def test_mtp_and_dflash_heads_pair_by_stem():
    assert find_draft_model_path("/m/gemma-4-12b-it-UD-Q4_K_XL.gguf", PATHS) == "/m/mtp-gemma-4-12b-it.gguf"
    assert find_draft_model_path("/m/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", PATHS) == "/m/dflash-Nemotron-3.5-Bolt-E4B.gguf"
    assert find_draft_model_path("/m/Qwen3-0.6B-Q4_K_M.gguf", PATHS) is None
