"""installer/offers.py -- what a fresh install offers and what it writes."""
import pytest

from installer import offers as o

LINUX_NV = o.Machine("Linux", "x86_64", nvidia=True)
MAC = o.Machine("Darwin", "arm64")
LINUX_CPU = o.Machine("Linux", "x86_64")


def test_the_basic_offers():
    ids = [x.id for x in o.offers(LINUX_NV)]
    assert ids == ["llm-gemma-12b", "llm-gemma-e4b", "llm-gemma-e2b", "tts-chatterbox", "tts-piper", "stt-whisper",
                   "image-iris", "image-comfyui"]


def test_gemma_pairs_its_vision_module_by_name():
    e4b = {x.id: x for x in o.offers(LINUX_NV)}["llm-gemma-e4b"]
    assert [f.save_as for f in e4b.files] == ["gemma-4-E4B-it-Q4_K_M.gguf", "gemma-4-E4B-it-mmproj-F16.gguf"]
    assert e4b.files[1].path == "mmproj-F16.gguf" and e4b.chat_model == "gemma-4-E4B-it-Q4_K_M.gguf"
    # ai-serve pairs an mmproj with a model whose name starts with the mmproj's stem
    assert e4b.files[0].save_as.startswith(e4b.files[1].save_as.split("-mmproj-")[0])


def test_mac_gets_chatterbox_through_mlx():
    tts = {x.id: x for x in o.offers(MAC)}["tts-chatterbox"]
    assert tts.setup[0] == "providers/chatterbox-mlx/setup.sh" and "MLX" in tts.what
    assert {x.id: x for x in o.offers(LINUX_CPU)}["tts-chatterbox"].setup[0] == "providers/chatterbox/setup.sh"


def test_choice_rules():
    assert o.validate_choice(["image-iris", "llm-gemma-e2b"], MAC) == ["llm-gemma-e2b", "image-iris"]
    for pair in (["llm-gemma-e4b", "llm-gemma-e2b"], ["llm-gemma-12b", "llm-gemma-e2b"]):
        with pytest.raises(ValueError):
            o.validate_choice(pair, MAC)
    with pytest.raises(ValueError):
        o.validate_choice(["rm -rf /"], MAC)


def test_env_updates_and_merge():
    chosen = [x for x in o.offers(MAC) if x.id in ("llm-gemma-e4b", "image-iris")]
    up = o.env_updates(chosen, "./runtimes/b9999-metal/llama-server")
    assert up == {"LLAMA_SERVER_BIN": "./runtimes/b9999-metal/llama-server",
                  "DEFAULT_CHAT_MODEL": "gemma-4-E4B-it-Q4_K_M.gguf"}
    text = "# mine\nNETWORK_PROVIDER=none\nLLAMA_SERVER_BIN=/old\n"
    merged = o.merge_env(text, up)
    assert merged.startswith("# mine\nNETWORK_PROVIDER=none\nLLAMA_SERVER_BIN=./runtimes/b9999-metal/llama-server\n")
    assert "DEFAULT_CHAT_MODEL=gemma-4-E4B-it-Q4_K_M.gguf" in merged and merged.count("LLAMA_SERVER_BIN") == 1


# ---- suggestions from the hardware found (2026-10-26, owner: "Vorschläge, die sich an der gefundenen Hardware
# orientieren") -- the defaults of the interactive setup and `--recommended`

def _yes(m):
    return [oid for oid, yes, why in o.recommend(m) if yes]


def _why(m, oid):
    return next(why for i, yes, why in o.recommend(m) if i == oid)


BIG_NV = o.Machine("Linux", "x86_64", nvidia=True, gpus_mb=(24576, 16380), ram_mb=65536, disk_free_gb=500)
SMALL_NV = o.Machine("Linux", "x86_64", nvidia=True, gpus_mb=(8192,), ram_mb=32768, disk_free_gb=500)
TINY_NV = o.Machine("Linux", "x86_64", nvidia=True, gpus_mb=(4096,), ram_mb=16384, disk_free_gb=500)
CPU = o.Machine("Linux", "x86_64", ram_mb=16384, disk_free_gb=500)
MAC24 = o.Machine("Darwin", "arm64", ram_mb=24576, disk_free_gb=500)
MAC16 = o.Machine("Darwin", "arm64", ram_mb=16384, disk_free_gb=500)
MAC8 = o.Machine("Darwin", "arm64", ram_mb=8192, disk_free_gb=500)


def test_every_offer_gets_a_yes_or_no_with_a_reason():
    for m in (BIG_NV, SMALL_NV, TINY_NV, CPU, MAC24, MAC16, MAC8, LINUX_NV, MAC, LINUX_CPU):
        rows = o.recommend(m)
        assert [r[0] for r in rows] == [x.id for x in o.offers(m)]
        assert all(why for _, _, why in rows)
        assert sum(1 for oid, yes, _ in rows if yes and oid.startswith("llm-")) == 1   # exactly one language model
        o.validate_choice(_yes(m), m)                                                    # always a valid choice


def test_the_gpu_memory_picks_the_language_model():
    assert "llm-gemma-12b" in _yes(BIG_NV) and "24.0 GB" in _why(BIG_NV, "llm-gemma-12b")
    assert "llm-gemma-e4b" in _yes(SMALL_NV)
    assert "llm-gemma-e2b" in _yes(TINY_NV) and "llm-gemma-e2b" in _yes(CPU)
    # Apple Silicon: the GPU may use ~2/3 of the RAM (Metal's limit) -- 24 GB -> 16 GB -> the 12B fits
    assert "llm-gemma-12b" in _yes(MAC24) and "llm-gemma-e4b" in _yes(MAC16) and "llm-gemma-e2b" in _yes(MAC8)


def test_a_big_nvidia_card_gets_image_editing_instead_of_iris():
    assert {"tts-chatterbox", "stt-whisper", "image-comfyui"} <= set(_yes(BIG_NV))
    assert "image-iris" not in _yes(BIG_NV) and "ComfyUI" in _why(BIG_NV, "image-iris")


def test_image_editing_follows_the_card():
    """>= 12 GB: Qwen-Image-Edit 2511 and FLUX.2 [klein]; 8-12 GB: Z-Image + klein only (klein shares Z-Image's text
    encoder, ~8.5 GB in use); less: not offered as a suggestion."""
    big = {x.id: x for x in o.offers(BIG_NV)}["image-comfyui"]
    small = {x.id: x for x in o.offers(SMALL_NV)}["image-comfyui"]
    assert big.setup == ("setup.sh", "--models", "z-image-turbo,qwen-edit-2511,flux2-klein-4b")
    assert small.setup == ("setup.sh", "--models", "z-image-turbo,flux2-klein-4b")
    assert small.size_gb < big.size_gb and "klein" in small.what and "later" in small.what
    assert "image-comfyui" in _yes(SMALL_NV) and "image-iris" not in _yes(SMALL_NV)
    assert "image-comfyui" not in _yes(TINY_NV) and "image-comfyui" not in _yes(MAC24)


def test_smaller_machines():
    assert {"tts-chatterbox", "stt-whisper", "image-comfyui"} <= set(_yes(SMALL_NV))
    assert _yes(CPU) == ["llm-gemma-e2b", "tts-piper", "stt-whisper"]       # Chatterbox and images: too slow on a CPU
    assert "CPU" in _why(CPU, "tts-chatterbox") and "CPU" in _why(CPU, "tts-piper")
    assert "tts-piper" in _yes(TINY_NV) and "tts-chatterbox" not in _yes(TINY_NV)
    assert "tts-piper" not in _yes(BIG_NV) and "tts-piper" not in _yes(MAC16)       # Chatterbox there; Piper any time later
    assert {"tts-chatterbox", "image-iris"} <= set(_yes(MAC16)) and "image-comfyui" not in _yes(MAC24)


def test_not_more_than_the_disk_holds():
    m = o.Machine("Linux", "x86_64", nvidia=True, gpus_mb=(24576,), ram_mb=65536, disk_free_gb=40)
    yes = _yes(m)
    assert "image-comfyui" not in yes and "disk" in _why(m, "image-comfyui")
    sizes = {x.id: x.size_gb for x in o.offers(m)}
    assert sum(sizes[i] for i in yes) <= 40 - o.DISK_RESERVE_GB
    assert any(i.startswith("llm-") for i in yes)                             # a language model still comes


def test_unknown_hardware_keeps_the_old_defaults():
    """No memory figures (detection failed): what the setup proposed before."""
    assert _yes(LINUX_NV) == ["llm-gemma-e4b", "tts-chatterbox", "stt-whisper", "image-iris"]


def test_the_machine_says_what_was_found():
    assert o.describe(BIG_NV) == "Linux x86_64, NVIDIA GPUs 24.0 + 16.0 GB, RAM 64 GB, 500 GB free"
    assert o.describe(MAC24) == "Darwin arm64 (Apple Silicon, ~16 GB for the GPU), RAM 24 GB, 500 GB free"
    assert o.describe(LINUX_CPU) == "Linux x86_64, no NVIDIA GPU"


def test_piper_comes_from_its_own_repository():
    """Piper is an external plugin (an example of `plugin add`), not part of the ai-serve tree: setup clones it."""
    piper = {x.id: x for x in o.offers(CPU)}["tts-piper"]
    assert piper.git == "https://github.com/geonom/ai-serve-piper" and piper.plugin == "ai-serve-piper"
    assert piper.setup == ("setup.sh", "--voices", "de_DE-thorsten-medium,en_US-ljspeech-medium")
    assert piper.size_gb < 1


def test_reasons_match_the_descriptions_on_a_mac():
    """Found on the owner's Mac (2026-10-26): Whisper's reason said 'GPU or CPU' under a description saying 'on the
    CPU', and ComfyUI's Mac note stood twice."""
    assert "CPU" in _why(MAC24, "stt-whisper") and "GPU" not in _why(MAC24, "stt-whisper")
    assert "GPU" in _why(BIG_NV, "stt-whisper")
    comfy = {x.id: x for x in o.offers(MAC24)}["image-comfyui"]
    assert _why(MAC24, "image-comfyui") not in comfy.what
