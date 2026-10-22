"""installer/offers.py -- what a fresh install offers and what it writes."""
import pytest

from installer import offers as o

LINUX_NV = o.Machine("Linux", "x86_64", nvidia=True)
MAC = o.Machine("Darwin", "arm64")
LINUX_CPU = o.Machine("Linux", "x86_64")


def test_the_basic_offers():
    ids = [x.id for x in o.offers(LINUX_NV)]
    assert ids == ["llm-gemma-e4b", "llm-gemma-e2b", "tts-chatterbox", "stt-whisper", "image-iris", "image-comfyui"]


def test_gemma_pairs_its_vision_module_by_name():
    e4b = o.offers(LINUX_NV)[0]
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
    with pytest.raises(ValueError):
        o.validate_choice(["llm-gemma-e4b", "llm-gemma-e2b"], MAC)
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
