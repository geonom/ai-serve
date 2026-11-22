"""The video prompt writer (backend/video_prompt.py, POST /v1/video/prompt in routers/video_jobs.py, 2026-11-13).
What matters: settings only from the published lists (free text only for dialogue/sound), the video model's own guide
as the system prompt (picture -> video guide with a start picture), the settings reach the chat model, a chat model
that can't see still gets a caption from the words, and the answer is one clean paragraph -- never a started video."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import video_prompt
from backend.cluster.service import Upstream
from backend.main import app
from backend.plugins import video_gen
from backend.routers import gateway

HAS_H3 = "minimax-h3" in video_prompt.GUIDES   # local only: the public tree ships without MiniMax's guide
needs_h3 = pytest.mark.skipif(not HAS_H3, reason="MiniMax H3's guide is local only")

client = TestClient(app, client=("127.0.0.1", 50000))


# ------------------------------------------------------------------ pure rules

def test_settings_come_from_the_lists_and_free_text_is_limited():
    r = video_prompt.parse({"prompt": "  Ein Fuchs  ", "settings": {
        "shot": "close-up", "camera_motion": "", "lighting": None, "dialogue": "  „Hallo“\n sagt er "}})
    assert r.prompt == "Ein Fuchs" and r.settings == {"shot": "close-up", "dialogue": "„Hallo“ sagt er"}
    for bad in ({"shot": "dutch angle"}, {"color": "red"}, {"sound": "x" * 501}, {"dialogue": 3}):
        with pytest.raises(video_prompt.PromptError):
            video_prompt.parse({"prompt": "x", "settings": bad})
    for bad in ({}, {"prompt": " "}, {"prompt": "x" * 2001}, {"prompt": "x", "seed": True}, {"prompt": "x", "llm": ""},
                {"prompt": "x", "image_base64": ""}, {"prompt": "x", "settings": []}, []):
        with pytest.raises(video_prompt.PromptError):
            video_prompt.parse(bad)


def test_messages_use_the_guide_and_carry_every_setting():
    r = video_prompt.parse({"prompt": "Zwei Marines", "settings": {"shot": "wide shot", "pace": "slow motion",
                                                                   "sound": "Brandung"}})
    m = video_prompt.messages(r, "ltx-2", "text_to_video", with_image=False)
    assert m[0] == {"role": "system", "content": video_prompt.GUIDES["ltx-2"]["text_to_video"]}
    u = m[1]["content"]
    assert u.startswith("user prompt: Zwei Marines")
    assert "- Shot type: wide shot" in u and "- Pace: slow motion" in u and "- Sound, music and ambience: Brandung" in u
    assert video_prompt.user_text(video_prompt.parse({"prompt": "x"}), "ltx-2", "text_to_video") == "user prompt: x"
    assert video_prompt.user_text(video_prompt.parse({"prompt": "x"}), "wan-2.2", "image_to_video") == "x"


def test_a_start_picture_picks_the_picture_guide_and_goes_along_as_an_image():
    r = video_prompt.parse({"prompt": "er winkt", "image_base64": "/9j/AAAA"})
    m = video_prompt.messages(r, "ltx-2", "image_to_video", with_image=True)
    assert m[0]["content"] == video_prompt.GUIDES["ltx-2"]["image_to_video"]
    assert m[1]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAAA"}}
    assert m[1]["content"][1]["text"] == "User Raw Input Prompt: er winkt."
    assert isinstance(video_prompt.messages(r, "ltx-2", "image_to_video", with_image=False)[1]["content"], str)


def test_the_answer_is_one_clean_paragraph():
    assert video_prompt.clean('```\nCaption: "A fox runs.\n\nSnow falls."\n```', "idea") == "A fox runs. Snow falls."
    assert video_prompt.clean("   ", "idea") == "idea" and video_prompt.clean(None, "idea") == "idea"


@needs_h3
def test_minimax_h3_follows_its_makers_guide_with_mode_and_length():
    """MiniMax H3 (2026-11-15): MiniMax's own VIDEO_PROMPT_WRITING_GUIDE_base_en.md verbatim, the mode (T2VA / I2VA
    first frame) said in the system prompt, the length in the request -- cut times must fall inside it."""
    from backend import video_prompt_guide_minimax as g
    t2v, i2v = video_prompt.GUIDES["minimax-h3"]["text_to_video"], video_prompt.GUIDES["minimax-h3"]["image_to_video"]
    assert t2v.endswith(g.MINIMAX_H3_BASE) and i2v.endswith(g.MINIMAX_H3_BASE)
    assert g.MINIMAX_H3_BASE.startswith("# Video Prompt Writing Guide (T2VA / I2VA / FL2VA / L2VA)")
    assert "T2VA" in t2v[:2000] and "I2VA" in i2v[:2000] and "first frame" in i2v[:2000]
    r = video_prompt.parse({"prompt": "Ein Bäcker öffnet", "seconds": 6})
    assert r.seconds == 6
    u = video_prompt.user_text(r, "minimax-h3", "text_to_video")
    assert "6 seconds" in u and "Ein Bäcker öffnet" in u
    assert video_prompt.user_text(r, "ltx-2", "text_to_video") == "user prompt: Ein Bäcker öffnet"   # others as before
    for bad in (0, 601, "4", True):
        with pytest.raises(video_prompt.PromptError):
            video_prompt.parse({"prompt": "x", "seconds": bad})


def test_h3_answers_keep_their_three_fields_as_paragraphs():
    raw = ("```text\nFor the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
           "referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action,\n cinematic.\n\n"
           "overall_soundscape: Rain.\n\nnon_diegetic_music: N/A\n```")
    assert video_prompt.clean(raw, "idea", paragraphs=True) == (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        "\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic.\n\noverall_soundscape: Rain."
        "\n\nnon_diegetic_music: N/A")


def test_options_list_settings_and_guides():
    o = client.get("/v1/video/prompt/options").json()
    assert "close-up" in o["settings"]["shot"]["choices"] and o["free_text"]["dialogue"]["max_length"] == 500
    assert o["guides"] == (["ltx-2", "minimax-h3", "wan-2.2"] if HAS_H3 else ["ltx-2", "wan-2.2"])


# ------------------------------------------------------------------ the route, with a fake chat model

ROWS = [{"id": "ltx-2.5", "label": "LTX-2.5", "plugin": "comfyui", "tasks": ["text_to_video", "image_to_video"],
         "default": True, "available": True, "prompt_guide": "ltx-2", "audio": True},
        {"id": "wan-2.2", "label": "Wan 2.2", "plugin": "comfyui", "tasks": ["image_to_video"], "default": False,
         "available": True, "prompt_guide": "wan-2.2", "audio": False},
        {"id": "minimax-h3", "label": "MiniMax H3", "plugin": "comfyui", "tasks": ["text_to_video", "image_to_video"],
         "default": False, "available": True, "prompt_guide": "minimax-h3", "audio": True, "min_seconds": 2,
         "max_seconds": 10, "default_seconds": 5},
        {"id": "other", "label": "Other", "plugin": "comfyui", "tasks": ["text_to_video"], "default": False,
         "available": True, "prompt_guide": None}]


class FakeChat:
    def __init__(self, blind=False):
        self.blind, self.calls = blind, []
        self.answer = "Caption: A wide shot frames two men."

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        if self.blind and isinstance(body["messages"][1]["content"], list):
            return httpx.Response(400, json={"error": {"message": "image input is not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": self.answer},
                                                      "finish_reason": "stop"}]})


@pytest.fixture
def chat(monkeypatch):
    fake = FakeChat()

    async def catalog():
        return ROWS, []

    async def resolve(name, **kw):
        fake.llm = name
        return Upstream(kind="plugin", base="http://127.0.0.1:9999", label="gemma-test")
    monkeypatch.setattr(video_gen, "catalog", catalog)
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    monkeypatch.setattr(gateway, "_upstream_client", lambda up, t: httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    return fake


def test_a_prompt_is_written_with_the_guide_and_settings(chat):
    r = client.post("/v1/video/prompt", json={"prompt": "Zwei Marines", "settings": {"shot": "wide shot"}, "seed": 4})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["prompt"] == "A wide shot frames two men." and out["model"] == "ltx-2.5"
    assert out["task"] == "text_to_video" and out["guide"] == "ltx-2" and out["llm"] == "gemma-test"
    sent = chat.calls[-1]
    assert sent["messages"][0]["content"] == video_prompt.GUIDES["ltx-2"]["text_to_video"]
    assert "- Shot type: wide shot" in sent["messages"][1]["content"] and sent["seed"] == 4 and not sent["stream"]


def test_a_chat_model_that_cannot_see_still_writes_from_the_words(chat):
    chat.blind = True
    out = client.post("/v1/video/prompt", json={"prompt": "er winkt", "image_base64": "iVBORw0"}).json()
    assert out["task"] == "image_to_video" and out["used_image"] is False and out["prompt"]
    assert isinstance(chat.calls[0]["messages"][1]["content"], list) and isinstance(chat.calls[1]["messages"][1]["content"], str)
    assert chat.calls[1]["messages"][0]["content"] == video_prompt.GUIDES["ltx-2"]["image_to_video"]


def test_models_without_a_guide_and_bad_settings_are_400(chat):
    assert client.post("/v1/video/prompt", json={"prompt": "x", "model": "other"}).status_code == 400
    assert client.post("/v1/video/prompt", json={"prompt": "x", "model": "sora"}).status_code == 400
    assert client.post("/v1/video/prompt", json={"prompt": "x", "settings": {"shot": "dutch"}}).status_code == 400
    assert chat.calls == []


def test_wan_gets_its_own_picture_guide_and_no_sound_settings(chat):
    out = client.post("/v1/video/prompt", json={"prompt": "er winkt", "model": "wan-2.2", "image_base64": "iVBORw0",
                                                "settings": {"camera_motion": "slow push-in"}}).json()
    assert out["guide"] == "wan-2.2" and out["task"] == "image_to_video"
    sent = chat.calls[-1]["messages"]
    assert sent[0]["content"] == video_prompt.GUIDES["wan-2.2"]["image_to_video"]
    assert sent[1]["content"][1]["text"].startswith("er winkt\n\n") and "- Camera movement: slow push-in" in sent[1]["content"][1]["text"]
    n = len(chat.calls)
    for settings in ({"sound": "Wellen"}, {"dialogue": "Hallo"}):
        r = client.post("/v1/video/prompt", json={"prompt": "x", "model": "wan-2.2", "image_base64": "iVBORw0",
                                                  "settings": settings})
        assert r.status_code == 400 and "no sound" in r.text
    r = client.post("/v1/video/prompt", json={"prompt": "x", "model": "wan-2.2"})       # Wan's guide needs the picture
    assert r.status_code == 400 and "start picture" in r.text
    assert len(chat.calls) == n


@needs_h3
def test_minimax_h3_gets_its_guide_the_length_and_paragraphs(chat):
    chat.answer = "integrated_multimodal_description: [Shot 1] Cinematic.\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: N/A"
    out = client.post("/v1/video/prompt", json={"prompt": "Ein Fuchs", "model": "minimax-h3"}).json()
    assert out["guide"] == "minimax-h3" and out["prompt"].count("\n\n") == 2
    sent = chat.calls[-1]["messages"]
    assert sent[0]["content"] == video_prompt.GUIDES["minimax-h3"]["text_to_video"]
    assert "5 seconds" in sent[1]["content"]                                   # the model's default length
    client.post("/v1/video/prompt", json={"prompt": "Ein Fuchs", "model": "minimax-h3", "seconds": 8})
    assert "8 seconds" in chat.calls[-1]["messages"][1]["content"]
    n = len(chat.calls)
    r = client.post("/v1/video/prompt", json={"prompt": "x", "model": "minimax-h3", "seconds": 30})
    assert r.status_code == 400 and "seconds" in r.text and len(chat.calls) == n


def test_without_minimaxs_guide_there_is_no_h3_guide_and_nothing_breaks(monkeypatch):
    """The public tree ships without video_prompt_guide_minimax.py (licence territory excludes the EU, 2026-11-22)."""
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "backend.video_prompt_guide_minimax", None)   # import fails, as when missing
    try:
        vp = importlib.reload(video_prompt)
        assert "minimax-h3" not in vp.GUIDES and {"ltx-2", "wan-2.2"} <= set(vp.GUIDES)
    finally:
        monkeypatch.delitem(sys.modules, "backend.video_prompt_guide_minimax")
        importlib.reload(video_prompt)
