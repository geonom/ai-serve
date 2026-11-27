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


def test_paragraph_guides_keep_their_blank_line_parts():
    raw = "```text\nFirst part,\n on two lines.\n\nSecond part.\n```"
    assert video_prompt.clean(raw, "idea", paragraphs=True) == "First part, on two lines.\n\nSecond part."
    assert video_prompt.clean(raw, "idea") == "First part, on two lines. Second part."


def test_the_length_is_checked_and_reaches_guides_that_ask_for_it(monkeypatch):
    r = video_prompt.parse({"prompt": "Ein Bäcker öffnet", "seconds": 6})
    assert r.seconds == 6
    monkeypatch.setitem(video_prompt.FRAMING, "timed", {"text_to_video": "Length {s} s. {p}"})
    assert video_prompt.user_text(r, "timed", "text_to_video") == "Length 6 s. Ein Bäcker öffnet"
    assert video_prompt.user_text(r, "ltx-2", "text_to_video") == "user prompt: Ein Bäcker öffnet"
    for bad in (0, 601, "4", True):
        with pytest.raises(video_prompt.PromptError):
            video_prompt.parse({"prompt": "x", "seconds": bad})


def test_guides_kept_on_this_machine_come_from_extra_modules(monkeypatch, tmp_path):
    """backend/video_prompt_extra_*.py (2026-11-27): guides whose licence doesn't allow publishing them stay local."""
    import sys
    import types
    mod = types.ModuleType("backend.video_prompt_extra_test")
    mod.GUIDES = {"test-guide": {"text_to_video": "SYSTEM"}}
    mod.FRAMING = {"test-guide": {"text_to_video": "{p}"}}
    mod.PARAGRAPH_GUIDES = {"test-guide"}
    monkeypatch.setitem(sys.modules, "backend.video_prompt_extra_test", mod)
    monkeypatch.setattr(video_prompt, "_extra_names", lambda: ["video_prompt_extra_test"])
    for d in ("GUIDES", "FRAMING"):
        monkeypatch.setattr(video_prompt, d, dict(getattr(video_prompt, d)))
    monkeypatch.setattr(video_prompt, "PARAGRAPH_GUIDES", set(video_prompt.PARAGRAPH_GUIDES))
    video_prompt._add_extras()
    assert video_prompt.GUIDES["test-guide"]["text_to_video"] == "SYSTEM" and "test-guide" in video_prompt.PARAGRAPH_GUIDES


def test_options_list_settings_and_guides():
    o = client.get("/v1/video/prompt/options").json()
    assert "close-up" in o["settings"]["shot"]["choices"] and o["free_text"]["dialogue"]["max_length"] == 500
    assert {"ltx-2", "wan-2.2"} <= set(o["guides"]) and o["guides"] == sorted(o["guides"])


# ------------------------------------------------------------------ the route, with a fake chat model

ROWS = [{"id": "ltx-2.5", "label": "LTX-2.5", "plugin": "comfyui", "tasks": ["text_to_video", "image_to_video"],
         "default": True, "available": True, "prompt_guide": "ltx-2", "audio": True},
        {"id": "wan-2.2", "label": "Wan 2.2", "plugin": "comfyui", "tasks": ["image_to_video"], "default": False,
         "available": True, "prompt_guide": "wan-2.2", "audio": False},
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


