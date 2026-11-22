"""Writing a video prompt for the caller (POST /v1/video/prompt, routers/video_jobs.py), pure rules.

The caller gives a short idea plus optional camera and look settings; a chat model expands it following the video
model's own prompt guide (`prompt_guide` in its catalog row; texts in video_prompt_guides.py). The answer is only a
proposal: the caller shows it, may edit it and then sends it to POST /v1/video/jobs itself -- nothing here starts a
video. Settings are fixed lists (OPTIONS) so callers can build menus from GET /v1/video/prompt/options; dialogue and
sound are free text. Tests: tests/unit/test_video_prompt.py."""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend import video_prompt_guides

# key -> (what the chat model is told, choices). The wording of the choices follows the guides' own terms (shot type,
# camera motion, viewpoint lists are LTX's).
OPTIONS: Dict[str, Dict[str, Any]] = {
    "shot": {"label": "Shot type", "choices": [
        "extreme wide shot", "wide shot", "medium shot", "medium close-up", "close-up", "extreme close-up"]},
    "camera_motion": {"label": "Camera movement", "choices": [
        "static camera", "slow push-in", "slow pull-back", "pan left", "pan right", "tilt up", "tilt down",
        "tracking shot following the subject", "orbit around the subject", "crane up", "crane down",
        "handheld camera", "slow zoom in", "slow zoom out"]},
    "viewpoint": {"label": "Camera viewpoint", "choices": [
        "front-facing", "back-facing", "side view", "over-the-shoulder", "top-down", "low-angle", "high-angle"]},
    "lens": {"label": "Lens and focus", "choices": [
        "wide-angle lens", "telephoto lens", "shallow depth of field", "deep focus", "macro lens"]},
    "lighting": {"label": "Lighting", "choices": [
        "natural daylight", "golden hour", "blue hour", "night with neon lights", "soft studio light",
        "hard, dramatic low-key light", "candlelight", "overcast light"]},
    "style": {"label": "Look", "choices": [
        "realistic, cinematic", "documentary", "vintage 16mm film", "black and white", "anime", "3D animation",
        "stop-motion"]},
    "pace": {"label": "Pace", "choices": ["slow motion", "calm", "energetic"]},
}
# free text the caller writes: (label for the chat model, max length)
FREE_TEXT = {"dialogue": ("Spoken words (quote them exactly, in their original language)", 500),
             "sound": ("Sound, music and ambience", 500)}

# MiniMax's guide is written for people, not as a system prompt: this line in front says what to do with it (ours)
_H3_TASK = ("You write the final prompt for the MiniMax H3 video model, following its makers' writing guide below "
            "exactly. The mode is {mode}. The user's request may be short and in any language: expand it faithfully, "
            "keep every element they asked for, and write in English -- spoken words and on-screen text stay verbatim "
            "in their original language, as the guide says. The user message gives the video's length; every cut time "
            "must fall inside it. Output ONLY the final prompt: {start}the three fields integrated_multimodal_"
            "description, overall_soundscape and non_diegetic_music, each beginning with its name and a colon, "
            "separated by blank lines. No markdown, no code fences, no commentary.\n\n--- WRITING GUIDE ---\n\n")
_H3_T2V = _H3_TASK.format(mode="T2VA (text only, no reference picture)", start="")
_H3_I2V = _H3_TASK.format(mode="I2VA (the attached picture is the first frame of the video, at 0.00 seconds)",
                          start="the I2VA first-frame instruction line, one blank line, then ")

# guide -> task -> system prompt; FRAMING: how the guide's makers hand over the user's words ({p}; {s} = seconds)
GUIDES = {"ltx-2": {"text_to_video": video_prompt_guides.LTX2_TEXT_TO_VIDEO,
                    "image_to_video": video_prompt_guides.LTX2_IMAGE_TO_VIDEO},
          "wan-2.2": {"image_to_video": video_prompt_guides.WAN22_IMAGE_TO_VIDEO}}
try:   # local only (licence territory excludes the EU) -- not in the public tree, see video_prompt_guide_minimax.py
    from backend.video_prompt_guide_minimax import MINIMAX_H3_BASE
    GUIDES["minimax-h3"] = {"text_to_video": _H3_T2V + MINIMAX_H3_BASE, "image_to_video": _H3_I2V + MINIMAX_H3_BASE}
except ImportError:
    pass
FRAMING = {"ltx-2": {"text_to_video": "user prompt: {p}", "image_to_video": "User Raw Input Prompt: {p}."},
           "wan-2.2": {"image_to_video": "{p}"},
           "minimax-h3": {"text_to_video": "Video length: {s} seconds.\n\nUser request: {p}",
                          "image_to_video": "Video length: {s} seconds.\n\nUser request: {p}"}}
PARAGRAPH_GUIDES = {"minimax-h3"}       # answers keep their blank-line structure (instruction line + three fields)
SOUND_SETTINGS = ("dialogue", "sound")      # only for models that make sound

MAX_PROMPT = 2000
MAX_IMAGE_B64 = 28_000_000          # ~20 MB picture
MAX_TOKENS = 1200                   # LTX asks for 150-220 words; H3's three fields run longer


class PromptError(ValueError):
    pass


@dataclass
class PromptRequest:
    prompt: str
    settings: Dict[str, str] = field(default_factory=dict)
    image: Optional[str] = None
    llm: Optional[str] = None
    seed: Optional[int] = None
    seconds: Optional[float] = None     # the video's length (H3's cut times); None = the model's default


def options() -> Dict[str, Any]:
    return {"settings": {k: dict(v) for k, v in OPTIONS.items()},
            "free_text": {k: {"label": lbl, "max_length": n} for k, (lbl, n) in FREE_TEXT.items()},
            "guides": sorted(GUIDES)}


def parse(body: Any) -> PromptRequest:
    if not isinstance(body, dict):
        raise PromptError("the body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptError("prompt: describe the scene")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT:
        raise PromptError(f"prompt: at most {MAX_PROMPT} characters")
    raw = body.get("settings")
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise PromptError("settings: an object")
    settings: Dict[str, str] = {}
    for k, v in raw.items():
        if v is None or v == "":
            continue                                        # "let the model choose"
        if k in OPTIONS:
            if v not in OPTIONS[k]["choices"]:
                raise PromptError(f"settings.{k}: one of {', '.join(OPTIONS[k]['choices'])}")
        elif k in FREE_TEXT:
            if not isinstance(v, str) or len(v) > FREE_TEXT[k][1]:
                raise PromptError(f"settings.{k}: text, at most {FREE_TEXT[k][1]} characters")
            v = " ".join(v.split())
            if not v:
                continue
        else:
            raise PromptError(f"settings.{k}: unknown (known: {', '.join(list(OPTIONS) + list(FREE_TEXT))})")
        settings[k] = v
    image = body.get("image_base64")
    if image is not None and (not isinstance(image, str) or not image or len(image) > MAX_IMAGE_B64):
        raise PromptError("image_base64: a base64 picture of at most 20 MB")
    llm = body.get("llm")
    if llm is not None and (not isinstance(llm, str) or not llm.strip() or len(llm) > 200):
        raise PromptError("llm: a chat model name")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise PromptError("seed: a whole number 0..4294967295")
    seconds = body.get("seconds")
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                or not 1 <= seconds <= 600):
        raise PromptError("seconds: the video's length, 1..600")
    return PromptRequest(prompt=prompt, settings=settings, image=image or None, llm=llm, seed=seed, seconds=seconds)


def _mime(b64: str) -> str:
    if b64.startswith("/9j/"):
        return "image/jpeg"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def user_text(r: PromptRequest, guide: str, task: str) -> str:
    # the guides' own framing of the request (ComfyUI's LTX node: "user prompt:" / "User Raw Input Prompt:")
    s = r.seconds if r.seconds is not None else "?"
    head = FRAMING[guide][task].format(p=r.prompt, s=f"{s:g}" if isinstance(s, (int, float)) else s)
    lines = [f"- {OPTIONS[k]['label']}: {v}" for k, v in r.settings.items() if k in OPTIONS]
    lines += [f"- {FREE_TEXT[k][0]}: {v}" for k, v in r.settings.items() if k in FREE_TEXT]
    if not lines:
        return head
    return (head + "\n\nThe user also chose these settings. They are part of the request: every one must show in "
                   "the caption, woven into the prose (never as labels), and they win over your own choices:\n"
            + "\n".join(lines))


def messages(r: PromptRequest, guide: str, task: str, with_image: bool) -> List[Dict[str, Any]]:
    system = GUIDES[guide][task]
    text = user_text(r, guide, task)
    if with_image and r.image:
        content: Any = [{"type": "image_url", "image_url": {"url": f"data:{_mime(r.image)};base64,{r.image}"}},
                        {"type": "text", "text": text}]
    else:
        content = text
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$")
_LABEL = re.compile(r"^(caption|prompt|video prompt|output)\s*:\s*", re.I)


def clean(text: Any, fallback: str, paragraphs: bool = False) -> str:
    """One paragraph of caption: no fences, labels, wrapping quotes or line breaks; nothing usable -> the idea itself
    (the guides say so too). `paragraphs` (PARAGRAPH_GUIDES): blank-line-separated parts stay apart, each one line."""
    if not isinstance(text, str):
        return fallback
    t = _FENCE.sub("", text.strip()).strip()
    t = _LABEL.sub("", t).strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    if paragraphs:
        parts = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t)]
        return "\n\n".join(p for p in parts if p) or fallback
    t = " ".join(t.split())
    return t or fallback
