"""The MOSS prompt enhancer -- pure rules, no I/O (tests: tests/test_moss_brief.py).

Vocabulary: a *medium* is what kind of picture it is (cover, title card, document, poster, item, sign); a *brief* is
what an app knows -- the literal *texts* to show, the *scene* (what should be seen), the *style*, and *must_not*
(what must not be in it). From both the enhancer builds a MOSS-Image *layout* (moss_layout.py elements + a summary
caption), the way MOSS's report says it works best: a JSON layout rather than prose, text elements apart from
objects, the literal text verbatim with explicit line breaks.

How: a language model (through ai-serve, any OpenAI-compatible server) proposes the layout under the medium's rules
and a JSON schema; `accept_llm_layout` checks it strictly, and anything it refuses falls back to `template_layout`, a
fixed layout per medium -- so a request always gets a picture.

Threat model / rules: the brief comes from apps (and through them from people): bounded sizes, no control characters
except line breaks, no unknown fields. The layout model's answer is untrusted too: the same layout rules as an app's
own layout (parse_layout), every literal text exactly once and verbatim -- the model only places the texts, the
enhancer puts them in, so it can't "improve" a title or add a tagline -- and no must_not word in what MOSS gets.
must_not only steers the layout model: MOSS has no negative prompt, and "no people" in a prompt puts people in.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from image_serve.moss_layout import LayoutError, MAX_CAPTION, MAX_TEXT, parse_layout

MAX_TEXTS = 8
MAX_SCENE = 600
MAX_STYLE = 300
MAX_MUST_NOT = 8
MAX_MUST_NOT_LEN = 100
MAX_SUMMARY = 600
_FIELDS = {"texts", "scene", "style", "must_not"}
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")   # line breaks (\n) are allowed: MOSS breaks lines there


class BriefError(ValueError):
    pass


@dataclass(frozen=True)
class Medium:
    label: str
    aspect: str
    rules: str


MEDIA: Dict[str, Medium] = {
    "cover": Medium("a book cover", "3:4", (
        "A book or story cover. The first text is the title: large, in the upper third, centred. Further texts "
        "(subtitle, series, author) are smaller, below the title or at the bottom edge. One strong central motif "
        "from the scene fills the middle; keep the text areas calm.")),
    "title_card": Medium("a chapter title card", "16:9", (
        "A cinematic chapter or title card. The texts are centred in the middle band, the first smaller (e.g. the "
        "book or part), the second the large chapter title. The scene is an atmospheric full-frame background "
        "behind the text, darker or blurred where the text sits.")),
    "document": Medium("a document", "3:4", (
        "A document seen from the front, filling the frame: a letter, an ID card, a form, a report. Texts are lines "
        "on it from top to bottom in reading order, left-aligned, in a typeface fitting the document. The paper or "
        "card is the main object; small details (stamp, photo, signature) only if the scene asks for them.")),
    "poster": Medium("a poster", "3:4", (
        "A poster. The first text is the headline at the top, large; further texts (date, place, tagline) smaller "
        "at the bottom. The motif from the scene sits between them.")),
    "item": Medium("an inventory item", "1:1", (
        "One object seen clearly, centred, on a plain or softly blurred background, like a game inventory item or "
        "an evidence photo. The texts are written ON the object (a label, an engraving, a printed line), small "
        "boxes inside the object's box.")),
    "sign": Medium("a sign", "16:9", (
        "A sign, screen or board in its surroundings. The sign is the main object, large and readable; the texts "
        "sit on it, centred, the first one largest. The surroundings come from the scene.")),
}

_BOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
        "description": "[left, top, right, bottom], each 0..1 of the canvas"}


def layout_schema(n_texts: int) -> Dict[str, Any]:
    """The answer the layout model must give. It never writes the texts (found 2026-10-24: a small model put them
    into the caption and left `text` out): it places them -- exactly one box per given text, in their order -- and
    the enhancer puts each text in verbatim."""
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One sentence: the whole picture, positively worded."},
            "texts": {"type": "array", "minItems": n_texts, "maxItems": n_texts, "items": {
                "type": "object",
                "properties": {"box": _BOX, "caption": {
                    "type": "string", "description": "how the text looks: typeface, size, weight, colour"}},
                "required": ["box", "caption"]}},
            "objects": {"type": "array", "maxItems": 12, "items": {
                "type": "object",
                "properties": {"box": _BOX, "kind": {"type": "string", "enum": ["object", "character"]},
                               "caption": {"type": "string", "description": "what it is and looks like"}},
                "required": ["box", "kind", "caption"]}},
        },
        "required": ["summary", "texts", "objects"],
    }


@dataclass(frozen=True)
class Brief:
    medium: str
    texts: Tuple[str, ...]
    scene: str
    style: str
    must_not: Tuple[str, ...]


def _text(value: Any, field: str, limit: int, required: bool = False) -> str:
    if value is None or (isinstance(value, str) and not value.strip() and not required):
        if required:
            raise BriefError(f"brief: {field} is required")
        return ""
    if not isinstance(value, str) or not value.strip() or len(value) > limit or _CONTROL.search(value):
        raise BriefError(f"brief: {field} must be text of at most {limit} characters, no control characters")
    return value.strip()


def _texts(value: Any, field: str, most: int, limit: int, required: bool) -> Tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, list) or (required and not value) or len(value) > most:
        raise BriefError(f"brief: {field} must be a list of 1..{most} texts" if required
                         else f"brief: {field} must be a list of at most {most} texts")
    return tuple(_text(v, field, limit, required=True) for v in value)


def parse_brief(medium: Any, brief: Any, prompt: str) -> Brief:
    """Untrusted medium + brief (+ the request's prompt, used as the scene when the brief has none) -> Brief."""
    if medium not in MEDIA:
        raise BriefError(f"medium: one of {', '.join(MEDIA)}")
    if not isinstance(brief, dict):
        raise BriefError("brief: an object with texts, scene, style, must_not")
    unknown = set(brief) - _FIELDS
    if unknown:
        raise BriefError(f"brief: unknown field(s) {', '.join(sorted(map(str, unknown)))}")
    texts = _texts(brief.get("texts"), "texts", MAX_TEXTS, MAX_TEXT, required=True)
    scene = _text(brief.get("scene"), "scene", MAX_SCENE) or _text(prompt, "prompt", MAX_SCENE)
    return Brief(medium=medium, texts=texts, scene=scene, style=_text(brief.get("style"), "style", MAX_STYLE),
                 must_not=_texts(brief.get("must_not"), "must_not", MAX_MUST_NOT, MAX_MUST_NOT_LEN, required=False))


def _summary(medium: str, b: Brief) -> str:
    label = MEDIA[medium].label
    parts = [label[0].upper() + label[1:] + (f": {b.scene}" if b.scene else "")]
    if b.style:
        parts.append(b.style)
    return ". ".join(p.rstrip(".") for p in parts)[:MAX_SUMMARY] + "."


def _stack(texts, left, top, right, bottom, first_share=None):
    """Text boxes from top to bottom in [top, bottom]; the first one gets first_share of the height if given."""
    n = len(texts)
    out, y = [], top
    for i, t in enumerate(texts):
        if first_share and n > 1:
            h = (bottom - top) * (first_share if i == 0 else (1 - first_share) / (n - 1))
        else:
            h = (bottom - top) / n
        out.append((t, [left, round(y, 4), right, round(y + h, 4)]))
        y += h
    return out


def template_layout(medium: str, b: Brief) -> Tuple[str, List[Dict[str, Any]]]:
    """A fixed layout per medium -- the fallback when the layout model's answer is refused or it is unreachable."""
    t, style = list(b.texts), (b.style or "clean, legible type")
    scene = b.scene or MEDIA[medium].label

    def text_el(text, box, size):
        return {"box": box, "kind": "text", "text": text, "caption": f"{size} text, {style}"[:MAX_CAPTION]}

    def obj(box, caption):
        return {"box": box, "kind": "object", "caption": caption[:MAX_CAPTION]}

    if medium in ("cover", "poster"):
        els = [text_el(t[0], [0.08, 0.05, 0.92, 0.27], "large title")]
        els += [text_el(x, box, "small") for x, box in _stack(t[1:], 0.12, 0.8, 0.88, 0.95)] if len(t) > 1 else []
        els.append(obj([0.1, 0.3, 0.9, 0.78], scene))
    elif medium == "title_card":
        els = [obj([0.0, 0.0, 1.0, 1.0], f"atmospheric background: {scene}")]
        sizes = ["small"] + ["large title"] * (len(t) - 1) if len(t) > 1 else ["large title"]
        els += [text_el(x, box, sizes[i]) for i, (x, box) in
                enumerate(_stack(t, 0.1, 0.32, 0.9, 0.68, first_share=0.3 if len(t) > 1 else None))]
    elif medium == "document":
        els = [obj([0.04, 0.03, 0.96, 0.97], f"the document itself: {scene}")]
        els += [text_el(x, box, "printed") for x, box in _stack(t, 0.1, 0.08, 0.9, 0.08 + min(0.8, 0.1 * len(t)))]
    elif medium == "item":
        els = [obj([0.1, 0.1, 0.9, 0.9], f"the object, centred: {scene}")]
        els += [text_el(x, box, "small printed") for x, box in _stack(t, 0.25, 0.4, 0.75, 0.4 + min(0.4, 0.08 * len(t)))]
    else:   # sign
        els = [obj([0.15, 0.15, 0.85, 0.85], f"the sign, large and readable, in its surroundings: {scene}")]
        els += [text_el(x, box, "large" if i == 0 else "medium") for i, (x, box) in
                enumerate(_stack(t, 0.22, 0.25, 0.78, 0.75, first_share=0.5 if len(t) > 1 else None))]
    return _summary(medium, b), els


def llm_messages(medium: str, b: Brief) -> List[Dict[str, str]]:
    system = (
        "You lay out pictures for the image model MOSS-Image. Answer with JSON only: a one-sentence summary of the "
        "whole picture; `texts`: for each given text, in the given order, its box [left, top, right, bottom] in 0..1 "
        "of the canvas and a caption describing how it looks (typeface, size, weight, colour) -- do not repeat the "
        "text itself; `objects`: the things and characters in the picture, each with a box and a caption.\n"
        f"Medium: {MEDIA[medium].label} ({MEDIA[medium].aspect}). {MEDIA[medium].rules}\n"
        "Rules: text boxes are wide and high enough for their text and do not overlap each other. Describe "
        "everything positively: never mention what must not be in the picture -- simply leave it out of the "
        "summary and every caption.")
    user = {"medium": medium, "texts": [{"index": i + 1, "text": t} for i, t in enumerate(b.texts)],
            "scene": b.scene, "style": b.style, "must_not": list(b.must_not)}
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def _mentions(text: str, words: Tuple[str, ...]) -> bool:
    low = text.lower()
    return any(w.lower() in low for w in words if w.strip())


def accept_llm_layout(answer: str, b: Brief) -> Tuple[str, List[Dict[str, Any]]]:
    """The layout model's JSON answer (layout_schema) -> (summary, elements), or BriefError. The texts are the
    brief's, verbatim, in order -- the model only placed them."""
    try:
        data = json.loads(answer)
    except (TypeError, ValueError):
        raise BriefError("the layout model's answer is not JSON")
    if not isinstance(data, dict) or not isinstance(data.get("texts"), list) or not isinstance(data.get("objects"), list):
        raise BriefError("the layout model's answer has no texts/objects")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MAX_SUMMARY or _CONTROL.search(summary):
        raise BriefError("the layout model's summary is missing or too long")
    if len(data["texts"]) != len(b.texts):
        raise BriefError(f"the layout model placed {len(data['texts'])} texts, not {len(b.texts)}")
    raw: List[Dict[str, Any]] = []
    for text, t in zip(b.texts, data["texts"]):
        if not isinstance(t, dict):
            raise BriefError("a text placement is not an object")
        raw.append({"box": t.get("box"), "kind": "text", "text": text, "caption": t.get("caption")})
    for o in data["objects"]:
        if not isinstance(o, dict) or o.get("kind") not in ("object", "character"):
            raise BriefError("an object must be of kind object or character")
        raw.append({"box": o.get("box"), "kind": o["kind"], "caption": o.get("caption")})
    try:
        elements = parse_layout(raw)
    except LayoutError as e:
        raise BriefError(f"the layout model's layout: {e}")
    if _mentions(summary, b.must_not) or any(_mentions(el.caption, b.must_not) for el in elements):
        raise BriefError("the layout model mentioned something that must not be in the picture")
    return summary.strip(), raw
