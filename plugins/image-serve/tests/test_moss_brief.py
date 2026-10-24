"""The MOSS prompt enhancer (image_serve/moss_brief.py, 2026-10-24; asked for by V-Noveller, the owner said build it):
apps send only content -- a `medium` (cover, title card, document, poster, item, sign) and a `brief` (the literal
texts, what should be seen, the style, what must not be in it); image-serve builds the layout MOSS-Image does best
with (MOSS report: JSON layouts beat prose, text kept apart from objects, explicit line breaks). A language model
through ai-serve proposes the layout under rules per medium; the answer is checked strictly -- every literal text
exactly once and verbatim, no invented text, valid boxes -- and anything else falls back to a fixed template, so
there is always a picture. `must_not` only steers the layout model: MOSS has no negative prompt, and words like
"no faces" would put faces in (the same lesson as for Z-Image)."""
import json

import pytest

from image_serve import moss_brief as mb
from image_serve.moss_layout import parse_layout

BRIEF = {"texts": ["Buch 1", "Die letzten 48 Stunden"], "scene": "an ash field at dusk, a lone tower",
         "style": "dark cinematic, warm white serif type", "must_not": ["people", "logos"]}


def test_media_and_their_shapes():
    assert set(mb.MEDIA) == {"cover", "title_card", "document", "poster", "item", "sign"}
    assert mb.MEDIA["cover"].aspect == "3:4" and mb.MEDIA["title_card"].aspect == "16:9"
    assert all(m.aspect in ("1:1", "4:3", "3:4", "16:9", "9:16") for m in mb.MEDIA.values())


def test_a_brief_is_checked_like_untrusted_input():
    b = mb.parse_brief("title_card", BRIEF, prompt="")
    assert b.texts == ("Buch 1", "Die letzten 48 Stunden") and b.must_not == ("people", "logos")
    assert mb.parse_brief("item", {"texts": ["ID"]}, prompt="a key card").scene == "a key card"   # prompt = scene
    assert mb.parse_brief("cover", {"texts": ["Zeile 1\nZeile 2"]}, prompt="x").texts == ("Zeile 1\nZeile 2",)
    for medium, brief in [("comic", BRIEF), ("cover", "text"), ("cover", {"texts": "Buch 1"}),
                          ("cover", {"texts": ["x" * 201]}), ("cover", {"texts": [f"t{i}" for i in range(9)]}),
                          ("cover", {"texts": ["a\x07b"]}), ("cover", {"texts": ["ok"], "evil": 1}),
                          ("cover", {"texts": ["ok"], "scene": "s" * 601}), ("cover", {"texts": []}),
                          ("cover", {"texts": ["ok"], "must_not": ["x" * 101]})]:
        with pytest.raises(mb.BriefError):
            mb.parse_brief(medium, brief, prompt="p")


@pytest.mark.parametrize("medium", sorted(mb.MEDIA))
def test_every_template_holds_every_text_verbatim_and_is_a_valid_layout(medium):
    b = mb.parse_brief(medium, {**BRIEF, "texts": ["Erste Zeile", "Zweite\nZeile", "Dritte"]}, prompt="")
    summary, elements = mb.template_layout(medium, b)
    parse_layout([e for e in elements])                                    # same rules as an app's layout
    assert sorted(e["text"] for e in elements if e["kind"] == "text") == sorted(b.texts)
    assert summary and "people" not in summary and "logos" not in summary  # must_not never reaches MOSS


def test_the_layout_model_gets_the_rules_the_texts_and_must_not():
    b = mb.parse_brief("cover", BRIEF, prompt="")
    msgs = mb.llm_messages("cover", b)
    system, user = msgs[0]["content"], json.loads(msgs[1]["content"])
    assert mb.MEDIA["cover"].rules in system
    assert user["texts"] == [{"index": 1, "text": "Buch 1"}, {"index": 2, "text": "Die letzten 48 Stunden"}]
    assert user["must_not"] == ["people", "logos"]


def test_the_schema_asks_for_exactly_one_box_per_text():
    """Found live 2026-10-24: a small model (Gemma 4 E4B) wrote the text into the caption and left `text` out. So
    the model never copies the texts: it places them -- one box per given text, in order -- and the enhancer puts
    the texts in verbatim itself."""
    schema = mb.layout_schema(3)
    t = schema["properties"]["texts"]
    assert t["minItems"] == 3 and t["maxItems"] == 3 and "text" not in t["items"]["properties"]
    assert schema["properties"]["objects"]["items"]["properties"]["kind"]["enum"] == ["object", "character"]


def _answer(texts, objects=None, summary="a dark cinematic cover, ash field at dusk"):
    return json.dumps({"summary": summary, "texts": texts,
                       "objects": objects if objects is not None else
                       [{"box": [0.2, 0.45, 0.8, 0.95], "kind": "object", "caption": "a lone tower on an ash field"}]})


GOOD = [{"box": [0.1, 0.05, 0.9, 0.2], "caption": "small serif"},
        {"box": [0.1, 0.2, 0.9, 0.4], "caption": "bold serif title"}]


def test_a_good_answer_gets_the_texts_verbatim_in_order():
    b = mb.parse_brief("cover", BRIEF, prompt="")
    summary, elements = mb.accept_llm_layout(_answer(GOOD), b)
    assert summary.startswith("a dark cinematic cover") and len(elements) == 3
    assert [(e["kind"], e.get("text")) for e in elements[:2]] == [("text", "Buch 1"), ("text", "Die letzten 48 Stunden")]
    parse_layout(elements)


@pytest.mark.parametrize("bad", [
    "not json",
    _answer(GOOD[:1]),                                                                         # a box missing
    _answer(GOOD + GOOD[:1]),                                                                  # one too many
    _answer([{**GOOD[0], "box": [0.5, 0.5, 2, 2]}, GOOD[1]]),                                  # box off canvas
    _answer(GOOD, objects=[{"box": [0, 0, 1, 1], "kind": "text", "caption": "x"}]),            # text as an object
    _answer(GOOD, summary=""),
    _answer(GOOD, summary="a cover without people"),                                           # a must_not word
    _answer(GOOD, objects=[{"box": [0, 0, 1, 1], "kind": "object", "caption": "no logos anywhere"}]),
])
def test_anything_else_is_refused(bad):
    b = mb.parse_brief("cover", BRIEF, prompt="")
    with pytest.raises(mb.BriefError):
        mb.accept_llm_layout(bad, b)


def test_summaries_read_right():
    b = mb.parse_brief("item", {"texts": ["ID"], "scene": "a key card"}, prompt="")
    assert mb.template_layout("item", b)[0].startswith("An inventory item: a key card")
