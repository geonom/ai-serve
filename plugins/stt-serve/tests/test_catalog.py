"""stt-serve describes its engines (owner, 2026-10-20: the Playground should offer whatever the gateway
can do, open for future models): which engines exist, whether they're set up, and the options each
takes as plain form fields -- every field name is a real STTTranscribeRequest field."""
from stt_serve.catalog import engine_catalog
from stt_serve.models import STTTranscribeRequest

FIELD_TYPES = {"bool", "choice", "int", "language", "list"}


def _cat(**ready):
    return engine_catalog(nemotron_ready=ready.get("nemotron", True), moss_ready=ready.get("moss", True),
                          loaded=ready.get("loaded", set()))


def test_both_engines_with_their_options():
    cat = _cat()
    assert [e["id"] for e in cat["engines"]] == ["whisperx", "moss"] and cat["default_engine"] == "whisperx"
    for e in cat["engines"]:
        assert e["label"] and e["description"] and e["available"] is True
        for f in e["fields"]:
            assert f["name"] in STTTranscribeRequest.model_fields, f["name"]
            assert f["type"] in FIELD_TYPES and f["label"]
            for k in (f.get("show_if") or {}):
                assert k in {x["name"] for x in e["fields"]}


def test_whisperx_diarize_backends_say_whether_they_are_set_up():
    wx = next(e for e in _cat(nemotron=False)["engines"] if e["id"] == "whisperx")
    backend = next(f for f in wx["fields"] if f["name"] == "diarize_backend")
    assert {c["value"]: c["available"] for c in backend["choices"]} == {"whisperx": True, "nemotron": False}
    assert backend["show_if"] == {"diarize": True}


def test_moss_not_set_up_is_listed_but_unavailable():
    moss = next(e for e in _cat(moss=False)["engines"] if e["id"] == "moss")
    assert moss["available"] is False and "not set up yet" in moss["description"].lower()
    assert {f["name"] for f in moss["fields"]} >= {"hotwords", "diarize"}


def test_loaded_state():
    cat = _cat(loaded={"moss"})
    assert {e["id"]: e["loaded"] for e in cat["engines"]} == {"whisperx": False, "moss": True}
