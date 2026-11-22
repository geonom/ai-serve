"""Speech-to-speech models on /v1/realtime (2026-11-03, owner via V-Noveller: LFM3-Omni for voice conversations):
recognised from the GGUF header, loaded with --reasoning off, the WebSocket passed through to llama-server's own
/v1/realtime."""
import asyncio
import base64
import json
import threading

import pytest
from aiohttp import web
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.cluster.service import Upstream
from backend.llm_catalog import traits
from backend.main import app
from backend.model_manager import speech_flags
from backend.routers import gateway

TTS_TEMPLATE = "{{- 'Perform TTS in ' + lang -}}"


def test_a_model_with_a_tts_template_speaks():
    meta = {"general.architecture": "lfm3", "tokenizer.chat_template": "{{ messages }}<|audio_out|>",
            "tokenizer.chat_template.tts": TTS_TEMPLATE}
    t = traits(meta, "LFM3-Omni-2B-Q8_0.gguf")
    assert t["speech"] is True and t["kind"] == "chat"
    assert traits({"general.architecture": "gemma4", "tokenizer.chat_template": "x"}, "g.gguf")["speech"] is False


def test_speech_models_load_without_reasoning():
    assert speech_flags({"kind": "chat", "speech": True}) == ["--reasoning", "off"]
    assert speech_flags({"kind": "chat", "speech": False}) == []
    assert speech_flags(None) == []


def realtime_upstream():
    """A stand-in for llama-server's /v1/realtime: session.created, then echoes every client event back."""
    seen = {"queries": []}
    ready = threading.Event()

    async def handler(request):
        seen["queries"].append(dict(request.query))
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "session.created", "session": {"model": "/mnt/x/LFM3-Omni.gguf"}}))
        async for msg in ws:
            ev = json.loads(msg.data)
            if ev["type"] == "bye":
                await ws.close()
                break
            await ws.send_str(json.dumps({"type": "echo", "of": ev}))
        return ws

    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        a = web.Application()
        a.router.add_get("/v1/realtime", handler)
        runner = web.AppRunner(a)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        seen["port"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()
    threading.Thread(target=run, daemon=True).start()
    ready.wait(5)
    return seen, loop


@pytest.fixture
def fake(monkeypatch):
    seen, loop = realtime_upstream()
    asked, books = {}, []

    async def resolve(model_name, **load):
        asked.update(model=model_name, **load)
        return Upstream(kind="local", base=f"http://127.0.0.1:{seen['port']}", label="LFM3-Omni-2B-Q8_0.gguf",
                        port=seen["port"])
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: books.append("begin"))
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: books.append("end"))
    monkeypatch.setattr(gateway, "_default_speech_model", lambda: "/m/LFM3-Omni-2B-Q8_0.gguf")
    yield seen, asked, books
    loop.call_soon_threadsafe(loop.stop)


def client():
    return TestClient(app, client=("127.0.0.1", 50000))


def test_events_go_both_ways_and_the_model_is_named_by_file(fake):
    seen, asked, books = fake
    with client().websocket_connect("/v1/realtime?model=LFM3-Omni-2B-Q8_0.gguf") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["model"] == "LFM3-Omni-2B-Q8_0.gguf"     # not the absolute path
        ws.send_json({"type": "input_audio_buffer.append", "audio": "AAAA"})
        assert ws.receive_json() == {"type": "echo", "of": {"type": "input_audio_buffer.append", "audio": "AAAA"}}
    assert asked["model"] == "LFM3-Omni-2B-Q8_0.gguf" and asked["use_mtp"] is False
    assert seen["queries"] == [{}]
    assert books == ["begin", "end"]                  # the instance counts as busy for the whole session


def test_without_a_model_the_speech_model_is_taken_and_intent_passes(fake):
    seen, asked, _ = fake
    with client().websocket_connect("/v1/realtime?intent=transcription") as ws:
        ws.receive_json()
    assert asked["model"] == "/m/LFM3-Omni-2B-Q8_0.gguf"
    assert seen["queries"] == [{"intent": "transcription"}]


def test_upstream_closing_closes_the_client(fake):
    with client().websocket_connect("/v1/realtime") as ws:
        ws.receive_json()
        ws.send_json({"type": "bye"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_no_speech_model_says_where_to_get_one(monkeypatch):
    monkeypatch.setattr(gateway, "_default_speech_model", lambda: None)
    with client().websocket_connect("/v1/realtime") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error" and "LFM3-Omni-2B-GGUF" in ev["error"]["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_unknown_intent_is_refused(fake):
    with client().websocket_connect("/v1/realtime?intent=../x") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error" and "intent" in ev["error"]["message"]


def test_a_model_on_another_device_is_refused(fake, monkeypatch):
    async def resolve(model_name, **load):
        return Upstream(kind="node", base="http://100.64.0.9:9", label="x")
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    with client().websocket_connect("/v1/realtime") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error" and "this machine" in ev["error"]["message"]


# /v1/audio/speech with a speech model's file name -> that model's own /tts (2026-11-03: in the realtime conversation
# Omni speaks German with an English accent; its TTS template with lang=de is clean German)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend import model_manager


def tts_upstream():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"RIFFwave")

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.fixture
def speech(monkeypatch):
    srv, seen = tts_upstream()
    asked = {}
    monkeypatch.setattr(model_manager, "scan_all_gguf_paths",
                        lambda: ["/m/LFM3-Omni-2B-Q8_0.gguf", "/m/gemma-4-12b.gguf"])
    monkeypatch.setattr(model_manager, "get_model_traits",
                        lambda p: {"kind": "chat", "speech": "LFM3" in p})

    async def resolve(model_name, **load):
        asked.update(model=model_name, **load)
        return Upstream(kind="local", base=f"http://127.0.0.1:{srv.server_port}", label="LFM3-Omni-2B-Q8_0.gguf",
                        port=srv.server_port)
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
    yield seen, asked
    srv.shutdown()


def test_only_a_speech_models_file_name_picks_it():
    assert gateway.speech_model_path("tts-1") is None


def test_a_speech_model_speaks_in_the_asked_language(speech):
    seen, asked = speech
    r = client().post("/v1/audio/speech", json={"model": "lfm3-omni-2b-q8_0", "input": "Die Leserin.",
                                                "voice": "male", "language": "de", "seed": 7})
    assert r.status_code == 200, r.text
    assert r.content == b"RIFFwave" and r.headers["content-type"].startswith("audio/wav")
    assert asked["model"] == "/m/LFM3-Omni-2B-Q8_0.gguf"
    assert seen["path"] == "/tts"
    assert seen["body"] == {"input": "Die Leserin.", "lang": "de", "voice": "male", "seed": 7,
                            "response_format": "wav", "stream": False}


def test_other_voice_names_become_the_default_voice_and_stream_passes(speech):
    seen, _ = speech
    r = client().post("/v1/audio/speech", json={"model": "LFM3-Omni-2B-Q8_0.gguf", "input": "Hallo",
                                                "voice": "alloy", "stream": True})
    assert r.status_code == 200
    assert seen["body"]["voice"] == "female" and seen["body"]["stream"] is True and seen["body"]["lang"] == "en"
    assert "seed" not in seen["body"]


def test_a_chat_model_is_no_speech_model(speech):
    assert gateway.speech_model_path("gemma-4-12b.gguf") is None


# A realtime conversation in another language than English (2026-11-03): Omni's conversation voice has an English
# accent; with session.audio.output.language the gateway lets it answer in text and speaks each sentence with its
# own TTS template in that language, as response.output_audio.delta events of the same protocol.

import struct

from backend import realtime_speech as rs


def test_sentences_come_out_as_they_complete():
    b = rs.SentenceBuffer()
    assert b.add("Hallo Welt. Wie") == ["Hallo Welt."]
    assert b.add(" geht es dir? Pi ist 3.14 und") == ["Wie geht es dir?"]
    assert b.add(" fertig") == []
    assert b.flush() == ["Pi ist 3.14 und fertig"]
    assert b.flush() == []
    assert rs.SentenceBuffer().add("... ") == []          # nothing to say


def test_float_samples_become_pcm16_across_chunk_edges():
    raw = struct.pack("<3f", 0.5, -1.5, 1.0)
    pcm, carry = rs.f32le_to_pcm16(raw[:5], b"")
    assert pcm == struct.pack("<h", 16384) and len(carry) == 1
    pcm, carry = rs.f32le_to_pcm16(raw[5:], carry)
    assert pcm == struct.pack("<2h", -32767, 32767) and carry == b""


def test_the_language_field_switches_speaking_on_but_not_for_english():
    ev = {"type": "session.update", "session": {"instructions": "x",
                                                "audio": {"output": {"language": "DE", "voice": "male"}}}}
    speak = rs.SpeakState()
    out = speak.client_event(ev)
    assert speak.language == "de" and speak.voice == "male" and speak.active
    assert out["session"]["output_modalities"] == ["text"] and "language" not in out["session"]["audio"]["output"]
    en = rs.SpeakState()
    out = en.client_event({"type": "session.update", "session": {"audio": {"output": {"language": "en-gb"}}}})
    assert not en.active and "output_modalities" not in out["session"]
    text_only = rs.SpeakState()
    text_only.client_event({"type": "session.update", "session": {"output_modalities": ["text"],
                                                                   "audio": {"output": {"language": "de"}}}})
    assert not text_only.active                         # the caller wants no voice at all
    r = speak.client_event({"type": "response.create", "response": {"output_modalities": ["text", "audio"]}})
    assert r["response"]["output_modalities"] == ["text"]


def speaking_upstream(tts_failures=0, slots=None, deltas=("Hallo Welt. Wie", " geht's?")):
    """Omni stand-in: answers response.create with two text sentences; /tts streams float32 samples (after failing
    `tts_failures` times); /slots lists `slots` slots when given."""
    seen = {"events": [], "tts": [], "fail": tts_failures}
    ready = threading.Event()

    async def realtime(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        await ws.send_str(json.dumps({"type": "session.created", "session": {"model": "/m/LFM3.gguf"}}))
        async for msg in ws:
            ev = json.loads(msg.data)
            seen["events"].append(ev)
            if ev["type"] == "response.create":
                ids = {"response_id": "r1", "item_id": "i1", "output_index": 0, "content_index": 0}
                for e in ({"type": "response.created", "response": {"id": "r1"}},
                          *({"type": "response.output_text.delta", **ids, "delta": d} for d in deltas),
                          {"type": "response.output_text.done", **ids, "text": "Hallo Welt. Wie geht's?"},
                          {"type": "response.output_item.done", "response_id": "r1", "output_index": 0},
                          {"type": "response.done", "response": {"id": "r1", "status": "completed"}}):
                    await ws.send_str(json.dumps(e))
        return ws

    async def slots_list(request):
        if slots is None:
            return web.Response(status=404)
        return web.json_response([{"id": i, "is_processing": False} for i in range(slots)])

    async def tts(request):
        seen["tts"].append(await request.json())
        if seen["fail"] > 0:
            seen["fail"] -= 1
            return web.Response(status=503, text="no free slot")
        resp = web.StreamResponse(headers={"Content-Type": "audio/pcm;rate=24000;encoding=float;bits=32"})
        await resp.prepare(request)
        await resp.write(struct.pack("<2f", 0.5, 0.5))
        await resp.write_eof()
        return resp

    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        a = web.Application()
        a.router.add_get("/v1/realtime", realtime)
        a.router.add_post("/tts", tts)
        a.router.add_get("/slots", slots_list)
        runner = web.AppRunner(a)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        seen["port"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()
    threading.Thread(target=run, daemon=True).start()
    ready.wait(5)
    return seen, loop


def test_a_german_session_speaks_each_sentence_in_german(monkeypatch):
    seen, loop = speaking_upstream()
    asked = {}

    async def resolve(model_name, **load):
        asked.update(load)
        return Upstream(kind="local", base=f"http://127.0.0.1:{seen['port']}", label="LFM3.gguf", port=seen["port"])
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
    monkeypatch.setattr(gateway, "_default_speech_model", lambda: "/m/LFM3.gguf")
    try:
        with client().websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.update",
                          "session": {"audio": {"output": {"language": "de", "voice": "male"}}}})
            ws.send_json({"type": "response.create"})
            got = []
            while not got or got[-1]["type"] != "response.done":
                got.append(ws.receive_json())
        types = [e["type"] for e in got]
        audio = [e for e in got if e["type"] == "response.output_audio.delta"]
        assert len(audio) == 2 and audio[0]["response_id"] == "r1" and audio[0]["content_index"] == 1
        assert base64.b64decode(audio[0]["delta"]) == struct.pack("<2h", 16384, 16384)
        assert types.index("response.output_audio.done") < types.index("response.output_item.done")
        assert [t["input"] for t in seen["tts"]] == ["Hallo Welt.", "Wie geht's?"]
        assert seen["tts"][0]["lang"] == "de" and seen["tts"][0]["voice"] == "male"
        assert seen["events"][0]["session"]["output_modalities"] == ["text"]
        assert asked["parallel"] == 3                      # conversations hold a slot each, /tts needs one more
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_a_conversation_beyond_the_free_slots_is_refused_at_once(fake):
    """One slot always stays free for /tts: with 3 slots at most 2 conversations; a third would leave German speech
    waiting forever for a slot (2026-11-03: two open tabs blocked every spoken answer)."""
    seen, asked, _ = fake
    assert asked == {} or True
    c = client()
    with c.websocket_connect("/v1/realtime") as one, c.websocket_connect("/v1/realtime") as two:
        one.receive_json(), two.receive_json()
        assert asked["parallel"] == 3
        with c.websocket_connect("/v1/realtime") as three:
            ev = three.receive_json()
            assert ev["type"] == "error" and "conversations" in ev["error"]["message"]
    with c.websocket_connect("/v1/realtime") as again:        # the places are free again
        assert again.receive_json()["type"] == "session.created"


def speaking_fixture(monkeypatch, **kw):
    seen, loop = speaking_upstream(**kw)

    async def resolve(model_name, **load):
        return Upstream(kind="local", base=f"http://127.0.0.1:{seen['port']}", label="LFM3.gguf", port=seen["port"])
    monkeypatch.setattr(gateway, "_resolve_upstream", resolve)
    monkeypatch.setattr(gateway, "_begin_request", lambda up: None)
    monkeypatch.setattr(gateway, "_end_request", lambda up, keep: None)
    monkeypatch.setattr(gateway, "_default_speech_model", lambda: "/m/LFM3.gguf")
    return seen, loop


def german_turn(ws):
    ws.receive_json()
    ws.send_json({"type": "session.update", "session": {"audio": {"output": {"language": "de"}}}})
    ws.send_json({"type": "response.create"})
    got = []
    while not got or got[-1]["type"] != "response.done":
        got.append(ws.receive_json())
    return got


def test_a_sentence_whose_speech_fails_is_tried_once_more(monkeypatch):
    seen, loop = speaking_fixture(monkeypatch, tts_failures=1)
    try:
        with client().websocket_connect("/v1/realtime") as ws:
            got = german_turn(ws)
        assert [e for e in got if e["type"] == "error"] == []
        assert len([e for e in got if e["type"] == "response.output_audio.delta"]) == 2
        assert [t["input"] for t in seen["tts"]] == ["Hallo Welt.", "Hallo Welt.", "Wie geht's?"]
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_a_sentence_that_fails_twice_says_why(monkeypatch):
    seen, loop = speaking_fixture(monkeypatch, tts_failures=2)
    try:
        with client().websocket_connect("/v1/realtime") as ws:
            got = german_turn(ws)
        errors = [e for e in got if e["type"] == "error"]
        assert len(errors) == 1 and "503" in errors[0]["error"]["message"]
        assert len([e for e in got if e["type"] == "response.output_audio.delta"]) == 1   # the next sentence speaks
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_the_limit_follows_the_instances_real_slots(monkeypatch):
    """An instance adopted after a restart has no recorded slot count: ask llama-server (/slots). With 2 slots only
    one conversation fits next to /tts (2026-11-03: the adopted 2-slot Omni took two and a German answer stalled)."""
    seen, loop = speaking_fixture(monkeypatch, slots=2)
    try:
        c = client()
        with c.websocket_connect("/v1/realtime") as one:
            one.receive_json()
            with c.websocket_connect("/v1/realtime") as two:
                ev = two.receive_json()
                assert ev["type"] == "error" and "1 conversation" in ev["error"]["message"]
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_a_tool_call_in_the_text_is_never_spoken(monkeypatch):
    """Omni streams its tool call as text too (<tool_call><function=write_text>...); spoken, that is markup read
    aloud (2026-11-03). Everything from <tool_call> on stays silent."""
    seen, loop = speaking_fixture(monkeypatch, deltas=("Gern, ich schreibe sie. <tool", "_call>\n<function=write_text>\n",
                                                       "<parameter=instruction>\nEin Fuchs. Er schwimmt.\n</parameter>"))
    try:
        with client().websocket_connect("/v1/realtime") as ws:
            german_turn(ws)
        assert [t["input"] for t in seen["tts"]] == ["Gern, ich schreibe sie."]
    finally:
        loop.call_soon_threadsafe(loop.stop)
