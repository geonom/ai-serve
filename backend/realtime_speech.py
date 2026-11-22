"""Speaking a realtime conversation in another language than English (2026-11-03, owner via V-Noveller).

LFM3-Omni's conversation voice has an English accent in German ("Leitschirmwörterin"); its own TTS template, which
names the language, speaks clean German (Whisper: word-perfect). With `session.audio.output.language` (our extension
of the OpenAI realtime session, next to `audio.output.voice`) the gateway lets the model answer in text and speaks
each finished sentence with llama-server's POST /tts in that language, as `response.output_audio.delta` events of the
same protocol -- the client sees an ordinary spoken answer. English stays the model's own voice (fastest, ~0.06 s).
The rules here are pure; the WebSocket relay is in backend/routers/gateway.py.
"""
import re
from array import array
from typing import Any, Dict, List, Optional, Tuple

VOICES = ("female", "male")
# a sentence ends at . ! ? … : ; followed by whitespace -- "3.14" stays whole
_END = re.compile(r"[.!?…:;]+[\"'»«”)]*\s")
_SAYABLE = re.compile(r"\w")


_TOOL_CALL = "<tool_call>"


class SentenceBuffer:
    """Text deltas in, finished sentences out. The model also streams its tool calls as text
    (<tool_call><function=...>); from that marker on nothing is said -- not even when it is split across deltas."""

    def __init__(self):
        self._text = ""
        self._all = ""        # everything received
        self._fed = 0         # how much of it went into _text
        self._muted = False

    def _sayable(self, final: bool = False) -> str:
        if self._muted:
            return ""
        cut = self._all.find(_TOOL_CALL)
        if cut >= 0:
            self._muted = True
            usable = self._all[:cut]
        elif final:
            usable = self._all
        else:   # hold back an ending that might be the start of the marker
            hold = next((n for n in range(min(len(_TOOL_CALL) - 1, len(self._all)), 0, -1)
                         if _TOOL_CALL.startswith(self._all[-n:])), 0)
            usable = self._all[:len(self._all) - hold]
        new = usable[self._fed:] if len(usable) > self._fed else ""
        self._fed = max(self._fed, len(usable))
        return new

    def add(self, delta: str) -> List[str]:
        self._all += delta
        self._text += self._sayable()
        out = []
        while True:
            m = _END.search(self._text)
            if not m:
                break
            sentence, self._text = self._text[:m.end()].strip(), self._text[m.end():]
            if _SAYABLE.search(sentence):
                out.append(sentence)
        return out

    def flush(self) -> List[str]:
        self._text += self._sayable(final=True)
        rest, self._text = self._text.strip(), ""
        return [rest] if _SAYABLE.search(rest) else []


def f32le_to_pcm16(data: bytes, carry: bytes) -> Tuple[bytes, bytes]:
    """llama-server's /tts pcm (float32 LE) -> PCM16 LE; a chunk may end inside a sample, that part is carried."""
    data = carry + data
    whole = len(data) - len(data) % 4
    samples = array("f")
    samples.frombytes(data[:whole])
    if samples.itemsize != 4:          # pragma: no cover -- every platform we run on has 4-byte floats
        raise RuntimeError("float is not 32 bit here")
    out = array("h", (int(round(max(-1.0, min(1.0, s)) * 32767)) for s in samples))
    return out.tobytes(), data[whole:]


def _wants_audio(modalities: Optional[List[str]]) -> bool:
    return modalities is None or "audio" in modalities


class SpeakState:
    """What one session asked for, and the client events rewritten for the model."""

    def __init__(self):
        self.language: Optional[str] = None
        self.voice = "female"
        self.modalities: Optional[List[str]] = None   # the client's, None = the model picks
        self.speaks_next = False                       # the response asked for last is spoken by the gateway

    @property
    def active(self) -> bool:
        lang = self.language or ""
        return bool(lang) and not (lang == "en" or lang.startswith("en-")) and _wants_audio(self.modalities)

    def client_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        t = ev.get("type")
        if t == "session.update" and isinstance(ev.get("session"), dict):
            session = ev["session"]
            if "output_modalities" in session:
                self.modalities = session["output_modalities"] if isinstance(session["output_modalities"], list) \
                    else None
            out = (session.get("audio") or {}).get("output") if isinstance(session.get("audio"), dict) else None
            if isinstance(out, dict):
                if isinstance(out.get("language"), str):
                    self.language = out.pop("language").strip().lower() or None
                if isinstance(out.get("voice"), str) and out["voice"].lower() in VOICES:
                    self.voice = out["voice"].lower()
            if self.active:
                session["output_modalities"] = ["text"]
        elif t == "response.create":
            resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            self.speaks_next = self.active and _wants_audio(resp.get("output_modalities", self.modalities))
            if self.speaks_next:
                resp["output_modalities"] = ["text"]
                ev["response"] = resp
        return ev
