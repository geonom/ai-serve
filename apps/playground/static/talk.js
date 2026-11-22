// "Speech > Talk": a voice conversation with a speech-to-speech model (LFM3-Omni) over the gateway's
// /v1/realtime WebSocket, tunnelled by this server (policy.wants_websocket).
// - English: the model listens and answers with its own voice (~0.06 s). Other languages: the gateway lets it
//   answer in text and speaks each sentence with the model's TTS in that language (session.audio.output.language;
//   its own conversation voice has an English accent in German, 2026-11-03).
// - Hands-free: the microphone stays open; a pause ends the turn, speaking while Omni talks interrupts it.
// - Tools: Omni decides what to do with the text beside the conversation (write, continue, change, look up);
//   a bigger chat model through the gateway does the writing (Omni's own writing was plain and it looped on
//   read/append in tests), Omni says what happens. One tool round per turn, then a spoken answer (tool_choice none).
// Pure rules first (tests: node --test tests/talk.test.mjs).

const TalkRules = (() => {
    const LANGUAGES = [   // the model's speech languages that have a name here; en = the model's own voice
        ['en', 'English'], ['de', 'German'], ['fr', 'French'], ['es', 'Spanish'], ['it', 'Italian'],
        ['nl', 'Dutch'], ['pl', 'Polish'], ['pt', 'Portuguese'], ['sv', 'Swedish'],
    ];
    const IN_RATE = 16000;
    const langName = (lang) => (LANGUAGES.find(([c]) => c === lang) || [lang, lang])[1];

    // Models that listen and speak (GET /v1/llm/models marks them `speech`); writers = other chat models.
    function speechModels(catalog) {
        return ((catalog && catalog.models) || []).filter(m => m.speech);
    }
    // Gemma 4 12B first, other Gemmas, the rest. Not "a loaded model first": a loaded model is usually someone's
    // running job (2026-11-03 it picked V-Noveller's gemma-4-31B mid-run and queued behind it). Loaded ones are
    // marked in the list, the person may pick one.
    function writerModels(catalog) {
        const rows = ((catalog && catalog.models) || []).filter(m => m.kind === 'chat' && !m.speech && !m.exclusive);
        // within one family a loaded variant goes first: picking another Gemma 4 12B file loaded a second 12B next
        // to the running one (2026-11-03)
        const score = (m) => (/gemma-4-12b/i.test(m.id) ? 1 : /gemma/i.test(m.id) ? 2 : 3);
        return rows.slice().sort((a, b) => score(a) - score(b) || (b.loaded ? 1 : 0) - (a.loaded ? 1 : 0));
    }

    // After write/continue/change Omni should only say a short sentence. In English it does when the tool output
    // says so; in the other languages (text mode, the gateway speaks) it kept calling tools instead (4 of 4,
    // 2026-11-03) -- there the page says a fixed sentence in the model's own voice (/v1/audio/speech).
    const CONFIRM = {
        de: { write: 'Gern, ich schreibe es jetzt.', continue: 'Gut, ich schreibe weiter.', change: 'In Ordnung, ich ändere das.' },
        fr: { write: 'Avec plaisir, je l\'écris maintenant.', continue: 'D\'accord, je continue.', change: 'D\'accord, je le modifie.' },
        es: { write: 'Claro, lo escribo ahora.', continue: 'Vale, sigo escribiendo.', change: 'De acuerdo, lo cambio.' },
        it: { write: 'Volentieri, lo scrivo adesso.', continue: 'Va bene, continuo.', change: 'D\'accordo, lo cambio.' },
    };
    function confirmation(kind, lang) {
        return lang === 'en' ? null : ((CONFIRM[lang] || {})[kind] || null);
    }
    const TOOL_DONE = 'Done: the writer has started and the text appears on the page. Reply with one short sentence ' +
                      'only, e.g. "I\'m writing it now." Do not tell the text yourself.';

    // What of an answer is shown: the model also streams its tool calls as text (<tool_call><function=...>).
    function visibleAnswer(raw) {
        const s = String(raw || '');
        const cut = s.indexOf('<tool_call>');
        if (cut >= 0) return s.slice(0, cut).trimEnd();
        const tail = ['<tool_call', '<tool_cal', '<tool_ca', '<tool_c', '<tool_', '<tool', '<too', '<to', '<t', '<']
            .find(p => s.endsWith(p));
        return tail ? s.slice(0, s.length - tail.length) : s;
    }

    const TOOLS = [
        { type: 'function', name: 'write_text',
          description: 'Start a new story or text on the user\'s page, replacing what is there. Use only when the user asks you to write something new.',
          parameters: { type: 'object', properties: {
              instruction: { type: 'string', description: 'What to write, in the user\'s words (topic, kind, length, style).' },
              title: { type: 'string' } }, required: ['instruction'] } },
        { type: 'function', name: 'continue_text',
          description: 'Continue the story or text on the user\'s page. Use when the user asks to go on or continue.',
          parameters: { type: 'object', properties: {
              instruction: { type: 'string', description: 'How to continue, e.g. "one more paragraph where she finds the key".' } },
              required: ['instruction'] } },
        { type: 'function', name: 'change_text',
          description: 'Change the story or text on the user\'s page: rewrite a part, change a detail, shorten, change the tone.',
          parameters: { type: 'object', properties: {
              instruction: { type: 'string', description: 'The change, in the user\'s words, e.g. "in the second paragraph the ship has red sails".' } },
              required: ['instruction'] } },
        { type: 'function', name: 'look_up',
          description: 'Look a topic up in Wikipedia and get a short summary. Use when the user asks for facts you should check.',
          parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } },
    ];

    // The session.update for a language: the model card's system prompt exactly ("Respond with interleaved text and
    // audio."; further instructions go into a user turn, guidance()) -- a German prompt of our own ("one to three
    // short sentences") made Omni tell the story itself instead of calling write_text (0 of 3 vs 3 of 3, 2026-11-03).
    // Other languages than English also set the gateway's voice. Tools ride along.
    function sessionUpdate(lang, voice, withTools = true) {
        const session = { instructions: 'Respond with interleaved text and audio.', max_output_tokens: 1024,
                          tools: withTools ? TOOLS : [] };
        if (lang !== 'en') session.audio = { output: { language: lang, voice: voice === 'male' ? 'male' : 'female' } };
        return { type: 'session.update', session };
    }

    // The house rules, sent once as the first user turn (the model card wants the system prompt exact).
    // An example sentence only in English: in a German conversation the English one was just said back ("I'll
    // write it now.", no tool) and a German one too ("Gern, ich schreibe …", no tool); without one, 4 of 5 German
    // spoken requests got the right tool (2026-11-03, the page's own audio).
    function guidance(lang) {
        const example = lang === 'en' ? 'I\'ll write it now.' : '';
        return 'Some notes for our conversation: we talk like two people. Keep your turns short and natural. ' +
            'Beside our conversation there is a text page. Use write_text, continue_text or change_text only when I ' +
            'ask you to write, continue or change the text, and look_up only when I ask for facts. Otherwise just talk. ' +
            'When you use one of these tools, do not write or tell the text yourself: say one short sentence' +
            (example ? `, like "${example}"` : '') + '. The text appears on the page. ' +
            (lang === 'en' ? '' : `Always answer in ${langName(lang)}. `) + 'You don\'t need to answer this note.';
    }

    // What the page knows about its text, told to Omni after a change (it doesn't read the whole text).
    function paragraphs(text) {
        return String(text || '').split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
    }
    function textNote(title, text) {
        const ps = paragraphs(text);
        if (!ps.length) return '(Note: the text page is empty now.)';
        const last = ps[ps.length - 1];
        return `(Note: the text page${title ? ` "${title}"` : ''} now has ${ps.length} paragraph${ps.length === 1 ? '' : 's'}. ` +
               `The last one begins: "${last.slice(0, 160)}${last.length > 160 ? '…' : ''}")`;
    }

    // The writer's messages for a tool call.
    function writerMessages(kind, text, instruction, lang) {
        const system = `You are a skilled author. Write in ${langName(lang)}. Output only the text itself: paragraphs ` +
            'separated by blank lines, no title line, no notes, no quotation marks around it.';
        const ask = instruction || (kind === 'continue' ? 'one more paragraph' : '');
        const user = kind === 'write' ? `Write this: ${ask}`
            : kind === 'continue' ? `Here is the text so far:\n\n${text}\n\nContinue it: ${ask}. Output only the new paragraphs.`
            : `Here is the text:\n\n${text}\n\nChange it: ${ask}. Output the whole revised text and nothing else.`;
        return [{ role: 'system', content: system }, { role: 'user', content: user }];
    }
    // The page's text after the writer answered (`out` may still be growing while it streams).
    function applyWriter(kind, before, out) {
        const o = String(out || '').trim();
        if (kind === 'continue') return before.trim() ? `${before.trim()}\n\n${o}` : o;
        return o;
    }

    function wikiUrl(lang, query) {
        const q = encodeURIComponent(String(query || '').slice(0, 200));
        const site = /^[a-z]{2}$/.test(lang) ? lang : 'en';
        return `https://${site}.wikipedia.org/w/api.php?action=query&format=json&origin=*&generator=search` +
               `&gsrsearch=${q}&gsrlimit=1&prop=extracts&exintro=1&explaintext=1&exchars=1200`;
    }
    function wikiSummary(data) {
        const pages = Object.values(((data || {}).query || {}).pages || {});
        if (!pages.length || !pages[0].extract) return null;
        return { title: pages[0].title, text: pages[0].extract.trim() };
    }

    // Who is talking, from the loudness of the microphone. Blocks (the AudioWorklet's ~2.7 ms) are pooled into
    // 30 ms windows -- judged one by one, every tiny gap between syllables reset the count and no turn ever started
    // (2026-11-03). A turn starts after `startS` of voice (short gaps only slow it down) and ends after `endS` of
    // quiet; while Omni plays, the bar is higher (its voice from the speakers isn't the user).
    class Turns {
        constructor({ startS = 0.25, endS = 0.8, minLevel = 0.012, windowS = 0.03 } = {}) {
            Object.assign(this, { startS, endS, minLevel, windowS, floor: 0.005, voiced: 0, quiet: 0, talking: false,
                                  acc: 0, accS: 0, accPlaying: false });
        }
        feed(level, seconds, playing = false) {
            this.acc += level * level * seconds;
            this.accS += seconds;
            this.accPlaying = this.accPlaying || playing;
            if (this.accS + 1e-9 < this.windowS) return null;
            const win = Math.sqrt(this.acc / this.accS), span = this.accS, wasPlaying = this.accPlaying;
            this.acc = 0; this.accS = 0; this.accPlaying = false;
            return this._judge(win, span, wasPlaying);
        }
        _judge(level, seconds, playing) {
            if (!this.talking) this.floor = level < this.floor ? level : this.floor * 0.98 + level * 0.02;
            const bar = Math.max(this.minLevel, this.floor * 3) * (playing ? 3 : 1);
            if (level > bar) {
                this.voiced += seconds;
                this.quiet = 0;
            } else {
                this.quiet += seconds;
                if (!this.talking) this.voiced = Math.max(0, this.voiced - seconds);
            }
            if (!this.talking && this.voiced >= this.startS) { this.talking = true; return 'start'; }
            if (this.talking && this.quiet >= this.endS) { this.talking = false; this.voiced = 0; return 'end'; }
            return null;
        }
    }
    function rms(samples) {
        let s = 0;
        for (let i = 0; i < samples.length; i++) s += samples[i] * samples[i];
        return samples.length ? Math.sqrt(s / samples.length) : 0;
    }

    // Microphone samples (the device's rate) -> 16 kHz, by averaging each step's samples.
    function downsample(samples, fromRate, toRate = IN_RATE) {
        if (fromRate === toRate) return Float32Array.from(samples);
        const ratio = fromRate / toRate;
        const out = new Float32Array(Math.floor(samples.length / ratio));
        for (let i = 0; i < out.length; i++) {
            const a = Math.floor(i * ratio), b = Math.min(samples.length, Math.floor((i + 1) * ratio));
            let sum = 0;
            for (let j = a; j < b; j++) sum += samples[j];
            out[i] = b > a ? sum / (b - a) : 0;
        }
        return out;
    }

    function toPcm16Base64(samples) {
        const bytes = new Uint8Array(samples.length * 2);
        const view = new DataView(bytes.buffer);
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(2 * i, Math.round(s * 32767), true);
        }
        let bin = '';
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        return btoa(bin);
    }

    function pcm16Base64ToFloat(b64) {
        const bin = atob(b64);
        const out = new Float32Array(bin.length >> 1);
        for (let i = 0; i < out.length; i++) {
            let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
            if (v >= 0x8000) v -= 0x10000;
            out[i] = v / 32768;
        }
        return out;
    }

    // "Talk with {model} in {lang}{voice}." -> pieces: plain text and the names of the choices, in order
    function sentencePieces(template) {
        return template.split(/(\{[a-z]+\})/).filter(Boolean).map(p => (/^\{[a-z]+\}$/.test(p) ? { pick: p.slice(1, -1) } : p));
    }

    return { LANGUAGES, IN_RATE, TOOLS, TOOL_DONE, confirmation, sentencePieces, speechModels, writerModels, visibleAnswer, sessionUpdate, guidance,
             paragraphs, textNote, writerMessages, applyWriter, wikiUrl, wikiSummary, Turns, rms,
             downsample, toPcm16Base64, pcm16Base64ToFloat };
})();
if (typeof module !== 'undefined') module.exports = TalkRules;

if (typeof document !== 'undefined') (function () {
    const R = TalkRules;
    const $ = (id) => document.getElementById(id);
    const STORE = 'playground-talk-text';
    const PREROLL_S = 0.4;          // audio kept from before a turn was recognised, so its first word isn't cut
    const IDLE_END_MS = 180000;     // an open conversation holds a slot of the model: a forgotten tab gives it back
    const state = { ws: null, mic: null, ctx: null, playAt: 0, sources: [], answering: false, answer: null,
                    turns: new R.Turns(), preroll: [], pending: [], talking: false, models: [], writers: [],
                    calls: [], afterTools: false, writing: null, sentSeconds: 0, frames: 0, loudest: 0 };
    let idleTimer = null;
    window.talkState = state;       // for a look from the browser console (frames, loudest level, turns)

    // ---------------------------------------------------------------- small helpers
    function choices(select, options, value) {
        select.innerHTML = '';
        for (const [v, label] of options) select.append(new Option(label, v, false, v === value));
    }
    function note(text, kind) {
        const s = $('talk-status');
        s.textContent = text || '';
        s.className = 'muted small' + (kind === 'error' ? ' talk-error' : '');
    }
    function turn(who, text) {
        const row = document.createElement('p');
        row.className = 'talk-turn';
        const name = document.createElement('span');
        name.className = 'talk-who';
        name.textContent = who + ' ';
        const body = document.createElement('span');
        body.textContent = text;
        row.append(name, body);
        $('talk-log').append(row);
        row.scrollIntoView({ block: 'nearest' });
        return body;
    }
    function send(ev) { if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(ev)); }
    function playing() { return !!state.ctx && state.playAt > state.ctx.currentTime; }
    function idleLater() {
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => { if (state.ws && !state.talking && !state.answering) end(tx('Conversation ended after a quiet while.')); }, IDLE_END_MS);
    }

    // ---------------------------------------------------------------- the text page
    // every change by the writer can be taken back: Omni sometimes picks a tool nobody asked for (write_text on
    // "how are you?" would replace the text)
    const undo = [];
    function keepForUndo() {
        undo.push({ title: $('talk-title').value, text: getText() });
        if (undo.length > 20) undo.shift();
        $('talk-undo').hidden = false;
    }
    function undoLast() {
        const last = undo.pop();
        if (last) { $('talk-title').value = last.title; setText(last.text); }
        $('talk-undo').hidden = !undo.length;
    }
    function getText() { return $('talk-text').value; }
    function setText(t) {
        $('talk-text').value = t;
        try { localStorage.setItem(STORE, JSON.stringify({ title: $('talk-title').value, text: t })); } catch (e) { /* private mode */ }
        $('talk-download').hidden = !t.trim();
    }
    function restoreText() {
        try {
            const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
            if (saved) { $('talk-title').value = saved.title || ''; setText(saved.text || ''); }
        } catch (e) { /* nothing saved */ }
        $('talk-download').hidden = !getText().trim();
    }
    function download() {
        const blob = new Blob([($('talk-title').value ? $('talk-title').value + '\n\n' : '') + getText()], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = ($('talk-title').value || 'text').replace(/[^\w\- äöüÄÖÜß]+/g, '').trim().slice(0, 60) + '.txt';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }

    // The writer model streams into the page; Omni hears a note when it is done.
    async function write(kind, instruction, title) {
        if (state.writing) return 'busy: the writer is still working on the last request';
        const model = $('talk-writer').value;
        if (!model) return 'no writer model available';
        const before = getText();
        if (before.trim() || $('talk-title').value) keepForUndo();
        if (kind === 'write' && title) $('talk-title').value = title;
        const lang = $('talk-lang').value;
        state.writing = { kind };
        $('talk-writing').textContent = { write: tx('writing…'), continue: tx('continuing…'), change: tx('changing…') }[kind];
        (async () => {
            let out = '';
            try {
                const r = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, messages: R.writerMessages(kind, before, instruction, lang), stream: true,
                                           thinking: false, max_tokens: 2000, temperature: 0.8 }) });
                if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || `HTTP ${r.status}`);
                const reader = r.body.getReader();
                const dec = new TextDecoder();
                let buf = '';
                for (;;) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    let i;
                    while ((i = buf.indexOf('\n')) >= 0) {
                        const line = buf.slice(0, i).trim();
                        buf = buf.slice(i + 1);
                        if (!line.startsWith('data:') || line === 'data: [DONE]') continue;
                        try {
                            const d = JSON.parse(line.slice(5)).choices?.[0]?.delta?.content;
                            if (d) { out += d; setText(R.applyWriter(kind, before, out)); }
                        } catch (e) { /* a keep-alive line */ }
                    }
                }
                setText(R.applyWriter(kind, before, out));
                send({ type: 'conversation.item.create', item: { type: 'message', role: 'user',
                       content: [{ type: 'input_text', text: R.textNote($('talk-title').value, getText()) }] } });
            } catch (err) {
                setText(before);
                note(tx('The writer failed: {err}', { err: err.message }), 'error');
            } finally {
                state.writing = null;
                $('talk-writing').textContent = '';
            }
        })();
        return R.TOOL_DONE;
    }

    async function lookUp(query) {
        try {
            const r = await fetch(R.wikiUrl($('talk-lang').value, query));
            const s = R.wikiSummary(await r.json());
            if (!s) return 'Nothing found in Wikipedia for that.';
            turn(tx('Wikipedia:'), `${s.title} – ${s.text.slice(0, 280)}${s.text.length > 280 ? '…' : ''}`);
            return `Wikipedia, "${s.title}": ${s.text}`;
        } catch (e) { return 'Wikipedia could not be reached.'; }
    }

    async function runTools(calls) {
        const lang = $('talk-lang').value;
        let say = null;
        for (const c of calls) {
            let args = {};
            try { args = JSON.parse(c.arguments || '{}'); } catch (e) { /* the model wrote broken JSON */ }
            const kind = { write_text: 'write', continue_text: 'continue', change_text: 'change' }[c.name];
            const output = kind ? await write(kind, args.instruction, args.title)
                : c.name === 'look_up' ? await lookUp(args.query) : 'unknown tool';
            if (kind && say === null) say = output === R.TOOL_DONE ? R.confirmation(kind, lang) : false;   // busy: Omni explains
            if (!kind) say = false;            // facts to tell: Omni answers itself
            send({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: c.call_id, output: String(output) } });
        }
        state.afterTools = true;            // now say what happened -- no second tool round
        if (say) return speakFixed(say, lang);
        state.answering = true;
        send({ type: 'response.create', response: { tool_choice: 'none', max_output_tokens: 80 } });
    }

    // A short sentence in the speech model's own voice, for languages where Omni won't confirm by itself.
    async function speakFixed(text, lang) {
        turn(tx('Omni:'), text);
        try {
            const r = await fetch('/v1/audio/speech', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: $('talk-model').value, input: text, language: lang, voice: $('talk-voice').value }) });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const audio = await state.ctx.decodeAudioData(await r.arrayBuffer());
            const src = state.ctx.createBufferSource();
            src.buffer = audio;
            src.connect(state.ctx.destination);
            const at = Math.max(state.ctx.currentTime + 0.02, state.playAt);
            src.start(at);
            state.playAt = at + audio.duration;
            state.sources.push(src);
            src.onended = () => { state.sources = state.sources.filter(x => x !== src); updateLine(); };
        } catch (e) { /* the text is in the log; speaking it is a nicety */ }
        state.answering = false;
        idleLater();
        updateLine();
    }

    // ---------------------------------------------------------------- audio
    function stopPlayback() {
        for (const s of state.sources) { try { s.stop(); } catch (e) { /* already ended */ } }
        state.sources = [];
        state.playAt = 0;
    }
    function play(b64, rate) {
        const samples = R.pcm16Base64ToFloat(b64);
        if (!samples.length) return;
        const buf = state.ctx.createBuffer(1, samples.length, rate);
        buf.copyToChannel(samples, 0);
        const src = state.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(state.ctx.destination);
        const at = Math.max(state.ctx.currentTime + 0.02, state.playAt);
        src.start(at);
        state.playAt = at + buf.duration;
        state.sources.push(src);
        src.onended = () => { state.sources = state.sources.filter(x => x !== src); updateLine(); };
        updateLine();
    }

    function sendAudio(samples) {
        state.sentSeconds += samples.length / R.IN_RATE;
        send({ type: 'input_audio_buffer.append', audio: R.toPcm16Base64(samples) });
    }
    function flushPending() {
        const n = state.pending.reduce((a, p) => a + p.length, 0);
        if (!n) return;
        const joined = new Float32Array(n);
        let at = 0;
        for (const p of state.pending) { joined.set(p, at); at += p.length; }
        state.pending = [];
        sendAudio(joined);
    }

    // every block of microphone samples: who is talking, and the audio of the user's turn to the model
    function onFrame(raw) {
        if (!state.ws) return;
        const pcm = R.downsample(raw, state.ctx.sampleRate);
        const seconds = pcm.length / R.IN_RATE;
        const level = R.rms(pcm);
        state.frames++;
        state.loudest = Math.max(state.loudest, level);
        const change = state.turns.feed(level, seconds, playing());
        if (change === 'start') {
            clearTimeout(idleTimer);
            if (state.answering || playing()) {         // the user speaks up: Omni stops
                send({ type: 'response.cancel' });
                stopPlayback();
                state.answering = false;
                state.answer = null;
            }
            state.talking = true;
            state.sentSeconds = 0;
            state.pending = state.preroll;
            state.preroll = [];
        }
        if (state.talking) {
            state.pending.push(pcm);
            if (state.pending.reduce((a, p) => a + p.length, 0) >= R.IN_RATE / 10) flushPending();
        } else {
            state.preroll.push(pcm);
            let n = state.preroll.reduce((a, p) => a + p.length, 0);
            while (n > PREROLL_S * R.IN_RATE && state.preroll.length > 1) n -= state.preroll.shift().length;
        }
        if (change === 'end') {
            flushPending();
            state.talking = false;
            turn(tx('You:'), tx('(spoken, {s} s)', { s: state.sentSeconds.toFixed(1) }));
            send({ type: 'input_audio_buffer.commit' });
            state.calls = [];
            state.afterTools = false;
            state.answering = true;
            send({ type: 'response.create' });
        }
        if (change) updateLine();
    }

    async function startMic() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true,
                                                                            noiseSuppression: true, autoGainControl: true } });
        await state.ctx.audioWorklet.addModule('talk-worklet.js');
        const source = state.ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(state.ctx, 'talk-capture');
        node.port.onmessage = (e) => onFrame(e.data);
        source.connect(node);
        state.mic = { stream, source, node };
    }

    // ---------------------------------------------------------------- the conversation
    function connect() {
        return new Promise((resolve, reject) => {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            const ws = new WebSocket(`${proto}://${location.host}/v1/realtime?model=${encodeURIComponent($('talk-model').value)}&keep_alive=1800`);
            let outRate = 24000, started = false;
            ws.onmessage = (msg) => {
                const ev = JSON.parse(msg.data);
                if (ev.type === 'session.created') {
                    started = true;
                    outRate = (((ev.session || {}).audio || {}).output || {}).format?.rate || 24000;
                    const lang = $('talk-lang').value;
                    ws.send(JSON.stringify(R.sessionUpdate(lang, $('talk-voice').value)));
                    const intro = R.guidance(lang) + (getText().trim() ? ' ' + R.textNote($('talk-title').value, getText()) : '');
                    ws.send(JSON.stringify({ type: 'conversation.item.create', item: { type: 'message', role: 'user',
                                             content: [{ type: 'input_text', text: intro }] } }));
                    idleLater();
                    resolve(ws);
                } else if (ev.type === 'response.output_text.delta') {
                    if (!state.answering) return;
                    state.answerRaw = (state.answer ? state.answerRaw : '') + (ev.delta || '');
                    const shown = R.visibleAnswer(state.answerRaw);
                    if (!state.answer && shown.trim()) state.answer = turn(tx('Omni:'), '');
                    if (state.answer) state.answer.textContent = shown;
                } else if (ev.type === 'response.output_audio.delta') {
                    if (state.answering) play(ev.delta, outRate);
                } else if (ev.type === 'response.output_item.done' && (ev.item || {}).type === 'function_call') {
                    state.calls.push(ev.item);
                } else if (ev.type === 'response.done') {
                    state.answer = null;
                    const calls = state.calls;
                    state.calls = [];
                    const status = (ev.response || {}).status;
                    if (calls.length && !state.afterTools && status !== 'cancelled' && state.answering) runTools(calls);
                    else { state.answering = false; idleLater(); }
                    updateLine();
                } else if (ev.type === 'error') {
                    const m = ((ev.error || {}).message) || tx('Something went wrong.');
                    if (!started) reject(new Error(m)); else note(m, 'error');
                }
            };
            ws.onerror = () => reject(new Error(tx('No connection to the gateway.')));
            ws.onclose = () => { if (state.ws === ws) end(); };
        });
    }

    function updateLine() {
        const on = !!state.ws;
        $('talk-btn').textContent = on ? tx('End conversation') : tx('Start conversation');
        $('talk-btn').classList.toggle('btn-primary', !on);
        if (!on) return;
        const t = state.talking ? tx('You are speaking…') : (playing() || state.answering) ? tx('Omni is speaking – just talk to interrupt.')
            : tx('Listening – just speak.');
        if (!$('talk-status').classList.contains('talk-error') || state.talking) note(t);
    }

    async function start() {
        try {
            state.ctx = state.ctx || new AudioContext();
            // not awaited: without an audio output device resume() never settles, and the page would just sit there
            state.ctx.resume().catch(() => {});
            note(tx('Connecting… (the first time the model loads, a few seconds)'));
            state.ws = await connect();
            if (!state.mic) await startMic();
            state.turns = new R.Turns();
            updateLine();
        } catch (err) {
            const ws = state.ws;
            state.ws = null;
            if (ws) ws.close();
            note(err.name === 'NotAllowedError' ? tx('The browser did not allow the microphone.') : err.message, 'error');
            updateLine();
        }
    }

    function end(message) {
        clearTimeout(idleTimer);
        const ws = state.ws;
        state.ws = null;
        state.talking = false;
        state.answering = false;
        state.answer = null;
        stopPlayback();
        if (ws) ws.close();
        if (state.mic) { state.mic.stream.getTracks().forEach(t => t.stop()); state.mic.source.disconnect(); state.mic = null; }
        note(message || tx('Conversation ended.'));
        updateLine();
    }

    // ---------------------------------------------------------------- the sentences with their choices
    function build(p, template, picks) {
        p.textContent = '';
        for (const piece of R.sentencePieces(template)) p.append(typeof piece === 'string' ? piece : picks[piece.pick]);
    }
    function buildSentences() {
        // take hold of the elements before emptying their parents: once out of the page, $() no longer finds them
        const model = $('talk-model'), lang = $('talk-lang'), voice = $('talk-voice'), voicePart = $('talk-voice-part');
        const writer = $('talk-writer');
        build(voicePart, tx(', with a {pick} voice'), { pick: voice });
        build($('talk-sentence'), tx('Talk with {model} in {lang}{voice}.'), { model, lang, voice: voicePart });
        build($('talk-writer-line'), tx('{writer} writes the text.'), { writer });
    }
    function updateSentence() {
        const english = $('talk-lang').value === 'en';
        $('talk-voice-part').hidden = english;
        $('talk-how').textContent = english ? tx('Omni listens and answers with its own voice, at once.')
            : tx('Omni answers in text; each sentence is read out in this language (a moment later, clean pronunciation).');
    }

    async function loadModels() {
        let catalog = null;
        try { catalog = await (await fetch('/v1/llm/models')).json(); } catch (e) { /* shown below */ }
        state.models = R.speechModels(catalog);
        state.writers = R.writerModels(catalog);
        if (!state.models.length) {
            choices($('talk-model'), [['', tx('no speech model')]], '');
            note(tx('No speech model on the gateway yet -- e.g. LiquidAI/LFM3-Omni-2B-GGUF (Q8_0 + its mmproj).'));
            $('talk-btn').disabled = true;
        } else {
            choices($('talk-model'), state.models.map(m => [m.id, m.label || m.id]), state.models[0].id);
            $('talk-btn').disabled = false;
        }
        choices($('talk-writer'), state.writers.length ? state.writers.map(m => [m.id, (m.label || m.id) + (m.loaded ? ' ' + tx('(loaded)') : '')]) : [['', tx('no writer model')]],
                (state.writers[0] || {}).id || '');
    }

    const german = typeof I18N !== 'undefined' && I18N.lang === 'de';
    choices($('talk-lang'), R.LANGUAGES.map(([c, n]) => [c, tx(n)]), 'en');
    choices($('talk-voice'), [['female', german ? 'weiblicher' : 'female'], ['male', german ? 'männlicher' : 'male']], 'female');
    buildSentences();
    // a new language, voice or model applies from the next conversation: the session prompt is set before the first turn
    for (const id of ['talk-lang', 'talk-voice', 'talk-model']) $(id).addEventListener('change', () => {
        updateSentence();
        if (state.ws) end(tx('Conversation ended – start again with the new choice.'));
    });
    $('talk-btn').addEventListener('click', () => (state.ws ? end() : start()));
    $('talk-download').addEventListener('click', download);
    $('talk-undo').addEventListener('click', undoLast);
    $('talk-text').addEventListener('input', () => setText(getText()));
    $('talk-title').addEventListener('input', () => setText(getText()));
    restoreText();
    updateSentence();
    updateLine();

    let loaded = false;
    $('speech-talk-btn').addEventListener('click', () => { if (!loaded) { loaded = true; loadModels(); } });
})();
