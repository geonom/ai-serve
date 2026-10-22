// "Speech": text-to-speech and transcription, built from the gateway's GET /v1/audio/models. Every engine
// tts-serve and stt-serve offer shows up with the inputs it describes for itself (modes, reference clip,
// transcript, style text or tags, preset speakers, numeric params; for transcription plain form fields), so
// a new engine needs no change here (owner, 2026-10-20: "möglichst offen für zukünftige Modelle").
// Pure rules first (tests: node --test tests/speech.test.mjs), the page below.

const SpeechRules = (() => {
    // Which transcription fields apply right now (show_if = only while other fields have these values).
    function visibleFields(fields, values) {
        return (fields || []).filter(f => !f.show_if || Object.entries(f.show_if).every(([k, v]) => values[k] === v));
    }

    // The /api/stt/transcribe body: engine, audio, and every visible field the person filled in.
    function sttPayload(engine, values, audio) {
        const body = { engine: engine.id, ...audio };
        for (const f of visibleFields(engine.fields, values)) {
            const v = values[f.name];
            if (v === undefined || v === null || v === '') continue;
            if (f.type === 'int') { const n = parseInt(v, 10); if (!Number.isNaN(n)) body[f.name] = n; }
            else if (f.type === 'list') {
                const items = String(v).split('\n').map(x => x.trim()).filter(Boolean);
                if (items.length) body[f.name] = items;
            } else body[f.name] = v;
        }
        return body;
    }

    // What a TTS mode still needs before it can run; [] = ready. `have` says what the person provided.
    function missingInputs(mode, have) {
        const names = { reference: tx('a recording'), ref_text: tx('the recording\'s transcript'),
                        instruct: tx('a style'), speaker: tx('a speaker') };
        return Object.entries(mode.inputs || {})
            .filter(([k, need]) => need === 'required' && !have[k])
            .map(([k]) => names[k] || k);
    }

    // The /api/tts/generate body. Params only when the person changed them (else the engine's default).
    function ttsPayload(engine, mode, v) {
        const body = { engine: engine.id, text: v.text, language: v.language || I18N.lang };
        if (mode.variant) body.variant = mode.variant;
        const inputs = mode.inputs || {};
        if (inputs.reference) {
            if (v.refVoicePath) body.ref_voice_path = v.refVoicePath;
            else if (v.refAudioBase64) body.ref_audio_base64 = v.refAudioBase64;
        }
        if (inputs.ref_text && v.refText) body.ref_text = v.refText;
        if (inputs.instruct && v.instruct) body.instruct = v.instruct;
        if (inputs.speaker && v.speaker) body.speaker = v.speaker;
        const params = {};
        for (const p of mode.params || []) if (v.params && v.params[p.name] !== undefined) params[p.name] = v.params[p.name];
        if (Object.keys(params).length) body.params = params;
        return body;
    }

    // Tag groups (e.g. OmniVoice): at most one tag per group; clicking a selected tag clears its group.
    function toggleTag(groups, selected, tag) {
        const group = groups.find(g => g.includes(tag)) || [tag];
        const without = selected.filter(t => !group.includes(t));
        return selected.includes(tag) ? without : [...without, tag];
    }

    function srtTime(s) {
        const ms = Math.max(0, Math.round((s || 0) * 1000));
        const p = (n, w = 2) => String(n).padStart(w, '0');
        return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
    }

    function toSrt(segments) {
        return (segments || []).filter(s => (s.text || '').trim()).map((s, i) =>
            `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.speaker ? s.speaker + ': ' : ''}${s.text.trim()}\n`
        ).join('\n');
    }

    // Readable result: with speakers one line per turn ("[0:12] SPEAKER_00: …"), else the plain text.
    function readable(resp) {
        const segs = resp.segments || [];
        if (!segs.some(s => s.speaker)) return resp.transcript || resp.text || '';
        const t = s => { const x = Math.floor(s || 0); return `${Math.floor(x / 60)}:${String(x % 60).padStart(2, '0')}`; };
        return segs.filter(s => (s.text || '').trim()).map(s => `[${t(s.start)}] ${s.speaker || '?'}: ${s.text.trim()}`).join('\n');
    }

    return { visibleFields, sttPayload, missingInputs, ttsPayload, toggleTag, toSrt, readable };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SpeechRules;

if (typeof document !== 'undefined' && document.getElementById('speech-tab')) (() => {
    const R = SpeechRules;
    const COMMON_LANGS = ['de', 'en', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ru', 'sv', 'tr', 'ja', 'zh'];
    const langName = (() => {
        try { const d = new Intl.DisplayNames([I18N.lang], { type: 'language' }); return c => d.of(c) || c; }
        catch (e) { return c => c; }
    })();
    let catalog = null, voices = [];
    const tts = { engine: null, mode: null, refAudioBase64: null, tags: [], params: {} };
    const stt = { engine: null, values: {}, audioBase64: null, last: null };

    function h(tag, attrs = {}, ...kids) {
        const el = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (k === 'on') for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
            else if (k === 'text') el.textContent = v;
            else if (v === true) el.setAttribute(k, '');
            else if (v !== false && v != null) el.setAttribute(k, v);
        }
        for (const k of kids.flat()) if (k != null) el.append(k);
        return el;
    }
    // labels and descriptions that come from the engines (plugins) are English; tx() shows them in German when known
    const group = (label, ...kids) => h('div', { class: 'input-group' }, h('label', { text: tx(label) }), ...kids);
    const note = text => h('div', { class: 'muted small', text: text && tx(text) });
    const need = v => v === 'required' ? '' : ' ' + tx('(optional)');

    function readFileBase64(file) {
        return new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(String(r.result).split(',')[1]);
            r.onerror = () => reject(r.error);
            r.readAsDataURL(file);
        });
    }

    function audioPicker(label, onChange, emptyLabel) {
        const sel = h('select', {}, h('option', { value: '', text: emptyLabel }),
            voices.map(v => h('option', { value: v.path, text: v.name.replace(/\.[a-z0-9]+$/i, '') })));
        const file = h('input', { type: 'file', accept: 'audio/*,video/*' });
        sel.addEventListener('change', () => { if (sel.value) file.value = ''; onChange({ path: sel.value || null }); });
        file.addEventListener('change', async () => {
            if (!file.files[0]) return onChange({});
            sel.value = '';
            onChange({ base64: await readFileBase64(file.files[0]) });
        });
        return group(label, sel, file);
    }

    function languageSelect(codes, value, onChange, auto) {
        const list = codes && codes.length ? codes : COMMON_LANGS;
        const sel = h('select', { on: { change: () => onChange(sel.value) } },
            auto ? h('option', { value: '', text: tx('detect automatically') }) : null,
            list.map(c => h('option', { value: c, text: langName(c) })));
        sel.value = value && (list.includes(value) || (auto && !value)) ? value : (auto ? '' : (list.includes(I18N.lang) ? I18N.lang : list[0]));
        onChange(sel.value);
        return sel;
    }

    // ------------------------------------------------------------------ text to speech
    function renderTts() {
        const form = document.getElementById('tts-form');
        const noteEl = document.getElementById('tts-engine-note');
        form.replaceChildren();
        const t = catalog && catalog.tts;
        if (!t || !t.available) { noteEl.textContent = (t && t.reason) || tx('Speech output unreachable.'); return; }
        noteEl.textContent = tx('{n} engines', { n: t.engines.length });
        if (!tts.engine || !t.engines.some(e => e.id === tts.engine.id)) {
            tts.engine = t.engines.find(e => e.id === t.default_engine) || t.engines[0];
            tts.mode = null;
        }
        const e = tts.engine;
        if (!tts.mode || !e.modes.some(m => m.variant === tts.mode.variant)) tts.mode = e.modes[0];
        const m = tts.mode;

        const engSel = h('select', { on: { change: () => {
            tts.engine = t.engines.find(x => x.id === engSel.value); tts.mode = null; tts.params = {}; tts.tags = []; renderTts();
        } } }, t.engines.map(x => h('option', { value: x.id, text: x.label + (x.loaded ? ' · ' + tx('loaded') : '') })));
        engSel.value = e.id;
        form.append(group('Engine', engSel, h('div', { class: 'muted small', text: [e.description && tx(e.description), e.license ? tx('License: {l}', { l: e.license }) : ''].filter(Boolean).join(' · ') })));

        if (e.modes.length > 1) {
            const modeSel = h('select', { on: { change: () => {
                tts.mode = e.modes.find(x => String(x.variant) === modeSel.value); tts.params = {}; renderTts();
            } } }, e.modes.map(x => h('option', { value: String(x.variant), text: tx(x.label) })));
            modeSel.value = String(m.variant);
            form.append(group('Kind', modeSel, m.description ? note(m.description) : null));
        } else if (m.description) form.append(note(m.description));

        form.append(group('Language', languageSelect(e.languages, tts.language, v => { tts.language = v; })));

        const inp = m.inputs || {};
        if (inp.reference) {
            form.append(audioPicker(tx('Recording to clone') + need(inp.reference) + (m.default_voice ? ' ' + tx('– empty = default voice') : ''),
                ({ path, base64 }) => {
                    tts.refVoicePath = path || null; tts.refAudioBase64 = base64 || null;
                    const v = voices.find(x => x.path === path);
                    if (inp.ref_text && v && v.transcript) { tts.refText = v.transcript; const ta = document.getElementById('tts-ref-text'); if (ta) ta.value = v.transcript; }
                }, inp.reference === 'required' ? tx('– choose or upload a recording –') : tx('– none –')));
        }
        if (inp.ref_text) {
            const ta = h('textarea', { id: 'tts-ref-text', rows: 3, placeholder: tx('What is said in the recording') });
            ta.value = tts.refText || '';
            ta.addEventListener('input', () => { tts.refText = ta.value.trim(); });
            form.append(group(tx('Transcript of the recording') + need(inp.ref_text), ta));
        }
        if (inp.speaker) {
            const sp = h('select', { on: { change: () => { tts.speaker = sp.value; } } },
                (m.speakers || []).map(x => h('option', { value: x, text: x.replace(/_/g, ' ') })));
            if (m.speakers.includes(tts.speaker)) sp.value = tts.speaker; else tts.speaker = sp.value;
            form.append(group(tx('Speaker') + need(inp.speaker), sp));
        }
        if (inp.instruct) form.append(instructInput(m.instruct || { kind: 'text' }, inp.instruct));
        for (const p of m.params || []) form.append(paramInput(p));
    }

    function instructInput(spec, required) {
        if (spec.kind === 'tags') {
            const box = h('div', { class: 'tag-selector-box' });
            const draw = () => {
                box.replaceChildren(...spec.groups.map(g => h('div', { class: 'tag-row' }, g.map(tag =>
                    h('button', { type: 'button', class: 'btn-tag' + (tts.tags.includes(tag) ? ' active' : ''), text: tag,
                                  on: { click: () => { tts.tags = R.toggleTag(spec.groups, tts.tags, tag); tts.instruct = tts.tags.join(', '); draw(); } } })))));
            };
            draw();
            return group(tx('Voice from keywords') + need(required), spec.hint ? note(spec.hint) : null, box);
        }
        const ta = h('textarea', { rows: 3, placeholder: spec.hint ? tx(spec.hint) : tx('Describe the style') });
        ta.value = tts.instruct || '';
        ta.addEventListener('input', () => { tts.instruct = ta.value.trim(); });
        const ex = (spec.examples || []).map(tx).map(x => h('button', { type: 'button', class: 'btn-tag', text: x,
            on: { click: () => { ta.value = x; tts.instruct = x; } } }));
        return group(tx('Style') + need(required), ta, ex.length ? h('div', { class: 'tag-row flex-wrap' }, ex) : null);
    }

    function paramInput(p) {
        const on = tts.params[p.name] !== undefined;
        const start = on ? tts.params[p.name] : (p.default !== undefined ? p.default : (p.min + p.max) / 2);
        const range = h('input', { type: 'range', min: p.min, max: p.max, step: p.step || (p.max - p.min) / 100, value: start, disabled: !on });
        const out = h('span', { class: 'muted small', text: on ? String(start) : tx('default') });
        const box = h('input', { type: 'checkbox', checked: on });
        box.addEventListener('change', () => {
            range.disabled = !box.checked;
            if (box.checked) { tts.params[p.name] = parseFloat(range.value); out.textContent = range.value; }
            else { delete tts.params[p.name]; out.textContent = tx('default'); }
        });
        range.addEventListener('input', () => { tts.params[p.name] = parseFloat(range.value); out.textContent = range.value; });
        return group(p.label, h('label', { class: 'small' }, box, ' ' + tx('adjust') + ' ', out), range);
    }

    document.getElementById('tts-generate-btn').addEventListener('click', async () => {
        const status = document.getElementById('tts-status');
        const text = document.getElementById('tts-text').value.trim();
        if (!tts.engine) return;
        if (!text) { status.textContent = tx('Enter a text first.'); return; }
        const m = tts.mode;
        const reference = !!(tts.refVoicePath || tts.refAudioBase64) || !!m.default_voice;
        const missing = R.missingInputs(m, { reference, ref_text: !!tts.refText, instruct: !!tts.instruct, speaker: !!tts.speaker });
        if (missing.length) { status.textContent = tx('Still missing: {what}.', { what: missing.join(', ') }); return; }
        const body = R.ttsPayload(tts.engine, m, { ...tts, text });
        const btn = document.getElementById('tts-generate-btn');
        btn.disabled = true;
        status.textContent = tx('{engine} is speaking… (the first time, the engine is loaded)', { engine: tts.engine.label });
        const t0 = performance.now();
        try {
            const r = await api('/api/tts/generate', 'POST', body);
            const src = r.audio_base64.startsWith('data:') ? r.audio_base64 : `data:audio/wav;base64,${r.audio_base64}`;
            document.getElementById('tts-result-audio').src = src;
            document.getElementById('tts-download').href = src;
            document.getElementById('tts-result').hidden = false;
            status.textContent = `${tx('Done')} · ${tts.engine.label} · ${tx(m.label)} · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
            addLog(tx('Speech generated ({engine}, {mode})', { engine: tts.engine.label, mode: tx(m.label) }), 'success');
        } catch (err) {
            status.textContent = tx('Failed: {err}', { err: err.message });
            addLog(tx('Speech: {err}', { err: err.message }), 'error');
        } finally {
            btn.disabled = false;
        }
    });

    // ------------------------------------------------------------------ transcription
    function renderStt() {
        const form = document.getElementById('stt-form');
        const noteEl = document.getElementById('stt-engine-note');
        form.replaceChildren();
        const s = catalog && catalog.stt;
        if (!s || !s.available) { noteEl.textContent = (s && s.reason) || tx('Transcription unreachable.'); return; }
        noteEl.textContent = tx('File or saved recording');
        if (!stt.engine || !s.engines.some(e => e.id === stt.engine.id)) {
            stt.engine = s.engines.find(e => e.id === s.default_engine && e.available) || s.engines.find(e => e.available) || s.engines[0];
            stt.values = {};
        }
        const e = stt.engine;
        for (const f of e.fields || []) if (stt.values[f.name] === undefined && f.default !== undefined) stt.values[f.name] = f.default;

        const engSel = h('select', { on: { change: () => { stt.engine = s.engines.find(x => x.id === engSel.value); stt.values = {}; renderStt(); } } },
            s.engines.map(x => h('option', { value: x.id, disabled: !x.available, text: x.label + (x.available ? '' : ' ' + tx('(not set up)')) })));
        engSel.value = e.id;
        form.append(group('Engine', engSel, note(e.description)));
        form.append(audioPicker(tx('Recording'), ({ path, base64 }) => { stt.audioPath = path || null; stt.audioBase64 = base64 || null; }, tx('– upload a file –')));
        for (const f of R.visibleFields(e.fields, stt.values)) form.append(fieldInput(f));
    }

    function fieldInput(f) {
        const set = v => { stt.values[f.name] = v; if (f.type === 'bool' || f.type === 'choice') renderStt(); };
        const v = stt.values[f.name];
        let el;
        if (f.type === 'bool') {
            el = h('input', { type: 'checkbox', checked: !!v, on: { change: () => set(el.checked) } });
            return h('div', { class: 'input-group' }, h('label', {}, el, ' ' + tx(f.label)), f.help ? note(f.help) : null);
        }
        if (f.type === 'choice') {
            el = h('select', { on: { change: () => set(el.value) } },
                f.choices.map(c => h('option', { value: c.value, disabled: c.available === false,
                                                  text: tx(c.label) + (c.available === false ? ' ' + tx('(not set up)') : '') })));
            el.value = v != null ? v : f.default;
        } else if (f.type === 'language') {
            el = languageSelect(null, v, x => { stt.values[f.name] = x || null; }, true);
        } else if (f.type === 'int') {
            el = h('input', { type: 'number', min: f.min, max: f.max, value: v != null ? v : '', on: { input: () => { stt.values[f.name] = el.value; } } });
        } else if (f.type === 'list') {
            el = h('textarea', { rows: 3, on: { input: () => { stt.values[f.name] = el.value; } } });
            el.value = v || '';
        } else {
            el = h('input', { type: 'text', value: v || '', on: { input: () => { stt.values[f.name] = el.value; } } });
        }
        return group(f.label, el, f.help ? note(f.help) : null);
    }

    document.getElementById('stt-srt-btn').addEventListener('click', () => {
        if (!stt.last) return;
        const blob = new Blob([R.toSrt(stt.last.segments)], { type: 'text/plain' });
        const a = h('a', { href: URL.createObjectURL(blob), download: tx('transcript') + '.srt' });
        document.body.append(a); a.click(); a.remove();
    });
    document.getElementById('stt-copy-btn').addEventListener('click', async () => {
        const text = document.getElementById('stt-result').textContent;
        if (text) logCopyResult(await copyTextToClipboard(text), tx('Transcript'));
    });

    async function transcribe() {
        const status = document.getElementById('stt-status');
        const out = document.getElementById('stt-result');
        if (!stt.engine) return;
        if (!stt.audioPath && !stt.audioBase64) { status.textContent = tx('Choose or upload a recording first.'); return; }
        const audio = stt.audioPath ? { audio_path: stt.audioPath } : { audio_base64: stt.audioBase64 };
        const body = R.sttPayload(stt.engine, stt.values, audio);
        status.textContent = tx('{engine} is transcribing… (the first time, the model is loaded)', { engine: stt.engine.label });
        const t0 = performance.now();
        try {
            const r = await api('/api/stt/transcribe', 'POST', body);
            stt.last = r;
            out.textContent = R.readable(r) || tx('(nothing recognised)');
            const speakers = new Set((r.segments || []).map(x => x.speaker).filter(Boolean));
            status.textContent = `${tx('Done')} · ${stt.engine.label}${r.language ? ' · ' + langName(r.language) : ''}`
                + `${speakers.size ? ' · ' + tx('{n} speakers', { n: speakers.size }) : ''} · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
            document.getElementById('stt-copy-btn').disabled = false;
            document.getElementById('stt-srt-btn').disabled = !(r.segments || []).length;
            addLog(tx('Transcribed ({engine})', { engine: stt.engine.label }), 'success');
        } catch (err) {
            status.textContent = tx('Failed: {err}', { err: err.message });
            addLog(tx('Transcribing: {err}', { err: err.message }), 'error');
        }
    }

    // ------------------------------------------------------------------ page
    function show(which) {
        document.getElementById('speech-tts').style.display = which === 'tts' ? '' : 'none';
        document.getElementById('speech-stt').style.display = which === 'stt' ? '' : 'none';
        document.getElementById('speech-stt').hidden = false;
        document.getElementById('speech-tts-btn').classList.toggle('active', which === 'tts');
        document.getElementById('speech-stt-btn').classList.toggle('active', which === 'stt');
    }
    document.getElementById('speech-tts-btn').addEventListener('click', () => show('tts'));
    document.getElementById('speech-stt-btn').addEventListener('click', () => show('stt'));
    show('tts');

    // The run button lives in the form's card (built per engine), so add it once below the form.
    const sttRun = h('button', { class: 'btn btn-primary', text: tx('Transcribe'), on: { click: transcribe } });
    document.getElementById('stt-form').after(h('div', { style: 'padding: 0 1.25rem 1.25rem;' }, sttRun));

    async function load() {
        try { voices = (await api('/api/voices')).voices || []; } catch (e) { voices = []; }
        try { catalog = await api('/v1/audio/models'); }
        catch (err) { catalog = null; addLog(tx('Couldn\'t load speech engines: {err}', { err: err.message }), 'error'); }
        renderTts(); renderStt();
    }
    // Loaded when the tab is first opened: asking starts tts-serve/stt-serve if they're down.
    let loaded = false;
    const tabLink = document.querySelector('.tab-link[data-tab="speech"]');
    const maybeLoad = () => { if (!loaded) { loaded = true; load(); } };
    tabLink && tabLink.addEventListener('click', maybeLoad);
    if (document.getElementById('speech-tab').classList.contains('active')) maybeLoad();
})();
