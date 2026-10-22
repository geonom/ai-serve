// "Overview", the start page: what the Spielwiese can do right now -- built from what the gateway offers
// (GET /v1/llm/models, /v1/images/models?task=all, /v1/audio/models), so new models and engines show up here
// by themselves (owner, 2026-10-20). Pure part first (tests: node --test tests/overview.test.mjs).

const OverviewRules = (() => {
    const reachable = m => (m.where || []).some(p => p.place !== 'device' || p.online);
    const plural = (n, one, many) => tx(n === 1 ? one : many, { n });   // one/many: '{n} …' templates

    // One card per page: {tab, group, title, text, facts[], ok}. `ok` false = the gateway can't do this now; then
    // the card names what could be downloaded for it (`downloads`: GET /api/plugins/installs items).
    function cards(llm, images, audio, downloads) {
        const out = baseCards(llm, images, audio);
        const CAPS = { 'image-gen': ['image'], layout: ['image'], compare: ['image'], speech: ['tts', 'stt'] };
        for (const c of out) {
            const can = (downloads || []).filter(d => d.state !== 'installed' && (CAPS[c.tab] || []).includes(d.capability));
            if (!c.ok && can.length) c.facts.push(tx('Can be downloaded: {list}', { list: can.map(d => d.label).join(', ') }));
        }
        return out;
    }

    function baseCards(llm, images, audio) {
        const llmOk = Array.isArray(llm), imgOk = Array.isArray(images);
        const models = llmOk ? llm.filter(reachable) : [];
        const chat = models.filter(m => m.kind === 'chat');
        const emb = models.filter(m => m.kind === 'embedding'), rr = models.filter(m => m.kind === 'rerank');
        const ocr = chat.filter(m => /nuextract|ocr/i.test(m.id));
        const onDevices = chat.filter(m => m.where.length && m.where.every(p => p.place === 'device'));
        const loaded = chat.filter(m => m.loaded).map(m => m.label);
        const imgs = imgOk ? images.filter(m => m.available) : [];
        const gen = imgs.filter(m => (m.tasks || ['generate']).includes('generate'));
        const edit = imgs.filter(m => (m.tasks || []).includes('edit'));
        const style = imgs.filter(m => (m.tasks || []).includes('style'));
        const layout = gen.filter(m => m.layout);
        const tts = audio && audio.tts && audio.tts.available ? audio.tts.engines : null;
        const stt = audio && audio.stt && audio.stt.available ? audio.stt.engines.filter(e => e.available) : null;
        const down = tx('Gateway unreachable right now.');
        const TEXT = tx('Text'), IMAGE = tx('Image'), SOUND = tx('Sound');
        return [
            { tab: 'gateway', group: TEXT, title: tx('Chat'), ok: chat.length > 0,
              text: tx('Write with language models – questions, texts, code, describing images.'),
              facts: llmOk ? [plural(chat.length, '{n} chat model', '{n} chat models'),
                              chat.filter(m => m.vision).length ? tx('{n} understand images', { n: chat.filter(m => m.vision).length }) : null,
                              chat.some(m => m.exclusive) ? tx('{names} (very large, takes all graphics cards)', { names: chat.filter(m => m.exclusive).map(m => m.label).join(', ') }) : null,
                              onDevices.length ? tx('{n} only on other machines in the network', { n: onDevices.length }) : null,
                              loaded.length ? tx('loaded: {names}', { names: loaded.slice(0, 2).join(', ') + (loaded.length > 2 ? ' …' : '') }) : null].filter(Boolean)
                           : [down] },
            { tab: 'texts', group: TEXT, title: tx('Compare texts'), ok: emb.length + rr.length > 0,
              text: tx('How similar texts are – and which fit a question best.'),
              facts: llmOk ? [plural(emb.length, '{n} embedding model', '{n} embedding models'), plural(rr.length, '{n} rerank model', '{n} rerank models')] : [down] },
            { tab: 'docs', group: TEXT, title: tx('Read documents'), ok: ocr.length > 0,
              text: tx('Transcribe photos and PDFs, or pull out specific details.'),
              facts: llmOk ? [ocr.length ? plural(ocr.length, '{n} reading model', '{n} reading models') : tx('no reading model available')] : [down] },
            { tab: 'image-gen', group: IMAGE, title: tx('Create images'), ok: gen.length > 0,
              text: tx('Images from a description, changing images by instruction, transferring a style.'),
              facts: imgOk ? [plural(gen.length, '{n} image model', '{n} image models'), tx('{n} for editing', { n: edit.length }), tx('{n} for style', { n: style.length })] : [down] },
            { tab: 'layout', group: IMAGE, title: tx('With positions'), ok: layout.length > 0,
              text: tx('Decide what goes where in the image – including lettering that appears exactly as written.'),
              facts: imgOk ? [layout.length ? plural(layout.length, '{n} variant', '{n} variants') + ' (MOSS-Image)' : tx('MOSS-Image not installed')] : [down] },
            { tab: 'compare', group: IMAGE, title: tx('Compare models'), ok: gen.length > 1,
              text: tx('Send the same prompt through several models and look at them side by side.'),
              facts: imgOk ? [tx('{n} models to choose from', { n: gen.length })] : [down] },
            { tab: 'mine', group: IMAGE, title: tx('My images'), ok: true,
              text: tx('Everything you\'ve made here – only for you.'), facts: [] },
            { tab: 'speech', group: SOUND, title: tx('Speech'), ok: !!(tts && tts.length) || !!(stt && stt.length),
              text: tx('Have texts read out (with a cloned voice too) and transcribe recordings.'),
              facts: [tts ? tx('Speaking: {names}', { names: tts.map(e => e.label).join(', ') }) : tx('Speech output unreachable'),
                      stt ? tx('Transcribing: {names}', { names: stt.map(e => e.label).join(', ') }) : tx('Transcription unreachable')] },
        ];
    }
    return { cards };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = OverviewRules;

if (typeof document !== 'undefined' && document.getElementById('overview-tab')) (() => {
    const grid = document.getElementById('ov-cards');
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

    function render(list) {
        grid.replaceChildren();
        let group = null;
        for (const c of list) {
            if (c.group !== group) { group = c.group; grid.append(el('h3', 'ov-group', group)); }
            const card = el('button', 'ov-card' + (c.ok ? '' : ' ov-off'));
            card.type = 'button';
            card.append(el('span', 'ov-title', c.title), el('span', 'ov-text', c.text));
            const ul = el('ul', 'ov-facts');
            for (const f of c.facts) ul.append(el('li', null, f));
            card.append(ul, el('span', 'ov-go', c.ok ? tx('Open →') : tx('not available right now')));
            card.addEventListener('click', () => document.querySelector(`.tab-link[data-tab="${c.tab}"]`)?.click());
            grid.append(card);
        }
    }

    async function load() {
        const status = document.getElementById('ov-status');
        status.textContent = tx('Asking the gateway what it can do right now …');
        const get = path => api(path).catch(() => null);
        const [llm, img, audio, backend] = await Promise.all([get('/v1/llm/models'), get('/v1/images/models?task=all'),
                                                              get('/v1/audio/models'), get('/sw/backend')]);
        const downloads = await Downloads.ready().then(() => Downloads.items()).catch(() => []);
        render(OverviewRules.cards(llm && llm.models, img && img.data, audio, downloads));
        const ok = llm || img || audio;
        status.textContent = !ok ? tx('The gateway isn\'t answering right now – the pages work again once it runs.')
            : backend && backend.kind === 'openai'
                ? tx('Connected to a standard OpenAI server (e.g. Ollama or LM Studio): chat and comparing texts work; images, speech and documents need ai-serve.')
                : tx('As of {time} – new models and engines show up here by themselves.', { time: new Date().toLocaleTimeString(I18N.locale) });
    }
    document.getElementById('ov-reload').addEventListener('click', load);
    let loaded = false;
    const maybe = () => { if (!loaded) { loaded = true; load(); } };
    document.querySelector('.tab-link[data-tab="overview"]')?.addEventListener('click', maybe);
    if (document.getElementById('overview-tab').classList.contains('active')) maybe();
})();
