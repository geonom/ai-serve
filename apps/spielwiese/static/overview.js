// "Übersicht", the start page: what the Spielwiese can do right now -- built from what the gateway offers
// (GET /v1/llm/models, /v1/images/models?task=all, /v1/audio/models), so new models and engines show up here
// by themselves (owner, 2026-10-20). Pure part first (tests: node --test tests/overview.test.mjs).

const OverviewRules = (() => {
    const reachable = m => (m.where || []).some(p => p.place !== 'device' || p.online);
    const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

    // One card per page: {tab, group, title, text, facts[], ok}. `ok` false = the gateway can't do this now.
    function cards(llm, images, audio) {
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
        const down = 'Gateway gerade nicht erreichbar.';
        return [
            { tab: 'gateway', group: 'Text', title: 'Chat', ok: chat.length > 0,
              text: 'Mit Sprachmodellen schreiben – Fragen, Texte, Code, Bilder beschreiben.',
              facts: llmOk ? [plural(chat.length, 'Chat-Modell', 'Chat-Modelle'),
                              chat.filter(m => m.vision).length ? `${chat.filter(m => m.vision).length} verstehen Bilder` : null,
                              chat.some(m => m.exclusive) ? `${chat.filter(m => m.exclusive).map(m => m.label).join(', ')} (sehr groß, belegt alle Grafikkarten)` : null,
                              onDevices.length ? `${onDevices.length} nur auf anderen Rechnern im Netzwerk` : null,
                              loaded.length ? `geladen: ${loaded.slice(0, 2).join(', ')}${loaded.length > 2 ? ' …' : ''}` : null].filter(Boolean)
                           : [down] },
            { tab: 'texts', group: 'Text', title: 'Texte vergleichen', ok: emb.length + rr.length > 0,
              text: 'Wie ähnlich sich Texte sind – und welche am besten zu einer Frage passen.',
              facts: llmOk ? [plural(emb.length, 'Embedding-Modell', 'Embedding-Modelle'), plural(rr.length, 'Rerank-Modell', 'Rerank-Modelle')] : [down] },
            { tab: 'docs', group: 'Text', title: 'Dokumente lesen', ok: ocr.length > 0,
              text: 'Fotos und PDFs abschreiben oder gezielt Angaben herausziehen.',
              facts: llmOk ? [ocr.length ? plural(ocr.length, 'Lese-Modell', 'Lese-Modelle') : 'kein Lese-Modell vorhanden'] : [down] },
            { tab: 'image-gen', group: 'Bild', title: 'Bild erstellen', ok: gen.length > 0,
              text: 'Bilder aus einer Beschreibung, Bilder per Anweisung ändern, Stil übertragen.',
              facts: imgOk ? [plural(gen.length, 'Bildmodell', 'Bildmodelle'), `${edit.length} zum Bearbeiten`, `${style.length} für Stil`] : [down] },
            { tab: 'layout', group: 'Bild', title: 'Mit Positionen', ok: layout.length > 0,
              text: 'Festlegen, was wo im Bild steht – auch Schrift, die genau so erscheint.',
              facts: imgOk ? [layout.length ? plural(layout.length, 'Variante', 'Varianten') + ' (MOSS-Image)' : 'MOSS-Image nicht installiert'] : [down] },
            { tab: 'compare', group: 'Bild', title: 'Modelle vergleichen', ok: gen.length > 1,
              text: 'Denselben Prompt durch mehrere Modelle schicken und nebeneinander ansehen.',
              facts: imgOk ? [`${gen.length} Modelle zur Auswahl`] : [down] },
            { tab: 'mine', group: 'Bild', title: 'Meine Bilder', ok: true,
              text: 'Alles, was du hier erzeugt hast – nur für dich.', facts: [] },
            { tab: 'speech', group: 'Ton', title: 'Sprache', ok: !!(tts && tts.length) || !!(stt && stt.length),
              text: 'Texte vorlesen lassen (auch mit geklonter Stimme) und Aufnahmen abschreiben.',
              facts: [tts ? `Sprechen: ${tts.map(e => e.label).join(', ')}` : 'Sprachausgabe nicht erreichbar',
                      stt ? `Abschreiben: ${stt.map(e => e.label).join(', ')}` : 'Transkription nicht erreichbar'] },
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
            card.append(ul, el('span', 'ov-go', c.ok ? 'Öffnen →' : 'gerade nicht verfügbar'));
            card.addEventListener('click', () => document.querySelector(`.tab-link[data-tab="${c.tab}"]`)?.click());
            grid.append(card);
        }
    }

    async function load() {
        const status = document.getElementById('ov-status');
        status.textContent = 'Frage das Gateway, was es gerade kann …';
        const get = path => api(path).catch(() => null);
        const [llm, img, audio, backend] = await Promise.all([get('/v1/llm/models'), get('/v1/images/models?task=all'),
                                                              get('/v1/audio/models'), get('/sw/backend')]);
        render(OverviewRules.cards(llm && llm.models, img && img.data, audio));
        const ok = llm || img || audio;
        status.textContent = !ok ? 'Das Gateway antwortet gerade nicht – die Seiten funktionieren erst wieder, wenn es läuft.'
            : backend && backend.kind === 'openai'
                ? 'Verbunden mit einem Standard-OpenAI-Server (z. B. Ollama oder LM Studio): Chat und Texte vergleichen gehen; Bilder, Sprache und Dokumente brauchen ai-serve.'
                : `Stand ${new Date().toLocaleTimeString()} – neue Modelle und Engines erscheinen hier von selbst.`;
    }
    document.getElementById('ov-reload').addEventListener('click', load);
    let loaded = false;
    const maybe = () => { if (!loaded) { loaded = true; load(); } };
    document.querySelector('.tab-link[data-tab="overview"]')?.addEventListener('click', maybe);
    if (document.getElementById('overview-tab').classList.contains('active')) maybe();
})();
