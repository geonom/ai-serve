// "Vergleich" (same prompt + seed through several image models, optionally several step counts and seeds)
// and "Meine Bilder" (each person's own images). The server keeps both per person (/sw/...); comparisons
// run there, so closing the page doesn't stop them. DOM only via textContent/attributes (no innerHTML).

const cmp = { catalog: [], job: null, poll: null, viewing: null };
const mine = { before: null, items: [], current: null };
const SW = (p) => `${BASE_URL}/sw${p}`;
const imgUrl = (id) => SW(`/gallery/${id}/image`);

function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (v === undefined || v === null || v === false) continue;
        if (k === 'text') e.textContent = v;
        else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
        else e.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children) if (c) e.append(c);
    return e;
}

async function sw(path, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await swFetch(SW(path), opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    return data;
}

// Called by the Bild and Positionen tabs after an image arrived: it goes into "Meine Bilder".
async function saveToGallery(b64, meta) {
    try {
        const clean = Object.fromEntries(Object.entries(meta).filter(([, v]) => v !== null && v !== undefined && v !== ''));
        await sw('/gallery', 'POST', { image: b64, meta: clean });
        mine.dirty = true;
    } catch (err) {
        addLog(`Could not keep the image in "Meine Bilder": ${err.message}`, 'warn');
    }
}

const numbers = (text) => (text || '').split(/[\s,;]+/).filter(Boolean).map(Number);
// what's wrong with a "4, 8, 16"-style field, or null
const badNumbers = (text, name, lo, hi) => {
    const parts = (text || '').split(/[\s,;]+/).filter(Boolean);
    const bad = parts.filter(p => !/^\d+$/.test(p) || Number(p) < lo || Number(p) > hi);
    return bad.length ? `${name}: „${bad.join(', ')}“ – nur ganze Zahlen ${lo}–${hi}, durch Komma getrennt` : null;
};

function cmpSay(text, kind) {
    const box = document.getElementById('cmp-status');
    box.textContent = text || '';
    box.className = `cmp-status${kind ? ' ' + kind : ''}`;
    box.hidden = !text;
}

// problems that make the server refuse, checked before sending (null = fine)
function cmpProblem() {
    const models = cmpChosen().map(id => cmp.catalog.find(m => m.id === id)).filter(Boolean);
    if (!document.getElementById('cmp-prompt').value.trim()) return 'Bitte einen Prompt eingeben.';
    if (!models.length) return 'Bitte mindestens ein Modell wählen.';
    if (models.length > 8) return 'Höchstens 8 Modelle pro Vergleich.';
    const steps = badNumbers(document.getElementById('cmp-steps').value, 'Schrittzahlen', 1, 60);
    if (steps) return steps;
    const seeds = badNumbers(document.getElementById('cmp-seeds').value, 'Seeds', 0, 2147483647);
    if (seeds) return seeds;
    if (numbers(document.getElementById('cmp-steps').value).length > 6) return 'Höchstens 6 Schrittzahlen.';
    if (numbers(document.getElementById('cmp-seeds').value).length > 4) return 'Höchstens 4 Seeds.';
    if (cmpCells() > 32) return `${cmpCells()} Bilder – höchstens 32 pro Vergleich (weniger Modelle, Schrittzahlen oder Seeds).`;
    return null;
}

function cmpCells() {
    const models = cmpChosen().map(id => cmp.catalog.find(m => m.id === id)).filter(Boolean);
    const steps = numbers(document.getElementById('cmp-steps').value);
    const seeds = Math.max(1, numbers(document.getElementById('cmp-seeds').value).length);
    let n = 0;
    for (const m of models) {
        const ok = m.steps.adjustable ? steps.filter(s => s >= 1 && s <= m.steps.max).length : 0;
        n += Math.max(1, ok) * seeds;
    }
    return n;
}
const ratio = (ar) => { const [w, h] = ar.split(':').map(Number); return w / h; };

// ------------------------------------------------------------------ Vergleich
async function cmpLoadCatalog() {
    const box = document.getElementById('cmp-models');
    try {
        const data = await api('/v1/images/models');
        cmp.catalog = data.data || [];
    } catch (err) {
        box.replaceChildren(el('span', { class: 'muted small', text: `Modellliste nicht verfügbar: ${err.message}` }));
        return;
    }
    box.replaceChildren(...cmp.catalog.map(m => {
        const st = m.steps || {};
        const note = `Standard ${st.default} Schritte${st.adjustable ? '' : ' (fest)'}` +
                     (m.aspect_ratios.length < 5 ? ` · nur ${m.aspect_ratios.join(', ')}` : '');
        return el('label', { class: m.available ? null : 'off', title: m.description },
            el('input', { type: 'checkbox', value: m.id, disabled: !m.available, checked: m.default || null,
                          onchange: cmpCount }),
            el('span', {}, document.createTextNode(m.available ? m.label : `${m.label} (nicht installiert)`),
               el('span', { class: 'muted', text: note })));
    }));
    cmpCount();
}

function cmpChosen() {
    return [...document.querySelectorAll('#cmp-models input:checked')].map(i => i.value);
}

// same rule as the server (compare.py closest_aspect): a model without the chosen format gets its nearest one
function closestAspect(wanted, offered) {
    if (offered.includes(wanted)) return wanted;
    const r = (a) => Math.log(ratio(a));
    return offered.reduce((best, a) => Math.abs(r(a) - r(wanted)) < Math.abs(r(best) - r(wanted)) ? a : best);
}

function cmpCount() {
    const box = document.getElementById('cmp-count');
    const problem = cmpProblem();
    const n = cmpCells();
    const aspect = document.getElementById('cmp-aspect').value;
    const other = cmpChosen().map(id => cmp.catalog.find(m => m.id === id)).filter(m => m && !m.aspect_ratios.includes(aspect))
        .map(m => `${m.label} in ${closestAspect(aspect, m.aspect_ratios)}`);
    box.textContent = problem && !/^Bitte/.test(problem) ? problem
        : n ? `${n} Bild${n === 1 ? '' : 'er'} werden erzeugt.${other.length ? ` ${other.join(', ')} (kann ${aspect} nicht).` : ''}` : '';
    box.classList.toggle('bad', !!problem && !/^Bitte/.test(problem));
}

async function cmpStart() {
    const problem = cmpProblem();
    if (problem) { cmpSay(problem, 'error'); return; }
    const body = { prompt: document.getElementById('cmp-prompt').value.trim(), models: cmpChosen(),
                   aspect_ratio: document.getElementById('cmp-aspect').value };
    const steps = numbers(document.getElementById('cmp-steps').value);
    const seeds = numbers(document.getElementById('cmp-seeds').value);
    if (steps.length) body.steps = steps;
    if (seeds.length) body.seeds = seeds;
    const btn = document.getElementById('cmp-run');
    btn.disabled = true;
    btn.textContent = 'Starte …';
    cmpSay('Vergleich wird gestartet …');
    try {
        const job = await sw('/compare', 'POST', body);
        cmpSay(`Gestartet: ${job.cells.length} Bild${job.cells.length === 1 ? '' : 'er'} – verschiedene Dienste (ComfyUI, MOSS, Iris) rechnen gleichzeitig. Läuft auf dem Server weiter, auch wenn du die Seite schließt.`, 'ok');
        cmpShow(job);
        cmpLoadHistory();
        const grid = document.getElementById('cmp-grid');
        if (grid.getBoundingClientRect().top > innerHeight * 0.7) grid.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
    } catch (err) {
        const msg = /still running/.test(err.message)
            ? 'Es läuft schon ein Vergleich – warte, bis er fertig ist, oder brich ihn ab.' : `Nicht gestartet: ${err.message}`;
        cmpSay(msg, 'error');
        if (!cmp.job || cmp.job.status !== 'running') {   // never cover a comparison that is still running
            document.getElementById('cmp-grid').replaceChildren(el('div', { class: 'cmp-error', text: msg }));
        }
        btn.disabled = false;
    } finally {
        btn.textContent = 'Vergleich starten';
    }
}

function cmpShow(job) {
    cmp.job = job;
    cmp.viewing = job.id;
    clearTimeout(cmp.poll);
    cmpRender();
    if (job.status === 'running') cmp.poll = setTimeout(() => cmpRefresh(job.id), 2000);
}

async function cmpRefresh(id) {
    try {
        const job = await sw(`/compare/${id}`);
        if (cmp.viewing !== id) return;
        const finished = job.status !== 'running';
        cmpShow(job);
        if (finished) { cmpLoadHistory(); mine.dirty = true; }
    } catch (err) {
        cmp.poll = setTimeout(() => cmpRefresh(id), 5000);   // e.g. a short gateway/door hiccup
    }
}

function cmpRender() {
    const job = cmp.job;
    const done = job.cells.filter(c => c.status === 'done').length;
    document.getElementById('cmp-title').textContent = 'Ergebnis';
    const state = { running: `läuft – ${done} von ${job.cells.length} fertig`, done: 'fertig', cancelled: 'abgebrochen',
                    interrupted: 'unterbrochen (Server neu gestartet)', failed: 'fehlgeschlagen' }[job.status] || job.status;
    document.getElementById('cmp-subtitle').textContent = `„${job.prompt}“ · ${job.aspect_ratio} · ${state}`;
    const cancelBtn = document.getElementById('cmp-cancel');
    cancelBtn.hidden = job.status !== 'running';
    if (job.status !== 'running') { cancelBtn.disabled = false; cancelBtn.textContent = 'Abbrechen'; }
    if (job.status === 'cancelled' && document.getElementById('cmp-status').classList.contains('ok')) cmpSay('Abgebrochen.', 'ok');
    document.getElementById('cmp-run').disabled = job.status === 'running';
    if (job.status === 'done' && document.getElementById('cmp-status').classList.contains('ok')) cmpSay('');

    // Wrapping tiles instead of a fixed table: as many side by side as the width allows, never wider
    // than the page. Models only: one block (per seed); with step counts: one block per model.
    const models = [...new Map(job.cells.map(c => [c.model, c.label])).entries()];
    const seeds = [...new Set(job.cells.map(c => c.seed))];
    const stepsOf = (m) => [...new Set(job.cells.filter(c => c.model === m).map(c => c.steps))];
    const stepsVary = models.some(([m]) => stepsOf(m).length > 1);
    const tiles = (cells, title) => el('div', { class: 'cmp-tiles' }, ...cells.map(c => cmpCell(c, job, title(c))));
    const blocks = [];
    if (!stepsVary) {
        for (const seed of seeds) {
            const cells = job.cells.filter(c => c.seed === seed);
            if (seeds.length > 1) blocks.push(el('h3', { class: 'cmp-block-title', text: `Seed ${seed}` }));
            blocks.push(tiles(cells, c => c.label));
        }
    } else {
        for (const [m, label] of models) {
            const fixed = stepsOf(m).length === 1 && cmp.catalog.find(x => x.id === m)?.steps?.adjustable === false;
            blocks.push(el('h3', { class: 'cmp-block-title', text: fixed ? `${label} (fest ${stepsOf(m)[0]} Schritte)` : label }));
            const cells = job.cells.filter(c => c.model === m).sort((a, b) => a.steps - b.steps || a.seed - b.seed);
            blocks.push(tiles(cells, c => `${c.steps} Schritte${seeds.length > 1 ? ` · Seed ${c.seed}` : ''}`));
        }
    }
    document.getElementById('cmp-grid').replaceChildren(...blocks);
}

// the comparison's finished images in the order the grid shows them (row by row)
function cmpSet(job) {
    const order = [...document.querySelectorAll('#cmp-grid img')].map(img => img.dataset.id);
    const byId = new Map(job.cells.filter(c => c.status === 'done' && c.image).map(c => [c.image,
        { id: c.image, prompt: job.prompt, label: c.label, model: c.model, steps: c.steps, seed: c.seed,
          seconds: c.seconds, aspect_ratio: c.aspect_ratio || job.aspect_ratio }]));
    return order.filter(id => byId.has(id)).map(id => byId.get(id));
}

function cmpCell(c, job, title) {
    const own = c.aspect_ratio || job.aspect_ratio;
    const fmt = own !== job.aspect_ratio ? ` · ${own}` : '';
    const details = `${c.steps} Schritte · Seed ${c.seed}${fmt}${c.seconds ? ` · ${c.seconds} s` : ''}`;
    let body;
    if (c.status === 'done' && c.image) {
        body = el('img', { src: imgUrl(c.image), alt: `${c.label}, ${c.steps} Schritte, Seed ${c.seed}`, loading: 'lazy',
                           'data-id': c.image, style: `aspect-ratio:${ratio(own)}`,
                           onclick: () => { const set = cmpSet(job); openViewer(set, Math.max(0, set.findIndex(x => x.id === c.image))); } });
    } else {
        const text = c.waiting ? 'wartet auf freie GPU bzw. den Gateway …'
            : { pending: 'wartet', running: 'wird erzeugt …', cancelled: 'abgebrochen' }[c.status] || c.error || c.status;
        body = el('div', { class: `ph${c.status === 'failed' ? ' failed' : ''}`, style: `aspect-ratio:${ratio(own)}`, text });
    }
    return el('figure', { class: 'cmp-tile' }, el('figcaption', { class: 'cmp-tile-title', text: title }), body,
              el('div', { class: 'cap', text: details }));
}

async function cmpLoadHistory() {
    const box = document.getElementById('cmp-history');
    let list = [];
    try { list = (await sw('/compare')).comparisons || []; } catch (err) { return; }
    if (!list.length) { box.replaceChildren(el('span', { class: 'muted small', text: 'Noch keine.' })); return; }
    box.replaceChildren(...list.map(j => el('button', {
        class: j.id === cmp.viewing ? 'active' : null,
        onclick: async () => { cmpShow(await sw(`/compare/${j.id}`)); cmpLoadHistory(); },
    }, document.createTextNode(j.prompt.length > 70 ? j.prompt.slice(0, 70) + ' …' : j.prompt),
       el('span', { class: 'muted', text: `${new Date(j.created * 1000).toLocaleString()} · ${j.cells} Bilder · ${j.models.join(', ')}` +
                                         (j.status === 'running' ? ' · läuft' : '') }))));
    const running = list.find(j => j.status === 'running');
    if (running && !cmp.viewing) cmpShow(await sw(`/compare/${running.id}`));
}

document.getElementById('cmp-run').addEventListener('click', cmpStart);
document.getElementById('cmp-cancel').addEventListener('click', async () => {
    if (!cmp.job) return;
    const btn = document.getElementById('cmp-cancel');
    btn.disabled = true;
    btn.textContent = 'Wird abgebrochen …';
    cmpSay('Wird abgebrochen – das Bild, das gerade entsteht, wird gestoppt.', 'ok');
    try {
        await sw(`/compare/${cmp.job.id}/cancel`, 'POST', {});
        cmpRefresh(cmp.job.id);
    } catch (err) {
        cmpSay(`Abbrechen ging nicht: ${err.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Abbrechen';
    }
});
['cmp-steps', 'cmp-seeds', 'cmp-prompt'].forEach(id => document.getElementById(id).addEventListener('input', () => { cmpCount(); cmpSay(''); }));
document.getElementById('cmp-aspect').addEventListener('change', cmpCount);

// ------------------------------------------------------------------ Meine Bilder
function mineCaption(e) {
    return [e.label || e.model, e.steps ? `${e.steps} Schritte` : null, e.seed !== undefined ? `Seed ${e.seed}` : null]
        .filter(Boolean).join(' · ') || 'Bild';
}

async function mineLoad(reset = true) {
    if (reset) { mine.before = null; mine.items = []; }
    let data;
    try {
        const q = new URLSearchParams({ limit: '48' });
        if (mine.before) q.set('before', mine.before);
        data = await sw(`/gallery?${q}`);
    } catch (err) {
        document.getElementById('mine-grid').replaceChildren(el('p', { class: 'muted', text: `Nicht verfügbar: ${err.message}` }));
        return;
    }
    mine.items.push(...data.images);
    mine.before = data.images.length ? data.images[data.images.length - 1].id : mine.before;
    document.getElementById('mine-more').hidden = data.images.length < 48;
    mine.dirty = false;
    mineRender();
}

function mineRender() {
    const filter = document.getElementById('mine-filter');
    const models = [...new Set(mine.items.map(e => e.label || e.model).filter(Boolean))].sort();
    const keep = filter.value;
    filter.replaceChildren(el('option', { value: '', text: 'alle' }), ...models.map(m => el('option', { value: m, text: m })));
    filter.value = models.includes(keep) ? keep : '';
    const shown = mine.items.filter(e => !filter.value || (e.label || e.model) === filter.value);
    const grid = document.getElementById('mine-grid');
    if (!shown.length) { grid.replaceChildren(el('p', { class: 'muted', text: 'Noch keine Bilder – erzeuge welche unter „Bild“, „Positionen“ oder „Vergleich“.' })); return; }
    grid.replaceChildren(...shown.map((e, i) => el('figure', { onclick: () => openViewer(shown, i) },
        el('img', { src: imgUrl(e.id), alt: e.prompt || 'Bild', loading: 'lazy' }),
        el('figcaption', { title: e.prompt || '', text: mineCaption(e) }))));
}

document.getElementById('mine-reload').addEventListener('click', () => mineLoad(true));
document.getElementById('mine-more').addEventListener('click', () => mineLoad(false));
document.getElementById('mine-filter').addEventListener('change', mineRender);

// ------------------------------------------------------------------ Viewer
// One fixed-size stage: every image is drawn into the same box (object-fit: contain), so images of the
// same format sit exactly on top of each other -- switching never moves anything. "Umschalten" flips
// through the set (← →, 1-9, space = back to the previous one); "Schieber" lays B over A and reveals it
// right of a divider you drag (or move with the range input / ← →).
const view = { set: [], i: 0, prev: null, mode: 'flip', a: 0, b: 1, split: 50, preload: [] };
const $ = (id) => document.getElementById(id);

function viewLabel(e) {
    return [e.label || e.model || 'Bild', e.steps ? `${e.steps} Schritte` : null,
            e.seed !== undefined && e.seed !== null ? `Seed ${e.seed}` : null].filter(Boolean).join(' · ');
}

function openViewer(set, index = 0) {
    view.set = set;
    view.prev = null;
    view.preload = set.map(e => { const im = new Image(); im.src = imgUrl(e.id); return im; });   // instant switching
    $('sw-lightbox').hidden = false;
    setMode(view.mode, index);
}
// kept for single images (older callers)
function openLightbox(e) { openViewer([e], 0); }

function closeLightbox() { $('sw-lightbox').hidden = true; view.preload = []; }

function setMode(mode, index = view.i) {
    view.mode = set_mode_allowed(mode);
    document.querySelectorAll('.sw-seg button').forEach(b => b.classList.toggle('active', b.dataset.mode === view.mode));
    const slider = view.mode === 'slider';
    $('sw-lb-stage').classList.toggle('slider', slider);
    ['sw-lb-img-b', 'sw-lb-divider', 'sw-lb-tag-a', 'sw-lb-tag-b', 'sw-lb-slider-row'].forEach(id => { $(id).hidden = !slider; });
    $('sw-lb-hint').textContent = slider ? 'Im Bild ziehen oder ← → verschieben · Bilder links/rechts wählen'
                                         : '← → wechseln · Leertaste: zum vorigen · 1–9 direkt';
    if (slider) {
        view.a = index;
        view.b = view.set.length > 1 ? (index + 1) % view.set.length : index;
        fillPickers();
        renderSlider();
    } else {
        show(index, false);
    }
    renderThumbs();
}

function set_mode_allowed(mode) { return mode === 'slider' && view.set.length < 2 ? 'flip' : mode; }

// Size the frame to the image's aspect, contained and centred in the stage. Same aspect -> same box,
// so switching between images of one comparison never moves them.
function fitFrame(e) {
    const stage = $('sw-lb-stage'), frame = $('sw-lb-frame');
    const img = view.preload[view.set.indexOf(e)];
    let ar = img && img.naturalWidth ? img.naturalWidth / img.naturalHeight : null;
    if (!ar && e.aspect_ratio) ar = ratio(e.aspect_ratio);
    if (!ar) ar = 1;
    const W = stage.clientWidth, H = stage.clientHeight;
    const w = Math.min(W, H * ar), h = w / ar;
    Object.assign(frame.style, { width: `${w}px`, height: `${h}px`, left: `${(W - w) / 2}px`, top: `${(H - h) / 2}px` });
    if (img && !img.naturalWidth) img.onload = () => { if (current() === e) fitFrame(e); };
}
const current = () => view.set[view.mode === 'slider' ? view.a : view.i];
window.addEventListener('resize', () => { if (!$('sw-lightbox').hidden && current()) fitFrame(current()); });

function show(i, remember = true) {
    if (!view.set.length) return;
    i = (i + view.set.length) % view.set.length;
    if (remember && i !== view.i) view.prev = view.i;
    view.i = i;
    const e = view.set[i];
    const img = $('sw-lb-img');
    img.src = imgUrl(e.id);
    img.alt = e.prompt || 'Bild';
    fitFrame(e);
    $('sw-lb-pos').textContent = view.set.length > 1 ? `${i + 1} / ${view.set.length} · ${viewLabel(e)}` : viewLabel(e);
    renderMeta(e);
    renderThumbs();
}

function renderMeta(e, other) {
    const lines = (x) => [x.prompt ? `„${x.prompt}“` : null, '', `Modell: ${x.label || x.model || '?'}`,
        x.steps ? `Schritte: ${x.steps}` : null, x.seed !== undefined && x.seed !== null ? `Seed: ${x.seed}` : null,
        x.aspect_ratio ? `Format: ${x.aspect_ratio}` : null, x.seconds ? `Dauer: ${x.seconds} s` : null,
        x.created ? `Erstellt: ${new Date(x.created * 1000).toLocaleString()}` : null].filter(l => l !== null);
    $('sw-lb-meta').textContent = other
        ? [`Links: ${viewLabel(e)}${e.seconds ? ` (${e.seconds} s)` : ''}`, `Rechts: ${viewLabel(other)}${other.seconds ? ` (${other.seconds} s)` : ''}`,
           '', e.prompt ? `„${e.prompt}“` : ''].join('\n')
        : lines(e).join('\n');
    const dl = $('sw-lb-download');
    dl.href = imgUrl(e.id);
    dl.download = `${(e.model || 'bild').replace(/[^\w.-]+/g, '_')}-${e.steps ?? ''}-${e.seed ?? e.id}.png`;
    mine.current = e;
}

function fillPickers() {
    const opts = () => view.set.map((e, i) => el('option', { value: String(i), text: `${i + 1}: ${viewLabel(e)}` }));
    $('sw-lb-a').replaceChildren(...opts());
    $('sw-lb-b').replaceChildren(...opts());
    $('sw-lb-a').value = String(view.a);
    $('sw-lb-b').value = String(view.b);
}

function renderSlider() {
    const a = view.set[view.a], b = view.set[view.b];
    $('sw-lb-img').src = imgUrl(a.id);
    $('sw-lb-img-b').src = imgUrl(b.id);
    fitFrame(a);
    $('sw-lb-tag-a').textContent = viewLabel(a);
    $('sw-lb-tag-b').textContent = viewLabel(b);
    $('sw-lb-pos').textContent = `${view.a + 1} ↔ ${view.b + 1}`;
    setSplit(view.split);
    renderMeta(a, b);
    renderThumbs();
}

function setSplit(pct) {
    view.split = Math.max(0, Math.min(100, pct));
    $('sw-lb-img-b').style.clipPath = `inset(0 0 0 ${view.split}%)`;
    $('sw-lb-divider').style.left = `${view.split}%`;
    $('sw-lb-range').value = String(view.split);
}

function renderThumbs() {
    const box = $('sw-lb-thumbs');
    if (view.set.length < 2) { box.replaceChildren(); return; }
    box.replaceChildren(...view.set.map((e, i) => {
        const cls = view.mode === 'slider' ? (i === view.a ? 'cur' : i === view.b ? 'b' : null) : (i === view.i ? 'cur' : null);
        return el('button', { type: 'button', class: cls, title: viewLabel(e),
                              onclick: (ev) => {
                                  if (view.mode === 'slider') {
                                      if (ev.shiftKey || ev.altKey) view.a = i; else view.b = i;
                                      fillPickers(); renderSlider();
                                  } else show(i);
                              } },
            el('img', { src: imgUrl(e.id), alt: viewLabel(e) }), el('span', { class: 'n', text: String(i + 1) }));
    }));
}

// slider: drag anywhere on the stage
(function () {
    const stage = $('sw-lb-stage');
    let dragging = false;
    const frame = $('sw-lb-frame');
    const at = (ev) => { const r = frame.getBoundingClientRect(); setSplit((ev.clientX - r.left) / r.width * 100); };
    stage.addEventListener('pointerdown', (ev) => { if (view.mode !== 'slider') return; dragging = true; stage.setPointerCapture(ev.pointerId); at(ev); });
    stage.addEventListener('pointermove', (ev) => { if (dragging) at(ev); });
    stage.addEventListener('pointerup', () => { dragging = false; });
    stage.addEventListener('pointercancel', () => { dragging = false; });
    // flip mode: a tap on the left/right third goes back/forward (phones have no arrow keys)
    stage.addEventListener('click', (ev) => {
        if (view.mode !== 'flip' || view.set.length < 2) return;
        const r = frame.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width;
        if (x < 1 / 3) show(view.i - 1); else if (x > 2 / 3) show(view.i + 1);
    });
})();

$('sw-lb-range').addEventListener('input', (ev) => setSplit(parseFloat(ev.target.value)));
$('sw-lb-a').addEventListener('change', (ev) => { view.a = parseInt(ev.target.value); renderSlider(); });
$('sw-lb-b').addEventListener('change', (ev) => { view.b = parseInt(ev.target.value); renderSlider(); });
document.querySelectorAll('.sw-seg button').forEach(b => b.addEventListener('click', () =>
    setMode(b.dataset.mode, view.mode === 'slider' ? view.a : view.i)));

document.addEventListener('keydown', (ev) => {
    if ($('sw-lightbox').hidden) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName) && ev.key !== 'Escape') return;
    if (ev.key === 'Escape') { closeLightbox(); return; }
    if (view.mode === 'slider') {
        if (ev.key === 'ArrowLeft') { setSplit(view.split - (ev.shiftKey ? 10 : 2)); ev.preventDefault(); }
        else if (ev.key === 'ArrowRight') { setSplit(view.split + (ev.shiftKey ? 10 : 2)); ev.preventDefault(); }
        else if (ev.key === ' ') { [view.a, view.b] = [view.b, view.a]; fillPickers(); renderSlider(); ev.preventDefault(); }
        return;
    }
    if (ev.key === 'ArrowLeft') { show(view.i - 1); ev.preventDefault(); }
    else if (ev.key === 'ArrowRight') { show(view.i + 1); ev.preventDefault(); }
    else if (ev.key === ' ' && view.prev !== null) { show(view.prev); ev.preventDefault(); }
    else if (/^[1-9]$/.test(ev.key) && parseInt(ev.key) <= view.set.length) { show(parseInt(ev.key) - 1); ev.preventDefault(); }
});

$('sw-lb-close').addEventListener('click', closeLightbox);
$('sw-lightbox').addEventListener('click', (ev) => { if (ev.target.id === 'sw-lightbox') closeLightbox(); });
$('sw-lb-reuse').addEventListener('click', () => {
    const e = mine.current;
    if (!e) return;
    $('cmp-prompt').value = e.prompt || '';
    if (e.seed !== undefined && e.seed !== null) $('cmp-seeds').value = String(e.seed);
    closeLightbox();
    document.querySelector('.tab-link[data-tab="compare"]').click();
    cmpCount();
});
$('sw-lb-delete').addEventListener('click', async () => {
    const e = mine.current;
    if (!e || !confirm(`„${viewLabel(e)}“ endgültig löschen?`)) return;
    try {
        await sw(`/gallery/${e.id}`, 'DELETE');
        mine.items = mine.items.filter(x => x.id !== e.id);
        mineRender();
        const left = view.set.filter(x => x.id !== e.id);
        if (!left.length) { closeLightbox(); return; }
        const idx = Math.min(view.mode === 'slider' ? view.a : view.i, left.length - 1);
        view.set = left;
        setMode(view.mode, idx);
        if (cmp.job) cmpRefresh(cmp.job.id);
    } catch (err) { addLog(`Löschen fehlgeschlagen: ${err.message}`, 'error'); }
});

// load lazily when a tab is opened
document.querySelectorAll('.tab-link[data-tab="mine"]').forEach(b => b.addEventListener('click', () => {
    if (!mine.items.length || mine.dirty) mineLoad(true);
}));
document.querySelectorAll('.tab-link[data-tab="compare"]').forEach(b => b.addEventListener('click', () => {
    if (!cmp.catalog.length) cmpLoadCatalog();
}));
cmpLoadCatalog();
cmpLoadHistory();
if (document.querySelector('.tab-link[data-tab="mine"].active')) mineLoad(true);
