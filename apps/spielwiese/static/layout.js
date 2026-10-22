// "Positionen": MOSS-Image with positions. Boxes are drawn on the canvas (relative 0..1, as the
// gateway's /v1/images/generations expects with model "moss" and `layout`), each with a kind,
// a caption and -- for text -- the exact text. DOM only via textContent/value (no innerHTML).

const LAY_COLORS = ['#e4572e', '#17bebb', '#ffc914', '#76b041', '#2e86ab', '#a23b72', '#f18f01', '#6a4c93'];
const LAY_KINDS = [['object', 'Objekt'], ['character', 'Figur'], ['text', 'Text']];
const lay = { elements: [], selected: -1, drag: null, image: null };

const layCanvas = document.getElementById('lay-canvas');
const layList = document.getElementById('lay-elements');
const layStatus = document.getElementById('lay-status');

function laySize() {
    const [w, h] = document.getElementById('lay-size').value.split('x').map(Number);
    return { w, h };
}

function layApplyAspect() {
    const { w, h } = laySize();
    layCanvas.style.aspectRatio = `${w} / ${h}`;
}

const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const round3 = (v) => Math.round(v * 1000) / 1000;

function layPoint(ev) {
    const r = layCanvas.getBoundingClientRect();
    return { x: clamp((ev.clientX - r.left) / r.width), y: clamp((ev.clientY - r.top) / r.height) };
}

function layColor(i) {
    return LAY_COLORS[i % LAY_COLORS.length];
}

function layEl(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'text') e.textContent = v;
        else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
        else if (k === 'value') e.value = v;
        else e.setAttribute(k, v);
    }
    for (const c of children) if (c) e.append(c);
    return e;
}

function laySelect(i) {
    lay.selected = i;
    layRender();
}

function layRenderBoxes() {
    layCanvas.querySelectorAll('.lay-box').forEach(b => b.remove());
    lay.elements.forEach((e, i) => {
        const [l, t, r, b] = e.box;
        const box = layEl('div', { class: `lay-box${i === lay.selected ? ' selected' : ''}`, 'data-i': i,
                                   style: `--lay-color:${layColor(i)};left:${l * 100}%;top:${t * 100}%;width:${(r - l) * 100}%;height:${(b - t) * 100}%` });
        const label = e.kind === 'text' ? `„${e.text || '…'}"` : (e.caption || LAY_KINDS.find(k => k[0] === e.kind)[1]);
        box.append(layEl('span', { class: 'lay-tag', text: `${i + 1} ${label}` }), layEl('span', { class: 'lay-handle', 'data-handle': '1' }));
        layCanvas.append(box);
    });
}

function layRenderList() {
    layList.replaceChildren();
    if (!lay.elements.length) {
        layList.append(layEl('div', { class: 'muted small', text: 'Noch keine Rahmen – ziehe auf der Fläche einen auf.' }));
        return;
    }
    lay.elements.forEach((e, i) => {
        const kind = layEl('select', { 'aria-label': `Art von Element ${i + 1}`, onchange: (ev) => { e.kind = ev.target.value; layRender(); } },
            ...LAY_KINDS.map(([v, l]) => { const o = layEl('option', { value: v, text: l }); o.selected = v === e.kind; return o; }));
        const caption = layEl('input', { type: 'text', maxlength: '400', placeholder: 'Was ist hier? z. B. Bäckerin mit Brotkorb',
                                         'aria-label': `Beschreibung von Element ${i + 1}`, value: e.caption,
                                         oninput: (ev) => { e.caption = ev.target.value; layRenderBoxes(); } });
        const text = e.kind === 'text' ? layEl('input', { type: 'text', maxlength: '200', class: 'lay-wide', placeholder: 'Genauer Text im Bild',
                                                           'aria-label': `Text von Element ${i + 1}`, value: e.text || '',
                                                           oninput: (ev) => { e.text = ev.target.value; layRenderBoxes(); } }) : null;
        const color = layEl('input', { type: 'color', value: e.palette || '#888888', title: 'Farbe (optional)',
                                       'aria-label': `Farbe von Element ${i + 1}`,
                                       onchange: (ev) => { e.palette = ev.target.value; } });
        const useColor = layEl('label', { class: 'small lay-wide' }, (() => {
            const c = layEl('input', { type: 'checkbox', onchange: (ev) => { e.usePalette = ev.target.checked; } });
            c.checked = !!e.usePalette; return c; })(), document.createTextNode(' Farbe vorgeben '), color);
        const del = layEl('button', { class: 'btn btn-secondary btn-sm', type: 'button', text: '✕', title: 'Rahmen löschen',
                                      onclick: () => { lay.elements.splice(i, 1); lay.selected = -1; layRender(); } });
        layList.append(layEl('div', { class: `lay-el${i === lay.selected ? ' selected' : ''}`, style: `--lay-color:${layColor(i)}`,
                                      onclick: (ev) => { if (ev.target === ev.currentTarget) laySelect(i); } },
            layEl('span', { class: 'lay-num', text: String(i + 1) }), kind, del,
            layEl('div', { class: 'lay-wide' }, caption), text, useColor));
    });
}

function layRender() {
    layRenderBoxes();
    layRenderList();
    document.getElementById('lay-api').textContent = JSON.stringify(layRequestBody(), null, 2);
}

// --- drawing, moving, resizing --------------------------------------------------------------

layCanvas.addEventListener('pointerdown', (ev) => {
    const p = layPoint(ev);
    const boxEl = ev.target.closest('.lay-box');
    layCanvas.setPointerCapture(ev.pointerId);
    if (boxEl) {
        const i = Number(boxEl.dataset.i);
        lay.selected = i;
        lay.drag = { mode: ev.target.dataset.handle ? 'resize' : 'move', i, start: p, box: [...lay.elements[i].box] };
        layRender();
        return;
    }
    if (lay.elements.length >= 20) { layStatus.textContent = 'Höchstens 20 Rahmen.'; return; }
    lay.elements.push({ box: [p.x, p.y, p.x, p.y], kind: 'object', caption: '', text: '', palette: '#888888' });
    lay.selected = lay.elements.length - 1;
    lay.drag = { mode: 'draw', i: lay.selected, start: p };
});

layCanvas.addEventListener('pointermove', (ev) => {
    if (!lay.drag) return;
    const p = layPoint(ev);
    const d = lay.drag;
    const e = lay.elements[d.i];
    if (d.mode === 'draw') {
        e.box = [Math.min(d.start.x, p.x), Math.min(d.start.y, p.y), Math.max(d.start.x, p.x), Math.max(d.start.y, p.y)];
    } else if (d.mode === 'move') {
        const [l, t, r, b] = d.box;
        const dx = clamp(p.x - d.start.x, -l, 1 - r), dy = clamp(p.y - d.start.y, -t, 1 - b);
        e.box = [l + dx, t + dy, r + dx, b + dy];
    } else {
        const [l, t] = d.box;
        e.box = [l, t, clamp(p.x, l + 0.02, 1), clamp(p.y, t + 0.02, 1)];
    }
    layRenderBoxes();
});

layCanvas.addEventListener('pointerup', () => {
    const d = lay.drag;
    lay.drag = null;
    if (!d) return;
    const e = lay.elements[d.i];
    e.box = e.box.map(round3);
    // a click without dragging, or a sliver, is no box
    if (d.mode === 'draw' && (e.box[2] - e.box[0] < 0.03 || e.box[3] - e.box[1] < 0.03)) {
        lay.elements.splice(d.i, 1);
        lay.selected = -1;
    }
    layRender();
    if (d.mode === 'draw' && lay.selected >= 0) {
        const inputs = layList.querySelectorAll('.lay-el');
        inputs[lay.selected]?.querySelector('input[type=text]')?.focus();
    }
});

// --- request ----------------------------------------------------------------------------------

function layRequestBody() {
    const body = { model: document.getElementById('lay-variant').value, prompt: document.getElementById('lay-prompt').value.trim(),
                   size: document.getElementById('lay-size').value, response_format: 'b64_json' };
    const steps = parseInt(document.getElementById('lay-steps').value, 10);
    if (steps) body.steps = steps;
    const seed = document.getElementById('lay-seed').value;
    if (seed !== '') body.seed = parseInt(seed, 10);
    body.layout = lay.elements.map(e => {
        const item = { box: e.box, kind: e.kind, caption: e.caption.trim() || (e.kind === 'text' ? 'Schriftzug' : '') };
        if (e.kind === 'text') item.text = (e.text || '').trim();
        if (e.usePalette && e.palette) item.palette = [e.palette];
        return item;
    });
    return body;
}

async function layGenerate() {
    const body = layRequestBody();
    if (!body.prompt) { layStatus.textContent = 'Beschreibe zuerst das ganze Bild.'; return; }
    const missing = body.layout.findIndex(e => !e.caption || (e.kind === 'text' && !e.text));
    if (missing >= 0) { layStatus.textContent = `Element ${missing + 1}: Beschreibung${body.layout[missing].kind === 'text' ? ' und Text' : ''} fehlt.`; return; }
    const btn = document.getElementById('lay-run-btn');
    btn.disabled = true;
    const t0 = Date.now();
    layStatus.textContent = 'Erzeuge … (der erste Aufruf lädt das Modell, das dauert länger)';
    try {
        const res = await swFetch('/v1/images/generations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
        const src = `data:image/png;base64,${data.data[0].b64_json}`;
        layCanvas.style.backgroundImage = `url("${src}")`;
        layCanvas.classList.add('has-image');
        const dl = document.getElementById('lay-download');
        dl.href = src;
        dl.hidden = false;
        layStatus.textContent = `Fertig nach ${Math.round((Date.now() - t0) / 1000)} s (${body.model}).`;
        if (typeof saveToGallery === 'function') {
            saveToGallery(data.data[0].b64_json, { prompt: body.prompt, model: body.model, label: (document.getElementById('lay-variant').selectedOptions[0] || {}).textContent || body.model, steps: body.steps,
                                                   seed: body.seed, seconds: (Date.now() - t0) / 1000, source: 'positionen' });
        }
        if (typeof addLog === 'function') addLog(`MOSS-Image: Bild mit ${body.layout.length} Rahmen erzeugt`, 'system');
    } catch (err) {
        layStatus.textContent = `Fehler: ${err.message}`;
        if (typeof addLog === 'function') addLog(`MOSS-Image: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
    }
}

document.getElementById('lay-run-btn').addEventListener('click', layGenerate);
document.getElementById('lay-size').addEventListener('change', () => { layApplyAspect(); layRender(); });
['lay-prompt', 'lay-steps', 'lay-seed'].forEach(id => document.getElementById(id).addEventListener('input', layRender));
document.getElementById('lay-variant').addEventListener('change', layRender);
document.getElementById('lay-show-boxes').addEventListener('change', (ev) => layCanvas.classList.toggle('hide-boxes', !ev.target.checked));
document.getElementById('lay-clear-btn').addEventListener('click', () => { lay.elements = []; lay.selected = -1; layRender(); });
document.getElementById('lay-api-btn').addEventListener('click', () => {
    const pre = document.getElementById('lay-api');
    pre.hidden = !pre.hidden;
});
document.addEventListener('keydown', (ev) => {
    if ((ev.key === 'Delete' || ev.key === 'Backspace') && lay.selected >= 0 && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)
        && document.getElementById('layout-tab').classList.contains('active')) {
        lay.elements.splice(lay.selected, 1);
        lay.selected = -1;
        layRender();
    }
});

layApplyAspect();
layRender();
