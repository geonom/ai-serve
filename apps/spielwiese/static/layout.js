// "With positions": MOSS-Image with positions. Boxes are drawn on the canvas (relative 0..1, as the
// gateway's /v1/images/generations expects with model "moss" and `layout`), each with a kind,
// a caption and -- for text -- the exact text. DOM only via textContent/value (no innerHTML).

const LAY_COLORS = ['#e4572e', '#17bebb', '#ffc914', '#76b041', '#2e86ab', '#a23b72', '#f18f01', '#6a4c93'];
const LAY_KINDS = [['object', tx('Object')], ['character', tx('Character')], ['text', tx('Text')]];
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
        layList.append(layEl('div', { class: 'muted small', text: tx('No boxes yet – draw one on the canvas.') }));
        return;
    }
    lay.elements.forEach((e, i) => {
        const kind = layEl('select', { 'aria-label': tx('Kind of element {n}', { n: i + 1 }), onchange: (ev) => { e.kind = ev.target.value; layRender(); } },
            ...LAY_KINDS.map(([v, l]) => { const o = layEl('option', { value: v, text: l }); o.selected = v === e.kind; return o; }));
        const caption = layEl('input', { type: 'text', maxlength: '400', placeholder: tx('What is here? e.g. baker with a bread basket'),
                                         'aria-label': tx('Description of element {n}', { n: i + 1 }), value: e.caption,
                                         oninput: (ev) => { e.caption = ev.target.value; layRenderBoxes(); } });
        const text = e.kind === 'text' ? layEl('input', { type: 'text', maxlength: '200', class: 'lay-wide', placeholder: tx('Exact text in the image'),
                                                           'aria-label': tx('Text of element {n}', { n: i + 1 }), value: e.text || '',
                                                           oninput: (ev) => { e.text = ev.target.value; layRenderBoxes(); } }) : null;
        const color = layEl('input', { type: 'color', value: e.palette || '#888888', title: tx('Colour (optional)'),
                                       'aria-label': tx('Colour of element {n}', { n: i + 1 }),
                                       onchange: (ev) => { e.palette = ev.target.value; } });
        const useColor = layEl('label', { class: 'small lay-wide' }, (() => {
            const c = layEl('input', { type: 'checkbox', onchange: (ev) => { e.usePalette = ev.target.checked; } });
            c.checked = !!e.usePalette; return c; })(), document.createTextNode(' ' + tx('set a colour') + ' '), color);
        const del = layEl('button', { class: 'btn btn-secondary btn-sm', type: 'button', text: '✕', title: tx('Delete box'),
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
    if (lay.elements.length >= 20) { layStatus.textContent = tx('At most 20 boxes.'); return; }
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
        const item = { box: e.box, kind: e.kind, caption: e.caption.trim() || (e.kind === 'text' ? 'lettering' : '') };
        if (e.kind === 'text') item.text = (e.text || '').trim();
        if (e.usePalette && e.palette) item.palette = [e.palette];
        return item;
    });
    return body;
}

async function layGenerate() {
    const body = layRequestBody();
    if (!body.prompt) { layStatus.textContent = tx('Describe the whole image first.'); return; }
    const missing = body.layout.findIndex(e => !e.caption || (e.kind === 'text' && !e.text));
    if (missing >= 0) { layStatus.textContent = body.layout[missing].kind === 'text' ? tx('Element {n}: description and text missing.', { n: missing + 1 }) : tx('Element {n}: description missing.', { n: missing + 1 }); return; }
    const btn = document.getElementById('lay-run-btn');
    btn.disabled = true;
    const t0 = Date.now();
    layStatus.textContent = tx('Generating … (the first call loads the model, which takes longer)');
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
        layStatus.textContent = tx('Done after {s} s ({model}).', { s: Math.round((Date.now() - t0) / 1000), model: body.model });
        if (typeof saveToGallery === 'function') {
            saveToGallery(data.data[0].b64_json, { prompt: body.prompt, model: body.model, label: (document.getElementById('lay-variant').selectedOptions[0] || {}).textContent || body.model, steps: body.steps,
                                                   seed: body.seed, seconds: (Date.now() - t0) / 1000, source: 'positionen' });
        }
        if (typeof addLog === 'function') addLog(tx('MOSS-Image: image with {n} boxes generated', { n: body.layout.length }), 'system');
    } catch (err) {
        layStatus.textContent = tx('Error: {err}', { err: err.message });
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

// The variants come from the gateway's catalog (models with `layout`); without one the page says what's missing.
async function layLoadVariants() {
    const sel = document.getElementById('lay-variant'), btn = document.getElementById('lay-run-btn');
    let rows = [];
    try { rows = ((await api('/v1/images/models')).data || []).filter(m => m.layout); } catch (err) { rows = []; }
    const keep = sel.value;
    sel.replaceChildren(...rows.map(m => {
        const o = document.createElement('option');
        o.value = m.id; o.disabled = !m.available;
        o.textContent = m.available ? m.label : tx('{label} (not installed)', { label: m.label });
        return o;
    }));
    const usable = rows.filter(m => m.available);
    const pick = usable.find(m => m.id === keep) || usable.find(m => m.id === 'moss-turbo-4-bf16') || usable[0];
    if (pick) sel.value = pick.id;
    btn.disabled = !usable.length;
    // only the hint: the image downloads on offer (Iris) can't place things either
    Downloads.panel(document.getElementById('dl-layout'), false, usable.length ? '' :
        tx('No model on this gateway can place things by position. That needs MOSS-Image (the comfyui-moss plugin) – it isn\'t part of this installation (docs/plugins.md).'));
    layRender();
}
window.addEventListener('sw-installed', (e) => { if (e.detail.capability === 'image') layLoadVariants(); });

layApplyAspect();
layRender();
layLoadVariants();
