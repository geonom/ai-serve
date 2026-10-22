// "Compute & models": this machine and the other devices as one pool.
// This machine's accelerators come from the central GPU manager (GET /api/gpu/consumers,
// with Pin / Unload per consumer); remote devices are rendered by devices.js
// into the same list. Below: every model on any machine, with where it lies and runs.

let computePollInterval = null;
let computeShow = 'all';           // all | local | remote
let computeLocalModels = null;     // /api/models
let computeStatus = null;          // /api/status (what runs here)

function setComputePolling(active) {
    if (computePollInterval) {
        clearInterval(computePollInterval);
        computePollInterval = null;
    }
    if (active) {
        renderComputeFilters();
        fetchCompute();
        computePollInterval = setInterval(fetchCompute, 5000);
    }
}

const CONSUMER_KIND = {
    llm: tx('language model'), image: tx('image models'), tts: tx('speech output'), stt: tx('speech recognition'), training: tx('training'),
    exclusive: tx('exclusive (all GPUs)'), other: tx('not managed'),
};
const CONSUMER_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)'];

function consumerColor(c, i) {
    return c.kind === 'other' ? 'var(--c-other)' : CONSUMER_COLORS[i % CONSUMER_COLORS.length];
}

function consumerLabel(c) {
    return c.kind === 'other' ? tx('other programs') : c.label;
}

function consumerState(c) {
    if (c.kind === 'other') return ['—', ''];
    if (c.busy) return [tx('working'), 'tone-busy'];
    if (c.pinned) return [tx('pinned'), 'tone-pin'];
    if (!c.evictable) return [tx('never unloaded'), ''];
    return [tx('idle'), ''];
}

function renderComputeFilters() {
    const box = document.getElementById('compute-filters');
    if (!box) return;
    const chip = (key, label) => el('button', {
        class: `chip${key === computeShow ? ' chip-active' : ''}`, 'aria-pressed': String(key === computeShow), text: label,
        onclick: () => { computeShow = key; renderComputeFilters(); applyComputeFilter(); } });
    box.replaceChildren(chip('all', tx('All machines')), chip('local', tx('This machine only')), chip('remote', tx('Remote only')));
}

function applyComputeFilter() {
    const local = document.getElementById('compute-local');
    const remote = document.getElementById('devices-list');
    if (local) local.hidden = computeShow === 'remote';
    if (remote) remote.hidden = computeShow === 'local';
}

// One accelerator: bar + consumer rows. rows: [{label, sub, mb, color, state, tone, actions[]}]
function renderAccel({ name, kind, totalMb, freeText, rows, reservedMb = 0, reservations = [] }) {
    const bar = el('div', { class: 'usage-bar', role: 'img',
                            'aria-label': `${name}: ${rows.map(r => `${r.label} ${fmtGb(r.mb)} GB`).join(', ') || tx('empty')}` });
    for (const r of rows) {
        bar.append(el('div', { style: `width:${Math.min(100, 100 * r.mb / Math.max(1, totalMb))}%;background:${r.color}` }));
    }
    if (reservedMb > 0) bar.append(el('div', { class: 'usage-reserved', style: `width:${Math.min(100, 100 * reservedMb / Math.max(1, totalMb))}%` }));
    const list = el('div', { class: 'consumer-list' });
    for (const r of rows) {
        list.append(el('div', { class: 'consumer-row' },
            el('span', { class: 'swatch', style: `background:${r.color}` }),
            el('span', { class: 'consumer-name' }, el('span', { class: 'strong', text: r.label }), el('span', { class: 'muted small', text: r.sub })),
            el('span', { class: 'mono small', text: `${fmtGb(r.mb)} GB` }),
            el('span', { class: `small ${r.tone || ''}`, text: r.state }),
            el('span', { class: 'consumer-actions' }, ...(r.actions || []))));
    }
    if (!rows.length) list.append(el('div', { class: 'consumer-row muted small', text: tx('Nothing loaded') }));
    for (const res of reservations) {
        list.append(el('div', { class: 'consumer-row small tone-pin' },
            el('span', { class: 'swatch usage-reserved' }),
            el('span', { text: (res.pid ? tx('Reserved: {gb} GB for {purpose}', { gb: fmtGb(res.mb), purpose: res.purpose })
                                     : tx('Reserved: {gb} GB for {purpose} — waiting for confirmation', { gb: fmtGb(res.mb), purpose: res.purpose })) })));
    }
    return el('div', { class: 'accel' },
        el('div', { class: 'accel-head' }, el('span', { class: 'strong', text: name }, el('span', { class: 'muted small', text: ` ${kind}` })),
           el('span', { class: 'mono small muted', text: freeText })),
        bar, list);
}

function machineCard({ id, name, where, sub, online, statusText, accels, facts = [], extra = [], menu = null, icon }) {
    const head = el('div', { class: 'machine-head' },
        el('div', { class: `machine-icon${where === 'local' ? ' machine-icon-local' : ''}` }, svgIcon(icon, 22)),
        el('div', { class: 'machine-title' },
            el('div', { class: 'machine-title-row' }, el('h2', { text: name }),
               el('span', { class: `where-tag ${where === 'local' ? 'where-local' : ''}`, text: where === 'local' ? tx('this machine') : tx('remote') })),
            el('span', { class: 'muted small', text: sub })),
        el('span', { class: `pill ${online ? 'pill-ok' : 'pill-off'}` }, el('span', { class: 'pill-dot' }), el('span', { text: statusText })),
        menu);
    return el('section', { class: `machine-card${where === 'local' ? ' machine-local' : ''}`, 'data-machine': id },
        head,
        accels.length ? el('div', { class: 'accel-grid' }, ...accels) : null,
        facts.length ? el('div', { class: 'machine-facts' }, ...facts.map(([k, v]) => el('span', {}, el('span', { class: 'muted', text: `${k} ` }), el('span', { text: v })))) : null,
        ...extra);
}

async function consumerAction(path, body, okText) {
    try {
        await api(path, 'POST', body || {});
        addLog(okText, 'system');
    } catch (err) {
        addLog(`GPU: ${err.message}`, 'error');
        if (typeof deviceHint === 'function') deviceHint(err.message);
    }
    fetchCompute();
}

function consumerRows(g) {
    return g.consumers.map((c, i) => {
        const [state, tone] = consumerState(c);
        const actions = [];
        if (!['other', 'training', 'exclusive'].includes(c.kind)) {
            actions.push(el('button', {
                class: `btn-icon-sm${c.pinned ? ' is-on' : ''}`, 'aria-pressed': String(!!c.pinned),
                title: c.pinned ? tx('Unpin') : tx('Pin (never unload automatically)'),
                'aria-label': c.pinned ? tx('Stop pinning {label}', { label: c.label }) : tx('Pin {label}', { label: c.label }),
                onclick: () => consumerAction(`/api/gpu/consumers/${encodeURIComponent(c.pin_key || c.id)}/pin`, { pinned: !c.pinned },
                                              c.pinned ? tx('{label}: no longer pinned', { label: c.label }) : tx('{label}: pinned', { label: c.label })),
            }, svgIcon('M9 4h6l-1 6 4 4H6l4-4zM12 14v6', 15, 2)));
        }
        if (c.evictable || (c.pinned && !c.busy)) {
            actions.push(el('button', {
                class: 'btn btn-danger-outline btn-sm', text: tx('Unload'),
                title: tx('Frees the GPU memory now (files stay; the next request loads again)'),
                onclick: () => consumerAction(`/api/gpu/consumers/${encodeURIComponent(c.id)}/unload`, null, tx('{label}: unloaded', { label: c.label })),
            }));
        }
        if (c.kind === 'exclusive' && c.plugin) {      // e.g. Strata: never unloaded automatically, stopped by hand
            actions.push(el('button', {
                class: 'btn btn-danger-outline btn-sm', text: tx('Stop'),
                title: tx('Stops the model server and frees all graphics cards (only while it isn\'t doing anything)'),
                onclick: () => consumerAction(`/api/plugins/${encodeURIComponent(c.plugin)}/release`, null, tx('{label}: stopped', { label: c.label })),
            }));
        }
        return { label: consumerLabel(c), sub: CONSUMER_KIND[c.kind] || c.kind, mb: c.mb, color: consumerColor(c, i), state, tone, actions };
    });
}

async function fetchCompute() {
    const box = document.getElementById('compute-local');
    if (!box) return;
    applyComputeFilter();
    let data;
    try {
        [data, computeStatus] = await Promise.all([api('/api/gpu/consumers'), api('/api/status').catch(() => null)]);
    } catch (err) {
        box.replaceChildren(el('div', { class: 'empty-state', text: tx('Couldn\'t load GPU usage: {err}', { err: err.message }) }));
        return;
    }
    if (box.contains(document.activeElement) && document.activeElement !== document.body) return;
    const accels = data.gpus.map(g => renderAccel({
        name: g.name || `GPU ${g.index}`, kind: `GPU ${g.index}`, totalMb: g.total_mb,
        freeText: tx('{free} of {total} GB free', { free: fmtGb(g.free_mb), total: fmtGb(g.total_mb) }),
        rows: consumerRows(g), reservedMb: g.reserved_mb || 0, reservations: g.reservations || [],
    }));
    const side = document.getElementById('side-host');
    if (side && !side.textContent.includes('·')) {
        try {
            const dns = (await api('/api/services')).tailscale?.dns_name;
            if (dns) side.textContent = `${dns.split('.')[0]} · ${tx('Coordinator')}`;
        } catch { /* keep the generic label */ }
    }
    const host = (side?.textContent || '').includes('·') ? side.textContent.split(' · ')[0] : tx('This machine');
    box.replaceChildren(machineCard({
        id: 'local', name: host, where: 'local', icon: 'M4 4h16v11H4zM9 20h6M12 15v5',
        sub: tx('Coordinator · spreads requests over all machines'), online: true, statusText: tx('online'), accels,
    }));
    renderComputeModels();
    updateComputeCount();
}

function updateComputeCount() {
    const c = document.getElementById('side-count-compute');
    if (!c) return;
    const nodes = typeof devicesLastDetails !== 'undefined' && devicesLastDetails ? devicesLastDetails : [];
    c.textContent = `${1 + nodes.filter(n => n.online).length}/${1 + nodes.length}`;
}

function modelTags(m) {
    const n = m.name.toLowerCase();
    if (n.includes('rerank')) return ['Reranker'];
    if (/embed|bge-m3|nomic/.test(n)) return ['Embedding'];
    if (n.includes('nuextract')) return [tx('Reading documents')];
    const t = ['Chat'];
    if (m.mmproj_path || m.vision_bundled) t.push(tx('Understands images'));
    if (m.mtp_bundled || m.mtp_draft_path) t.push('MTP');
    return t;
}

async function renderComputeModels() {
    const box = document.getElementById('compute-models');
    if (!box) return;
    if (!computeLocalModels) {
        try { computeLocalModels = (await api('/api/models')).models || []; } catch { computeLocalModels = []; }
    }
    if (box.contains(document.activeElement) && document.activeElement !== document.body) return;
    const q = (document.getElementById('compute-model-search')?.value || '').trim().toLowerCase();
    const nodes = (typeof devicesLastDetails !== 'undefined' && devicesLastDetails) || [];
    const base = (p) => (p || '').split('/').pop();
    const running = new Map();
    for (const i of (computeStatus && computeStatus.instances) || []) {
        if (i.port === 8189 || !i.model) continue;
        running.set(base(i.model_path || i.model), i.gpu_id !== null && i.gpu_id !== undefined ? tx('loaded · GPU {gpu}', { gpu: i.gpu_id }) : tx('loaded'));
    }
    const byName = new Map();
    for (const m of computeLocalModels) byName.set(m.name, { ...m, on: ['local'] });
    for (const n of nodes) {
        if (!n.online) continue;
        for (const f of (n.downloads && n.downloads.files) || []) {
            const e = byName.get(f.name) || { name: f.name, on: [] };
            if (!e.on.includes(n.id)) e.on.push(n.id);
            byName.set(f.name, e);
        }
        for (const i of n.instances || []) {
            if (!running.has(base(i.model))) running.set(base(i.model), tx('running · {node}', { node: n.name || n.id }));
        }
    }
    let rows = [...byName.values()];
    if (q) rows = rows.filter(m => m.name.toLowerCase().includes(q));
    rows.sort((a, b) => (running.has(b.name) - running.has(a.name)) || a.name.localeCompare(b.name));
    const shown = rows.slice(0, q ? 200 : 40);
    const table = el('table', { class: 'r-table' },
        el('thead', {}, el('tr', {}, ...[tx('Model'), tx('Context'), tx('Can'), tx('Lies on'), tx('Status'), ''].map(h => el('th', { scope: 'col', text: h })))),
        el('tbody', {}, ...shown.map(m => el('tr', {},
            el('td', { class: 'mono small', text: m.name }),
            el('td', { class: 'mono small', text: m.context_length ? `${Math.round(m.context_length / 1024)}k` : '—' }),
            el('td', {}, ...modelTags(m).map(t => el('span', { class: 'tag', text: t }))),
            el('td', {}, ...m.on.map(o => el('span', { class: `where-tag ${o === 'local' ? 'where-local' : ''}`,
                                                      text: o === 'local' ? tx('this machine') : o }))),
            el('td', { class: `small ${running.has(m.name) ? 'tone-ok' : 'muted'}`, text: running.get(m.name) || '—' }),
            el('td', { class: 'right' }, m.on.includes('local') && !running.has(m.name)
                ? el('button', { class: 'btn btn-secondary btn-sm', text: tx('Load …'), onclick: () => openLaunchFor(m.path) })
                : null)))));
    box.replaceChildren(el('div', { class: 'table-scroll' }, table),
        rows.length > shown.length ? el('p', { class: 'muted small', text: tx('{n} more -- find them through the search.', { n: rows.length - shown.length }) }) : null);
}

function openLaunchFor(path) {
    const details = document.getElementById('compute-launch');
    const select = document.getElementById('model-select');
    if (path && select) {
        select.value = path;
        select.dispatchEvent(new Event('change'));
    }
    if (details) {
        details.open = true;
        details.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('compute-launch-btn')?.addEventListener('click', () => openLaunchFor(null));
    document.getElementById('compute-model-search')?.addEventListener('input', renderComputeModels);
});
