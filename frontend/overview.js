// "Overview": boxes the owner arranges (order, width, hidden -- saved in the backend,
// GET/PUT /api/dashboard/layout) showing key figures, who uses the compute power on
// every machine, apps that start automatically, pinned apps and recent events.

let overviewPollInterval = null;
let overviewLayout = null;
let overviewArranging = false;
let overviewPicking = false;
let overviewDragging = null;
let overviewData = null;
let overviewSaveTimer = null;

const WIDGET_TITLES = {
    stats: tx('Key figures'), compute: tx('Compute power'), autostart: tx('Starts automatically'),
    pinned: tx('Pinned apps'), events: tx('Recent events'),
};
const WIDGET_SIZES = { narrow: tx('Narrow'), half: tx('Half'), wide: tx('Wide'), full: tx('Full width') };

function setOverviewPolling(active) {
    if (overviewPollInterval) {
        clearInterval(overviewPollInterval);
        overviewPollInterval = null;
    }
    if (active) {
        refreshOverview();
        overviewPollInterval = setInterval(refreshOverview, 5000);
    }
}

async function loadOverviewData() {
    const [services, consumers, status, nodes, events] = await Promise.all([
        api('/api/services').catch(() => null),
        api('/api/gpu/consumers').catch(() => null),
        api('/api/status').catch(() => null),
        api('/api/cluster/nodes').then(r => r.nodes).catch(() => []),
        api('/api/events?limit=8').then(r => r.events).catch(() => []),
    ]);
    const details = await Promise.all(nodes.map(n => api(`/api/cluster/nodes/${encodeURIComponent(n.id)}`)
        .then(d => ({ name: n.name, ...d })).catch(() => ({ id: n.id, name: n.name, online: false }))));
    // facts for apps shown on the overview that report them
    const shown = new Set((overviewLayout?.pinned || []));
    for (const s of services?.services || []) if (s.autostart) shown.add(s.id);
    const facts = {};
    await Promise.all((services?.services || []).filter(s => shown.has(s.id) && s.has_status && s.instances.length)
        .map(s => api(`/api/services/${encodeURIComponent(s.id)}/status`).then(r => { facts[s.id] = r.facts; }).catch(() => {})));
    if (services?.tailscale) servicesTailscale = services.tailscale;
    const host = services?.tailscale?.dns_name ? services.tailscale.dns_name.split('.')[0] : null;
    const side = document.getElementById('side-host');
    if (host && side) side.textContent = `${host} · ${tx('Coordinator')}`;
    return { services: services?.services || [], consumers: consumers?.gpus || [], status: status?.instances || [],
             nodes: details, events, facts };
}

async function refreshOverview() {
    if (!document.getElementById('overview-tab')?.classList.contains('active')) return;
    if (!overviewLayout) {
        try { overviewLayout = await api('/api/dashboard/layout'); } catch { return; }
    }
    overviewData = await loadOverviewData();
    if (typeof updateServicesCount === 'function' && overviewData.services.length) updateServicesCount({ services: overviewData.services });
    const c = document.getElementById('side-count-compute');
    if (c) c.textContent = `${1 + overviewData.nodes.filter(n => n.online).length}/${1 + overviewData.nodes.length}`;
    if (overviewBusy()) return;
    renderOverview();
}

// Don't rebuild under the owner's hands.
function overviewBusy() {
    const box = document.getElementById('overview-widgets');
    const a = document.activeElement;
    return overviewDragging || overviewPicking || (box && a && a !== document.body && box.contains(a) && a.tagName === 'SELECT');
}

function saveOverviewLayout() {
    clearTimeout(overviewSaveTimer);
    overviewSaveTimer = setTimeout(async () => {
        try {
            overviewLayout = await api('/api/dashboard/layout', 'PUT', overviewLayout);
        } catch (err) {
            addLog(tx('Arrangement not saved: {err}', { err: err.message }), 'error');
        }
    }, 400);
}

function renderOverview() {
    if (!overviewLayout || !overviewData) return;
    const box = document.getElementById('overview-widgets');
    const bar = document.getElementById('overview-arrange-bar');
    const btn = document.getElementById('overview-arrange-btn');
    btn.textContent = overviewArranging ? tx('Done') : tx('Arrange');
    btn.setAttribute('aria-pressed', String(overviewArranging));
    btn.classList.toggle('btn-primary', overviewArranging);
    btn.classList.toggle('btn-secondary', !overviewArranging);
    document.getElementById('overview-sub').textContent = overviewArranging
        ? tx('Drag boxes or move them with the arrows, pick a width, hide them. Saved automatically.')
        : tx('Everything that\'s running right now — at a glance.');

    const hidden = overviewLayout.widgets.filter(w => w.hidden);
    bar.hidden = !overviewArranging;
    bar.replaceChildren(el('span', { text: tx('Arrange: order, width and visibility apply to this dashboard.') }),
        ...(hidden.length ? [el('span', { class: 'muted', text: tx('Hidden:') }),
            ...hidden.map(w => el('button', { class: 'chip', text: `+ ${WIDGET_TITLES[w.id]}`,
                onclick: () => { w.hidden = false; saveOverviewLayout(); renderOverview(); } }))] : []));

    box.replaceChildren(...overviewLayout.widgets.filter(w => !w.hidden).map(renderWidget));
}

function moveWidget(id, delta) {
    const ws = overviewLayout.widgets;
    const visible = ws.filter(w => !w.hidden);
    const i = visible.findIndex(w => w.id === id), j = i + delta;
    if (j < 0 || j >= visible.length) return;
    const a = ws.indexOf(visible[i]), b = ws.indexOf(visible[j]);
    [ws[a], ws[b]] = [ws[b], ws[a]];
    saveOverviewLayout();
    renderOverview();
}

function renderWidget(w) {
    const section = el('section', { class: `widget widget-${w.size}${overviewArranging ? ' widget-arranging' : ''}`,
                                    'aria-label': WIDGET_TITLES[w.id], 'data-widget': w.id });
    if (overviewArranging) {
        section.draggable = true;
        section.addEventListener('dragstart', (e) => { overviewDragging = w.id; e.dataTransfer.effectAllowed = 'move'; section.classList.add('widget-dragging'); });
        section.addEventListener('dragend', () => { overviewDragging = null; renderOverview(); });
        section.addEventListener('dragover', (e) => { if (overviewDragging && overviewDragging !== w.id) { e.preventDefault(); section.classList.add('widget-drop'); } });
        section.addEventListener('dragleave', () => section.classList.remove('widget-drop'));
        section.addEventListener('drop', (e) => {
            e.preventDefault();
            const from = overviewDragging;
            overviewDragging = null;
            if (!from || from === w.id) return;
            const ws = overviewLayout.widgets;
            const moved = ws.splice(ws.findIndex(x => x.id === from), 1)[0];
            ws.splice(ws.findIndex(x => x.id === w.id), 0, moved);
            saveOverviewLayout();
            renderOverview();
        });
        const size = el('select', { class: 'r-select-sm', 'aria-label': tx('Width of {box}', { box: WIDGET_TITLES[w.id] }),
            onchange: (e) => { w.size = e.target.value; saveOverviewLayout(); renderOverview(); } });
        for (const [k, label] of Object.entries(WIDGET_SIZES)) {
            const o = el('option', { value: k, text: label });
            if (k === w.size) o.selected = true;
            size.append(o);
        }
        section.append(el('div', { class: 'widget-toolbar' },
            el('span', { class: 'drag-handle', 'aria-hidden': 'true', text: '⠿' }),
            el('span', { class: 'strong small', text: WIDGET_TITLES[w.id] }),
            el('span', { class: 'spacer' }),
            el('button', { class: 'btn-icon-sm', text: '←', 'aria-label': tx('Move {box} forward', { box: WIDGET_TITLES[w.id] }), onclick: () => moveWidget(w.id, -1) }),
            el('button', { class: 'btn-icon-sm', text: '→', 'aria-label': tx('Move {box} back', { box: WIDGET_TITLES[w.id] }), onclick: () => moveWidget(w.id, 1) }),
            size,
            el('button', { class: 'btn btn-secondary btn-sm', text: tx('Hide'),
                           onclick: () => { w.hidden = true; saveOverviewLayout(); renderOverview(); } })));
    }
    const body = { stats: widgetStats, compute: widgetCompute, autostart: widgetAutostart, pinned: widgetPinned, events: widgetEvents }[w.id];
    section.append(...body());
    return section;
}

// ---------------------------------------------------------------- figures

function overviewAccelerators() {
    const d = overviewData;
    const out = [];
    for (const g of d.consumers) {
        out.push({ where: 'local', machine: tx('this machine'), name: g.name || `GPU ${g.index}`, total: g.total_mb,
                   limit: g.gpu_limit_mb || null, free: Math.max(0, g.free_mb - (g.reserved_mb || 0)),
                   parts: g.consumers.map((c, i) => ({ label: consumerLabel(c), mb: c.mb, color: consumerColor(c, i),
                       state: consumerState(c)[0] })) });
    }
    for (const n of d.nodes) {
        if (!n.online) continue;
        const sizes = {};
        for (const f of (n.downloads && n.downloads.files) || []) sizes[f.name] = Math.round(f.size_bytes / 1048576);
        for (const a of n.accelerators || []) {
            const parts = (n.instances || []).filter(i => i.gpu === null || i.gpu === undefined || i.gpu === a.index)
                .map((i, k) => ({ label: i.model, mb: sizes[i.model] || 0, color: CONSUMER_COLORS[k % CONSUMER_COLORS.length],
                                  state: i.state === 'ready' ? tx('ready') : i.state }));
            const used = parts.reduce((t, p) => t + p.mb, 0);
            out.push({ where: 'remote', machine: n.name || n.id, name: a.name, total: a.budget_mb,
                       free: Math.max(0, a.budget_mb - used), parts, budget: true });
        }
    }
    return out;
}

function widgetStats() {
    const d = overviewData;
    const accels = overviewAccelerators();
    const free = accels.reduce((t, a) => t + a.free, 0), total = accels.reduce((t, a) => t + a.total, 0);
    const online = d.nodes.filter(n => n.online);
    const offline = d.nodes.filter(n => !n.online);
    const models = [];
    for (const g of d.consumers) for (const c of g.consumers) if (c.kind === 'llm' || (c.kind === 'image' && c.mb >= 1024)) models.push(c.label);
    for (const n of online) for (const i of n.instances || []) models.push(`${i.model} @ ${n.name || n.id}`);
    const running = d.services.filter(s => s.instances.length);
    const stat = (label, value, of, note, goto, tone) => el('button', { class: `stat stat-${tone}`, 'data-goto': goto },
        el('span', { class: 'muted small', text: label }),
        el('span', { class: 'stat-value' }, el('span', { text: value }), of ? el('span', { class: 'stat-of', text: ` ${of}` }) : null),
        el('span', { class: 'muted small stat-note', text: note }));
    return [el('div', { class: 'stat-grid' },
        stat(tx('Free on all machines'), fmtGb(free), `/ ${fmtGb(total)} GB`,
             (online.length ? tx('{a} accelerators on {m} machines', { a: accels.length, m: 1 + online.length })
                            : tx('{a} accelerators on 1 machine', { a: accels.length })), 'compute', 'a'),
        stat(tx('Models loaded'), String(models.length), '', models.slice(0, 3).join(' · ') || tx('none'), 'compute', 'b'),
        stat(tx('Machines online'), String(1 + online.length), `/ ${1 + d.nodes.length}`,
             offline.length ? tx('{names} unreachable', { names: offline.map(n => n.name || n.id).join(', ') }) : tx('all reachable'), 'compute', 'c'),
        stat(tx('Apps active'), String(running.length), `/ ${d.services.length}`, running.slice(0, 3).map(s => s.name).join(', ') || tx('none'), 'services', 'd'))];
}

function widgetCompute() {
    const accels = overviewAccelerators();
    return [el('div', { class: 'widget-head' }, el('h2', { text: tx('Who uses the compute power') }),
               el('button', { class: 'link-btn', 'data-goto': 'compute', text: tx('Details') })),
        ...accels.map(a => {
            const bar = el('div', { class: 'usage-bar usage-bar-thin', role: 'img', 'aria-label': `${a.name}: ${a.parts.map(p => p.label).join(', ') || tx('empty')}` });
            for (const p of a.parts) bar.append(el('div', { style: `width:${Math.min(100, 100 * p.mb / Math.max(1, a.total))}%;background:${p.color}` }));
            return el('div', { class: 'ov-accel' },
                el('div', { class: 'ov-accel-head' },
                    el('span', { class: 'strong' }, el('span', { text: a.name }),
                       el('span', { class: `where-tag ${a.where === 'local' ? 'where-local' : ''}`, text: a.where === 'local' ? tx('this machine') : `${a.machine} · ${tx('remote')}` })),
                    el('span', { class: 'mono small muted', text: a.budget ? tx('{free} GB free · budget {total} GB', { free: fmtGb(a.free), total: fmtGb(a.total) })
                                                                  : freeMemText(a.free, a.total, a.limit) })),
                bar,
                el('div', { class: 'legend' }, ...(a.parts.length ? a.parts.map(p => el('span', {},
                    el('span', { class: 'swatch', style: `background:${p.color}` }),
                    el('span', { text: `${p.label} · ${fmtGb(p.mb)} GB${p.state && p.state !== '—' ? ` · ${p.state}` : ''}` })))
                    : [el('span', { class: 'muted', text: tx('nothing loaded') })])));
        })];
}

function appTileCard(s, withShot) {
    const best = typeof serviceBestUrl === 'function' ? serviceBestUrl(s) : null;
    const facts = overviewData.facts[s.id] || [];
    const actions = [];
    const open = s.instances.length && best ? serviceOpenControl(s, 'btn btn-secondary btn-sm') : null;
    if (open) actions.push(open);
    if (s.instances.length === 1) {
        const inst = s.instances[0];
        actions.push(el('button', { class: 'btn btn-danger-outline btn-sm', text: s.category === 'integrations' ? tx('Stop (close access)') : tx('Stop'),
            onclick: () => { if (confirmStop(s, inst)) serviceAction(s.id, 'stop', { pid: inst.pid }); } }));
    }
    if (!s.instances.length) actions.push(el('button', { class: 'btn btn-outline btn-sm', text: s.status === 'starting' ? tx('Starting…') : tx('Start'),
        disabled: s.status === 'starting', onclick: () => serviceAction(s.id, 'start') }));
    return el('article', { class: 'ov-app' },
        withShot && s.has_screenshot ? appScreenshot(s, best && !best.unreachable ? best.url : null) : null,
        el('div', { class: 'ov-app-head' }, appTile(s, 34), el('span', { class: 'strong ov-app-name', text: s.name }), maturityPill(s), statusPill(s)),
        facts.length ? el('dl', { class: 'facts' }, ...facts.flatMap(f => [el('dt', { text: tx(f.label) }), el('dd', { text: tx(f.value) })])) : null,
        actions.length ? el('div', { class: 'app-actions' }, ...actions) : null);
}

function widgetAutostart() {
    const apps = overviewData.services.filter(s => s.autostart);
    return [el('div', { class: 'widget-head' },
            el('div', {}, el('h2', { text: tx('Starts automatically') }), el('span', { class: 'muted small', text: tx('Follows the autostart checkbox in Apps & Services') })),
            el('button', { class: 'link-btn', 'data-goto': 'services', text: tx('Change') })),
        el('div', { class: 'ov-app-grid' }, ...(apps.length ? apps.map(s => appTileCard(s, false))
            : [el('p', { class: 'muted small', text: tx('No app starts automatically.') })]))];
}

function widgetPinned() {
    const byId = new Map(overviewData.services.map(s => [s.id, s]));
    const apps = overviewLayout.pinned.map(id => byId.get(id)).filter(Boolean);
    const head = el('div', { class: 'widget-head' },
        el('div', {}, el('h2', { text: tx('Pinned apps') }), el('span', { class: 'muted small', text: tx('Chosen by you') })),
        el('button', { class: `btn btn-sm ${overviewPicking ? 'btn-primary' : 'btn-secondary'}`, 'aria-pressed': String(overviewPicking),
                       text: overviewPicking ? tx('Done') : tx('Choose'),
                       onclick: () => { overviewPicking = !overviewPicking; renderOverview(); } }));
    const parts = [head];
    if (overviewPicking) {
        const fs = el('fieldset', { class: 'picker' }, el('legend', { text: tx('Pinned apps') }));
        for (const s of overviewData.services) {
            const box = el('input', { type: 'checkbox', onchange: (e) => {
                const p = overviewLayout.pinned.filter(x => x !== s.id);
                if (e.target.checked) p.push(s.id);
                overviewLayout.pinned = p;
                saveOverviewLayout();
            } });
            box.checked = overviewLayout.pinned.includes(s.id);
            fs.append(el('label', { class: 'picker-row' }, box, el('span', { text: s.name })));
        }
        parts.push(fs);
    }
    parts.push(el('div', { class: 'ov-app-grid' }, ...(apps.length ? apps.map(s => appTileCard(s, true))
        : [el('p', { class: 'muted small', text: tx('Nothing pinned yet -- “Choose”.') })])));
    return parts;
}

function widgetEvents() {
    const ev = overviewData.events;
    const time = (t) => new Date(t * 1000).toLocaleTimeString(I18N.locale, { hour: '2-digit', minute: '2-digit' });
    return [el('h2', { text: tx('Recent events') }),
        el('div', { class: 'events' }, ...(ev.length ? ev.map(e => el('div', { class: `event event-${e.level}` },
            el('span', { class: 'mono small muted', text: time(e.t) }), el('span', { text: e.template ? tx(e.template, e.vars) : e.text })))
            : [el('p', { class: 'muted small', text: tx('Nothing has happened since the last restart.') })]))];
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('overview-arrange-btn')?.addEventListener('click', () => {
        overviewArranging = !overviewArranging;
        overviewPicking = false;
        renderOverview();
    });
    setOverviewPolling(true);  // the overview is the start page
});
