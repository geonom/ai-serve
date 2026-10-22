// Remote machines on "Compute & models": other machines (Macs, PCs, Pis, phones)
// running the node agent, used as optional AI resources (GET/POST /api/cluster/...).
// Rendered as machine cards next to this machine's (compute.js). Uses el()/api()/
// addLog() from services.js/app.js. All text from nodes goes through textContent only.

let devicesPollInterval = null;
let devicesLocalModels = null;
let devicesLastDetails = null;  // last /api/cluster/nodes/{id} answers (compute.js reads them)

function setDevicesPolling(active) {
    if (devicesPollInterval) {
        clearInterval(devicesPollInterval);
        devicesPollInterval = null;
    }
    if (active) {
        fetchDevices();
        devicesPollInterval = setInterval(fetchDevices, 5000);
    }
}

function deviceHint(msg, isError = true) {
    const box = document.getElementById('devices-messages');
    box.replaceChildren(el('div', { class: `service-hint${isError ? ' service-error' : ''}`, text: msg }));
    setTimeout(() => { if (box.textContent === msg) box.replaceChildren(); }, 10000);
}

function fmtMb(mb) {
    if (mb === null || mb === undefined) return '?';
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

async function addDevice() {
    const box = document.getElementById('devices-enroll');
    try {
        const e = await api('/api/cluster/enrollments', 'POST');
        const mins = Math.round((e.expires_at - Date.now() / 1000) / 60);
        const cmd = el('pre', { class: 'code-snippet', text: e.command });
        const copy = el('button', {
            class: 'btn btn-secondary btn-sm', text: tx('Copy command'),
            onclick: () => navigator.clipboard.writeText(e.command).then(() => { copy.textContent = tx('Copied'); }),
        });
        box.replaceChildren(el('section', { class: 'r-card r-card-accent' },
            el('h2', { text: tx('Add a machine') }),
            el('p', { class: 'service-desc', text: tx('Run this on the new device (needs Python 3.9+ and Tailscale; the device must belong to your Tailscale account). The code works once and expires in {mins} minutes. Then let it start with the device: "python3 agent.py --install-service" (macOS, Linux, Android/Termux). Macs and other devices with shared memory use at most half of their RAM unless you change the budget.', { mins }) }),
            cmd, copy,
        ));
    } catch (err) {
        deviceHint(tx('No one-time code possible: {err}', { err: err.message }));
    }
}

function renderPlaceForm() {
    const box = document.getElementById('devices-place');
    if (box.childElementCount) return;  // built once; keeps what was typed
    const local = el('select', { class: 'service-exposure-select' }, el('option', { value: '', text: tx('Local model…') }));
    (devicesLocalModels || []).forEach(m => local.append(el('option', { value: m, text: m })));
    const repo = el('input', { type: 'text', placeholder: tx('or HF repo'), class: 'device-input' });
    const file = el('input', { type: 'text', placeholder: tx('File in the repo (.gguf)'), class: 'device-input' });
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: tx('Send to the best-fitting device'),
        onclick: async () => {
            const body = local.value ? { model: local.value } : { hf_repo: repo.value.trim(), hf_file: file.value.trim() };
            btn.disabled = true;
            try {
                const r = await api('/api/cluster/models', 'POST', body);
                addLog(r.state === 'present' ? tx('{name} is already on {node}', { name: r.name, node: r.node })
                                             : tx('Copying {name} to {node} (if it refuses, the next device gets its turn)', { name: r.name, node: r.node }), 'system');
                local.value = ''; repo.value = ''; file.value = '';
            } catch (err) {
                deviceHint(tx('Sending failed: {err}', { err: err.message }));
            }
            btn.disabled = false;
            fetchDevices();
        },
    });
    box.replaceChildren(el('div', { class: 'device-models', title:
        tx('The coordinator picks a device with room; if one refuses the download (quota, disk), the next one gets its turn.') },
        el('span', { text: tx('Copy a model to a device') }), local, repo, file, btn));
}

async function renderCopies() {
    const box = document.getElementById('devices-copies');
    let data;
    try { data = await api('/api/cluster/copies'); } catch { return; }
    const rows = [];
    for (const [name, c] of Object.entries(data.copies || {})) {
        const text = c.state === 'copying' ? tx('Copying {name} to {node}…', { name, node: c.node })
            : c.state === 'done' ? tx('{name} is now on {node}', { name, node: c.node })
            : tx('{name}: no device could take it', { name });
        rows.push(el('div', { class: `service-hint${c.state === 'failed' ? ' service-error' : ''}`, text }));
        for (const e of c.errors || []) rows.push(el('div', { class: 'service-dir', text: tx('skipped: {e}', { e }) }));
    }
    box.replaceChildren(...rows);
}

async function fetchDevices() {
    renderPlaceForm();
    renderCopies();
    const list = document.getElementById('devices-list');
    let nodes;
    try {
        nodes = (await api('/api/cluster/nodes')).nodes;
    } catch (err) {
        list.replaceChildren(el('div', { class: 'empty-state', text: tx('Couldn\'t load devices: {err}', { err: err.message }) }));
        return;
    }
    if (!nodes.length) {
        devicesLastDetails = [];
        list.replaceChildren(el('p', { class: 'muted small machine-empty', text: tx('No remote machines yet -- “Add a machine” adds one.') }));
        if (typeof updateComputeCount === 'function') updateComputeCount();
        return;
    }
    const details = await Promise.all(nodes.map(n => api(`/api/cluster/nodes/${encodeURIComponent(n.id)}`)
        .catch(err => ({ id: n.id, url: n.url, online: false, error: err.message }))));
    devicesLastDetails = details.map((d, i) => ({ name: nodes[i].name, ...d }));
    if (typeof updateComputeCount === 'function') updateComputeCount();
    if (typeof renderComputeModels === 'function') renderComputeModels();
    // Refresh card by card. A card the user is working in -- an open/focused
    // dropdown (found on the owner's Mac: rebuilding every 5 s closed the long model
    // list while scrolling it) or a field with unsent text -- is left alone until
    // they're done; the other cards (install progress etc.) keep updating.
    const active = document.activeElement;
    const old = new Map([...list.querySelectorAll('[data-node]')].map(c => [c.dataset.node, c]));
    const inUse = (card) => card && card.contains(active) && active !== document.body;
    const cards = details.map(d => {
        const prev = old.get(d.id);
        if (inUse(prev)) return prev;           // dropdown open / typing: wait
        const card = renderDeviceCard(d);
        card.dataset.node = d.id;
        // unsent edits without focus survive the rebuild
        for (const f of prev ? prev.querySelectorAll('[data-key]') : []) {
            const changed = f.tagName === 'SELECT' ? f.value !== f.dataset.initial : f.value !== f.defaultValue;
            const twin = changed && card.querySelector(`[data-key="${CSS.escape(f.dataset.key)}"]`);
            if (twin) twin.value = f.value;
        }
        return card;
    });
    list.replaceChildren(...cards);
}

const DEVICE_ICONS = {
    laptop: 'M4 5h16v11H4zM2 19h20', desktop: 'M4 4h16v11H4zM9 20h6M12 15v5',
    phone: 'M8 2h8a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zM11 18h2',
};

function deviceIcon(d) {
    const p = ((d.facts && d.facts.platform) || d.platform || '').toLowerCase();
    if (p.includes('android')) return DEVICE_ICONS.phone;
    if (p.includes('darwin') || p.includes('mac')) return DEVICE_ICONS.laptop;
    return DEVICE_ICONS.desktop;
}

function renderDeviceCard(d) {
    const name = d.name || d.id;
    const menu = el('button', { class: 'btn-icon-sm', title: tx('Remove device'), 'aria-label': tx('Remove {name}', { name }),
                                onclick: () => removeDevice(d.id) }, svgIcon('M5 7h14M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3', 16, 2));
    if (!d.online) {
        return machineCard({ id: d.id, name, where: 'remote', icon: deviceIcon(d), sub: d.url, online: false,
                             statusText: tx('unreachable'), accels: [], menu,
                             extra: [el('div', { class: 'service-hint', text: d.error || tx('The agent doesn\'t answer.') })] });
    }
    const f = d.facts || {};
    const sizes = {};
    for (const file of (d.downloads && d.downloads.files) || []) sizes[file.name] = Math.round(file.size_bytes / 1048576);
    const accels = (d.accelerators || []).map(a => {
        const rows = (d.instances || []).filter(i => i.gpu === undefined || i.gpu === null || i.gpu === a.index)
            .map((i, n) => ({
                label: i.model, sub: `${i.mode && i.mode !== 'chat' ? i.mode : tx('language model')} · ${tx('context {n}', { n: i.ctx_size })}`,
                mb: sizes[i.model] || 0, color: CONSUMER_COLORS[n % CONSUMER_COLORS.length],
                state: i.state === 'ready' ? tx('ready') : i.state === 'loading' ? tx('loading') : i.state, tone: i.state === 'loading' ? 'tone-busy' : '',
            }));
        const used = rows.reduce((t, r) => t + r.mb, 0);
        return el('div', { class: 'accel-wrap' },
            renderAccel({ name: a.name, kind: a.kind.toUpperCase(), totalMb: a.budget_mb, rows,
                          freeText: tx('{free} GB free · budget {budget} of {total} GB', { free: fmtGb(Math.max(0, a.budget_mb - used)), budget: fmtGb(a.budget_mb), total: fmtGb(a.total_mb) }) }),
            renderBudget(d, a));
    });
    const rt = d.runtimes || {};
    const inst = rt.install || {};
    let rtText = rt.active ? `${(d.runtime && d.runtime.llama_server || rt.active).split('/').slice(-2, -1)[0] || rt.active}` : tx('not installed yet');
    const extra = [];
    if (inst.state === 'installing') {
        const pct = inst.total_bytes ? Math.round(100 * inst.done_bytes / inst.total_bytes) : 0;
        rtText = tx('installing {tag} {variant}: {pct} %', { tag: inst.tag, variant: inst.variant, pct });
    } else if (inst.state === 'failed') {
        extra.push(el('div', { class: 'service-hint service-error', text: tx('llama.cpp installation failed: {err}', { err: inst.error }) }));
    }
    const files = (d.downloads && d.downloads.files) || [];
    const diskMb = files.reduce((t, x) => t + x.size_bytes / 1048576, 0);
    const manage = el('details', { class: 'r-details r-details-inner' }, el('summary', { text: tx('Manage: llama.cpp, models, files') }),
        renderRuntimeControls(d), renderModelControls(d), renderDownloads(d));
    extra.push(manage);
    return machineCard({
        id: d.id, name, where: 'remote', icon: deviceIcon(d),
        sub: `${f.platform || d.platform}/${f.machine || d.machine}` + (f.cuda_driver ? ` · ${tx('CUDA driver {v}', { v: f.cuda_driver.join('.') })}` : '') +
             (f.vulkan ? ' · Vulkan' : '') + (f.rocm ? ' · ROCm' : '') + ` · Agent ${d.agent_version}`,
        online: true, statusText: 'online', accels, menu,
        facts: [['llama.cpp', rtText], [tx('Model files'), tx('{gb} GB · deleted only by hand', { gb: fmtGb(diskMb) })]],
        extra,
    });
}

function renderBudget(d, a) {
    const shared = a.kind === 'metal' || a.kind === 'cpu';
    const input = el('input', { type: 'number', min: 1, max: a.total_mb, step: 256, value: a.budget_mb,
                                class: 'device-budget', title: tx('MB the device may use for models'),
                                'data-key': `${d.id}:budget:${a.index}` });
    const save = el('button', {
        class: 'btn btn-secondary btn-sm', text: tx('Save'),
        onclick: async () => {
            const v = parseInt(input.value, 10);
            const body = {}; body[String(a.index)] = (v === a.budget_default_mb) ? null : v;
            try {
                await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/budgets`, 'PUT', body);
                addLog(tx('{id}: budget for {kind} {index} set to {mb}', { id: d.id, kind: a.kind, index: a.index, mb: fmtMb(v) }), 'system');
            } catch (err) {
                deviceHint(tx('Budget on {id} not set: {err}', { id: d.id, err: err.message }));
            }
            input.blur();
            fetchDevices();
        },
    });
    const hint = shared ? tx('Default {mb} = half the RAM', { mb: fmtMb(a.budget_default_mb) }) : tx('Default {mb} = all of the VRAM', { mb: fmtMb(a.budget_default_mb) });
    return el('div', { class: 'device-budget-row' },
        el('span', { class: 'small', text: 'Budget' }), input, el('span', { class: 'small', text: 'MB' }), save,
        el('span', { class: 'muted small', text: hint }));
}

function renderRuntimeControls(d) {
    const select = el('select', { class: 'service-exposure-select', 'data-key': `${d.id}:variant`, 'data-initial': 'auto' });
    for (const v of ['auto', 'cpu', 'vulkan', 'cuda-12.8', 'cuda-13.4', 'rocm-10.0', 'metal']) {
        select.append(el('option', { value: v, text: v === 'auto' ? tx('Best for the device') : v }));
    }
    const busy = (d.runtimes && d.runtimes.install && d.runtimes.install.state === 'installing');
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: tx('Install llama.cpp'), disabled: busy,
        onclick: async () => {
            btn.disabled = true;
            try {
                const body = select.value === 'auto' ? {} : { variant: select.value };
                const r = await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/runtime`, 'POST', body);
                addLog(tx('{id}: installing llama.cpp {tag} {variant} ({mb})', { id: d.id, tag: r.tag, variant: r.variant, mb: fmtMb(r.size_mb) }), 'system');
            } catch (err) {
                deviceHint(tx('llama.cpp installation on {id} failed: {err}', { id: d.id, err: err.message }));
            }
            fetchDevices();
        },
    });
    return el('div', { class: 'service-exposure-row' }, el('span', { text: 'llama.cpp' }), select, btn);
}

function renderModelControls(d) {
    const local = el('select', { class: 'service-exposure-select', 'data-key': `${d.id}:local`, 'data-initial': '' },
        el('option', { value: '', text: tx('Local model…') }));
    (devicesLocalModels || []).forEach(m => local.append(el('option', { value: m, text: m })));
    const repo = el('input', { type: 'text', placeholder: tx('or HF repo, e.g. unsloth/Qwen3-0.6B-GGUF'), class: 'device-input',
                               'data-key': `${d.id}:repo` });
    const file = el('input', { type: 'text', placeholder: tx('File in the repo, e.g. Qwen3-0.6B-Q4_K_M.gguf'), class: 'device-input',
                               'data-key': `${d.id}:file` });
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: tx('Send model'),
        onclick: async () => {
            const body = local.value ? { model: local.value } : { hf_repo: repo.value.trim(), hf_file: file.value.trim() };
            btn.disabled = true;
            try {
                const r = await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/models`, 'POST', body);
                addLog(tx('{id}: downloading {name} ({mb}) from {n} source(s)', { id: d.id, name: r.name, mb: fmtMb(r.size >> 20), n: r.sources.length }), 'system');
            } catch (err) {
                deviceHint(tx('Model not sent to {id}: {err}', { id: d.id, err: err.message }));
            }
            btn.disabled = false;
        },
    });
    return el('div', { class: 'device-models' }, el('span', { text: tx('Model') }), local, repo, file, btn);
}

function renderDownloads(d) {
    const dl = d.downloads || {};
    const rows = [];
    for (const [name, st] of Object.entries(dl.downloads || {})) {
        if (st.state === 'downloading') {
            const pct = st.total_bytes ? Math.round(100 * st.done_bytes / st.total_bytes) : 0;
            rows.push(el('div', { class: 'service-instance', text: tx('Downloading {name}: {pct} %', { name, pct }) }));
        } else if (st.state === 'failed') {
            rows.push(el('div', { class: 'service-hint service-error', text: `${name}: ${st.error}` }));
        }
    }
    for (const f of dl.files || []) {
        rows.push(el('div', { class: 'service-instance' },
            el('span', { text: `${f.name} · ${fmtMb(Math.round(f.size_bytes / 1048576))}` }),
            el('button', {
                class: 'btn btn-danger-outline btn-sm', text: tx('Delete file'),
                onclick: () => deleteDownload(d.id, f.name),
            })));
    }
    return rows.length ? el('div', { class: 'device-downloads' }, ...rows) : null;
}

async function deleteDownload(id, name) {
    if (!window.confirm(tx('Delete {name} on {id}? To use it there again it would have to be downloaded again.', { name, id }))) return;
    try {
        await api(`/api/cluster/nodes/${encodeURIComponent(id)}/downloads/${encodeURIComponent(name)}`, 'DELETE');
        addLog(tx('{id}: {name} deleted', { id, name }), 'system');
    } catch (err) {
        deviceHint(tx('{name} on {id} not deleted: {err}', { name, id, err: err.message }));
    }
    fetchDevices();
}

async function removeDevice(id) {
    if (!window.confirm(tx('Remove device {id}? Its agent keeps running until you stop it there; it just won\'t be used any more.', { id }))) return;
    try {
        await api(`/api/cluster/nodes/${encodeURIComponent(id)}`, 'DELETE');
        addLog(tx('Device {id} removed', { id }), 'system');
    } catch (err) {
        deviceHint(tx('{id} not removed: {err}', { id, err: err.message }));
    }
    fetchDevices();
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('devices-add-btn')?.addEventListener('click', addDevice);
    api('/api/models').then(d => { devicesLocalModels = (d.models || []).map(m => m.name).sort(); }).catch(() => {});
});
