// Remote machines on "Rechner & Modelle": other machines (Macs, PCs, Pis, phones)
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
            class: 'btn btn-secondary btn-sm', text: 'Befehl kopieren',
            onclick: () => navigator.clipboard.writeText(e.command).then(() => { copy.textContent = 'Kopiert'; }),
        });
        box.replaceChildren(el('section', { class: 'r-card r-card-accent' },
            el('h2', { text: 'Rechner hinzufügen' }),
            el('p', { class: 'service-desc', text:
                `Auf dem neuen Gerät ausführen (Python 3.9+ und Tailscale nötig; das Gerät muss zu deinem ` +
                `Tailscale-Konto gehören). Der Code gilt einmal und läuft in ${mins} Minuten ab. Danach mit dem Gerät ` +
                `starten lassen: "python3 agent.py --install-service" (macOS, Linux, Android/Termux). Macs und andere ` +
                `Geräte mit gemeinsamem Speicher nutzen höchstens den halben Arbeitsspeicher, solange du das Budget nicht änderst.` }),
            cmd, copy,
        ));
    } catch (err) {
        deviceHint(`Kein Einmal-Code möglich: ${err.message}`);
    }
}

function renderPlaceForm() {
    const box = document.getElementById('devices-place');
    if (box.childElementCount) return;  // built once; keeps what was typed
    const local = el('select', { class: 'service-exposure-select' }, el('option', { value: '', text: 'Lokales Modell…' }));
    (devicesLocalModels || []).forEach(m => local.append(el('option', { value: m, text: m })));
    const repo = el('input', { type: 'text', placeholder: 'oder HF-Repo', class: 'device-input' });
    const file = el('input', { type: 'text', placeholder: 'Datei im Repo (.gguf)', class: 'device-input' });
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: 'An das passendste Gerät senden',
        onclick: async () => {
            const body = local.value ? { model: local.value } : { hf_repo: repo.value.trim(), hf_file: file.value.trim() };
            btn.disabled = true;
            try {
                const r = await api('/api/cluster/models', 'POST', body);
                addLog(r.state === 'present' ? `${r.name} liegt schon auf ${r.node}`
                                             : `Kopiere ${r.name} nach ${r.node} (lehnt es ab, kommt das nächste Gerät dran)`, 'system');
                local.value = ''; repo.value = ''; file.value = '';
            } catch (err) {
                deviceHint(`Senden fehlgeschlagen: ${err.message}`);
            }
            btn.disabled = false;
            fetchDevices();
        },
    });
    box.replaceChildren(el('div', { class: 'device-models', title:
        'Der Koordinator wählt ein Gerät mit Platz; lehnt eines den Download ab (Kontingent, Platte), kommt das nächste dran.' },
        el('span', { text: 'Modell auf ein Gerät kopieren' }), local, repo, file, btn));
}

async function renderCopies() {
    const box = document.getElementById('devices-copies');
    let data;
    try { data = await api('/api/cluster/copies'); } catch { return; }
    const rows = [];
    for (const [name, c] of Object.entries(data.copies || {})) {
        const text = c.state === 'copying' ? `Kopiere ${name} nach ${c.node}…`
            : c.state === 'done' ? `${name} liegt jetzt auf ${c.node}`
            : `${name}: kein Gerät konnte es aufnehmen`;
        rows.push(el('div', { class: `service-hint${c.state === 'failed' ? ' service-error' : ''}`, text }));
        for (const e of c.errors || []) rows.push(el('div', { class: 'service-dir', text: `übersprungen: ${e}` }));
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
        list.replaceChildren(el('div', { class: 'empty-state', text: `Geräte nicht ladbar: ${err.message}` }));
        return;
    }
    if (!nodes.length) {
        devicesLastDetails = [];
        list.replaceChildren(el('p', { class: 'muted small machine-empty', text: 'Noch keine entfernten Rechner -- mit „Rechner hinzufügen“ kommt einer dazu.' }));
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
    const menu = el('button', { class: 'btn-icon-sm', title: 'Gerät entfernen', 'aria-label': `${name} entfernen`,
                                onclick: () => removeDevice(d.id) }, svgIcon('M5 7h14M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3', 16, 2));
    if (!d.online) {
        return machineCard({ id: d.id, name, where: 'remote', icon: deviceIcon(d), sub: d.url, online: false,
                             statusText: 'nicht erreichbar', accels: [], menu,
                             extra: [el('div', { class: 'service-hint', text: d.error || 'Der Agent antwortet nicht.' })] });
    }
    const f = d.facts || {};
    const sizes = {};
    for (const file of (d.downloads && d.downloads.files) || []) sizes[file.name] = Math.round(file.size_bytes / 1048576);
    const accels = (d.accelerators || []).map(a => {
        const rows = (d.instances || []).filter(i => i.gpu === undefined || i.gpu === null || i.gpu === a.index)
            .map((i, n) => ({
                label: i.model, sub: `${i.mode && i.mode !== 'chat' ? i.mode : 'Sprachmodell'} · Kontext ${i.ctx_size}`,
                mb: sizes[i.model] || 0, color: CONSUMER_COLORS[n % CONSUMER_COLORS.length],
                state: i.state === 'ready' ? 'bereit' : i.state === 'loading' ? 'lädt' : i.state, tone: i.state === 'loading' ? 'tone-busy' : '',
            }));
        const used = rows.reduce((t, r) => t + r.mb, 0);
        return el('div', { class: 'accel-wrap' },
            renderAccel({ name: a.name, kind: a.kind.toUpperCase(), totalMb: a.budget_mb, rows,
                          freeText: `${fmtGb(Math.max(0, a.budget_mb - used))} GB frei · Budget ${fmtGb(a.budget_mb)} von ${fmtGb(a.total_mb)} GB` }),
            renderBudget(d, a));
    });
    const rt = d.runtimes || {};
    const inst = rt.install || {};
    let rtText = rt.active ? `${(d.runtime && d.runtime.llama_server || rt.active).split('/').slice(-2, -1)[0] || rt.active}` : 'noch nicht installiert';
    const extra = [];
    if (inst.state === 'installing') {
        const pct = inst.total_bytes ? Math.round(100 * inst.done_bytes / inst.total_bytes) : 0;
        rtText = `installiere ${inst.tag} ${inst.variant}: ${pct} %`;
    } else if (inst.state === 'failed') {
        extra.push(el('div', { class: 'service-hint service-error', text: `llama.cpp-Installation fehlgeschlagen: ${inst.error}` }));
    }
    const files = (d.downloads && d.downloads.files) || [];
    const diskMb = files.reduce((t, x) => t + x.size_bytes / 1048576, 0);
    const manage = el('details', { class: 'r-details r-details-inner' }, el('summary', { text: 'Verwalten: llama.cpp, Modelle, Dateien' }),
        renderRuntimeControls(d), renderModelControls(d), renderDownloads(d));
    extra.push(manage);
    return machineCard({
        id: d.id, name, where: 'remote', icon: deviceIcon(d),
        sub: `${f.platform || d.platform}/${f.machine || d.machine}` + (f.cuda_driver ? ` · CUDA-Treiber ${f.cuda_driver.join('.')}` : '') +
             (f.vulkan ? ' · Vulkan' : '') + (f.rocm ? ' · ROCm' : '') + ` · Agent ${d.agent_version}`,
        online: true, statusText: 'online', accels, menu,
        facts: [['llama.cpp', rtText], ['Modelldateien', `${fmtGb(diskMb)} GB · löschen nur von Hand`]],
        extra,
    });
}

function renderBudget(d, a) {
    const shared = a.kind === 'metal' || a.kind === 'cpu';
    const input = el('input', { type: 'number', min: 1, max: a.total_mb, step: 256, value: a.budget_mb,
                                class: 'device-budget', title: 'MB, die das Gerät für Modelle nutzen darf',
                                'data-key': `${d.id}:budget:${a.index}` });
    const save = el('button', {
        class: 'btn btn-secondary btn-sm', text: 'Speichern',
        onclick: async () => {
            const v = parseInt(input.value, 10);
            const body = {}; body[String(a.index)] = (v === a.budget_default_mb) ? null : v;
            try {
                await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/budgets`, 'PUT', body);
                addLog(`${d.id}: Budget für ${a.kind} ${a.index} auf ${fmtMb(v)} gesetzt`, 'system');
            } catch (err) {
                deviceHint(`Budget auf ${d.id} nicht gesetzt: ${err.message}`);
            }
            input.blur();
            fetchDevices();
        },
    });
    const hint = shared ? `Vorgabe ${fmtMb(a.budget_default_mb)} = halber Arbeitsspeicher` : `Vorgabe ${fmtMb(a.budget_default_mb)} = ganzer VRAM`;
    return el('div', { class: 'device-budget-row' },
        el('span', { class: 'small', text: 'Budget' }), input, el('span', { class: 'small', text: 'MB' }), save,
        el('span', { class: 'muted small', text: hint }));
}

function renderRuntimeControls(d) {
    const select = el('select', { class: 'service-exposure-select', 'data-key': `${d.id}:variant`, 'data-initial': 'auto' });
    for (const v of ['auto', 'cpu', 'vulkan', 'cuda-12.8', 'cuda-13.4', 'rocm-10.0', 'metal']) {
        select.append(el('option', { value: v, text: v === 'auto' ? 'Passend für das Gerät' : v }));
    }
    const busy = (d.runtimes && d.runtimes.install && d.runtimes.install.state === 'installing');
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: 'llama.cpp installieren', disabled: busy,
        onclick: async () => {
            btn.disabled = true;
            try {
                const body = select.value === 'auto' ? {} : { variant: select.value };
                const r = await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/runtime`, 'POST', body);
                addLog(`${d.id}: installiere llama.cpp ${r.tag} ${r.variant} (${fmtMb(r.size_mb)})`, 'system');
            } catch (err) {
                deviceHint(`llama.cpp-Installation auf ${d.id} fehlgeschlagen: ${err.message}`);
            }
            fetchDevices();
        },
    });
    return el('div', { class: 'service-exposure-row' }, el('span', { text: 'llama.cpp' }), select, btn);
}

function renderModelControls(d) {
    const local = el('select', { class: 'service-exposure-select', 'data-key': `${d.id}:local`, 'data-initial': '' },
        el('option', { value: '', text: 'Lokales Modell…' }));
    (devicesLocalModels || []).forEach(m => local.append(el('option', { value: m, text: m })));
    const repo = el('input', { type: 'text', placeholder: 'oder HF-Repo, z. B. unsloth/Qwen3-0.6B-GGUF', class: 'device-input',
                               'data-key': `${d.id}:repo` });
    const file = el('input', { type: 'text', placeholder: 'Datei im Repo, z. B. Qwen3-0.6B-Q4_K_M.gguf', class: 'device-input',
                               'data-key': `${d.id}:file` });
    const btn = el('button', {
        class: 'btn btn-secondary btn-sm', text: 'Modell senden',
        onclick: async () => {
            const body = local.value ? { model: local.value } : { hf_repo: repo.value.trim(), hf_file: file.value.trim() };
            btn.disabled = true;
            try {
                const r = await api(`/api/cluster/nodes/${encodeURIComponent(d.id)}/models`, 'POST', body);
                addLog(`${d.id}: lädt ${r.name} (${fmtMb(r.size >> 20)}) von ${r.sources.length} Quelle(n)`, 'system');
            } catch (err) {
                deviceHint(`Modell an ${d.id} nicht gesendet: ${err.message}`);
            }
            btn.disabled = false;
        },
    });
    return el('div', { class: 'device-models' }, el('span', { text: 'Modell' }), local, repo, file, btn);
}

function renderDownloads(d) {
    const dl = d.downloads || {};
    const rows = [];
    for (const [name, st] of Object.entries(dl.downloads || {})) {
        if (st.state === 'downloading') {
            const pct = st.total_bytes ? Math.round(100 * st.done_bytes / st.total_bytes) : 0;
            rows.push(el('div', { class: 'service-instance', text: `Lädt ${name}: ${pct} %` }));
        } else if (st.state === 'failed') {
            rows.push(el('div', { class: 'service-hint service-error', text: `${name}: ${st.error}` }));
        }
    }
    for (const f of dl.files || []) {
        rows.push(el('div', { class: 'service-instance' },
            el('span', { text: `${f.name} · ${fmtMb(Math.round(f.size_bytes / 1048576))}` }),
            el('button', {
                class: 'btn btn-danger-outline btn-sm', text: 'Datei löschen',
                onclick: () => deleteDownload(d.id, f.name),
            })));
    }
    return rows.length ? el('div', { class: 'device-downloads' }, ...rows) : null;
}

async function deleteDownload(id, name) {
    if (!window.confirm(`${name} auf ${id} löschen? Um es dort wieder zu nutzen, müsste es neu heruntergeladen werden.`)) return;
    try {
        await api(`/api/cluster/nodes/${encodeURIComponent(id)}/downloads/${encodeURIComponent(name)}`, 'DELETE');
        addLog(`${id}: ${name} gelöscht`, 'system');
    } catch (err) {
        deviceHint(`${name} auf ${id} nicht gelöscht: ${err.message}`);
    }
    fetchDevices();
}

async function removeDevice(id) {
    if (!window.confirm(`Gerät ${id} entfernen? Sein Agent läuft weiter, bis du ihn dort stoppst; er wird nur nicht mehr genutzt.`)) return;
    try {
        await api(`/api/cluster/nodes/${encodeURIComponent(id)}`, 'DELETE');
        addLog(`Gerät ${id} entfernt`, 'system');
    } catch (err) {
        deviceHint(`${id} nicht entfernt: ${err.message}`);
    }
    fetchDevices();
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('devices-add-btn')?.addEventListener('click', addDevice);
    api('/api/models').then(d => { devicesLocalModels = (d.models || []).map(m => m.name).sort(); }).catch(() => {});
});
