// "Apps & Services": overview / start / stop / open of sibling apps (GET/POST /api/services),
// grouped by the category from each app's manifest. DOM is built with textContent
// only -- names and descriptions come from files in other repos.

let servicesPollInterval = null;
const servicesOpenLogs = new Set();
const servicesOpenDetails = new Set();
let servicesFilter = 'all';
let servicesLast = null;
let servicesFacts = {};  // id -> [{label, value}] from each app's own status_path
let servicesDoor = null;  // GET /api/door: shares through the door (Nextcloud login), people seen
const servicesDoorDrafts = new Map();  // id -> {name, groups: Set, users: Set} while editing a share

function setServicesPolling(active) {
    if (servicesPollInterval) {
        clearInterval(servicesPollInterval);
        servicesPollInterval = null;
    }
    if (active) {
        fetchServices();
        servicesPollInterval = setInterval(fetchServices, 5000);
    }
}

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
        else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
}

function serviceUptime(startedAt) {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
    if (s < 60) return `${s} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
    return `${Math.floor(s / 86400)} d ${Math.floor((s % 86400) / 3600)} h`;
}

// A service bound to 127.0.0.1 is only reachable from this machine.
function serviceOpenUrl(open) {
    const here = window.location.hostname;
    const isLocal = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(here);
    const host = open.local_only ? 'localhost' : here;
    return { url: `http://${host}:${open.port}${open.path}`, unreachable: open.local_only && !isLocal };
}

// Where to open a running service from this browser: the tailnet URL when the page
// isn't on this machine and the app is exposed, else its own port.
function serviceBestUrl(s) {
    const inst = s.instances[0];
    if (!inst) return null;
    const { url, unreachable } = serviceOpenUrl(inst.open);
    if (unreachable && s.exposure && s.exposure.url) return { url: s.exposure.url, unreachable: false };
    return { url, unreachable };
}

// "Öffnen" for a running app. From another device a loopback-only app has no working
// address -- instead of a dead localhost link, offer to make it reachable for the
// owner via Tailscale (an explicit, confirmed choice; never automatic).
function serviceOpenControl(s, cls = 'btn btn-primary btn-sm') {
    if (s.ui === false) return null;  // an API without a page: nothing to open or share
    const best = serviceBestUrl(s);
    if (!best) return null;
    if (!best.unreachable) {
        return el('a', { class: cls, href: best.url, target: '_blank', rel: 'noopener', title: best.url, text: 'Öffnen' });
    }
    if (!servicesTailscale.available) {
        return el('button', { class: cls, disabled: true, text: 'Öffnen',
                              title: 'Nur auf dem Rechner selbst erreichbar (lauscht auf 127.0.0.1); Tailscale ist hier nicht verfügbar.' });
    }
    return el('button', {
        class: cls, text: 'Für mich freigeben',
        title: 'Lauscht nur auf 127.0.0.1 -- von diesem Gerät aus nur über Tailscale erreichbar',
        onclick: async (e) => {
            const who = servicesTailscale.owner ? `nur für ${servicesTailscale.owner}` : 'nur für dich';
            if (!window.confirm(`${s.name} läuft nur auf dem Rechner selbst. Über Tailscale ${who} erreichbar machen (nie öffentlich)?`)) return;
            e.target.disabled = true;
            try {
                await api(`/api/services/${encodeURIComponent(s.id)}/exposure`, 'POST', { level: 'owner' });
                addLog(`App ${s.id}: für ${servicesTailscale.owner || 'dich'} über Tailscale erreichbar`, 'system');
            } catch (err) {
                alertServiceError(err.message);
                addLog(`App ${s.id}: Freigabe fehlgeschlagen -- ${err.message}`, 'error');
            }
            fetchServices();
            if (typeof refreshOverview === 'function') refreshOverview();
        },
    });
}

const SERVICE_STATUS = {
    running: ['läuft', 'ok'], starting: ['startet…', 'busy'], stopped: ['gestoppt', 'off'], exited: ['abgestürzt', 'bad'],
};

async function serviceAction(id, action, body = null) {
    try {
        await api(`/api/services/${encodeURIComponent(id)}/${action}`, 'POST', body || {});
        addLog(`App ${id}: ${action === 'start' ? 'Start' : 'Stopp'} angefordert`, 'system');
    } catch (err) {
        addLog(`App ${id}: ${action} fehlgeschlagen -- ${err.message}`, 'error');
        alertServiceError(err.message);
    }
    fetchServices();
    if (typeof refreshOverview === 'function') refreshOverview();
}

function alertServiceError(msg) {
    const box = document.getElementById('services-errors');
    if (!box) return;
    box.replaceChildren(el('div', { class: 'service-hint service-error', text: msg }));
    setTimeout(() => { if (box.textContent === msg) box.replaceChildren(); }, 8000);
}

async function toggleServiceLog(id, pre) {
    if (servicesOpenLogs.has(id)) {
        servicesOpenLogs.delete(id);
        pre.hidden = true;
        return;
    }
    servicesOpenLogs.add(id);
    pre.hidden = false;
    await loadServiceLog(id, pre);
}

async function loadServiceLog(id, pre) {
    try {
        const data = await api(`/api/services/${encodeURIComponent(id)}/log?tail=200`);
        pre.textContent = data.log || '(leer)';
        pre.scrollTop = pre.scrollHeight;
    } catch (err) {
        pre.textContent = `Protokoll nicht lesbar: ${err.message}`;
    }
}

function statusPill(s) {
    const [text, tone] = SERVICE_STATUS[s.status] || [s.status, 'off'];
    return el('span', { class: `pill pill-${tone}` }, el('span', { class: 'pill-dot' }),
              el('span', { text: s.status === 'exited' ? `${text} (${s.exit_code})` : text }));
}

const IDENTITY_TEXT = {
    none: 'Personen: speichert keine Daten pro Person -- kann mit anderen geteilt werden.',
    single: 'Personen: speichert Daten nur für eine Person -- nur für dich, bis die App Daten pro Person trennt (docs/identity.md).',
    multi: 'Personen: trennt Daten pro Person -- kann mit anderen geteilt werden.',
};

const EXPOSURE_LABELS = {
    local: 'Nur dieser Rechner',
    tailnet: 'Ganzes Tailnet',
    owner: 'Nur ich (Tailscale)',
};
let servicesTailscale = { available: false, owner: null };

function renderServiceCard(s) {
    const running = s.instances.length > 0;
    const best = serviceBestUrl(s);
    const parts = [];
    if (s.has_screenshot) parts.push(appScreenshot(s, best && !best.unreachable ? best.url : null));

    parts.push(el('div', { class: 'app-head' }, appTile(s),
        el('div', { class: 'app-head-text' },
            el('h3', { text: s.name }),
            s.description ? el('p', { class: 'app-desc', text: s.description }) : null)));

    const ex = (s.exposure && s.exposure.level) || 'local';
    parts.push(el('div', { class: 'pill-row' }, statusPill(s), maturityPill(s),
        el('span', { class: 'pill', text: s.ui === false ? 'Dienst ohne eigene Seite' : (EXPOSURE_LABELS[ex] || ex) }),
        doorShare(s) ? el('span', { class: 'pill pill-accent', text: 'Mit Login geteilt', title: doorShare(s).url }) : null,
        el('span', { class: 'mono muted', text: `:${s.port}` })));
    const facts = servicesFacts[s.id] || [];
    if (facts.length) parts.push(el('dl', { class: 'facts' }, ...facts.flatMap(f => [el('dt', { text: f.label }), el('dd', { text: f.value })])));

    for (const inst of s.instances) {
        if (inst.exposed && inst.exposed.length) {
            parts.push(el('div', { class: 'service-hint service-error', text:
                `Lauscht auf ${inst.exposed.join(', ')} -- im LAN/Tailnet erreichbar, am Dashboard vorbei. Hier stoppen und von hier neu starten.` }));
        }
        if (inst.healthy === false) parts.push(el('div', { class: 'service-hint', text: `Kopie auf :${inst.ports.join(', :')} antwortet nicht gesund.` }));
    }

    const actions = [];
    if (running && best) {
        const open = serviceOpenControl(s);
        if (open) actions.push(open);
        if (s.instances.length === 1) {
            const inst = s.instances[0];
            actions.push(el('button', { class: 'btn btn-secondary btn-sm', text: 'Stoppen',
                onclick: () => { if (confirmStop(s, inst)) serviceAction(s.id, 'stop', { pid: inst.pid }); } }));
        }
    }
    const portBusy = s.instances.some(i => i.ports.includes(s.port));
    if (!portBusy) {
        actions.push(el('button', { class: 'btn btn-outline btn-sm', disabled: s.status === 'starting',
                                    text: s.status === 'starting' ? 'Startet…' : 'Starten',
                                    onclick: () => serviceAction(s.id, 'start') }));
    }
    const more = el('button', { class: 'btn-icon-sm', 'aria-expanded': String(servicesOpenDetails.has(s.id)),
                                title: 'Erreichbarkeit, Kopien, Protokoll', 'aria-label': `Mehr zu ${s.name}`, text: '⋯',
                                onclick: () => {
                                    if (servicesOpenDetails.has(s.id)) servicesOpenDetails.delete(s.id);
                                    else servicesOpenDetails.add(s.id);
                                    fetchServices();
                                } });
    parts.push(el('div', { class: 'app-actions' }, ...actions, renderServiceAutostart(s), more));
    if (s.instances.length > 1) parts.push(el('div', { class: 'service-hint', text: `${s.instances.length} Kopien laufen -- Details unter „⋯“.` }));
    if (servicesOpenDetails.has(s.id)) parts.push(renderServiceDetails(s));
    return el('article', { class: 'app-card', 'data-service': s.id }, ...parts);
}

function renderServiceDetails(s) {
    const rows = s.instances.map(inst => {
        const { url, unreachable } = serviceOpenUrl(inst.open);
        return el('div', { class: 'service-instance' },
            el('span', { class: 'mono', text: `:${inst.ports.join(', :')}` }),
            el('span', { text: `PID ${inst.pid}` }),
            el('span', { text: `seit ${serviceUptime(inst.started_at)}` }),
            inst.managed ? el('span', { class: 'pill', text: 'von hier gestartet' }) : null,
            el('span', { class: 'spacer' }),
            unreachable ? null : el('a', { class: 'btn btn-secondary btn-sm', href: url, target: '_blank', rel: 'noopener', text: 'Öffnen', title: url }),
            el('button', { class: 'btn btn-danger btn-sm', text: 'Stoppen',
                           onclick: () => { if (confirmStop(s, inst)) serviceAction(s.id, 'stop', { pid: inst.pid }); } }));
    });
    const pre = el('pre', { class: 'code-snippet service-log', hidden: !servicesOpenLogs.has(s.id) });
    if (servicesOpenLogs.has(s.id)) loadServiceLog(s.id, pre);
    return el('div', { class: 'app-details' },
        ...rows,
        s.ui === false ? el('div', { class: 'muted small', text: 'Ein Dienst für andere Apps, ohne eigene Seite -- bleibt auf diesem Rechner (keine Tailscale-Freigabe).' })
                       : renderServiceExposure(s),
        s.ui === false ? null : renderServiceDoor(s),
        el('div', { class: 'muted small', text: IDENTITY_TEXT[s.identity] || '' }),
        el('div', { class: 'service-dir', text: s.project_dir, title: `gefunden über ${s.source}` }),
        s.has_log ? el('button', { class: 'btn btn-secondary btn-sm', text: 'Protokoll', onclick: () => toggleServiceLog(s.id, pre) }) : null,
        pre);
}

function renderServiceAutostart(s) {
    const box = el('input', {
        type: 'checkbox',
        onchange: async (e) => {
            box.disabled = true;
            try {
                await api(`/api/services/${encodeURIComponent(s.id)}/autostart`, 'POST', { enabled: e.target.checked });
                addLog(`App ${s.id}: Autostart ${e.target.checked ? 'an' : 'aus'}`, 'system');
            } catch (err) {
                addLog(`App ${s.id}: Autostart-Änderung fehlgeschlagen -- ${err.message}`, 'error');
                alertServiceError(err.message);
            }
            fetchServices();
        },
    });
    box.checked = !!s.autostart;
    return el('label', { class: 'app-autostart', title: 'Startet mit dem ai-serve-Backend' }, box, el('span', { text: 'Autostart' }));
}

function renderServiceExposure(s) {
    const ex = s.exposure || { level: 'local' };
    const select = el('select', {
        class: 'service-exposure-select',
        title: 'Wer die App über Tailscale erreicht (nie das öffentliche Internet)',
        onchange: (e) => setServiceExposure(s, e.target.value, select, ex.level),
    });
    for (const [value, label] of Object.entries(EXPOSURE_LABELS)) {
        // docs/identity.md: an app that keeps one person's data is only for the owner
        const single = value === 'tailnet' && s.identity === 'single';
        const disabled = (value !== 'local' && !servicesTailscale.available) || single;
        const opt = el('option', { value, text: single ? `${label} (nur mit Daten pro Person)` : label, disabled });
        if (value === ex.level) opt.selected = true;
        select.append(opt);
    }
    const parts = [el('div', { class: 'service-exposure-row' }, el('span', { text: 'Erreichbar für' }), select)];
    if (ex.url) parts.push(el('a', { class: 'service-exposure-url', href: ex.url, target: '_blank', rel: 'noopener', text: ex.url }));
    if (ex.warning) parts.push(el('div', { class: 'service-hint', text: ex.warning }));
    if (ex.error) parts.push(el('div', { class: 'service-hint service-error', text: ex.error }));
    return el('div', { class: 'service-exposure' }, ...parts);
}

async function setServiceExposure(s, level, select, previous) {
    if (level !== 'local') {
        const who = level === 'owner'
            ? `nur die Geräte von ${servicesTailscale.owner}`
            : 'jedes Gerät im Tailnet (auch von anderen geteilte Geräte)';
        if (!window.confirm(`${s.name} erreichbar machen für ${who}?`)) {
            select.value = previous;
            return;
        }
    }
    select.disabled = true;
    try {
        await api(`/api/services/${encodeURIComponent(s.id)}/exposure`, 'POST', { level });
        addLog(`App ${s.id}: Erreichbarkeit ${level}`, 'system');
    } catch (err) {
        addLog(`App ${s.id}: Erreichbarkeit nicht geändert -- ${err.message}`, 'error');
        alertServiceError(err.message);
    }
    fetchServices();
}

// ---- sharing through the door (https://<name>.<domain>/, Nextcloud login) -------------------

function doorRerender() {
    // the list keeps a card the user is working in -- leave it so the editor really redraws
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    renderServices(servicesLast);
}

function doorShare(s) {
    return servicesDoor && servicesDoor.shares ? servicesDoor.shares[s.id] : null;
}

function doorPeopleLabel(sub) {
    if (servicesDoor && sub === servicesDoor.owner) return 'Ich';
    const p = (servicesDoor?.people || []).find(x => x.sub === sub);
    return p && p.name ? `${p.name} (${sub})` : sub;
}

function renderServiceDoor(s) {
    if (servicesDoor && servicesDoor.configured === false) return document.createDocumentFragment();   // optional extension, not set up
    const box = el('div', { class: 'service-exposure door-share' },
        el('div', { class: 'service-exposure-row' }, el('strong', { text: 'Von außen, mit Nextcloud-Login' })));
    if (!servicesDoor) return box;
    if (!servicesDoor.available) {
        box.append(el('div', { class: 'muted small', text: 'Der Türsteher (edge-door) läuft gerade nicht.' }));
    }
    const share = doorShare(s);
    const draft = servicesDoorDrafts.get(s.id);
    if (draft) {
        box.append(renderDoorEditor(s, draft));
        return box;
    }
    if (share) {
        const who = [...share.groups.map(g => `Gruppe ${g}`), ...share.users.map(doorPeopleLabel)].join(', ');
        box.append(el('a', { class: 'service-exposure-url', href: share.url, target: '_blank', rel: 'noopener', text: share.url }),
                   el('div', { class: 'small', text: `Für: ${who}` }));
        if (share.error) box.append(el('div', { class: 'service-hint service-error', text: `Nicht aktiv: ${share.error}` }));
        box.append(el('div', { class: 'app-actions' },
            el('button', { class: 'btn btn-secondary btn-sm', text: 'Ändern', onclick: () => {
                servicesDoorDrafts.set(s.id, { name: share.name, groups: new Set(share.groups), users: new Set(share.users) });
                doorRerender();
            } }),
            el('button', { class: 'btn btn-danger btn-sm', text: 'Nicht mehr teilen', onclick: () => deleteDoorShare(s) })));
    } else {
        box.append(el('div', { class: 'muted small', text: s.identity === 'single'
            ? 'Diese App trennt Daten nicht pro Person – sie lässt sich nur für dich selbst von unterwegs freigeben.'
            : 'Bekannte melden sich mit ihrem Nextcloud-Konto an und haben hier ihre eigenen Daten.' }),
            el('div', { class: 'app-actions' }, el('button', { class: 'btn btn-outline btn-sm', text: 'Freigeben …',
                disabled: s.identity === 'single' && !servicesDoor.owner,
                onclick: () => {
                    const single = s.identity === 'single';
                    servicesDoorDrafts.set(s.id, { name: s.id.replace(/[^a-z0-9-]/g, '-').replace(/^-+|-+$/g, '').slice(0, 31),
                                                   groups: new Set(single ? [] : ['apps']),
                                                   users: new Set(single ? [servicesDoor.owner] : []) });
                    doorRerender();
                } })));
    }
    if (servicesDoor.door_error) box.append(el('div', { class: 'service-hint service-error', text: servicesDoor.door_error }));
    return box;
}

function renderDoorEditor(s, draft) {
    const single = s.identity === 'single';
    const name = el('input', { type: 'text', class: 'door-name', value: draft.name, maxlength: '31',
                               'aria-label': 'Adresse', oninput: (e) => { draft.name = e.target.value.trim().toLowerCase(); } });
    const parts = [el('label', { class: 'door-address' }, el('span', { class: 'mono', text: 'https://' }), name,
                      el('span', { class: 'mono', text: `.${servicesDoor.domain}/` }))];
    const check = (set, value, label, disabled = false) => {
        const c = el('input', { type: 'checkbox', disabled, onchange: (e) => { e.target.checked ? set.add(value) : set.delete(value); } });
        c.checked = set.has(value);
        return el('label', { class: 'door-check' }, c, el('span', { text: label }));
    };
    if (single) {
        parts.push(el('div', { class: 'small', text: `Nur für dich (${servicesDoor.owner}) – die App trennt Daten nicht pro Person.` }));
    } else {
        const groups = new Set(['apps', ...draft.groups]);
        for (const p of servicesDoor.people || []) for (const g of p.groups || []) groups.add(g);
        const people = new Set([servicesDoor.owner, ...draft.users, ...(servicesDoor.people || []).map(p => p.sub)].filter(Boolean));
        const addField = (placeholder, set) => {
            const input = el('input', { type: 'text', placeholder, class: 'door-add' });
            return el('span', { class: 'door-add-row' }, input, el('button', { class: 'btn btn-secondary btn-sm', text: '+', type: 'button',
                onclick: () => { const v = input.value.trim(); if (v) { set.add(v); doorRerender(); } } }));
        };
        parts.push(el('div', { class: 'door-col' }, el('div', { class: 'muted small', text: 'Nextcloud-Gruppen' }),
                      ...[...groups].sort().map(g => check(draft.groups, g, g)), addField('weitere Gruppe', draft.groups)),
                   el('div', { class: 'door-col' }, el('div', { class: 'muted small', text: 'Personen (Nextcloud-Konto)' }),
                      ...[...people].sort().map(u => check(draft.users, u, doorPeopleLabel(u))), addField('Kontoname', draft.users),
                      el('div', { class: 'muted small', text: 'Hier erscheint, wer sich schon einmal angemeldet hat; andere per Kontoname hinzufügen.' })));
    }
    parts.push(el('div', { class: 'app-actions' },
        el('button', { class: 'btn btn-primary btn-sm', text: 'Speichern', onclick: () => saveDoorShare(s, draft) }),
        el('button', { class: 'btn btn-secondary btn-sm', text: 'Abbrechen', onclick: () => { servicesDoorDrafts.delete(s.id); doorRerender(); } })));
    return el('div', { class: 'door-editor' }, ...parts);
}

async function saveDoorShare(s, draft) {
    const body = { name: draft.name, groups: [...draft.groups], users: [...draft.users] };
    const who = [...body.groups.map(g => `alle in Gruppe „${g}“`), ...body.users.map(doorPeopleLabel)].join(', ');
    if (!window.confirm(`${s.name} unter https://${draft.name}.${servicesDoor.domain}/ freigeben für: ${who || 'niemand'}?`)) return;
    try {
        const r = await api(`/api/door/shares/${encodeURIComponent(s.id)}`, 'PUT', body);
        servicesDoorDrafts.delete(s.id);
        addLog(`App ${s.id}: von außen freigegeben (${r.share.url})`, 'system');
        if (r.door_error) alertServiceError(r.door_error);
        if (r.restart_needed && s.instances.length) {
            alertServiceError(`${s.name} bitte einmal stoppen und neu starten, damit sie Anmeldungen unter der neuen Adresse erkennt.`);
        }
    } catch (err) {
        addLog(`App ${s.id}: Freigabe nicht gespeichert -- ${err.message}`, 'error');
        alertServiceError(err.message);
    }
    fetchServices();
}

async function deleteDoorShare(s) {
    if (!window.confirm(`${s.name} nicht mehr von außen freigeben? Wer gerade angemeldet ist, kommt danach nicht mehr hinein.`)) return;
    try {
        await api(`/api/door/shares/${encodeURIComponent(s.id)}`, 'DELETE');
        addLog(`App ${s.id}: Freigabe von außen beendet`, 'system');
    } catch (err) {
        alertServiceError(err.message);
    }
    fetchServices();
}

function confirmStop(s, inst) {
    // Instances not started from here may belong to someone's terminal session.
    if (inst.managed) return true;
    return window.confirm(`${s.name} (PID ${inst.pid}, Port ${inst.ports.join(', ')}) wurde nicht von diesem Dashboard gestartet. Trotzdem stoppen?`);
}

function renderServices(data) {
    const list = document.getElementById('services-list');
    const filters = document.getElementById('services-filters');
    const q = (document.getElementById('services-search')?.value || '').trim().toLowerCase();
    const services = data.services.filter(s => !q || `${s.name} ${s.description}`.toLowerCase().includes(q));
    const groups = Object.keys(APP_CATEGORIES)
        .map(key => ({ key, ...APP_CATEGORIES[key], items: services.filter(s => (s.category || 'tools') === key) }))
        .filter(g => g.items.length);
    if (servicesFilter !== 'all' && !groups.some(g => g.key === servicesFilter)) servicesFilter = 'all';

    const chip = (key, label, count) => el('button', {
        class: `chip${key === servicesFilter ? ' chip-active' : ''}`, 'aria-pressed': String(key === servicesFilter),
        onclick: () => { servicesFilter = key; renderServices(servicesLast); } },
        el('span', { text: label }), el('span', { class: 'chip-count', text: String(count) }));
    filters.replaceChildren(chip('all', 'Alle', services.length), ...groups.map(g => chip(g.key, g.label, g.items.length)));

    if (!data.services.length) {
        list.replaceChildren(el('div', { class: 'empty-state', text: 'Noch keine Apps -- ein Projekt mit llama-serve.service.json in einem Service-Ordner anlegen.' }));
        return;
    }
    // keep a card the user is working in (open dropdown, focused field)
    const active = document.activeElement;
    const old = new Map([...list.querySelectorAll('.app-card[data-service]')].map(c => [c.dataset.service, c]));
    const shown = groups.filter(g => servicesFilter === 'all' || g.key === servicesFilter);
    list.replaceChildren(...shown.map(g => el('section', { class: 'app-group' },
        el('div', { class: 'app-group-head' }, el('h2', { text: g.label }), g.hint ? el('span', { class: 'muted', text: g.hint }) : null),
        el('div', { class: 'app-grid' }, ...g.items.map(s => {
            const prev = old.get(s.id);
            const busy = prev && ((prev.contains(active) && active !== document.body) || prev.querySelector('.door-editor input:focus'));
            return busy ? prev : renderServiceCard(s);
        })))));
    if (!shown.length) list.replaceChildren(el('div', { class: 'empty-state', text: 'Keine App passt zur Suche.' }));
}

async function fetchServices() {
    const list = document.getElementById('services-list');
    const errors = document.getElementById('services-errors');
    try {
        const data = await api('/api/services');
        servicesLast = data;
        const withFacts = data.services.filter(s => s.has_status && s.instances.length);
        const facts = {};
        await Promise.all(withFacts.map(s => api(`/api/services/${encodeURIComponent(s.id)}/status`)
            .then(r => { facts[s.id] = r.facts; }).catch(() => {})));
        servicesFacts = facts;
        servicesDoor = await api('/api/door').catch(() => null);
        servicesTailscale = data.tailscale || servicesTailscale;
        const roots = document.getElementById('services-roots');
        if (roots && data.roots) roots.textContent = data.roots.join(', ');
        updateServicesCount(data);
        renderServices(data);
        if (data.errors.length) {
            errors.replaceChildren(...data.errors.map(e =>
                el('div', { class: 'service-hint', text: `${e.dir}: ${e.error}` })));
        } else if (!errors.querySelector('.service-error')) {
            errors.replaceChildren();
        }
    } catch (err) {
        list.replaceChildren(el('div', { class: 'empty-state', text: `Apps nicht ladbar: ${err.message}` }));
    }
}

function updateServicesCount(data) {
    const c = document.getElementById('side-count-services');
    if (c) c.textContent = `${data.services.filter(s => s.instances.length).length}/${data.services.length}`;
}

document.getElementById('services-refresh-btn').addEventListener('click', fetchServices);
document.getElementById('services-search').addEventListener('input', () => { if (servicesLast) renderServices(servicesLast); });
