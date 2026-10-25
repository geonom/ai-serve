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

// "Open" for a running app. From another device a loopback-only app has no working
// address -- instead of a dead localhost link, offer to make it reachable for the
// owner via Tailscale (an explicit, confirmed choice; never automatic).
function serviceOpenControl(s, cls = 'btn btn-primary btn-sm') {
    if (s.ui === false) return null;  // an API without a page: nothing to open or share
    const best = serviceBestUrl(s);
    if (!best) return null;
    if (!best.unreachable) {
        return el('a', { class: cls, href: best.url, target: '_blank', rel: 'noopener', title: best.url, text: tx('Open') });
    }
    if (!servicesTailscale.available) {
        return el('button', { class: cls, disabled: true, text: tx('Open'),
                              title: tx('Only reachable on the machine itself (listens on 127.0.0.1); Tailscale isn\'t available here.') });
    }
    return el('button', {
        class: cls, text: tx('Share with me'),
        title: tx('Listens on 127.0.0.1 only -- from this device only reachable through Tailscale'),
        onclick: async (e) => {
            const who = servicesTailscale.owner ? tx('only for {owner}', { owner: servicesTailscale.owner }) : tx('only for you');
            if (!window.confirm(tx('{name} only runs on the machine itself. Make it reachable through Tailscale {who} (never public)?', { name: s.name, who }))) return;
            e.target.disabled = true;
            try {
                await api(`/api/services/${encodeURIComponent(s.id)}/exposure`, 'POST', { level: 'owner' });
                addLog(tx('App {id}: reachable through Tailscale for {who}', { id: s.id, who: servicesTailscale.owner || tx('you') }), 'system');
            } catch (err) {
                alertServiceError(err.message);
                addLog(tx('App {id}: sharing failed -- {err}', { id: s.id, err: err.message }), 'error');
            }
            fetchServices();
            if (typeof refreshOverview === 'function') refreshOverview();
        },
    });
}

const SERVICE_STATUS = {
    running: [tx('running'), 'ok'], starting: [tx('starting…'), 'busy'], stopped: [tx('stopped'), 'off'], exited: [tx('crashed'), 'bad'],
};

async function serviceAction(id, action, body = null) {
    try {
        await api(`/api/services/${encodeURIComponent(id)}/${action}`, 'POST', body || {});
        addLog(action === 'start' ? tx('App {id}: start requested', { id }) : tx('App {id}: stop requested', { id }), 'system');
    } catch (err) {
        addLog(tx('App {id}: {action} failed -- {err}', { id, action, err: err.message }), 'error');
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
        pre.textContent = data.log || tx('(empty)');
        pre.scrollTop = pre.scrollHeight;
    } catch (err) {
        pre.textContent = tx('Log not readable: {err}', { err: err.message });
    }
}

function statusPill(s) {
    const [text, tone] = SERVICE_STATUS[s.status] || [s.status, 'off'];
    return el('span', { class: `pill pill-${tone}` }, el('span', { class: 'pill-dot' }),
              el('span', { text: s.status === 'exited' ? `${text} (${s.exit_code})` : text }));
}

const IDENTITY_TEXT = {
    none: tx('People: keeps no data per person -- can be shared with others.'),
    single: tx('People: keeps data for one person only -- only for you until the app separates data per person (docs/identity.md).'),
    multi: tx('People: keeps data separate per person -- can be shared with others.'),
};

const EXPOSURE_LABELS = {
    local: tx('This machine only'),
    tailnet: tx('Whole tailnet'),
    owner: tx('Only me (Tailscale)'),
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
        el('span', { class: 'pill', text: s.ui === false ? tx('Service without its own page') : (EXPOSURE_LABELS[ex] || ex) }),
        doorShare(s) ? el('span', { class: 'pill pill-accent', text: tx('Shared with login'), title: doorShare(s).url }) : null,
        el('span', { class: 'mono muted', text: `:${s.port}` })));
    const facts = servicesFacts[s.id] || [];
    if (facts.length) parts.push(el('dl', { class: 'facts' }, ...facts.flatMap(f => [el('dt', { text: tx(f.label) }), el('dd', { text: tx(f.value) })])));

    for (const inst of s.instances) {
        if (inst.exposed && inst.exposed.length) {
            parts.push(el('div', { class: 'service-hint service-error', text:
                tx('Listens on {addr} -- reachable in the LAN/tailnet, bypassing the dashboard. Stop it here and start it again from here.', { addr: inst.exposed.join(', ') }) }));
        }
        if (inst.dev_server && doorShare(s)) {
            parts.push(el('div', { class: 'service-hint service-error', text:
                tx('Runs the Vite development server while shared through the door -- it can hand out any file of the project (people\'s data too). Start it in production mode (build + its own server, see docs/service-manifest.md).') }));
        }
        if (inst.healthy === false) parts.push(el('div', { class: 'service-hint', text: tx('The copy on :{ports} doesn\'t answer healthy.', { ports: inst.ports.join(', :') }) }));
    }

    const actions = [];
    if (running && best) {
        const open = serviceOpenControl(s);
        if (open) actions.push(open);
        if (s.instances.length === 1) {
            const inst = s.instances[0];
            actions.push(el('button', { class: 'btn btn-secondary btn-sm', text: tx('Stop'),
                onclick: () => { if (confirmStop(s, inst)) serviceAction(s.id, 'stop', { pid: inst.pid }); } }));
        }
    }
    const portBusy = s.instances.some(i => i.ports.includes(s.port));
    if (!portBusy) {
        actions.push(el('button', { class: 'btn btn-outline btn-sm', disabled: s.status === 'starting',
                                    text: s.status === 'starting' ? tx('Starting…') : tx('Start'),
                                    onclick: () => serviceAction(s.id, 'start') }));
    }
    const more = el('button', { class: 'btn-icon-sm', 'aria-expanded': String(servicesOpenDetails.has(s.id)),
                                title: tx('Reachability, copies, log'), 'aria-label': tx('More about {name}', { name: s.name }), text: '⋯',
                                onclick: () => {
                                    if (servicesOpenDetails.has(s.id)) servicesOpenDetails.delete(s.id);
                                    else servicesOpenDetails.add(s.id);
                                    fetchServices();
                                } });
    parts.push(el('div', { class: 'app-actions' }, ...actions, renderServiceAutostart(s), more));
    if (s.instances.length > 1) parts.push(el('div', { class: 'service-hint', text: tx('{n} copies running -- details under “⋯”.', { n: s.instances.length }) }));
    if (servicesOpenDetails.has(s.id)) parts.push(renderServiceDetails(s));
    return el('article', { class: 'app-card', 'data-service': s.id }, ...parts);
}

function renderServiceDetails(s) {
    const rows = s.instances.map(inst => {
        const { url, unreachable } = serviceOpenUrl(inst.open);
        return el('div', { class: 'service-instance' },
            el('span', { class: 'mono', text: `:${inst.ports.join(', :')}` }),
            el('span', { text: `PID ${inst.pid}` }),
            el('span', { text: tx('for {t}', { t: serviceUptime(inst.started_at) }) }),
            inst.managed ? el('span', { class: 'pill', text: tx('started from here') }) : null,
            el('span', { class: 'spacer' }),
            unreachable ? null : el('a', { class: 'btn btn-secondary btn-sm', href: url, target: '_blank', rel: 'noopener', text: tx('Open'), title: url }),
            el('button', { class: 'btn btn-danger btn-sm', text: tx('Stop'),
                           onclick: () => { if (confirmStop(s, inst)) serviceAction(s.id, 'stop', { pid: inst.pid }); } }));
    });
    const pre = el('pre', { class: 'code-snippet service-log', hidden: !servicesOpenLogs.has(s.id) });
    if (servicesOpenLogs.has(s.id)) loadServiceLog(s.id, pre);
    return el('div', { class: 'app-details' },
        ...rows,
        s.ui === false ? el('div', { class: 'muted small', text: tx('A service for other apps, without a page of its own -- stays on this machine (no Tailscale sharing).') })
                       : renderServiceExposure(s),
        s.ui === false ? null : renderServiceDoor(s),
        el('div', { class: 'muted small', text: IDENTITY_TEXT[s.identity] || '' }),
        el('div', { class: 'service-dir', text: s.project_dir, title: tx('found through {src}', { src: s.source }) }),
        s.has_log ? el('button', { class: 'btn btn-secondary btn-sm', text: tx('Log'), onclick: () => toggleServiceLog(s.id, pre) }) : null,
        pre);
}

function renderServiceAutostart(s) {
    const box = el('input', {
        type: 'checkbox',
        onchange: async (e) => {
            box.disabled = true;
            try {
                await api(`/api/services/${encodeURIComponent(s.id)}/autostart`, 'POST', { enabled: e.target.checked });
                addLog(e.target.checked ? tx('App {id}: autostart on', { id: s.id }) : tx('App {id}: autostart off', { id: s.id }), 'system');
            } catch (err) {
                addLog(tx('App {id}: changing autostart failed -- {err}', { id: s.id, err: err.message }), 'error');
                alertServiceError(err.message);
            }
            fetchServices();
        },
    });
    box.checked = !!s.autostart;
    return el('label', { class: 'app-autostart', title: tx('Starts with the ai-serve backend') }, box, el('span', { text: tx('Autostart') }));
}

function renderServiceExposure(s) {
    const ex = s.exposure || { level: 'local' };
    const select = el('select', {
        class: 'service-exposure-select',
        title: tx('Who reaches the app through Tailscale (never the public internet)'),
        onchange: (e) => setServiceExposure(s, e.target.value, select, ex.level),
    });
    for (const [value, label] of Object.entries(EXPOSURE_LABELS)) {
        // docs/identity.md: an app that keeps one person's data is only for the owner
        const single = value === 'tailnet' && s.identity === 'single';
        const disabled = (value !== 'local' && !servicesTailscale.available) || single;
        const opt = el('option', { value, text: single ? tx('{label} (only with data per person)', { label }) : label, disabled });
        if (value === ex.level) opt.selected = true;
        select.append(opt);
    }
    const parts = [el('div', { class: 'service-exposure-row' }, el('span', { text: tx('Reachable for') }), select)];
    if (ex.url) parts.push(el('a', { class: 'service-exposure-url', href: ex.url, target: '_blank', rel: 'noopener', text: ex.url }));
    if (ex.warning) parts.push(el('div', { class: 'service-hint', text: ex.warning }));
    if (ex.error) parts.push(el('div', { class: 'service-hint service-error', text: ex.error }));
    return el('div', { class: 'service-exposure' }, ...parts);
}

async function setServiceExposure(s, level, select, previous) {
    if (level !== 'local') {
        const who = level === 'owner'
            ? tx('only the devices of {owner}', { owner: servicesTailscale.owner })
            : tx('every device in the tailnet (including devices shared by others)');
        if (!window.confirm(tx('Make {name} reachable for {who}?', { name: s.name, who }))) {
            select.value = previous;
            return;
        }
    }
    select.disabled = true;
    try {
        await api(`/api/services/${encodeURIComponent(s.id)}/exposure`, 'POST', { level });
        addLog(tx('App {id}: reachability {level}', { id: s.id, level }), 'system');
    } catch (err) {
        addLog(tx('App {id}: reachability not changed -- {err}', { id: s.id, err: err.message }), 'error');
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
    if (servicesDoor && sub === servicesDoor.owner) return tx('Me');
    const p = (servicesDoor?.people || []).find(x => x.sub === sub);
    return p && p.name ? `${p.name} (${sub})` : sub;
}

function renderServiceDoor(s) {
    if (servicesDoor && servicesDoor.configured === false) return document.createDocumentFragment();   // optional extension, not set up
    const box = el('div', { class: 'service-exposure door-share' },
        el('div', { class: 'service-exposure-row' }, el('strong', { text: tx('From outside, with Nextcloud login') })));
    if (!servicesDoor) return box;
    if (!servicesDoor.available) {
        box.append(el('div', { class: 'muted small', text: tx('The door (edge-door) isn\'t running right now.') }));
    }
    const share = doorShare(s);
    const draft = servicesDoorDrafts.get(s.id);
    if (draft) {
        box.append(renderDoorEditor(s, draft));
        return box;
    }
    if (share) {
        const who = [...share.groups.map(g => tx('group {g}', { g })), ...share.users.map(doorPeopleLabel)].join(', ');
        box.append(el('a', { class: 'service-exposure-url', href: share.url, target: '_blank', rel: 'noopener', text: share.url }),
                   el('div', { class: 'small', text: tx('For: {who}', { who }) }));
        if (share.error) box.append(el('div', { class: 'service-hint service-error', text: tx('Not active: {err}', { err: share.error }) }));
        box.append(el('div', { class: 'app-actions' },
            el('button', { class: 'btn btn-secondary btn-sm', text: tx('Change'), onclick: () => {
                servicesDoorDrafts.set(s.id, { name: share.name, groups: new Set(share.groups), users: new Set(share.users) });
                doorRerender();
            } }),
            el('button', { class: 'btn btn-danger btn-sm', text: tx('Stop sharing'), onclick: () => deleteDoorShare(s) })));
    } else {
        box.append(el('div', { class: 'muted small', text: s.identity === 'single'
            ? tx('This app doesn\'t separate data per person – it can only be shared for yourself, to use on the go.')
            : tx('People you know sign in with their Nextcloud account and have their own data here.') }),
            el('div', { class: 'app-actions' }, el('button', { class: 'btn btn-outline btn-sm', text: tx('Share …'),
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
                               'aria-label': tx('Address'), oninput: (e) => { draft.name = e.target.value.trim().toLowerCase(); } });
    const parts = [el('label', { class: 'door-address' }, el('span', { class: 'mono', text: 'https://' }), name,
                      el('span', { class: 'mono', text: `.${servicesDoor.domain}/` }))];
    const check = (set, value, label, disabled = false) => {
        const c = el('input', { type: 'checkbox', disabled, onchange: (e) => { e.target.checked ? set.add(value) : set.delete(value); } });
        c.checked = set.has(value);
        return el('label', { class: 'door-check' }, c, el('span', { text: label }));
    };
    if (single) {
        parts.push(el('div', { class: 'small', text: tx('Only for you ({owner}) – the app doesn\'t separate data per person.', { owner: servicesDoor.owner }) }));
    } else {
        const groups = new Set(['apps', ...draft.groups]);
        for (const p of servicesDoor.people || []) for (const g of p.groups || []) groups.add(g);
        const people = new Set([servicesDoor.owner, ...draft.users, ...(servicesDoor.people || []).map(p => p.sub)].filter(Boolean));
        const addField = (placeholder, set) => {
            const input = el('input', { type: 'text', placeholder, class: 'door-add' });
            return el('span', { class: 'door-add-row' }, input, el('button', { class: 'btn btn-secondary btn-sm', text: '+', type: 'button',
                onclick: () => { const v = input.value.trim(); if (v) { set.add(v); doorRerender(); } } }));
        };
        parts.push(el('div', { class: 'door-col' }, el('div', { class: 'muted small', text: tx('Nextcloud groups') }),
                      ...[...groups].sort().map(g => check(draft.groups, g, g)), addField(tx('another group'), draft.groups)),
                   el('div', { class: 'door-col' }, el('div', { class: 'muted small', text: tx('People (Nextcloud account)') }),
                      ...[...people].sort().map(u => check(draft.users, u, doorPeopleLabel(u))), addField(tx('account name'), draft.users),
                      el('div', { class: 'muted small', text: tx('Everyone who has signed in before shows up here; add others by account name.') })));
    }
    parts.push(el('div', { class: 'app-actions' },
        el('button', { class: 'btn btn-primary btn-sm', text: tx('Save'), onclick: () => saveDoorShare(s, draft) }),
        el('button', { class: 'btn btn-secondary btn-sm', text: tx('Cancel'), onclick: () => { servicesDoorDrafts.delete(s.id); doorRerender(); } })));
    return el('div', { class: 'door-editor' }, ...parts);
}

async function saveDoorShare(s, draft) {
    const body = { name: draft.name, groups: [...draft.groups], users: [...draft.users] };
    const who = [...body.groups.map(g => tx('everyone in group “{g}”', { g })), ...body.users.map(doorPeopleLabel)].join(', ');
    if (!window.confirm(tx('Share {name} at https://{host}/ for: {who}?', { name: s.name, host: `${draft.name}.${servicesDoor.domain}`, who: who || tx('nobody') }))) return;
    try {
        const r = await api(`/api/door/shares/${encodeURIComponent(s.id)}`, 'PUT', body);
        servicesDoorDrafts.delete(s.id);
        addLog(tx('App {id}: shared from outside ({url})', { id: s.id, url: r.share.url }), 'system');
        if (r.door_error) alertServiceError(r.door_error);
        if (r.restart_needed && s.instances.length) {
            alertServiceError(tx('Please stop and start {name} once so it recognises sign-ins at the new address.', { name: s.name }));
        }
    } catch (err) {
        addLog(tx('App {id}: share not saved -- {err}', { id: s.id, err: err.message }), 'error');
        alertServiceError(err.message);
    }
    fetchServices();
}

async function deleteDoorShare(s) {
    if (!window.confirm(tx('Stop sharing {name} from outside? Whoever is signed in right now won\'t get in afterwards.', { name: s.name }))) return;
    try {
        await api(`/api/door/shares/${encodeURIComponent(s.id)}`, 'DELETE');
        addLog(tx('App {id}: no longer shared from outside', { id: s.id }), 'system');
    } catch (err) {
        alertServiceError(err.message);
    }
    fetchServices();
}

function confirmStop(s, inst) {
    // Instances not started from here may belong to someone's terminal session.
    if (inst.managed) return true;
    return window.confirm(tx('{name} (PID {pid}, port {ports}) wasn\'t started from this dashboard. Stop it anyway?', { name: s.name, pid: inst.pid, ports: inst.ports.join(', ') }));
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
    filters.replaceChildren(chip('all', tx('All'), services.length), ...groups.map(g => chip(g.key, g.label, g.items.length)));

    if (!data.services.length) {
        list.replaceChildren(el('div', { class: 'empty-state', text: tx('No apps yet -- put a project with a llama-serve.service.json into a service folder.') }));
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
    if (!shown.length) list.replaceChildren(el('div', { class: 'empty-state', text: tx('No app matches the search.') }));
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
        list.replaceChildren(el('div', { class: 'empty-state', text: tx('Couldn\'t load apps: {err}', { err: err.message }) }));
    }
}

function updateServicesCount(data) {
    const c = document.getElementById('side-count-services');
    if (c) c.textContent = `${data.services.filter(s => s.instances.length).length}/${data.services.length}`;
}

document.getElementById('services-refresh-btn').addEventListener('click', fetchServices);
document.getElementById('services-search').addEventListener('input', () => { if (servicesLast) renderServices(servicesLast); });
