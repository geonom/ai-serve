// Kontingente: guests see what's left today in the top bar; the owner edits the limits in their own tab.
// The server (spielwiese/quota.py) is the authority -- this only shows and edits. DOM via textContent.

const quota = { me: null, cfg: null };

const minutes = (s) => Math.round((s || 0) / 60);

async function quotaBadge() {
    const badge = document.getElementById('quota-badge');
    try {
        const q = await sw('/quota');
        if (q.unlimited) { badge.hidden = true; return; }
        const L = q.limits, u = q.usage, parts = [];
        if (L.images_per_day !== null) parts.push(tx('{used}/{max} images', { used: u.images, max: L.images_per_day }));
        if (L.gpu_seconds_per_day !== null) parts.push(tx('{used}/{max} min', { used: minutes(u.gpu_seconds), max: minutes(L.gpu_seconds_per_day) }));
        badge.textContent = parts.length ? tx('Today: {parts}', { parts: parts.join(' · ') }) : tx('Today: unlimited');
        const low = (L.images_per_day !== null && u.images >= L.images_per_day * 0.9) ||
                    (L.gpu_seconds_per_day !== null && u.gpu_seconds >= L.gpu_seconds_per_day * 0.9);
        badge.classList.toggle('low', low);
        badge.hidden = false;
    } catch (err) { badge.hidden = true; }
}

const numOrNull = (v) => (v === '' || v === null || v === undefined) ? undefined : Math.max(0, Math.round(Number(v)));

function quotaPersonRow(p) {
    const own = p.limits || {};
    const unlimited = p.limits && own.images_per_day === null && own.gpu_seconds_per_day === null;
    const mk = (cls, val, label, min) => el('label', {}, document.createTextNode(label),
        el('input', { type: 'number', class: cls, min, step: '1', placeholder: tx('default'),
                      value: val === undefined || val === null ? '' : String(val), disabled: unlimited || null }));
    const unl = el('input', { type: 'checkbox', class: 'q-unl', checked: unlimited || null });
    const row = el('div', { class: 'quota-person', 'data-sub': p.sub },
        el('div', { class: 'who' }, el('b', { text: p.name || p.sub }),
           el('span', { text: (p.owner ? tx('owner') + ' · ' : '') + tx('today {images} images, {min} min', { images: p.usage.images, min: minutes(p.usage.gpu_seconds) }) +
                              (p.last_seen ? ' · ' + tx('last seen {when}', { when: new Date(p.last_seen * 1000).toLocaleString(I18N.locale) }) : '') })),
        mk('q-minutes', own.gpu_seconds_per_day === undefined || own.gpu_seconds_per_day === null ? undefined
                                                              : minutes(own.gpu_seconds_per_day), tx('minutes/day'), '0'),
        mk('q-images', own.images_per_day, tx('images/day'), '0'),
        el('label', { class: 'unl' }, unl, document.createTextNode(tx('unlimited'))));
    unl.addEventListener('change', () => row.querySelectorAll('input[type=number]').forEach(i => { i.disabled = unl.checked; }));
    if (p.owner) row.querySelectorAll('input').forEach(i => { i.disabled = true; i.title = tx('You aren\'t limited'); });
    return row;
}

async function quotaLoadAdmin() {
    const status = document.getElementById('q-status');
    try {
        quota.cfg = await sw('/admin/quotas');
    } catch (err) { status.textContent = tx('Not available: {err}', { err: err.message }); return; }
    const d = quota.cfg.defaults;
    document.getElementById('q-def-minutes').value = d.gpu_seconds_per_day === null ? '' : String(minutes(d.gpu_seconds_per_day));
    document.getElementById('q-def-images').value = d.images_per_day === null ? '' : String(d.images_per_day);
    document.getElementById('q-def-concurrent').value = d.concurrent === null ? '' : String(d.concurrent);
    const people = quota.cfg.people;
    document.getElementById('quota-people').replaceChildren(...(people.length ? people.map(quotaPersonRow)
        : [el('p', { class: 'muted small', text: tx('Nobody but you has used the Spielwiese yet.') })]));
    status.textContent = '';
}

async function quotaSave() {
    const status = document.getElementById('q-status');
    const defaults = {};
    const dm = numOrNull(document.getElementById('q-def-minutes').value);
    const di = numOrNull(document.getElementById('q-def-images').value);
    const dc = numOrNull(document.getElementById('q-def-concurrent').value);
    if (dm !== undefined) defaults.gpu_seconds_per_day = dm * 60;
    if (di !== undefined) defaults.images_per_day = di;
    if (dc !== undefined) defaults.concurrent = Math.max(1, dc);
    const people = {};
    document.querySelectorAll('.quota-person').forEach(row => {
        const sub = row.dataset.sub;
        if ((quota.cfg.people.find(p => p.sub === sub) || {}).owner) return;
        if (row.querySelector('.q-unl').checked) { people[sub] = { gpu_seconds_per_day: null, images_per_day: null }; return; }
        const m = numOrNull(row.querySelector('.q-minutes').value), i = numOrNull(row.querySelector('.q-images').value);
        const own = {};
        if (m !== undefined) own.gpu_seconds_per_day = m * 60;
        if (i !== undefined) own.images_per_day = i;
        if (Object.keys(own).length) people[sub] = own;
    });
    try {
        await sw('/admin/quotas', 'PUT', { defaults, people });
        status.textContent = tx('Saved.');
        quotaLoadAdmin();
    } catch (err) { status.textContent = tx('Not saved: {err}', { err: err.message }); }
}

document.getElementById('q-save').addEventListener('click', quotaSave);
document.getElementById('quota-tab-link').addEventListener('click', quotaLoadAdmin);

(async function quotaInit() {
    try { quota.me = await sw('/me'); } catch (err) { return; }
    document.getElementById('quota-tab-group').hidden = !quota.me.is_owner;
    quotaBadge();
    setInterval(quotaBadge, 30000);
})();
// refresh the badge after anything that may have used the GPU
document.addEventListener('click', (ev) => {
    if (ev.target.closest('#comfy-generate-btn, #cmp-run, #lay-run-btn, .btn-primary')) setTimeout(quotaBadge, 4000);
});
