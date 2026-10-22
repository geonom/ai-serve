// "Door: access and refusals" (Access & API): what the door let through and turned away per day,
// from edge-door's /api/security via GET /api/door/security. The number that matters most: requests that
// reached an app without a signed-in person -- must be 0. DOM only via textContent.

(function () {
    const $ = (id) => document.getElementById(id);
    const el = (tag, attrs = {}, text) => {
        const e = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        if (text !== undefined) e.textContent = text;
        return e;
    };
    const COLS = [
        ['day', tx('Day')], ['total', tx('Requests')], ['signed_in', tx('signed in to apps')], ['people', tx('People')],
        ['to_login', tx('sent to login')], ['denied', tx('refused 401/403')], ['not_found', tx('unknown 404')],
        ['bad', tx('malformed 400')], ['probes', tx('of which attack patterns')], ['door', tx('login page/public')],
        ['leak', tx('through without sign-in')],
    ];

    async function load() {
        let data;
        try {
            data = await (await fetch(`${window.location.origin}/api/door/security?days=14`)).json();
        } catch (err) { data = { available: false, detail: err.message }; }
        const alarm = $('door-sec-alarm');
        if (data.configured === false) {
            alarm.className = 'door-sec-alarm';
            alarm.textContent = tx('The door (access for others through Nextcloud) isn\'t set up – an optional extension.');
            $('door-sec-table').replaceChildren();
            return;
        }
        if (!data.available) {
            alarm.className = 'door-sec-alarm bad';
            alarm.textContent = tx('Overview not available: {err}', { err: data.detail || tx('door unreachable') });
            $('door-sec-table').replaceChildren();
            return;
        }
        if (data.leak_total > 0) {
            alarm.className = 'door-sec-alarm bad';
            const last = (data.leaks || []).slice(-3).map(l => `${l.time} ${l.upstream}${l.path}`).join(' · ');
            alarm.textContent = tx('WARNING: {n} request(s) reached an app without sign-in. Latest: {last}', { n: data.leak_total, last });
        } else {
            alarm.className = 'door-sec-alarm ok';
            alarm.textContent = tx('No request reached an app without sign-in.');
        }
        const table = el('table', { class: 'door-sec-table' });
        const head = el('tr');
        COLS.forEach(([, label]) => head.append(el('th', {}, label)));
        const thead = el('thead');
        thead.append(head);
        table.append(thead);
        const body = el('tbody');
        for (const d of data.days || []) {
            const tr = el('tr');
            COLS.forEach(([k]) => {
                const v = k === 'day' ? new Date(d.day + 'T12:00').toLocaleDateString(I18N.locale) : String(d[k] ?? 0);
                tr.append(el('td', k === 'leak' && d.leak > 0 ? { class: 'bad' } : {}, v));
            });
            body.append(tr);
        }
        table.append(body);
        $('door-sec-table').replaceChildren(table);
        const probes = $('door-sec-probes');
        probes.replaceChildren();
        if ((data.top_probes || []).length) {
            probes.append(el('strong', {}, tx('Most common attack patterns (7 days): ')));
            data.top_probes.forEach(([p, n]) => probes.append(el('code', {}, `${p} ×${n}`)));
        }
        $('door-sec-note').textContent = data.note || '';
    }

    $('door-sec-reload').addEventListener('click', load);
    document.querySelectorAll('.tab-link[data-tab="api-doc"]').forEach(b => b.addEventListener('click', load));
    if (document.querySelector('.tab-link[data-tab="api-doc"].active')) load();
})();
