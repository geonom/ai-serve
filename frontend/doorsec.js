// "Tür: Zugriffe und Abweisungen" (Zugang & API): what the door let through and turned away per day,
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
        ['day', 'Tag'], ['total', 'Anfragen'], ['signed_in', 'angemeldet in Apps'], ['people', 'Personen'],
        ['to_login', 'zum Login umgeleitet'], ['denied', 'abgewiesen 401/403'], ['not_found', 'unbekannt 404'],
        ['bad', 'fehlerhaft 400'], ['probes', 'davon Angriffsmuster'], ['door', 'Login-Seite/öffentlich'],
        ['leak', 'ohne Anmeldung durch'],
    ];

    async function load() {
        let data;
        try {
            data = await (await fetch(`${window.location.origin}/api/door/security?days=14`)).json();
        } catch (err) { data = { available: false, detail: err.message }; }
        const alarm = $('door-sec-alarm');
        if (data.configured === false) {
            alarm.className = 'door-sec-alarm';
            alarm.textContent = 'Die Tür (Zugang für andere über Nextcloud) ist nicht eingerichtet – eine optionale Erweiterung.';
            $('door-sec-table').replaceChildren();
            return;
        }
        if (!data.available) {
            alarm.className = 'door-sec-alarm bad';
            alarm.textContent = `Übersicht nicht verfügbar: ${data.detail || 'Türsteher nicht erreichbar'}`;
            $('door-sec-table').replaceChildren();
            return;
        }
        if (data.leak_total > 0) {
            alarm.className = 'door-sec-alarm bad';
            const last = (data.leaks || []).slice(-3).map(l => `${l.time} ${l.upstream}${l.path}`).join(' · ');
            alarm.textContent = `ACHTUNG: ${data.leak_total} Anfrage(n) haben eine App ohne Anmeldung erreicht. Zuletzt: ${last}`;
        } else {
            alarm.className = 'door-sec-alarm ok';
            alarm.textContent = 'Keine Anfrage hat eine App ohne Anmeldung erreicht.';
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
                const v = k === 'day' ? new Date(d.day + 'T12:00').toLocaleDateString() : String(d[k] ?? 0);
                tr.append(el('td', k === 'leak' && d.leak > 0 ? { class: 'bad' } : {}, v));
            });
            body.append(tr);
        }
        table.append(body);
        $('door-sec-table').replaceChildren(table);
        const probes = $('door-sec-probes');
        probes.replaceChildren();
        if ((data.top_probes || []).length) {
            probes.append(el('strong', {}, 'Häufigste Angriffsmuster (7 Tage): '));
            data.top_probes.forEach(([p, n]) => probes.append(el('code', {}, `${p} ×${n}`)));
        }
        $('door-sec-note').textContent = data.note || '';
    }

    $('door-sec-reload').addEventListener('click', load);
    document.querySelectorAll('.tab-link[data-tab="api-doc"]').forEach(b => b.addEventListener('click', load));
    if (document.querySelector('.tab-link[data-tab="api-doc"].active')) load();
})();
