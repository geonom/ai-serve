// "Weitere Modellserver": standard OpenAI-compatible servers the owner already runs (Ollama, LM Studio ...), added
// by port (GET/PUT/DELETE /api/model-servers; backend/plugins/external.py). Found ones are offered with one click.
async function fetchModelServers() {
    const box = document.getElementById('model-servers');
    if (!box || (box.contains(document.activeElement) && document.activeElement !== document.body)) return;
    let data;
    try {
        data = await api('/api/model-servers');
    } catch (err) {
        box.replaceChildren(el('div', { class: 'empty-state', text: `Nicht ladbar: ${err.message}` }));
        return;
    }
    const act = async (path, method, body, okText) => {
        try {
            await api(path, method, body);
            addLog(okText, 'system');
        } catch (err) {
            addLog(`Modellserver: ${err.message}`, 'error');
        }
        fetchModelServers();
        if (typeof renderComputeModels === 'function') renderComputeModels();
    };
    const rows = [];
    for (const s of data.servers) {
        rows.push(el('div', { class: 'ms-row' },
            el('div', {}, el('strong', { text: s.name }), el('span', { class: 'muted', text: `  Port ${s.port}` })),
            el('span', { class: s.running ? 'tone-ok' : 'muted',
                         text: s.running ? `läuft · ${s.models} Modell${s.models === 1 ? '' : 'e'}` : 'antwortet gerade nicht' }),
            el('button', { class: 'btn btn-danger-outline btn-sm', text: 'Entfernen',
                           title: 'ai-serve benutzt ihn nicht mehr; der Server selbst läuft weiter',
                           onclick: () => act(`/api/model-servers/${encodeURIComponent(s.id)}`, 'DELETE', null, `${s.name} entfernt`) })));
    }
    for (const d of data.detected) {
        rows.push(el('div', { class: 'ms-row' },
            el('div', {}, el('strong', { text: d.name }), el('span', { class: 'muted', text: `  gefunden auf Port ${d.port}` })),
            el('span', { class: 'muted', text: `${d.models} Modell${d.models === 1 ? '' : 'e'}` }),
            el('button', { class: 'btn btn-primary btn-sm', text: 'Einbinden',
                           onclick: () => act(`/api/model-servers/${encodeURIComponent(d.id)}`, 'PUT',
                                              { name: d.name, port: d.port }, `${d.name} eingebunden`) })));
    }
    if (!rows.length) rows.push(el('div', { class: 'empty-state', text: 'Keiner eingebunden, keiner auf den üblichen Ports (Ollama 11434, LM Studio 1234) gefunden.' }));
    const name = el('input', { class: 'r-input', placeholder: 'Name, z. B. vLLM', 'aria-label': 'Name', 'data-key': 'ms-name' });
    const port = el('input', { class: 'r-input', type: 'number', min: '1024', max: '65535', placeholder: 'Port', 'aria-label': 'Port', 'data-key': 'ms-port' });
    const add = el('button', { class: 'btn btn-secondary btn-sm', text: 'Hinzufügen', onclick: () => {
        const n = name.value.trim() || 'Server', p = parseInt(port.value, 10);
        const id = n.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 32) || 'server';
        act(`/api/model-servers/${id}`, 'PUT', { name: n, port: p }, `${n} eingebunden`);
    } });
    rows.push(el('div', { class: 'ms-row' }, el('span', { class: 'muted', text: 'Anderer Server auf diesem Rechner (OpenAI-API unter /v1):' }), name, port, add));
    if (data.errors && data.errors.length) rows.push(el('div', { class: 'muted', text: data.errors.join(' · ') }));
    // keep what was typed but not added yet across the refresh
    for (const old of box.querySelectorAll('input[data-key]')) {
        const fresh = rows.flatMap(r => [...r.querySelectorAll(`input[data-key="${old.dataset.key}"]`)])[0];
        if (fresh && old.value) fresh.value = old.value;
    }
    box.replaceChildren(...rows);
}

document.addEventListener('DOMContentLoaded', () => {
    fetchModelServers();
    setInterval(() => { if (document.getElementById('compute-tab')?.classList.contains('active')) fetchModelServers(); }, 10000);
});
