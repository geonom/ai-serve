// Chat model list from the gateway's GET /v1/llm/models: every model that can chat -- on this machine, on
// enrolled devices (the gateway routes there), other model servers from the gateway's llm plugins (e.g. Strata) -- grouped, with what each can do. Embedding, rerank
// and base models (no chat template) stay out of the chat list (owner, 2026-10-20).
// Pure; tests: node --test tests/models.test.mjs
const ChatModels = (() => {
    function optionText(m) {
        const badges = [m.vision && 'Bild', m.thinking && 'Denken', m.mtp && 'MTP'].filter(Boolean);
        const devices = m.where.filter(p => p.place === 'device').map(p => p.node);
        const local = m.where.some(p => p.place === 'local');
        return m.label + (m.quant ? ` · ${m.quant}` : '') + (badges.length ? ` (${badges.join(', ')})` : '')
            + (!local && devices.length ? ` @ ${devices.join(', ')}` : '');
    }

    function groups(models) {
        const chat = (models || []).filter(m => m.kind === 'chat');
        const out = [
            { label: 'Läuft gerade', items: chat.filter(m => m.loaded) },
            { label: 'Exklusiv (belegt alle Grafikkarten)', items: chat.filter(m => !m.loaded && m.exclusive) },
            { label: 'Auf dem Gateway-Rechner', items: chat.filter(m => !m.loaded && !m.exclusive && m.where.some(p => p.place === 'local')) },
            { label: 'Weitere Modellserver', items: chat.filter(m => !m.loaded && !m.exclusive && m.where.some(p => p.place === 'plugin')) },
            { label: 'Auf anderen Rechnern im Netzwerk', items: chat.filter(m => !m.loaded && m.where.length && m.where.every(p => p.place === 'device')) },
        ];
        return out.filter(g => g.items.length).map(g => ({
            label: g.label,
            options: g.items.map(m => ({
                value: m.id, text: optionText(m),
                // a model only on devices that are all offline can't run anywhere right now
                disabled: !m.where.some(p => p.place !== 'device' || p.online),
            })),
        }));
    }

    function contextFor(m) {
        if (!m) return null;
        const running = m.where.find(p => p.loaded && p.ctx_size);
        return (running && running.ctx_size) || m.context_length || null;
    }

    function note(m) {
        if (!m) return '';
        const server = m.where.find(p => p.place === 'plugin');
        if (m.exclusive) return `${(server && server.name) || m.label} belegt alle Grafikkarten für sich; das Laden kann Minuten dauern, andere Modelle werden dafür entladen.`;
        if (server) return `Läuft auf ${server.name || server.plugin}, das Gateway leitet dorthin.`;
        if (!m.where.some(p => p.place === 'local')) return `Liegt nur auf ${m.where.map(p => p.node).join(', ')} – das Gateway leitet dorthin.`;
        return '';
    }

    return { groups, optionText, contextFor, note };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ChatModels;
