// Chat model list from the gateway's GET /v1/llm/models: every model that can chat -- on this machine, on
// enrolled devices (the gateway routes there), other model servers from the gateway's llm plugins (e.g. Strata) -- grouped, with what each can do. Embedding, rerank
// and base models (no chat template) stay out of the chat list (owner, 2026-10-20).
// Pure; tests: node --test tests/models.test.mjs
const ChatModels = (() => {
    function optionText(m) {
        const badges = [m.vision && tx('image'), m.thinking && tx('thinking'), m.mtp && 'MTP'].filter(Boolean);
        const devices = m.where.filter(p => p.place === 'device').map(p => p.node);
        const local = m.where.some(p => p.place === 'local');
        return m.label + (m.quant ? ` · ${m.quant}` : '') + (badges.length ? ` (${badges.join(', ')})` : '')
            + (!local && devices.length ? ` @ ${devices.join(', ')}` : '');
    }

    function groups(models) {
        const chat = (models || []).filter(m => m.kind === 'chat');
        const out = [
            { label: tx('Running now'), items: chat.filter(m => m.loaded) },
            { label: tx('Exclusive (takes all graphics cards)'), items: chat.filter(m => !m.loaded && m.exclusive) },
            { label: tx('On the gateway machine'), items: chat.filter(m => !m.loaded && !m.exclusive && m.where.some(p => p.place === 'local')) },
            { label: tx('More model servers'), items: chat.filter(m => !m.loaded && !m.exclusive && m.where.some(p => p.place === 'plugin')) },
            { label: tx('On other machines in the network'), items: chat.filter(m => !m.loaded && m.where.length && m.where.every(p => p.place === 'device')) },
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
        if (m.exclusive) return tx('{name} takes all graphics cards for itself; loading can take minutes, other models are unloaded for it.', { name: (server && server.name) || m.label });
        if (server) return tx('Runs on {server}; the gateway passes requests there.', { server: server.name || server.plugin });
        if (!m.where.some(p => p.place === 'local')) return tx('Only on {nodes} – the gateway passes requests there.', { nodes: m.where.map(p => p.node).join(', ') });
        return '';
    }

    return { groups, optionText, contextFor, note };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ChatModels;
