// Missing models and engines (owner, 2026-10-22): a page shows what it could use but isn't installed, with its size
// and -- for the owner -- a "Download" button and progress; others are told to ask the owner. When a whole ability is
// missing (no plugin offers it), the page says so plainly instead of showing an empty list. The gateway lists what
// plugins can download (GET /api/plugins/installs) and starts one (POST /api/plugins/{plugin}/install {"id"}); the
// Spielwiese forwards the start only for its owner. Pure part first (tests: node --test tests/downloads.test.mjs).

const DownloadRules = (() => {
    // what a page can still download: items of its capability that aren't installed (`only`: just these item ids)
    function missing(items, capability, only = null) {
        return (items || []).filter(i => (!capability || i.capability === capability) && i.state !== 'installed'
            && (!only || only.includes(i.id)));
    }
    function sizeText(mb) {
        if (mb === null || mb === undefined) return '';
        return mb >= 1024 ? `~${(mb / 1024).toLocaleString(I18N.locale, { maximumFractionDigits: 1 })} GB` : `~${mb} MB`;
    }
    function stateText(i) {
        if (i.state === 'downloading') {
            return i.done_mb !== null && i.done_mb !== undefined && i.size_mb
                ? tx('Downloading … {done} of {total}', { done: sizeText(i.done_mb).replace('~', ''), total: sizeText(i.size_mb).replace('~', '') })
                : tx('Downloading and setting up … this can take a while');
        }
        if (i.state === 'failed') return tx('Failed: {err}', { err: i.error || '?' });
        return '';
    }
    return { missing, sizeText, stateText };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = DownloadRules;

const Downloads = (() => {
    if (typeof document === 'undefined') return null;
    let items = [], isOwner = false, loading = null, poll = null;
    const panels = new Map();          // element -> {capability, hint}

    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

    async function refresh() {
        try {
            const [list, me] = await Promise.all([api('/api/plugins/installs'),
                swFetch('/sw/me').then(r => r.ok ? r.json() : {}).catch(() => ({}))]);
            const before = new Map(items.map(i => [`${i.plugin}/${i.id}`, i.state]));
            items = list.items || [];
            isOwner = !!me.is_owner;
            for (const i of items) {
                if (before.get(`${i.plugin}/${i.id}`) === 'downloading' && i.state === 'installed') {
                    addLog(tx('{label} is installed.', { label: i.label }), 'success');
                    if (typeof showToast === 'function') showToast(tx('{label} is installed.', { label: i.label }), 'success');
                    window.dispatchEvent(new CustomEvent('sw-installed', { detail: { capability: i.capability } }));
                } else if (before.get(`${i.plugin}/${i.id}`) === 'downloading' && i.state === 'failed') {
                    addLog(tx('{label}: download failed: {err}', { label: i.label, err: i.error || '?' }), 'error');
                }
            }
        } catch (err) {
            items = [];
        }
        renderAll();
        clearTimeout(poll);
        if (items.some(i => i.state === 'downloading')) poll = setTimeout(refresh, 3000);
    }

    function ready() {
        if (!loading) loading = refresh();
        return loading;
    }

    async function start(i, btn) {
        btn.disabled = true;
        try {
            await api(`/api/plugins/${encodeURIComponent(i.plugin)}/install`, 'POST', { id: i.id });
            addLog(tx('{label}: download started ({size}).', { label: i.label, size: DownloadRules.sizeText(i.size_mb) || '?' }), 'system');
        } catch (err) {
            addLog(tx('{label}: download not started: {err}', { label: i.label, err: err.message }), 'error');
        }
        refresh();
    }

    function render(box, { capability, hint, only }) {
        const missing = capability === false ? [] : DownloadRules.missing(items, capability, only);
        box.replaceChildren();
        if (hint) {   // a string, or {text, href, link}: the text plus a link to how to get it
            const h = el('div', 'dl-hint', typeof hint === 'string' ? hint : hint.text);
            if (hint.href) {
                const a = el('a', null, hint.link || hint.href);
                a.href = hint.href; a.target = '_blank'; a.rel = 'noopener';
                h.append(' ', a);
            }
            box.append(h);
        }
        for (const i of missing) {
            const row = el('div', 'dl-row' + (i.state === 'failed' ? ' failed' : ''));
            const text = el('div', 'dl-text');
            text.append(el('strong', null, i.label), el('span', 'muted small', ' ' + [DownloadRules.sizeText(i.size_mb), i.license].filter(Boolean).join(' · ')));
            if (i.description) text.append(el('div', 'muted small', tx(i.description)));
            const state = DownloadRules.stateText(i);
            if (state) text.append(el('div', 'small dl-state', state));
            row.append(text);
            if (i.state !== 'downloading') {
                if (isOwner) {
                    const btn = el('button', 'btn btn-secondary btn-sm', i.state === 'failed' ? tx('Try again') : tx('Download'));
                    btn.type = 'button';
                    btn.addEventListener('click', () => start(i, btn));
                    row.append(btn);
                } else {
                    row.append(el('span', 'muted small', tx('Not installed – the owner can download it.')));
                }
            }
            box.append(row);
        }
        box.hidden = !hint && !missing.length;
    }

    function renderAll() {
        for (const [box, opts] of panels) if (box.isConnected) render(box, opts);
    }

    // Show what `capability` could still download in `box`; `hint` (optional) explains a missing ability;
    // `only` (optional) narrows the offer to these item ids.
    function panel(box, capability, hint = '', only = null) {
        if (!box) return;
        panels.set(box, { capability, hint, only });
        ready().then(() => render(box, { capability, hint, only }));
    }

    return { panel, ready, refresh, items: () => items, isOwner: () => isOwner };
})();
