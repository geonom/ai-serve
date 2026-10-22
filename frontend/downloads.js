// "Downloads" on Compute & models (owner, 2026-10-22): models and engines the plugins can download -- their size,
// a Download button and progress (GET /api/plugins/installs, POST /api/plugins/{plugin}/install {"id"}). The plugin
// picks the files and checks them; the dashboard only names the item. Uses el()/api()/addLog() from services.js/app.js.
let downloadsPoll = null;

function downloadSize(mb) {
    if (mb === null || mb === undefined) return '';
    return mb >= 1024 ? `~${(mb / 1024).toLocaleString(I18N.locale, { maximumFractionDigits: 1 })} GB` : `~${mb} MB`;
}

async function fetchDownloads() {
    const box = document.getElementById('plugin-downloads');
    if (!box || (box.contains(document.activeElement) && document.activeElement !== document.body)) return;
    let data;
    try {
        data = await api('/api/plugins/installs');
    } catch (err) {
        box.replaceChildren(el('div', { class: 'empty-state', text: tx('Couldn\'t load: {err}', { err: err.message }) }));
        return;
    }
    const open = data.items.filter(i => i.state !== 'installed');
    const done = data.items.filter(i => i.state === 'installed');
    const rows = open.map(i => {
        const state = i.state === 'downloading'
            ? (i.done_mb !== null && i.size_mb ? tx('Downloading … {done} of {total}', { done: downloadSize(i.done_mb).slice(1), total: downloadSize(i.size_mb).slice(1) })
                                               : tx('Downloading and setting up … this can take a while'))
            : i.state === 'failed' ? tx('Failed: {err}', { err: i.error || '?' }) : '';
        const btn = i.state === 'downloading' ? null : el('button', {
            class: 'btn btn-secondary btn-sm', text: i.state === 'failed' ? tx('Try again') : tx('Download'),
            onclick: async (e) => {
                e.target.disabled = true;
                try {
                    await api(`/api/plugins/${encodeURIComponent(i.plugin)}/install`, 'POST', { id: i.id });
                    addLog(tx('{label}: download started ({size}).', { label: i.label, size: downloadSize(i.size_mb) || '?' }), 'system');
                } catch (err) {
                    addLog(tx('{label}: download not started: {err}', { label: i.label, err: err.message }), 'error');
                }
                fetchDownloads();
            } });
        return el('div', { class: 'ms-row' },
            el('div', {}, el('strong', { text: i.label }),
               el('span', { class: 'muted', text: `  ${[downloadSize(i.size_mb), i.plugin_name || i.plugin, i.license].filter(Boolean).join(' · ')}` }),
               i.description ? el('div', { class: 'muted small', text: tx(i.description) }) : null),
            el('span', { class: i.state === 'failed' ? 'tone-bad small' : 'muted small', text: state }),
            btn);
    });
    if (!rows.length) rows.push(el('div', { class: 'empty-state', text: data.items.length
        ? tx('Everything the plugins offer for this machine is installed.') : tx('No plugin offers downloads here.') }));
    if (done.length) rows.push(el('div', { class: 'muted small', text: tx('Installed: {list}', { list: done.map(i => i.label).join(', ') }) }));
    if (data.errors && data.errors.length) rows.push(el('div', { class: 'muted small', text: data.errors.join(' · ') }));
    box.replaceChildren(...rows);
    clearTimeout(downloadsPoll);
    if (open.some(i => i.state === 'downloading')) downloadsPoll = setTimeout(fetchDownloads, 3000);
}

document.addEventListener('DOMContentLoaded', () => {
    fetchDownloads();
    document.querySelector('.tab-link[data-tab="compute"]')?.addEventListener('click', fetchDownloads);
});
