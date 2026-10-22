// "Language model from Hugging Face" on Compute & models (owner, 2026-10-22): paste a repo's address, pick the
// quantization, download with checksum (POST /api/models/hf/inspect, /api/models/hf/download, GET
// /api/models/hf/downloads; rules in backend/hf_models.py). Uses el()/api()/addLog() from services.js/app.js.
let hfPoll = null, hfLast = null;

function hfGb(bytes) {
    return `${(bytes / 2 ** 30).toLocaleString(I18N.locale, { maximumFractionDigits: 1 })} GB`;
}

function renderHfForm() {
    const box = document.getElementById('hf-download');
    if (!box || box.childElementCount) return;          // built once; keeps what was typed
    const input = el('input', { type: 'text', class: 'device-input', id: 'hf-url',
                                placeholder: 'https://huggingface.co/unsloth/Qwen3-4B-GGUF' });
    const look = el('button', { class: 'btn btn-secondary btn-sm', text: tx('Look up') });
    const result = el('div', { id: 'hf-result' });
    look.onclick = async () => {
        look.disabled = true;
        result.replaceChildren(el('div', { class: 'muted small', text: tx('Asking Hugging Face …') }));
        try {
            hfLast = await api('/api/models/hf/inspect', 'POST', { url: input.value.trim() });
            renderHfChoice(result, input.value.trim(), hfLast);
        } catch (err) {
            result.replaceChildren(el('div', { class: 'small tone-bad', text: err.message }));
        }
        look.disabled = false;
    };
    input.addEventListener('keydown', e => { if (e.key === 'Enter') look.click(); });
    box.replaceChildren(el('div', { class: 'hf-row' }, input, look), result, el('div', { id: 'hf-jobs' }));
    fetchHfJobs();
}

function renderHfChoice(box, url, info) {
    // the file the address named; else Q4_K_M (a good default), else the first
    const preset = info.selected || (info.models.find(m => m.quant === 'Q4_K_M') || info.models[0] || {}).path;
    const rows = info.models.map((m, i) => {
        const radio = el('input', { type: 'radio', name: 'hf-model', value: m.path });
        if (m.path === preset) radio.checked = true;
        const parts = m.parts > 1 ? ' · ' + tx('{n} parts', { n: m.parts }) : '';
        return el('label', { class: 'hf-option' }, radio,
                  el('span', { class: 'mono small', text: m.path }), el('span', { class: 'muted small', text: ` ${hfGb(m.size)}${parts}` }));
    });
    let proj = null;
    if (info.mmproj.length) {
        proj = el('select', { class: 'service-exposure-select' },
                  el('option', { value: '', text: tx('without vision projector') }),
                  ...info.mmproj.map(p => el('option', { value: p.path, text: `${p.path} (${hfGb(p.size)})` })));
        const f16 = info.mmproj.find(p => /(^|[^b])f16/i.test(p.path)) || info.mmproj[0];   // F16, not BF16: runs everywhere
        proj.value = f16.path;
    }
    const go = el('button', { class: 'btn btn-primary btn-sm', text: tx('Download') });
    go.onclick = async () => {
        const model = (box.querySelector('input[name="hf-model"]:checked') || {}).value;
        go.disabled = true;
        try {
            await api('/api/models/hf/download', 'POST', { url, model, mmproj: proj && proj.value ? proj.value : null });
            addLog(tx('{name}: download started', { name: model }), 'system');
            box.replaceChildren();
        } catch (err) {
            box.append(el('div', { class: 'small tone-bad', text: err.message }));
        }
        go.disabled = false;
        fetchHfJobs();
    };
    box.replaceChildren(
        el('div', { class: 'muted small', text: tx('{repo} – choose a file (Q4_K_M is a good balance of size and quality):', { repo: info.repo }) }),
        el('div', { class: 'hf-options' }, ...rows),
        proj ? el('div', { class: 'hf-row' }, el('span', { class: 'small', text: tx('Vision projector (lets the model see images):') }), proj) : null,
        el('div', { class: 'muted small', text: tx('Saved in {dir}', { dir: info.dest }) }),
        go);
}

async function fetchHfJobs() {
    const box = document.getElementById('hf-jobs');
    if (!box) return;
    let data;
    try { data = await api('/api/models/hf/downloads'); } catch (err) { return; }
    box.replaceChildren(...data.downloads.map(j => el('div', { class: 'small' },
        el('span', { class: 'mono', text: j.model }), ' ',
        el('span', { class: j.state === 'failed' ? 'tone-bad' : 'muted', text:
            j.state === 'downloading' ? tx('Downloading … {done} of {total}', { done: hfGb(j.done), total: hfGb(j.size) })
            : j.state === 'failed' ? tx('Failed: {err}', { err: j.error })
            : tx('ready – use it with "model": "{name}"', { name: j.model }) }))));
    clearTimeout(hfPoll);
    if (data.downloads.some(j => j.state === 'downloading')) hfPoll = setTimeout(fetchHfJobs, 2000);
}

document.addEventListener('DOMContentLoaded', () => {
    renderHfForm();
    document.querySelector('.tab-link[data-tab="compute"]')?.addEventListener('click', fetchHfJobs);
});
