// "Dokumente lesen": an image or PDF through the gateway's /api/ocr (NuExtract3).
// The gateway renders PDFs page by page and reads each page on its own.

let ocrResultText = '';

function populateOcrModels(models) {
    const sel = document.getElementById('ocr-model');
    const ocr = (models || []).filter(m => /nuextract/i.test(m.name));
    for (const m of ocr) {
        const o = document.createElement('option');
        o.value = m.name;
        o.textContent = m.name;
        sel.append(o);
    }
}

function readFileBase64(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result).split(',')[1]);
        r.onerror = () => reject(r.error);
        r.readAsDataURL(file);
    });
}

document.getElementById('ocr-mode').addEventListener('change', (e) => {
    document.getElementById('ocr-template-group').hidden = e.target.value !== 'structured';
});

document.getElementById('ocr-run-btn').addEventListener('click', async () => {
    const file = document.getElementById('ocr-file').files[0];
    const status = document.getElementById('ocr-status');
    const out = document.getElementById('ocr-result');
    const btn = document.getElementById('ocr-run-btn');
    if (!file) { status.textContent = tx('Choose a file first.'); return; }
    const mode = document.getElementById('ocr-mode').value;
    const body = { images_base64: [await readFileBase64(file)], mode };
    if (mode === 'structured') {
        body.template = document.getElementById('ocr-template').value.trim();
        const instr = document.getElementById('ocr-instructions').value.trim();
        if (instr) body.instructions = instr;
        if (!body.template) { status.textContent = tx('“Fields from a template” needs a template.'); return; }
    }
    const model = document.getElementById('ocr-model').value;
    if (model) body.model = model;
    btn.disabled = true;
    status.textContent = tx('Reading… (the first time, the model is loaded)');
    const t0 = performance.now();
    try {
        const r = await api('/api/ocr', 'POST', body);
        const pages = (r.pages || []).length;
        ocrResultText = r.content != null ? r.content : JSON.stringify(r.parsed, null, 2);
        out.textContent = ocrResultText || tx('(empty)');
        status.textContent = `${r.status === 'completed' ? tx('Done') : r.status === 'partial' ? tx('Partly') : r.status} · ${pages === 1 ? tx('1 page') : tx('{n} pages', { n: pages })} · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
        document.getElementById('ocr-copy-btn').disabled = !ocrResultText;
        addLog(tx('Document read: {name}', { name: file.name }), 'info');
    } catch (err) {
        status.textContent = tx('Failed: {err}', { err: err.message });
        addLog(tx('Document: {err}', { err: err.message }), 'error');
    }
    btn.disabled = false;
});

document.getElementById('ocr-copy-btn').addEventListener('click', async () => {
    logCopyResult(await copyTextToClipboard(ocrResultText), tx('Result'));
});
