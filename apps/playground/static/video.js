// "Create videos": the gateway's video jobs (/v1/video/models, /v1/video/jobs), 2026-11-13.
// Text -> video (LTX-2.5, with sound) or picture -> video (Wan 2.2); the start picture is uploaded or made here
// with the default image model. Jobs are polled until done; the MP4 comes through the gateway's own path.
// Prompt helper (models with a `prompt_guide`): camera/look settings + the description -> POST /v1/video/prompt writes
// a detailed prompt, shown and editable; that text (when present) is what the video job gets.

const VIDEO = { models: [], image: null, job: null, timer: null, started: 0, options: null, written: null };

function videoModel() {
    return VIDEO.models.find(m => m.id === document.getElementById('video-model').value) || null;
}

function needsPicture(m) {
    return !!m && m.tasks.includes('image_to_video') && !m.tasks.includes('text_to_video');
}

function takesPicture(m) {
    return !!m && m.tasks.includes('image_to_video');
}

function fillVideoForm() {
    const m = videoModel();
    if (!m) return;
    const info = [m.description, m.audio ? tx('with sound') : tx('no sound'), `${m.fps} fps`];
    if (m.license) info.push(m.license);
    document.getElementById('video-model-info').textContent = info.filter(Boolean).join(' · ');
    document.getElementById('video-image-group').hidden = !takesPicture(m);
    document.getElementById('video-image-label').textContent = needsPicture(m) ? tx('Start picture') : tx('Start picture (optional)');
    const secs = document.getElementById('video-seconds');
    secs.min = m.min_seconds || 1;
    secs.max = m.max_seconds || 10;
    secs.value = m.default_seconds || secs.min;
    const res = document.getElementById('video-resolution');
    res.innerHTML = '';
    for (const r of m.resolutions || []) res.append(new Option(r.replace('x', '×'), r));
    const lora = document.getElementById('video-lora');
    lora.innerHTML = '';
    lora.append(new Option(tx('none'), ''));
    for (const l of m.loras || []) lora.append(new Option(l, l));
    document.getElementById('video-lora-group').hidden = !(m.loras || []).length;
    const steps = document.getElementById('video-steps');
    document.getElementById('video-steps-group').hidden = !m.steps;
    if (m.steps) {
        steps.min = m.steps.min;
        steps.max = m.steps.max;
        steps.value = m.steps.default;
        steps.title = tx('Default {n}; more steps can add detail and steadier motion, each one costs time', { n: m.steps.default });
    }
    const sage = typeof m.sage_attention === 'boolean';
    document.getElementById('video-sage-group').hidden = !sage;
    document.getElementById('video-sage').checked = sage && m.sage_attention;
    document.getElementById('video-direction-group').hidden = !m.prompt_guide || !VIDEO.options;
    document.getElementById('video-sound-group').hidden = !m.audio;
    if (!m.prompt_guide) clearWrittenPrompt();
    else markWrittenStale();
}

// ---------------------------------------------------------------- prompt helper

async function loadPromptOptions() {
    if (VIDEO.options) return;
    try {
        VIDEO.options = await api('/v1/video/prompt/options');
    } catch (err) {
        return;                                   // no helper -- the plain description still works
    }
    const box = document.getElementById('video-direction-fields');
    box.innerHTML = '';
    for (const [key, opt] of Object.entries(VIDEO.options.settings || {})) {
        const wrap = document.createElement('div');
        const label = document.createElement('label');
        label.htmlFor = `video-dir-${key}`;
        label.textContent = tx(opt.label);
        const sel = document.createElement('select');
        sel.id = `video-dir-${key}`;
        sel.dataset.key = key;
        sel.append(new Option(tx('model decides'), ''));
        for (const c of opt.choices) sel.append(new Option(tx(c), c));
        sel.addEventListener('change', markWrittenStale);
        wrap.append(label, sel);
        box.append(wrap);
    }
}

function directionSettings() {
    const out = {};
    for (const sel of document.querySelectorAll('#video-direction-fields select')) {
        if (sel.value) out[sel.dataset.key] = sel.value;
    }
    const m = videoModel();
    if (m && m.audio) {                           // spoken words and sound only for models that make sound
        const dialogue = document.getElementById('video-dialogue').value.trim();
        if (dialogue) out.dialogue = dialogue;
        const sound = document.getElementById('video-sound').value.trim();
        if (sound) out.sound = sound;
    }
    return out;
}

// what the written prompt was made from -- to say when the description or settings changed since
function promptSource() {
    return JSON.stringify([document.getElementById('video-prompt').value.trim(), directionSettings(),
                           videoModel() ? videoModel().id : '', VIDEO.image ? VIDEO.image.length : 0]);
}

function clearWrittenPrompt() {
    VIDEO.written = null;
    document.getElementById('video-final-prompt').value = '';
    document.getElementById('video-final-group').hidden = true;
}

function markWrittenStale() {
    if (!VIDEO.written) return;
    const note = document.getElementById('video-final-note');
    const stale = promptSource() !== VIDEO.written.source;
    note.classList.toggle('stale', stale);
    note.textContent = stale ? tx('Description, picture or settings changed since – write again, or keep this text.')
                             : VIDEO.written.note;
}

document.getElementById('video-write-btn').addEventListener('click', async (e) => {
    const m = videoModel();
    const prompt = document.getElementById('video-prompt').value.trim();
    const status = document.getElementById('video-write-note');
    if (!m) return;
    if (!prompt) { showToast(tx('Describe the scene first.'), 'warn'); return; }
    if (needsPicture(m) && !VIDEO.image) { showToast(tx('Choose the start picture first – the prompt is written to match it.'), 'warn'); return; }
    const body = { model: m.id, prompt, settings: directionSettings() };
    if (takesPicture(m) && VIDEO.image) body.image_base64 = VIDEO.image;
    e.target.disabled = true;
    const t0 = performance.now();
    status.textContent = tx('Writing the prompt… (the first time loads the language model)');
    try {
        const r = await api('/v1/video/prompt', 'POST', body);
        const parts = [tx('Written by {llm} in {s} s', { llm: r.llm, s: r.runtime_seconds })];
        if (body.image_base64) parts.push(r.used_image ? tx('it looked at the start picture') : tx('it could not see the start picture – written from the words only'));
        const note = parts.join(' · ') + '. ' + tx('This text is what the video model gets.');
        VIDEO.written = { source: promptSource(), note };
        document.getElementById('video-final-prompt').value = r.prompt;
        document.getElementById('video-final-group').hidden = false;
        markWrittenStale();
        status.textContent = tx('Done after {s} s.', { s: Math.round((performance.now() - t0) / 1000) });
    } catch (err) {
        status.textContent = tx('Failed: {err}', { err: err.message });
    }
    e.target.disabled = false;
});

document.getElementById('video-final-clear').addEventListener('click', clearWrittenPrompt);
for (const id of ['video-prompt', 'video-dialogue', 'video-sound']) {
    document.getElementById(id).addEventListener('input', markWrittenStale);
}

async function loadVideoModels() {
    const sel = document.getElementById('video-model');
    try {
        const r = await api('/v1/video/models');
        VIDEO.models = (r.data || []).filter(m => m.available);
    } catch (err) {
        VIDEO.models = [];
        document.getElementById('video-status').textContent = tx('No video models: {err}', { err: err.message });
    }
    const keep = sel.value;
    sel.innerHTML = '';
    for (const m of VIDEO.models) {
        const kind = needsPicture(m) ? tx('picture → video') : takesPicture(m) ? tx('text or picture → video') : tx('text → video');
        sel.append(new Option(`${m.label} (${kind})`, m.id));
    }
    if (keep && VIDEO.models.some(m => m.id === keep)) sel.value = keep;
    if (!VIDEO.models.length) sel.append(new Option(tx('no video model installed'), ''));
    fillVideoForm();
}

function showVideoPicture(b64, note) {
    VIDEO.image = b64;
    const img = document.getElementById('video-image-preview');
    img.src = `data:image/png;base64,${b64}`;
    img.hidden = false;
    document.getElementById('video-image-clear').hidden = false;
    document.getElementById('video-image-note').textContent = note;
    markWrittenStale();
}

document.getElementById('video-image-clear').addEventListener('click', () => {
    VIDEO.image = null;
    document.getElementById('video-image-preview').hidden = true;
    document.getElementById('video-image-clear').hidden = true;
    document.getElementById('video-image-file').value = '';
    document.getElementById('video-image-note').textContent = tx('Any picture works – a photo or one made with AI.');
    markWrittenStale();
});

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result).split(',')[1]);
        r.onerror = () => reject(r.error);
        r.readAsDataURL(file);
    });
}

document.getElementById('video-model').addEventListener('change', fillVideoForm);
document.getElementById('video-lora-strength').addEventListener('input', (e) => {
    document.getElementById('video-lora-strength-value').textContent = Number(e.target.value).toFixed(2);
});

document.getElementById('video-image-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) { showToast(tx('The picture may be at most 20 MB.'), 'warn'); return; }
    showVideoPicture(await fileToBase64(file), tx('Uploaded: {name}', { name: file.name }));
});

document.getElementById('video-image-generate').addEventListener('click', async (e) => {
    const prompt = document.getElementById('video-prompt').value.trim();
    if (!prompt) { showToast(tx('Describe the scene first.'), 'warn'); return; }
    const res = document.getElementById('video-resolution').value || '832x480';
    const [w, h] = res.split('x').map(Number);
    const aspect = w > h ? '16:9' : w < h ? '9:16' : '1:1';
    e.target.disabled = true;
    document.getElementById('video-image-note').textContent = tx('Creating the start picture…');
    try {
        const r = await api('/v1/images/generations', 'POST', { prompt, aspect_ratio: aspect, response_format: 'b64_json' });
        showVideoPicture(r.data[0].b64_json, tx('Created with {model}', { model: r.model || tx('the default image model') }));
    } catch (err) {
        document.getElementById('video-image-note').textContent = tx('Failed: {err}', { err: err.message });
    }
    e.target.disabled = false;
});

function videoBusy(on) {
    document.getElementById('video-run-btn').disabled = on;
    document.getElementById('video-cancel-btn').hidden = !on;
    document.getElementById('video-progress').hidden = !on;
}

async function pollVideo() {
    const status = document.getElementById('video-status');
    try {
        const st = await api(`/v1/video/jobs/${encodeURIComponent(VIDEO.job)}`);
        const secs = Math.round((performance.now() - VIDEO.started) / 1000);
        document.getElementById('video-progress').value = st.progress || 0;
        if (st.status === 'done') {
            const f = st.result.files[0];
            const player = document.getElementById('video-player');
            player.src = f.url;
            player.hidden = false;
            const dl = document.getElementById('video-download');
            dl.href = f.url;
            dl.download = f.name;
            dl.hidden = false;
            const d = st.result.data || {};
            const parts = [d.model, d.resolution, d.seconds != null ? `${d.seconds} s` : null,
                           d.runtime_seconds != null ? tx('made in {s} s', { s: d.runtime_seconds }) : null,
                           d.steps != null ? tx('{n} steps', { n: d.steps }) : null, d.seed != null ? `seed ${d.seed}` : null, d.lora ? `LoRA ${d.lora} (${d.lora_strength})` : null,
                           d.sage_attention ? 'SageAttention' : null, VIDEO.usedWritten ? tx('written prompt') : null];
            document.getElementById('video-meta').textContent = parts.filter(Boolean).join(' · ');
            status.textContent = tx('Done after {s} s.', { s: secs });
            addLog(tx('Video ready: {model}', { model: d.model || '' }), 'info');
            VIDEO.job = null;
            videoBusy(false);
            return;
        }
        if (st.status === 'failed' || st.status === 'cancelled') {
            status.textContent = st.status === 'failed' ? tx('Failed: {err}', { err: st.error || '' }) : tx('Cancelled.');
            VIDEO.job = null;
            videoBusy(false);
            return;
        }
        status.textContent = `${st.message || tx('waiting')} · ${secs} s`;
    } catch (err) {
        status.textContent = tx('Status not reachable: {err}', { err: err.message });
    }
    VIDEO.timer = setTimeout(pollVideo, 2000);
}

document.getElementById('video-run-btn').addEventListener('click', async () => {
    const m = videoModel();
    const status = document.getElementById('video-status');
    const written = document.getElementById('video-final-group').hidden ? '' : document.getElementById('video-final-prompt').value.trim();
    const prompt = written || document.getElementById('video-prompt').value.trim();
    if (!m) { status.textContent = tx('No video model installed.'); return; }
    if (!prompt) { status.textContent = tx('Describe what happens in the video.'); return; }
    if (needsPicture(m) && !VIDEO.image) { status.textContent = tx('This model needs a start picture – upload or create one.'); return; }
    const body = { model: m.id, prompt, seconds: Number(document.getElementById('video-seconds').value),
                   resolution: document.getElementById('video-resolution').value };
    const seed = document.getElementById('video-seed').value;
    if (seed !== '') body.seed = Number(seed);
    const neg = document.getElementById('video-negative').value.trim();
    if (neg) body.negative_prompt = neg;
    const lora = document.getElementById('video-lora').value;
    if (lora) { body.lora = lora; body.lora_strength = Number(document.getElementById('video-lora-strength').value); }
    const steps = Number(document.getElementById('video-steps').value);
    if (m.steps && Number.isInteger(steps) && steps !== m.steps.default) body.steps = steps;
    if (typeof m.sage_attention === 'boolean') body.sage_attention = document.getElementById('video-sage').checked;
    if (takesPicture(m) && VIDEO.image) body.image_base64 = VIDEO.image;
    VIDEO.usedWritten = !!written;
    videoBusy(true);
    document.getElementById('video-player').hidden = true;
    document.getElementById('video-download').hidden = true;
    document.getElementById('video-meta').textContent = '';
    status.textContent = tx('Starting… (the first video loads the model, about a minute more)');
    try {
        const r = await api('/v1/video/jobs', 'POST', body);
        VIDEO.job = r.id;
        VIDEO.started = performance.now();
        pollVideo();
    } catch (err) {
        status.textContent = tx('Failed: {err}', { err: err.message });
        videoBusy(false);
    }
});

document.getElementById('video-cancel-btn').addEventListener('click', async () => {
    if (!VIDEO.job) return;
    try { await api(`/v1/video/jobs/${encodeURIComponent(VIDEO.job)}/cancel`, 'POST'); } catch (err) { /* polling shows it */ }
});

document.querySelector('[data-tab="video"]').addEventListener('click', async () => {
    await loadPromptOptions();
    if (!VIDEO.models.length) loadVideoModels();
    else fillVideoForm();
});
