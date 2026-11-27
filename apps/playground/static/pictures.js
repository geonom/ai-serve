// Pictures made here, at hand everywhere (2026-11-26, owner): every picture field has "Choose from My images", and a
// finished picture (a result on Create images, a picture in My images) offers its next steps -- edit it, add it to an
// edit, use it as a style, make a video, enlarge, cut out. Rules in pictureuse.js; DOM via textContent only.
// Needs app.js (setEditPicture, addEditExtraPictures, setStyleContent, setStyleRef, setComfyMode), compare.js
// (el, sw, imgUrl, mine) and video.js (showVideoPicture, VIDEO).

const offers = { edit: false, style: false, video: false, upscale: false, cutout: false };

window.addEventListener('pg-image-models', (e) => {
    const usable = rows => (rows || []).some(m => m.available !== false);
    offers.edit = usable(e.detail.edit);
    offers.style = usable(e.detail.style);
    offers.upscale = usable(e.detail.upscale);
    offers.cutout = usable(e.detail.cutout);
    refreshResultActions();
});
window.addEventListener('pg-video-models', (e) => {
    videoAsk.known = true;
    offers.video = (e.detail || []).some(m => (m.tasks || []).includes('image_to_video'));
    refreshResultActions();
});

function openTab(id) {
    const link = document.querySelector(`.tab-link[data-tab="${id}"]`);
    if (link) link.click();
}

// put a picture (data URL) where the chosen step wants it, then go there
function usePicture(action, dataUrl, name, prompt) {
    if (!$id('sw-lightbox').hidden) closeLightbox();
    if (action === 'video') {
        openTab('video');
        showVideoPicture(PictureUse.b64(dataUrl), tx('From My images: {name}', { name }));
        document.getElementById('video-image-group').scrollIntoView({ block: 'nearest' });
        return;
    }
    openTab('image-gen');
    if (action === 'edit') { setComfyMode('edit'); setEditPicture(dataUrl, name, prompt); }
    else if (action === 'upscale' || action === 'cutout') { setComfyMode(action); setEditPicture(dataUrl, name, prompt); }
    else if (action === 'edit-extra') { setComfyMode('edit'); addEditExtraPictures([dataUrl]); }
    else if (action === 'style') { setComfyMode('style'); setStyleRef(dataUrl); }
    else if (action === 'style-content') { setComfyMode('style'); setStyleContent(dataUrl); }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
const $id = (id) => document.getElementById(id);

function actionButtons(box, madeBy, getPicture) {
    askVideoModels();
    box.replaceChildren();
    const list = PictureUse.actions(offers, madeBy);
    if (!list.length) { box.hidden = true; return; }
    box.append(el('span', { class: 'muted small', text: tx('Use this picture:') }));
    for (const a of list) {
        box.append(el('button', { type: 'button', class: 'btn btn-secondary btn-sm', text: a.label,
            onclick: async () => {
                try { const { url, name, prompt } = await getPicture(); usePicture(a.id, url, name, prompt); }
                catch (err) { addLog(tx('Failed: {err}', { err: err.message }), 'error'); }
            } }));
    }
    box.hidden = false;
}

// ------------------------------------------------ under a result on Create images
let resultMadeBy = null;
function showResultActions(madeBy) {
    resultMadeBy = madeBy;
    refreshResultActions();
}
function refreshResultActions() {
    const box = $id('comfy-result-actions');
    if (!box || !resultMadeBy || UI.comfyResultImg.style.display === 'none') { if (box) box.hidden = true; return; }
    actionButtons(box, resultMadeBy, async () => ({ url: UI.comfyResultImg.src, name: tx('the last result'), prompt: lastResultPrompt }));
}

// ------------------------------------------------ in My images' viewer
async function galleryDataUrl(id) {
    const res = await swFetch(imgUrl(id));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = () => reject(r.error);
        r.readAsDataURL(blob);
    });
}
const galleryName = (e) => (e.prompt || e.label || e.model || tx('Image')).slice(0, 60);

function refreshViewerActions(entry) {
    const box = $id('sw-lb-use');
    if (!box) return;
    if (!entry) { box.replaceChildren(); return; }
    actionButtons(box, entry.source, async () => ({ url: await galleryDataUrl(entry.id), name: galleryName(entry), prompt: entry.prompt }));
}

// ------------------------------------------------ "Choose from My images"
const pick = { slot: null, before: null };

async function pickLoad(reset) {
    if (reset) { pick.before = null; $id('pick-grid').replaceChildren(); }
    const q = new URLSearchParams({ limit: '30' });
    if (pick.before) q.set('before', pick.before);
    let data;
    try { data = await sw(`/gallery?${q}`); }
    catch (err) { $id('pick-empty').hidden = false; $id('pick-empty').textContent = tx('Not available: {err}', { err: err.message }); return; }
    const images = data.images || [];
    for (const e of images) {
        $id('pick-grid').append(el('button', { type: 'button', class: 'pick-item', title: galleryName(e),
            onclick: async () => {
                try {
                    const url = await galleryDataUrl(e.id);
                    $id('pick-dialog').close();
                    usePicture(pick.slot, url, galleryName(e), e.prompt);
                } catch (err) { addLog(tx('Failed: {err}', { err: err.message }), 'error'); }
            } }, el('img', { src: imgUrl(e.id), alt: galleryName(e), loading: 'lazy' })));
    }
    pick.before = images.length ? images[images.length - 1].id : pick.before;
    $id('pick-more').hidden = images.length < 30;
    $id('pick-empty').hidden = $id('pick-grid').children.length > 0;
    if ($id('pick-empty').hidden === false) $id('pick-empty').textContent = tx('Nothing here yet – pictures you make in the Playground show up here.');
}

document.querySelectorAll('.pick-mine').forEach(b => b.addEventListener('click', (ev) => {
    ev.preventDefault();
    ev.stopPropagation();   // the buttons sit next to upload zones that open the file chooser on click
    pick.slot = b.dataset.slot;
    $id('pick-dialog').showModal();
    pickLoad(true);
}));
$id('pick-more').addEventListener('click', () => pickLoad(false));
$id('pick-close').addEventListener('click', () => $id('pick-dialog').close());
$id('pick-dialog').addEventListener('click', (ev) => { if (ev.target.id === 'pick-dialog') $id('pick-dialog').close(); });

// video models are loaded by video.js only when its page opens -- ask here so "Make a video" shows from the start;
// a list that didn't come (the plugin busy or starting) is asked for again when buttons are shown, at most every 30 s
const videoAsk = { known: false, at: 0 };
function askVideoModels() {
    if (videoAsk.known || Date.now() - videoAsk.at < 30000) return;
    videoAsk.at = Date.now();
    api('/v1/video/models').then(r => {
        videoAsk.known = true;
        offers.video = (r.data || []).some(m => m.available && (m.tasks || []).includes('image_to_video'));
        refreshResultActions();
        if (!$id('sw-lightbox').hidden && mine.current) refreshViewerActions(mine.current);
    }).catch(() => {});
}
askVideoModels();
