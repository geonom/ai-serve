// What a finished picture can be used for next (2026-11-26, owner: generated pictures should be at hand for edits
// and the like). A result on the Create images page, a picture in "My images" and every picture field (edit, style,
// video start picture, enlarge, cut out) share these rules; the buttons and the "From My images" chooser are in
// pictures.js. Pure (tests: tests/pictureuse.test.mjs).
const PictureUse = (() => {
    const MAX_OUTPUT = 4096 * 4096;   // the gateway's cap for an enlarged picture (image-serve UPSCALE_MAX_OUTPUT_PIXELS)
    const ORDER = [
        ['edit', 'edit', 'Edit it'],
        ['edit-extra', 'edit', 'Add to an edit'],
        ['style', 'style', 'Use as a style'],
        ['video', 'video', 'Make a video'],
        ['upscale', 'upscale', 'Enlarge'],
        ['cutout', 'cutout', 'Cut out'],
    ];
    const KEEP = ['prompt', 'model', 'label', 'steps', 'seed', 'aspect_ratio', 'seconds'];
    const SOURCE = { edit: 'edit', style: 'style', upscale: 'upscale', cutout: 'cutout' };

    // offers: {edit, style, video, upscale, cutout} -- what the gateway can do right now; madeBy: the step that made it
    function actions(offers, madeBy) {
        const o = offers || {};
        return ORDER.filter(([id, need]) => o[need] && !(id === 'cutout' && madeBy === 'cutout'))
                    .map(([id, , label]) => ({ id, label: tx(label) }));
    }
    // which factors keep the result within the gateway's limit (size unknown -> both; the gateway decides)
    function upscaleFactors(w, h) {
        if (!(w > 0 && h > 0)) return [2, 4];
        return [2, 4].filter(f => w * f * h * f <= MAX_OUTPUT);
    }
    function galleryMeta(mode, info) {
        const out = { source: SOURCE[mode] || 'bild' };
        for (const k of KEEP) {
            const v = (info || {})[k];
            if (v !== null && v !== undefined && v !== '') out[k] = v;
        }
        return out;
    }
    const b64 = (s) => String(s || '').replace(/^data:[^,]*,/, '');
    const dataUrl = (s) => String(s || '').startsWith('data:') ? s : `data:image/png;base64,${s}`;
    return { actions, upscaleFactors, galleryMeta, b64, dataUrl };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = PictureUse;
