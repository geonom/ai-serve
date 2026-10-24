// The Bild tab's "Steps" field follows the chosen model's catalog entry (GET /v1/images/models: steps = {default,
// adjustable, max}). Found 2026-10-22 (owner): Iris took the same time for 4, 16 and 24 steps -- it has a fixed 4-step
// schedule, and the field was simply not sent for gateway models. Pure (tests: tests/imagesteps.test.mjs).
const ImageSteps = (() => {
    // what the field shows for a model; steps: the catalog's object, or undefined (unknown model: free, up to 100)
    function field(steps) {
        if (!steps) return { disabled: false, value: null, placeholder: tx('default'), max: 100, note: '' };
        if (steps.adjustable === false) {
            return { disabled: true, value: steps.default, placeholder: String(steps.default), max: steps.default,
                     note: tx('This model has a fixed {n}-step schedule – other step counts aren\'t possible.', { n: steps.default }) };
        }
        return { disabled: false, value: null, placeholder: tx('default {n}', { n: steps.default }),
                 max: steps.max || 100, note: '' };
    }
    // what to send: nothing for a fixed schedule or an empty field; else the number, kept within the model's max
    function request(steps, raw) {
        if (steps && steps.adjustable === false) return null;
        const n = parseInt(raw, 10);
        if (!Number.isFinite(n) || n < 1) return null;
        return Math.min(n, (steps && steps.max) || 100);
    }
    return { field, request };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = ImageSteps;
