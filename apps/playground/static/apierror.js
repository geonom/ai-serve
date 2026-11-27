// A readable line out of a gateway's error answer (2026-11-26: FastAPI's field errors are a list, which showed as
// "[object Object]"), and what to tell a visitor (2026-11-27: "needs 18271 MB, 6427 MB free …" in a toast that
// disappeared, while the result area just reset). Pure (tests: tests/apierror.test.mjs).
const _tx = (typeof tx === 'function') ? tx : (s, v) => s.replace(/\{(\w+)\}/g, (_, k) => (v || {})[k] ?? `{${k}}`);

function errorText(body, status) {
    const d = body && (body.detail !== undefined ? body.detail : body.error);
    if (typeof d === 'string' && d) return d;
    if (Array.isArray(d) && d.length) {
        return d.map(e => {
            if (!e || typeof e !== 'object') return String(e);
            const where = Array.isArray(e.loc) ? e.loc.filter(p => p !== 'body' && p !== 'query').join('.') : '';
            return where ? `${where}: ${e.msg}` : String(e.msg || '');
        }).join('; ');
    }
    if (d && typeof d === 'object' && typeof d.message === 'string') return d.message;
    return `HTTP error ${status}`;
}

// {text: what a visitor reads, details: the technical message (for the log), retry: whether trying again later helps}
function friendly(status, text) {
    const details = String(text || '');
    if (status === 507) return { text: _tx('The graphics cards are busy right now – please try again in a minute.'), details, retry: true };
    if (status === 503) return { text: _tx('This part is starting or not running – please try again in a moment.'), details, retry: true };
    if (status === 409) return { text: _tx('Busy with another request – please try again shortly.'), details, retry: true };
    if (status >= 500) return { text: _tx('Something went wrong on the server.'), details, retry: true };
    return { text: details || _tx('The request was refused.'), details, retry: false };
}

errorText.friendly = friendly;
if (typeof module !== 'undefined' && module.exports) { module.exports = errorText; module.exports.friendly = friendly; }
