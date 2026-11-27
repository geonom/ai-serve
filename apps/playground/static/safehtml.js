// Text that goes into an HTML string (2026-11-27: chat answers, reasoning, file and model names went into innerHTML
// unescaped -- markup in a model's answer would have run in the page). Prefer textContent; where HTML is built from
// strings, every outside value goes through this. Pure (tests: tests/safehtml.test.mjs).
function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

if (typeof module !== 'undefined' && module.exports) module.exports = escapeHtml;
