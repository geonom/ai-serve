// A read document shown as a document (2026-11-27: NuExtract's markdown with HTML tables came out as raw
// <table><tr><td> text). Everything is escaped first; only bare, attribute-free table/text tags are let back in,
// plus a little markdown (headings, **bold**, "- " lists, paragraphs, page breaks). Nothing else can reach the page
// as markup. Pure (tests: tests/docsrender.test.mjs); the copy button keeps the raw text.
function renderDocument(text) {
    const esc = s => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const TAGS = /&lt;(\/?)(table|thead|tbody|tfoot|tr|th|td|b|strong|i|em|u|br|p|ul|ol|li|h[1-6])\s*\/?&gt;/gi;
    const allow = s => esc(s).replace(TAGS, (_, slash, tag) => `<${slash}${tag.toLowerCase()}>`);
    const inline = s => allow(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    const blocks = String(text || '').replace(/\r\n/g, '\n').replace(/<!--\s*PAGE BREAK\s*-->/g, '\n\n\u0000\n\n')
        .split(/\n\s*\n/).map(b => b.trim()).filter(Boolean);
    return blocks.map(b => {
        if (b === '\u0000') return '<hr>';
        if (/^<table\b/i.test(b)) return allow(b.replace(/\n/g, ''));
        const h = b.match(/^(#{1,4})\s+(.+)$/);
        if (h && !b.includes('\n')) return h[1].length === 1 ? `<h3>${inline(h[2])}</h3>` : `<h4>${inline(h[2])}</h4>`;
        const lines = b.split('\n');
        if (lines.every(l => /^\s*[-*]\s+/.test(l))) return `<ul>${lines.map(l => `<li>${inline(l.replace(/^\s*[-*]\s+/, ''))}</li>`).join('')}</ul>`;
        return `<p>${lines.map(inline).join('<br>')}</p>`;
    }).join('');
}

if (typeof module !== 'undefined' && module.exports) module.exports = renderDocument;
