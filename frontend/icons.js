// Icons and categories for apps (manifest fields `icon`, `category`). Manifests only
// name an icon; the page draws it from this fixed set -- no markup from other repos.

const APP_ICONS = {
    speaker: 'M4 9v6h4l5 4V5L8 9zM16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12',
    mic: 'M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5 11a7 7 0 0 0 14 0M12 18v3',
    image: 'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM21 17l-5-5-9 8M9 8a2 2 0 1 1 0 4 2 2 0 0 1 0-4z',
    book: 'M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 19V5M8 7h7',
    branch: 'M6 3a2 2 0 1 1 0 4 2 2 0 0 1 0-4zM6 17a2 2 0 1 1 0 4 2 2 0 0 1 0-4zM18 6a2 2 0 1 1 0 4 2 2 0 0 1 0-4zM6 7v10M18 10c0 4-6 3-12 7',
    crown: 'M3 8l4.5 4L12 5l4.5 7L21 8l-2 11H5z',
    waves: 'M2 9c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0M2 15c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0',
    bubbles: 'M4 5h9v7H8l-4 3zM13 9h7v7l-3-2h-4',
    magnifier: 'M11 5a6 6 0 1 1 0 12 6 6 0 0 1 0-12zM20 20l-4.5-4.5',
    archive: 'M3 4h18v4H3zM5 8v11h14V8M10 12h4',
    bars: 'M5 20V11M12 20V5M19 20v-7',
    cloud: 'M7 18h10a4 4 0 0 0 .5-8A6 6 0 0 0 6 9.5 4.3 4.3 0 0 0 7 18z',
    chat: 'M4 5h16v11H8l-4 4z',
    trend: 'M3 17l6-6 4 4 8-8M15 7h6v6',
    code: 'm8 7-5 5 5 5M16 7l5 5-5 5',
    music: 'M9 18V5l11-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0zM20 16a3 3 0 1 1-6 0 3 3 0 0 1 6 0z',
    video: 'M3 6h12v12H3zM15 10l6-3v10l-6-3',
    globe: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18',
    puzzle: 'M10 3h4v3a2 2 0 1 0 4 0V6h3v5h-3a2 2 0 1 0 0 4h3v6h-6v-3a2 2 0 1 0-4 0v3H4v-6h3a2 2 0 1 0 0-4H4V6h6z',
    game: 'M6 9h12a4 4 0 0 1 0 8H6a4 4 0 0 1 0-8zM8 11v4M6 13h4M16 12h.01M18 14h.01',
    camera: 'M4 7h4l2-3h4l2 3h4v13H4zM12 10a4 4 0 1 1 0 8 4 4 0 0 1 0-8z',
    pen: 'M4 20l4-1 11-11-3-3L5 16zM14 6l3 3',
    map: 'M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3zM9 3v15M15 6v15',
    heart: 'M12 20s-8-5-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 9c0 6-8 11-8 11z',
    star: 'M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z',
    tool: 'M14 6a4 4 0 0 0 5 5l-9 9a2 2 0 0 1-3-3l9-9a4 4 0 0 0-2-2z',
    sun: 'M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
    home: 'M3 10.5 12 3l9 7.5V21h-6v-6H9v6H3z',
};

const APP_CATEGORIES = {
    ai: { label: 'KI-Backends', hint: 'starten bei Bedarf von selbst', tint: 'var(--cat-ai-bg)', ink: 'var(--cat-ai)' },
    stories: { label: 'Geschichten & Spiele', hint: '', tint: 'var(--cat-stories-bg)', ink: 'var(--cat-stories)' },
    learning: { label: 'Lernen', hint: '', tint: 'var(--cat-learning-bg)', ink: 'var(--cat-learning)' },
    knowledge: { label: 'Wissen', hint: '', tint: 'var(--cat-knowledge-bg)', ink: 'var(--cat-knowledge)' },
    tools: { label: 'Werkzeuge', hint: '', tint: 'var(--cat-tools-bg)', ink: 'var(--cat-tools)' },
    integrations: { label: 'Integrationen', hint: 'Starten öffnet ihren Zugang', tint: 'var(--cat-integrations-bg)', ink: 'var(--cat-integrations)' },
};

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgIcon(path, size = 20, strokeWidth = 1.8) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', strokeWidth);
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('d', path);
    svg.append(p);
    return svg;
}

// Icon tile for an app: its icon, or its first letter when it names none.
function appTile(s, size = 48) {
    const cat = APP_CATEGORIES[s.category] || APP_CATEGORIES.tools;
    const tile = el('div', { class: 'app-tile', style: `width:${size}px;height:${size}px;background:${cat.tint};color:${cat.ink}` });
    if (s.icon && APP_ICONS[s.icon]) tile.append(svgIcon(APP_ICONS[s.icon], Math.round(size / 2)));
    else tile.append(el('span', { class: 'app-tile-letter', text: (s.name || '?').trim().charAt(0).toUpperCase() }));
    return tile;
}

// How far an app is (manifest "maturity"): a small pill, nothing when unrated.
const MATURITY = {
    alpha: ['Alpha', 'pill-maturity-alpha', 'Funktioniert, aber noch roh'],
    beta: ['Beta', 'pill-maturity-beta', 'Benutzbar, wird gerade getestet'],
    stable: ['Getestet', 'pill-maturity-stable', 'Vollständig getestet'],
};

function maturityPill(s) {
    const m = MATURITY[s.maturity];
    return m ? el('span', { class: `pill ${m[1]}`, text: m[0], title: m[2] }) : null;
}

// Loaded images are kept and reused, so the 5 s refresh of a page doesn't restart
// (and never finish) their download.
const appShotCache = new Map();

function appScreenshot(s, href = null) {
    let img = appShotCache.get(s.id);
    if (!img) {
        img = el('img', { src: `/api/services/${encodeURIComponent(s.id)}/screenshot`, alt: `Screenshot von ${s.name}`, decoding: 'async' });
        img.addEventListener('error', () => { appShotCache.delete(s.id); img.closest('.app-shot')?.remove(); });
        appShotCache.set(s.id, img);
    }
    img.remove();
    const wrap = href ? el('a', { class: 'app-shot', href, target: '_blank', rel: 'noopener' }, img)
                      : el('div', { class: 'app-shot' }, img);
    return wrap;
}

function fmtGb(mb) {
    if (mb === null || mb === undefined) return '?';
    return (mb / 1024).toFixed(1).replace('.', ',');
}
