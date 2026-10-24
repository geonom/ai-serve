/* i18n.js -- English is the source language of this UI; German lives in i18n.de.js (a copy of ai-serve's frontend/i18n.js; keep them in step) as a map English -> German.
 *   tx("3 models", ...)       -> JS-made text:  tx("{n} models", {n: 3})  (tx, not t: t is a common local name)
 *   static HTML               -> translated once at load: text nodes and placeholder/title/aria-label whose text is
 *                                a key, text-field values likewise, and elements marked data-i18n by their whole
 *                                (whitespace-normalised) innerHTML
 * The choice (EN | DE switcher, top right) is kept per browser; default English. Switching reloads the page. */
(function () {
    const KEY = "ai-serve-lang";
    let lang = "en";
    try { lang = localStorage.getItem(KEY) === "de" ? "de" : "en"; } catch (e) { /* storage blocked, or node (tests) */ }
    const DE = {};
    const norm = (s) => s.replace(/\s+/g, " ").trim();
    const fill = (s, vars) => vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m)) : s;

    function t(s, vars) {
        const out = lang === "de" && Object.prototype.hasOwnProperty.call(DE, s) ? DE[s] : s;
        return fill(out, vars);
    }

    function apply(root) {
        if (lang !== "de" || !root || typeof document === "undefined") return;
        root.querySelectorAll("[data-i18n]").forEach((el) => {
            const k = norm(el.innerHTML);
            if (Object.prototype.hasOwnProperty.call(DE, k)) el.innerHTML = DE[k];
        });
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        for (const n of nodes) {
            const p = n.parentElement;
            if (!p || p.closest("script, style, code, pre, [data-i18n], [data-no-i18n]")) continue;
            const k = norm(n.nodeValue);
            if (k && Object.prototype.hasOwnProperty.call(DE, k)) {
                const lead = n.nodeValue.match(/^\s*/)[0], trail = n.nodeValue.match(/\s*$/)[0];
                n.nodeValue = lead + DE[k] + trail;
            }
        }
        root.querySelectorAll('input[type="text"][value]').forEach((el) => {   // demo values in text fields
            const v = norm(el.value);
            if (v && Object.prototype.hasOwnProperty.call(DE, v)) el.value = DE[v];
        });
        root.querySelectorAll("[placeholder], [title], [aria-label]").forEach((el) => {
            for (const a of ["placeholder", "title", "aria-label"]) {
                const v = el.getAttribute(a);
                if (v && Object.prototype.hasOwnProperty.call(DE, norm(v))) el.setAttribute(a, DE[norm(v)]);
            }
        });
    }

    function setLang(l) {
        try { localStorage.setItem(KEY, l); } catch (e) { /* storage blocked: stays for this page only */ }
        location.reload();
    }

    function switcher() {
        const box = document.createElement("div");
        box.className = "lang-switch";
        box.setAttribute("role", "group");
        box.setAttribute("aria-label", "Language");
        for (const [code, label] of [["en", "EN"], ["de", "DE"]]) {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            b.lang = code;
            b.setAttribute("aria-pressed", String(code === lang));
            b.title = code === "en" ? "English" : "Deutsch";
            if (code !== lang) b.addEventListener("click", () => setLang(code));
            box.appendChild(b);
        }
        return box;
    }

    if (typeof document !== "undefined") document.documentElement.lang = lang;
    const g = typeof window !== "undefined" ? window : globalThis;   // node (unit tests): English, no DOM
    g.I18N = {
        lang, t, apply, setLang, switcher,
        add(dict) { Object.assign(DE, dict); },
        locale: lang === "de" ? "de-DE" : "en-US",
        has(s) { return Object.prototype.hasOwnProperty.call(DE, s); },
    };
    g.tx = t;
})();
