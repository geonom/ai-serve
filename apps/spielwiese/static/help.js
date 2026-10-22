// Help per page (owner, 2026-10-20): what a page can do, a screenshot of it in use, and first steps. Opens by
// itself the first time someone opens a page (remembered in their browser; if the browser keeps nothing, once
// per visit), and again from the "?" in the bar. Screenshots: static/help/<tab>.jpg.
// Pure part (tests: node --test tests/help.test.mjs): which pages have help, and when it opens by itself.

// English; the German texts are in i18n.de.js (shown through tx())
const HELP = {
    overview: {
        title: "Welcome to the Spielwiese",
        intro: "The Spielwiese is a place to try things out: chat with language models, create and edit images, have texts read out and transcribe recordings, read documents – everything runs on private machines that were added to the network.",
        caption: "The overview shows what the gateway offers right now.",
        steps: [
            "Click a card on the overview – or pick a page in the bar at the top.",
            "Every page explains itself the first time you open it; the “?” at the top right shows the explanation again.",
            "The first time, a model takes a few seconds to minutes to load – after that it's faster.",
        ],
        tip: "Greyed-out cards are things the gateway can't do right now – e.g. because a service isn't running.",
    },
    gateway: {
        title: "Chat",
        intro: "Write with language models – with all the ones the gateway reaches: on the machines in the network and on more model servers like Strata. The chosen model is loaded automatically when you first write.",
        caption: "A conversation with Gemma 4; model and settings on the left.",
        steps: [
            "Choose one under “Model” on the left – or leave it at “Auto”, then the one that's running answers.",
            "Write a message at the bottom and press “Send”.",
            "Models marked “image” understand images too: add one under “Attach an image or audio”.",
            "“Thinking” models think first; the reasoning can be expanded below the answer.",
        ],
        tip: "For answers in a fixed shape (JSON), enter a schema – “Build a schema” helps with that.",
    },
    texts: {
        title: "Compare texts",
        intro: "How similar texts are in meaning – and which texts fit a question best. That's the basis of every search “by meaning instead of by words”.",
        caption: "Dog and puppy are close together, stock market and shares too – across, hardly at all.",
        steps: [
            "Enter texts at the top left, one per line, and press “Compare”.",
            "On the right, the table shows how close each pair of texts is (1 = same meaning).",
            "At the bottom left, enter a question and several texts and press “Sort”.",
            "The rerank model scores each text together with the question – the best fit is at the top.",
        ],
        tip: "Works across languages: a German and an English sentence with the same meaning are close together.",
    },
    docs: {
        title: "Read documents",
        intro: "Have a photo or PDF transcribed – as clean text with headings and tables, or pull out specific details (e.g. the amount and date of an invoice).",
        caption: "A page transcribed as Markdown.",
        steps: [
            "Choose a file: an image or a PDF (up to 50 pages, every page is read on its own).",
            "Under “Kind”, choose: transcribe as Markdown, free content, or “Fields from a template”.",
            "For fields, give a template, e.g. {\"amount\": \"number\", \"date\": \"string\"}.",
            "Press “Read”; the result can be copied on the right.",
        ],
        tip: "If the same numbers appear in a table and in running text, say in “Instructions” where they should come from.",
    },
    "image-gen": {
        title: "Create images",
        intro: "Create images from a description, change an existing image by instruction, or transfer the style of one image onto another.",
        caption: "An image with Z-Image Turbo, the fast default model.",
        steps: [
            "Choose the kind at the top: “Generate”, “Edit” (change an image) or “Transfer style”.",
            "Choose a model – the description below says what it's good at.",
            "Describe what should be in the picture (English usually works best), pick an aspect ratio, “Generate image”.",
            "Every image also ends up under “My images”.",
        ],
        tip: "Same seed + same prompt = same image. That way you can try out single settings on purpose.",
    },
    layout: {
        title: "Image with positions",
        intro: "Decide what goes where in the image: draw boxes and describe them – including lettering that appears in the image exactly as written (MOSS-Image).",
        caption: "Two boxes: a sign with text at the top, a figure at the bottom left.",
        steps: [
            "Under “Whole image”, describe what the image shows overall.",
            "Draw a box on the canvas and enter what goes in it – for “Text”, the exact wording.",
            "Add more boxes; “Delete all boxes” starts over.",
            "Choose format and variant (Turbo is faster) and press “Generate image”.",
        ],
        tip: "“API call” shows how to request the same image from your own program.",
    },
    compare: {
        title: "Compare models",
        intro: "Send the same prompt with the same seed through several image models – also with different step counts – and compare the results side by side or on top of each other.",
        caption: "A fox in the snow with three models.",
        steps: [
            "Enter a prompt and pick an aspect ratio.",
            "Tick models; if you like, give several step counts or seeds.",
            "“Start comparison” – models on different graphics cards compute at the same time.",
            "In the result, switch between the images or blend them with the slider.",
        ],
        tip: "The comparison keeps running on the server even if you close the page; earlier ones are listed below.",
    },
    mine: {
        title: "My images",
        intro: "Everything you've made on the Spielwiese – with prompt, model and seed, visible only to you.",
        caption: "The gallery, filterable by model.",
        steps: [
            "Filter by model at the top, or show all.",
            "Click an image to see it large – with prompt, seed and duration.",
            "With the prompt and seed, an image can be made again exactly under “Create images”.",
        ],
        tip: "",
    },
    speech: {
        title: "Speech",
        intro: "Have text read out – with built-in voices, described voices or a voice cloned from a recording – and transcribe recordings, with separated speakers if you like.",
        caption: "“Speak” with CosyVoice 3; on the left, the fields adapt to the chosen engine.",
        steps: [
            "Choose “Speak” or “Transcribe” at the top.",
            "Choose an engine and the kind – the form shows what it needs (recording, style, speaker …).",
            "To speak, enter the text on the right and press “Speak”; the result can be downloaded.",
            "To transcribe, choose or upload a recording; the result is also available as subtitles (SRT).",
        ],
        tip: "Sliders (“adjust”) only apply when ticked – otherwise the engine's default applies.",
    },
    quota: {
        title: "Quotas",
        intro: "How much compute time and how many images guests may use per day. You yourself aren't limited.",
        caption: "",
        steps: [
            "Enter the defaults for all guests at the top; empty = unlimited.",
            "Below, give single people their own values.",
            "“Save” – the counters start again every day at midnight.",
        ],
        tip: "",
    },
};

const HelpRules = (() => {
    const KEY = 'spielwiese-help-seen';
    // Open by itself only on a page with help that this browser hasn't seen yet.
    function autoOpen(tab, seen) {
        return !!HELP[tab] && !seen.includes(tab);
    }
    function markSeen(tab, seen) {
        return seen.includes(tab) ? seen : [...seen, tab];
    }
    return { KEY, autoOpen, markSeen, pages: () => Object.keys(HELP) };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = { HELP, HelpRules };

if (typeof document !== 'undefined' && document.getElementById('help-dialog')) (() => {
    const dlg = document.getElementById('help-dialog');
    let memory = [];   // when the browser keeps nothing: once per visit
    const readSeen = () => { try { return JSON.parse(localStorage.getItem(HelpRules.KEY) || '[]'); } catch (e) { return memory; } };
    const writeSeen = list => { memory = list; try { localStorage.setItem(HelpRules.KEY, JSON.stringify(list)); } catch (e) { /* kept in memory */ } };

    function show(tab) {
        const h = HELP[tab];
        if (!h) return;
        document.getElementById('help-title').textContent = tx(h.title);
        document.getElementById('help-intro').textContent = tx(h.intro);
        const fig = document.getElementById('help-figure'), img = document.getElementById('help-img');
        fig.hidden = !h.caption;
        if (h.caption) {
            img.src = `help/${tab}.jpg?v=2`;
            img.alt = tx(h.caption);
            img.onerror = () => { fig.hidden = true; };
            document.getElementById('help-caption').textContent = tx(h.caption);
        }
        const ol = document.getElementById('help-steps');
        ol.replaceChildren(...h.steps.map(s => { const li = document.createElement('li'); li.textContent = tx(s); return li; }));
        document.getElementById('help-tip').textContent = h.tip ? tx('Tip: {tip}', { tip: tx(h.tip) }) : '';
        if (!dlg.open) dlg.showModal();
        writeSeen(HelpRules.markSeen(tab, readSeen()));
    }

    const current = () => (document.querySelector('.tab-link.active') || {}).dataset?.tab;
    const maybe = tab => { if (HelpRules.autoOpen(tab, readSeen())) show(tab); };
    document.getElementById('help-btn').addEventListener('click', () => show(current()));
    document.querySelectorAll('.tab-link[data-tab]').forEach(l => l.addEventListener('click', () => maybe(l.dataset.tab)));
    dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });   // click outside closes
    // the page the app opened on (app.js restores the last one before this runs)
    setTimeout(() => maybe(current()), 300);
})();
