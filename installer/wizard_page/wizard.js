// ai-serve's setup page (installer/wizard.py). The token comes in the URL's fragment and is sent with every API call.
'use strict';

const DE = {
  'Set up ai-serve': 'ai-serve einrichten',
  intro: 'Alles ist freiwillig und lässt sich später ergänzen. Die Haken sind Vorschläge für diesen Rechner.',
  'Language model:': 'Sprachmodell:',
  'Set up': 'Einrichten',
  'Back to the choice': 'Zurück zur Auswahl',
  Details: 'Details',
  'ai-serve is set up and running.': 'ai-serve ist eingerichtet und läuft.',
  'done-hint': 'Beim nächsten Start öffnet ai-serve gleich das Dashboard.',
  'Open the dashboard': 'Dashboard öffnen',
  'none': 'keins',
  'This computer: {m}': 'Dieser Rechner: {m}',
  'Together about {n} GB, {free} GB free on the disk.': 'Zusammen etwa {n} GB, {free} GB frei auf der Platte.',
  'That is more than fits ({keep} GB should stay free).': 'Das ist mehr, als passt ({keep} GB sollen frei bleiben).',
  "Only ai-serve itself and llama.cpp (about 0.5 GB).": 'Nur ai-serve selbst und llama.cpp (etwa 0,5 GB).',
  'licence {l}': 'Lizenz {l}',
  'The setup page lost its key. Start the setup again.': 'Die Einrichtungsseite hat ihren Schlüssel verloren. Starte die Einrichtung neu.',
  'The setup has ended. You can close this tab.': 'Die Einrichtung ist beendet. Du kannst diesen Tab schließen.',
  'Waiting for the setup …': 'Warte auf die Einrichtung …',
  'Setup failed: {e}': 'Die Einrichtung ist fehlgeschlagen: {e}',
  'Starting ai-serve': 'ai-serve wird gestartet',
  "ai-serve's Python environment": 'Die Python-Umgebung von ai-serve',
  'llama.cpp for this machine': 'llama.cpp für diesen Rechner',
  '{done} of {total} MB': '{done} von {total} MB',
  // offers (installer/offers.py)
  'Language model Gemma 4 12B (also reads images)': 'Sprachmodell Gemma 4 12B (liest auch Bilder)',
  'Language model Gemma 4 E4B (also reads images)': 'Sprachmodell Gemma 4 E4B (liest auch Bilder)',
  'Language model Gemma 4 E2B (also reads images)': 'Sprachmodell Gemma 4 E2B (liest auch Bilder)',
  'Speech output (Chatterbox Multilingual)': 'Sprachausgabe (Chatterbox Multilingual)',
  'Fast speech output on the CPU (Piper)': 'Schnelle Sprachausgabe auf dem Prozessor (Piper)',
  'Speech recognition (Whisper, with speaker diarization)': 'Spracherkennung (Whisper, mit Sprechertrennung)',
  'Images from text (Iris-3B Turbo)': 'Bilder aus Text (Iris-3B Turbo)',
  'Image editing + more image models (ComfyUI)': 'Bildbearbeitung und weitere Bildmodelle (ComfyUI)',
  'Reads text aloud in many languages, with a cloned voice': 'Liest Text in vielen Sprachen vor, mit einer geklonten Stimme',
  'Reads text aloud in many languages, with a cloned voice -- through MLX': 'Liest Text in vielen Sprachen vor, mit einer geklonten Stimme – über MLX',
  'Reads text aloud quickly without a GPU (a Raspberry Pi too); fixed voices -- German Thorsten, English LJSpeech, more in Downloads --, no cloning; a separate plugin from github.com/geonom/ai-serve-piper':
    'Liest schnell vor, ohne Grafikkarte (auch auf einem Raspberry Pi); feste Stimmen – Deutsch Thorsten, Englisch LJSpeech, weitere unter Downloads –, kein Klonen; ein eigenes Plugin von github.com/geonom/ai-serve-piper',
  'Images from descriptions in ~5-15 s; English lettering in images comes out clean, German does not':
    'Bilder aus Beschreibungen in etwa 5–15 s; englische Schrift im Bild wird sauber, deutsche nicht',
  'Editing by instruction, style transfer (Qwen-Image-Edit 2511), FLUX.2 [klein] 4B, Z-Image Turbo; MOSS-Image (text in images, positions) as a later download':
    'Bearbeiten nach Anweisung, Stil übertragen (Qwen-Image-Edit 2511), FLUX.2 [klein] 4B, Z-Image Turbo; MOSS-Image (Text im Bild, Positionen) als späterer Download',
  'Z-Image Turbo, editing from up to four images (FLUX.2 [klein] 4B); Qwen-Image-Edit 2511 and MOSS-Image as later downloads':
    'Z-Image Turbo, Bearbeiten aus bis zu vier Bildern (FLUX.2 [klein] 4B); Qwen-Image-Edit 2511 und MOSS-Image als spätere Downloads',
};
// texts with figures in them: [pattern, German]
const DE_PATTERNS = [
  [/^Chat, texts, describing images -- (\w+), 4-bit; runs with llama\.cpp on a GPU, a Mac or the CPU$/,
    'Chat, Texte, Bilder beschreiben – $1, 4 Bit; läuft mit llama.cpp auf einer Grafikkarte, einem Mac oder dem Prozessor'],
  [/^Transcribes recordings \(WhisperX small\)( -- on the CPU)?; telling speakers apart with pyannote needs a Hugging Face token \(HF_TOKEN, see the README\)(; without a token: MOSS-Transcribe or Nemotron from Downloads)?$/,
    (m, cpu, alt) => 'Schreibt Aufnahmen ab (WhisperX small)' + (cpu ? ' – auf dem Prozessor' : '') +
      '; Sprecher zu trennen braucht für pyannote einen Hugging-Face-Schlüssel (HF_TOKEN, siehe README)' +
      (alt ? '; ohne Schlüssel: MOSS-Transcribe oder Nemotron unter Downloads' : '')],
  [/^fits in ([\d.]+) GB on the GPU -- the most capable one here$/, 'passt in $1 GB der Grafikkarte – das fähigste hier'],
  [/^fits in ([\d.]+) GB on the GPU; the 12B needs ~12 GB$/, 'passt in $1 GB der Grafikkarte; das 12B braucht etwa 12 GB'],
  [/^no GPU: on the CPU the smallest one answers fastest$/, 'keine Grafikkarte: auf dem Prozessor antwortet das kleinste am schnellsten'],
  [/^only ([\d.]+) GB on the GPU$/, 'nur $1 GB auf der Grafikkarte'],
  [/^the usual choice \(GPU memory not measured\)$/, 'die übliche Wahl (Grafikspeicher nicht gemessen)'],
  [/^the smaller one: only (\d+) GB free on the disk$/, 'das kleinere: nur $1 GB frei auf der Platte'],
  [/^another size fits this machine better$/, 'eine andere Größe passt besser zu diesem Rechner'],
  [/^very slow on the CPU -- better on a GPU or a Mac$/, 'sehr langsam auf dem Prozessor – besser mit Grafikkarte oder auf einem Mac'],
  [/^runs on the GPU(?: \(([\d.]+) GB on the GPU\))?$/, (m, g) => 'läuft auf der Grafikkarte' + (g ? ` (${g} GB)` : '')],
  [/^needs ~(\d+) GB on the GPU, ([\d.]+) GB on the GPU here$/, 'braucht etwa $1 GB auf der Grafikkarte, hier $2 GB'],
  [/^no GPU needed: fast on the CPU$/, 'keine Grafikkarte nötig: schnell auf dem Prozessor'],
  [/^Chatterbox needs ~5 GB on the GPU, ([\d.]+) GB on the GPU here; Piper runs on the CPU$/,
    'Chatterbox braucht etwa 5 GB auf der Grafikkarte, hier $1 GB; Piper läuft auf dem Prozessor'],
  [/^Chatterbox is suggested here \(cloned voices\); Piper can be added any time$/,
    'hier wird Chatterbox vorgeschlagen (geklonte Stimmen); Piper lässt sich jederzeit ergänzen'],
  [/^the small Whisper also runs well on the CPU$/, 'das kleine Whisper läuft auch auf dem Prozessor gut'],
  [/^both image editors fit \(([\d.]+) GB on the GPU\)$/, 'beide Bildbearbeiter passen ($1 GB auf der Grafikkarte)'],
  [/^the smaller editor \(FLUX\.2 \[klein\]\) fits in ([\d.]+) GB on the GPU; Qwen-Image-Edit wants ~12 GB$/,
    'der kleinere Bearbeiter (FLUX.2 [klein]) passt in $1 GB der Grafikkarte; Qwen-Image-Edit will etwa 12 GB'],
  [/^slow on a Mac, editing untried there$/, 'langsam auf einem Mac, Bearbeiten dort unerprobt'],
  [/^needs an NVIDIA GPU with ~8 GB(?:, ([\d.]+) GB on the GPU here)?$/,
    (m, g) => 'braucht eine NVIDIA-Grafikkarte mit etwa 8 GB' + (g ? `, hier ${g} GB` : '')],
  [/^too slow on the CPU$/, 'zu langsam auf dem Prozessor'],
  [/^fast images from text(?: \(([\d.]+) GB on the GPU\))?$/, (m, g) => 'schnelle Bilder aus Text' + (g ? ` (${g} GB auf der Grafikkarte)` : '')],
  [/^Z-Image in ComfyUI \(image editing\) already makes images from text$/, 'Z-Image in ComfyUI (Bildbearbeitung) macht schon Bilder aus Text'],
  [/^not enough disk space: (\d+) GB free, ~([\d.]+) GB needed after the rest \(and (\d+) GB kept free\)$/,
    'nicht genug Platz: $1 GB frei, nach dem Rest wären etwa $2 GB nötig (und $3 GB bleiben frei)'],
  [/^not on Windows yet$/, 'unter Windows noch nicht'],
  [/, no NVIDIA GPU/, ', keine NVIDIA-Grafikkarte'], [/, NVIDIA GPUs /, ', NVIDIA-Grafikkarten '], [/, NVIDIA GPU /, ', NVIDIA-Grafikkarte '],
  [/ \(Apple Silicon, ~(\d+) GB for the GPU\)/, ' (Apple Silicon, etwa $1 GB für die Grafik)'],
  [/ GB free$/, ' GB frei'],
];

let lang = 'en';
try { lang = localStorage.getItem('ai-serve-lang') || (navigator.language || '').slice(0, 2); } catch (e) { /* none */ }
if (lang !== 'de') lang = 'en';

function tx(text, vars) {
  let out = text;
  if (lang === 'de') {
    if (DE[text]) out = DE[text];
    else {
      for (const [re, de] of DE_PATTERNS) out = out.replace(re, de);
      out = out.replace(/(\d)\.(\d)/g, '$1,$2');   // 24.0 GB -> 24,0 GB
    }
  }
  for (const [k, v] of Object.entries(vars || {})) out = out.replace(`{${k}}`, v);
  return out;
}
const $ = (id) => document.getElementById(id);
const gb = (n) => (lang === 'de' ? String(n).replace('.', ',') : String(n));

let token = '';
const m = location.hash.match(/t=([A-Za-z0-9_-]+)/);
if (m) {
  token = m[1];
  try { sessionStorage.setItem('setup-token', token); } catch (e) { /* the page still works until reloaded */ }
  history.replaceState(null, '', location.pathname);
} else {
  try { token = sessionStorage.getItem('setup-token') || ''; } catch (e) { token = ''; }
}

async function api(path, body) {
  const opts = { headers: { 'X-Setup-Token': token } };
  if (body !== undefined) {
    opts.method = 'POST';
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

let state = null;
let built = false;
let ended = false;

function translateStatic() {
  document.documentElement.lang = lang;
  document.title = tx('Set up ai-serve');
  for (const el of document.querySelectorAll('[data-t]')) {
    if (el.dataset.en === undefined) el.dataset.en = el.textContent;   // the page's own text is the English one
    el.textContent = lang === 'de' && DE[el.dataset.t] ? DE[el.dataset.t] : el.dataset.en;
  }
  for (const b of document.querySelectorAll('[data-lang]')) b.classList.toggle('on', b.dataset.lang === lang);
}

function llmOffers() {
  const group = state.groups[0] || [];
  return state.offers.filter((o) => group.includes(o.id));
}

function buildChoice() {
  const sel = $('llm');
  sel.replaceChildren();
  for (const o of llmOffers()) {
    const opt = document.createElement('option');
    opt.value = o.id;
    opt.textContent = `${o.label.replace(/^Language model /, '').replace(/ \(.*\)$/, '')} · ${gb(o.size_gb)} GB`;
    opt.disabled = !o.available;
    opt.selected = o.suggested;
    sel.append(opt);
  }
  const none = document.createElement('option');
  none.value = '';
  none.textContent = tx('none');
  sel.append(none);
  const ul = $('offers');
  ul.replaceChildren();
  const group = state.groups[0] || [];
  for (const o of state.offers.filter((x) => !group.includes(x.id))) {
    const li = document.createElement('li');
    li.classList.toggle('off', !o.available);
    const label = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.value = o.id;
    box.checked = o.suggested;
    box.disabled = !o.available;
    box.addEventListener('change', showTotal);
    const name = document.createElement('span');
    name.textContent = tx(o.label);
    const size = document.createElement('span');
    size.className = 'size';
    size.textContent = `${gb(o.size_gb)} GB`;
    label.append(box, name, size);
    const what = document.createElement('p');
    what.className = 'sub';
    what.textContent = tx(o.what);
    const why = document.createElement('p');
    why.className = 'sub';
    why.textContent = `${o.suggested ? '→ ' : ''}${tx(o.why)} · ${tx('licence {l}', { l: o.license })}`;
    li.append(label, what, why);
    ul.append(li);
  }
  sel.onchange = showTotal;
  showTotal();
}

function chosen() {
  const ids = [...document.querySelectorAll('#offers input:checked')].map((b) => b.value);
  if ($('llm').value) ids.unshift($('llm').value);
  return ids;
}

function showTotal() {
  const llm = state.offers.find((o) => o.id === $('llm').value);
  $('llm-why').textContent = llm ? tx(llm.why) : '';
  const ids = chosen();
  const total = state.offers.filter((o) => ids.includes(o.id)).reduce((s, o) => s + o.size_gb, 0);
  if (!ids.length) {
    $('total').textContent = tx('Only ai-serve itself and llama.cpp (about 0.5 GB).');
    return;
  }
  let text = tx('Together about {n} GB, {free} GB free on the disk.', { n: gb(Math.round(total * 10) / 10), free: gb(Math.round(state.disk_free_gb)) });
  if (state.disk_free_gb && total > state.disk_free_gb - state.reserve_gb) text += ' ' + tx('That is more than fits ({keep} GB should stay free).', { keep: state.reserve_gb });
  $('total').textContent = text;
}

function showWork() {
  const j = state.job;
  const steps = $('steps');
  steps.replaceChildren();
  for (const s of j.finished.filter(Boolean)) {
    const li = document.createElement('li');
    li.className = 'done';
    li.textContent = tx(s);
    steps.append(li);
  }
  if (j.step) {
    const li = document.createElement('li');
    li.className = j.state === 'running' ? 'now' : '';
    li.textContent = tx(j.step);
    steps.append(li);
  }
  const bar = j.state === 'running' && j.bar && j.bar.total ? j.bar : null;
  $('bar').hidden = !bar;
  if (bar) {
    document.querySelector('#bar .fill').style.width = `${Math.min(100, (bar.done * 100) / bar.total)}%`;
    $('bar-text').textContent = `${bar.label} · ${tx('{done} of {total} MB', { done: bar.done >> 20, total: bar.total >> 20 })}`;
  }
  $('failed').hidden = j.state !== 'failed';
  $('retry-row').hidden = j.state !== 'failed';
  if (j.state === 'failed') $('failed').textContent = tx('Setup failed: {e}', { e: j.error });
  const log = $('log');
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = j.log.join('\n');
  if (atEnd) log.scrollTop = log.scrollHeight;
}

let showingChoice = true;

function render() {
  translateStatic();
  if (!state) return;
  $('machine').textContent = tx('This computer: {m}', { m: tx(state.machine) });
  const j = state.job.state;
  if (!built) { buildChoice(); built = true; }
  if (j === 'running') showingChoice = false;
  $('choose').hidden = !(showingChoice && (j === 'idle' || j === 'failed'));
  $('work').hidden = !$('choose').hidden || j === 'done';
  $('done').hidden = j !== 'done';
  $('go').disabled = j === 'running';
  if (!$('work').hidden) showWork();
}

async function poll() {
  if (ended) return;
  try {
    state = await api('/api/state');
    $('problem').hidden = true;
  } catch (e) {
    $('problem').hidden = false;
    $('problem').textContent = /token/.test(e.message) ? tx('The setup page lost its key. Start the setup again.')
      : (state ? tx('The setup has ended. You can close this tab.') : tx('Waiting for the setup …'));
  }
  render();
  setTimeout(poll, state && state.job.state === 'running' ? 700 : 2000);
}

$('go').addEventListener('click', async () => {
  $('go').disabled = true;
  try {
    await api('/api/install', { offers: chosen() });
    showingChoice = false;
  } catch (e) {
    $('problem').hidden = false;
    $('problem').textContent = e.message;
    $('go').disabled = false;
  }
});
$('retry').addEventListener('click', () => { showingChoice = true; render(); });
$('open').addEventListener('click', async () => {
  const url = state.gateway.url;
  try { await api('/api/finish', {}); } catch (e) { /* the dashboard opens anyway */ }
  ended = true;
  location.href = url;
});
for (const b of document.querySelectorAll('[data-lang]')) {
  b.addEventListener('click', () => {
    lang = b.dataset.lang;
    try { localStorage.setItem('ai-serve-lang', lang); } catch (e) { /* only this visit */ }
    built = false;
    render();
  });
}
translateStatic();
poll();
