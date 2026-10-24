// "Texte vergleichen": what embedding and rerank models do, through the gateway's /v1/embeddings and
// /v1/rerank (owner, 2026-10-20: everything that runs through the gateway should be playable). Models come
// from GET /v1/llm/models (kind embedding / rerank). Without a rerank model, documents can still be sorted by
// embedding similarity -- labelled as such. Pure rules first (tests: node --test tests/texts.test.mjs).
const TextRules = (() => {
    function lines(text) {
        return String(text || '').split('\n').map(x => x.trim()).filter(Boolean);
    }

    function cosine(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return na && nb ? dot / Math.sqrt(na * nb) : 0;
    }

    function matrix(vectors) {
        return vectors.map(a => vectors.map(b => cosine(a, b)));
    }

    // The most similar pairs (i < j), best first.
    function closestPairs(m, n = 5) {
        const pairs = [];
        for (let i = 0; i < m.length; i++) for (let j = i + 1; j < m.length; j++) pairs.push({ i, j, score: m[i][j] });
        return pairs.sort((x, y) => y.score - x.score).slice(0, n);
    }

    // Documents sorted for the query by embedding similarity (first vector = the query).
    function rankByEmbedding(vectors, documents) {
        const [q, ...docs] = vectors;
        return docs.map((v, index) => ({ index, document: documents[index], score: cosine(q, v) }))
            .sort((a, b) => b.score - a.score);
    }

    // /v1/rerank's results ({index, relevance_score, document?}) sorted best first.
    function rankFromRerank(results, documents) {
        return (results || []).map(r => ({ index: r.index, document: r.document ?? documents[r.index], score: r.relevance_score }))
            .sort((a, b) => b.score - a.score);
    }

    function modelsOfKind(models, kind) {
        return (models || []).filter(m => m.kind === kind && m.where.some(p => p.place !== 'device' || p.online));
    }

    return { lines, cosine, matrix, closestPairs, rankByEmbedding, rankFromRerank, modelsOfKind };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = TextRules;

if (typeof document !== 'undefined' && document.getElementById('texts-tab')) (() => {
    const R = TextRules;
    const $ = id => document.getElementById(id);
    let models = [];

    function fill(select, rows, emptyText) {
        select.replaceChildren(...(rows.length ? rows : [null]).map(m => {
            const o = document.createElement('option');
            if (!m) { o.value = ''; o.textContent = emptyText; return o; }
            const dev = m.where.every(p => p.place === 'device') ? ` @ ${m.where.map(p => p.node).join(', ')}` : '';
            o.value = m.id; o.textContent = m.label + (m.quant ? ` · ${m.quant}` : '') + dev;
            return o;
        }));
    }

    async function load() {
        try { models = (await api('/v1/llm/models')).models || []; } catch (err) { addLog(tx('Models: {err}', { err: err.message }), 'error'); }
        const emb = R.modelsOfKind(models, 'embedding'), rr = R.modelsOfKind(models, 'rerank');
        fill($('emb-model'), emb, tx('no embedding model available'));
        fill($('rr-model'), rr.concat(emb.map(m => ({ ...m, viaEmbedding: true }))), tx('no suitable model available'));
        // options from embedding models rank by similarity: mark them
        [...$('rr-model').options].forEach((o, i) => { if (i >= rr.length && o.value) { o.dataset.emb = '1'; o.textContent += ' ' + tx('(by similarity)'); } });
        $('rr-note').textContent = rr.length ? '' : tx('No rerank model reachable – sorting uses an embedding model\'s similarity. A real rerank model (e.g. bge-reranker-v2-m3) scores question and text together and is usually more accurate.');
    }

    $('emb-run').addEventListener('click', async () => {
        const texts = R.lines($('emb-texts').value), status = $('emb-status'), out = $('emb-result');
        if (!$('emb-model').value) { status.textContent = tx('No embedding model available.'); return; }
        if (texts.length < 2) { status.textContent = tx('At least two texts, one per line.'); return; }
        status.textContent = tx('Computing… (the first time, the model is loaded)');
        const t0 = performance.now();
        try {
            const r = await api('/v1/embeddings', 'POST', { model: $('emb-model').value, input: texts });
            const vecs = (r.data || []).sort((a, b) => a.index - b.index).map(d => d.embedding);
            const m = R.matrix(vecs);
            const short = s => s.length > 40 ? s.slice(0, 38) + '…' : s;
            const table = document.createElement('table');
            table.className = 'sw-matrix';
            const head = table.insertRow();
            head.insertCell().textContent = '';
            texts.forEach((t, i) => { const c = head.insertCell(); c.textContent = i + 1; c.title = t; });
            m.forEach((row, i) => {
                const tr = table.insertRow();
                const label = tr.insertCell(); label.textContent = `${i + 1}. ${short(texts[i])}`; label.title = texts[i];
                row.forEach((v, j) => {
                    const c = tr.insertCell();
                    c.textContent = v.toFixed(2);
                    c.style.background = i === j ? 'transparent' : `rgba(53, 80, 216, ${Math.max(0, v) * 0.55})`;
                });
            });
            const pairs = R.closestPairs(m).map(p => `${p.score.toFixed(2)}  ${p.i + 1} ↔ ${p.j + 1}: „${short(texts[p.i])}“ / „${short(texts[p.j])}“`);
            const pre = document.createElement('pre');
            pre.className = 'sw-result';
            pre.textContent = tx('Most similar:') + '\n' + pairs.join('\n');
            out.replaceChildren(table, pre);
            status.textContent = tx('Done · {n} vectors of {d} numbers · {s} s', { n: vecs.length, d: vecs[0] ? vecs[0].length : 0, s: ((performance.now() - t0) / 1000).toFixed(1) });
        } catch (err) {
            status.textContent = tx('Failed: {err}', { err: err.message });
        }
    });

    $('rr-run').addEventListener('click', async () => {
        const query = $('rr-query').value.trim(), docs = R.lines($('rr-docs').value), status = $('rr-status');
        const opt = $('rr-model').selectedOptions[0];
        if (!opt || !opt.value) { status.textContent = tx('No suitable model available.'); return; }
        if (!query || docs.length < 2) { status.textContent = tx('A question and at least two texts (one per line).'); return; }
        status.textContent = tx('Scoring… (the first time, the model is loaded)');
        const t0 = performance.now();
        try {
            let ranked;
            if (opt.dataset.emb) {
                const r = await api('/v1/embeddings', 'POST', { model: opt.value, input: [query, ...docs] });
                ranked = R.rankByEmbedding((r.data || []).sort((a, b) => a.index - b.index).map(d => d.embedding), docs);
            } else {
                const r = await api('/v1/rerank', 'POST', { model: opt.value, query, documents: docs });
                ranked = R.rankFromRerank(r.results, docs);
            }
            $('rr-result').textContent = ranked.map((x, i) => `${i + 1}. ${x.score.toFixed(3)}  ${x.document}`).join('\n');
            status.textContent = tx('Done · {how} · {s} s', { how: opt.dataset.emb ? tx('by similarity') : 'Rerank', s: ((performance.now() - t0) / 1000).toFixed(1) });
        } catch (err) {
            status.textContent = tx('Failed: {err}', { err: err.message });
        }
    });

    let loaded = false;
    const maybeLoad = () => { if (!loaded) { loaded = true; load(); } };
    const link = document.querySelector('.tab-link[data-tab="texts"]');
    link && link.addEventListener('click', maybeLoad);
    if ($('texts-tab').classList.contains('active')) maybeLoad();
})();
