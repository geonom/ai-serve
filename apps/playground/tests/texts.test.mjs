// node --test tests/texts.test.mjs -- "Texte vergleichen" (static/texts.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const R = createRequire(import.meta.url)('../static/texts.js');

test('lines, cosine, matrix', () => {
    assert.deepEqual(R.lines(' a \n\n b\n'), ['a', 'b']);
    assert.equal(R.cosine([1, 0], [1, 0]), 1);
    assert.equal(R.cosine([1, 0], [0, 1]), 0);
    assert.equal(R.cosine([0, 0], [1, 1]), 0);
    const m = R.matrix([[1, 0], [1, 1], [0, 1]]);
    assert.ok(Math.abs(m[0][1] - Math.SQRT1_2) < 1e-9 && m[0][2] === 0 && m[1][1] > 0.9999);
    assert.deepEqual(R.closestPairs(m, 2).map(p => [p.i, p.j]), [[0, 1], [1, 2]]);
});

test('ranking by embedding: first vector is the query', () => {
    const r = R.rankByEmbedding([[1, 0], [0, 1], [1, 0.1]], ['weit weg', 'nah']);
    assert.deepEqual(r.map(x => x.document), ['nah', 'weit weg']);
});

test('ranking from /v1/rerank', () => {
    const r = R.rankFromRerank([{ index: 0, relevance_score: -7.8 }, { index: 1, relevance_score: 7.9, document: 'Paris' }], ['Berlin', 'x']);
    assert.deepEqual(r.map(x => [x.index, x.document]), [[1, 'Paris'], [0, 'Berlin']]);
});

test('models of a kind, reachable only', () => {
    const ms = [{ id: 'a', kind: 'embedding', where: [{ place: 'local' }] },
                { id: 'b', kind: 'embedding', where: [{ place: 'device', online: false }] },
                { id: 'c', kind: 'rerank', where: [{ place: 'device', online: true }] }];
    assert.deepEqual(R.modelsOfKind(ms, 'embedding').map(m => m.id), ['a']);
    assert.deepEqual(R.modelsOfKind(ms, 'rerank').map(m => m.id), ['c']);
});

test('scores a visitor can read: rerank logits and similarities as percent (2026-11-27)', () => {
    // a cross-encoder answers with logits (-1.55, -9.4, ...) -- shown raw they meant nothing
    assert.equal(R.percent(0, 'rerank'), 50);
    assert.equal(R.percent(-1.55, 'rerank'), 18);
    assert.equal(R.percent(-9.4, 'rerank'), 0);
    assert.equal(R.percent(4, 'rerank'), 98);
    assert.equal(R.percent(0.92, 'similarity'), 92);
    assert.equal(R.percent(-0.2, 'similarity'), 0);      // cosine below zero: no likeness
});
