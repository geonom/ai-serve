// node --test tests/downloads.test.mjs -- what a page can still download (static/downloads.js), and the overview's hint
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
req('../static/i18n.js');
const D = req('../static/downloads.js');
const R = req('../static/overview.js');

const ITEMS = [
    { plugin: 'sd-serve', id: 'iris-turbo', label: 'Iris-3B Turbo', capability: 'image', state: 'missing', size_mb: 5300 },
    { plugin: 'tts-serve', id: 'chatterbox', label: 'Chatterbox', capability: 'tts', state: 'downloading', size_mb: 6500, done_mb: null },
    { plugin: 'tts-serve', id: 'ready', label: 'Ready', capability: 'tts', state: 'installed', size_mb: 1 },
];

test('a page sees only its own missing items', () => {
    assert.deepEqual(D.missing(ITEMS, 'image').map(i => i.id), ['iris-turbo']);
    assert.deepEqual(D.missing(ITEMS, 'tts').map(i => i.id), ['chatterbox']);
    assert.equal(D.missing(ITEMS, 'stt').length, 0);
});

test('a page can narrow its offer to certain items (the layout page: only MOSS)', () => {
    const items = [...ITEMS, { plugin: 'image-serve', id: 'moss', capability: 'image', state: 'missing' },
                   { plugin: 'image-serve', id: 'z-image-turbo', capability: 'image', state: 'missing' }];
    assert.deepEqual(D.missing(items, 'image', ['moss', 'moss-bf16']).map(i => i.id), ['moss']);
    assert.deepEqual(D.missing(items, 'image').map(i => i.id), ['iris-turbo', 'moss', 'z-image-turbo']);
});

test('sizes and states read plainly', () => {
    assert.equal(D.sizeText(5300), '~5.2 GB');
    assert.equal(D.sizeText(700), '~700 MB');
    assert.equal(D.sizeText(null), '');
    assert.match(D.stateText(ITEMS[1]), /this can take a while/);
    assert.equal(D.stateText({ state: 'downloading', done_mb: 1024, size_mb: 5300 }), 'Downloading … 1 GB of 5.2 GB');
    assert.equal(D.stateText({ state: 'failed', error: 'disk full' }), 'Failed: disk full');
});

test('an unavailable overview card names what could be downloaded', () => {
    const c = Object.fromEntries(R.cards([], [], { tts: { available: true, engines: [] }, stt: null }, ITEMS).map(x => [x.tab, x]));
    assert.ok(c['image-gen'].facts.includes('Can be downloaded: Iris-3B Turbo'));
    assert.ok(c.speech.facts.includes('Can be downloaded: Chatterbox'));
    assert.ok(!c.gateway.facts.some(f => f.startsWith('Can be downloaded')));
});
