// node --test tests/overview.test.mjs -- the start page's cards (static/overview.js) from the gateway's catalogs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const R = createRequire(import.meta.url)('../static/overview.js');
const local = { place: 'local', loaded: false };
const LLM = [
    { id: 'gemma', label: 'gemma', kind: 'chat', vision: true, loaded: true, where: [{ place: 'local', loaded: true }] },
    { id: 'strata', label: 'Strata', kind: 'chat', exclusive: true, where: [{ place: 'plugin', plugin: 'strata', name: 'Strata' }] },
    { id: 'mac-only', label: 'q', kind: 'chat', where: [{ place: 'device', node: 'mac', online: true }] },
    { id: 'pi-only', label: 'p', kind: 'chat', where: [{ place: 'device', node: 'pi', online: false }] },
    { id: 'NuExtract3-Q8_0.gguf', label: 'NuExtract3', kind: 'chat', vision: true, where: [local] },
    { id: 'bge-m3', label: 'bge-m3', kind: 'embedding', where: [local] },
];
const IMG = [
    { id: 'z', available: true, tasks: ['generate'] }, { id: 'moss', available: true, tasks: ['generate'], layout: true },
    { id: 'ideo', available: false, tasks: ['generate'] }, { id: 'qe', available: true, tasks: ['edit', 'style'] },
];
const AUDIO = { tts: { available: true, engines: [{ label: 'CosyVoice 3' }] },
                stt: { available: true, engines: [{ label: 'WhisperX', available: true }, { label: 'MOSS', available: false }] } };

test('cards describe what the gateway offers now', () => {
    const c = Object.fromEntries(R.cards(LLM, IMG, AUDIO).map(x => [x.tab, x]));
    assert.deepEqual(c.gateway.facts, ['4 Chat-Modelle', '2 verstehen Bilder', 'Strata (sehr groß, belegt alle Grafikkarten)',
                                       '1 nur auf anderen Rechnern im Netzwerk', 'geladen: gemma']);     // pi offline: not counted
    assert.deepEqual(c.texts.facts, ['1 Embedding-Modell', '0 Rerank-Modelle']);
    assert.equal(c.docs.ok, true);
    assert.deepEqual(c['image-gen'].facts, ['2 Bildmodelle', '1 zum Bearbeiten', '1 für Stil']);   // not installed: out
    assert.equal(c.layout.ok, true);
    assert.deepEqual(c.speech.facts, ['Sprechen: CosyVoice 3', 'Abschreiben: WhisperX']);
});

test('a gateway that does not answer', () => {
    const c = R.cards(null, null, null);
    assert.ok(c.filter(x => x.tab !== 'mine').every(x => !x.ok));
    assert.deepEqual(c[0].facts, ['Gateway gerade nicht erreichbar.']);
});
