// node --test tests/overview.test.mjs -- the start page's cards (static/overview.js) from the gateway's catalogs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

createRequire(import.meta.url)('../static/i18n.js');   // tx() -- English in node
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
    assert.deepEqual(c.gateway.facts, ['4 chat models', '2 understand images', 'Strata (very large, takes all graphics cards)',
                                       '1 only on other machines in the network', 'loaded: gemma']);     // pi offline: not counted
    assert.deepEqual(c.texts.facts, ['1 embedding model', '0 rerank models']);
    assert.equal(c.docs.ok, true);
    assert.deepEqual(c['image-gen'].facts, ['2 image models', '1 for editing', '1 for style']);   // not installed: out
    assert.equal(c.layout.ok, true);
    assert.deepEqual(c.speech.facts, ['Speaking: CosyVoice 3', 'Transcribing: WhisperX']);
});

test('a gateway that does not answer', () => {
    const c = R.cards(null, null, null);
    assert.ok(c.filter(x => x.tab !== 'mine').every(x => !x.ok));
    assert.deepEqual(c[0].facts, ['Gateway unreachable right now.']);
});
