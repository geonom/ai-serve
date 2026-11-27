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

test('a card for videos, and enlarging / cutting out among the picture facts (2026-11-26)', () => {
    const VIDEO = [{ id: 'ltx', available: true, tasks: ['text_to_video', 'image_to_video'], audio: true },
                   { id: 'wan', available: true, tasks: ['image_to_video'], audio: false },
                   { id: 'gone', available: false, tasks: ['text_to_video'] }];
    const img = [...IMG, { id: 'esr', available: true, tasks: ['upscale'] }, { id: 'bir', available: true, tasks: ['remove_background'] }];
    const c = Object.fromEntries(R.cards(LLM, img, AUDIO, [], VIDEO).map(x => [x.tab, x]));
    assert.equal(c.video.ok, true);
    assert.equal(c.video.title, 'Create videos');
    assert.deepEqual(c.video.facts, ['2 video models', '2 from a picture', '1 with sound']);
    assert.deepEqual(c['image-gen'].facts, ['2 image models', '1 for editing', '1 for style', 'enlarge and cut out']);
    // no video plugin, or the gateway didn't answer
    assert.equal(Object.fromEntries(R.cards(LLM, IMG, AUDIO, [], []).map(x => [x.tab, x])).video.ok, false);
    assert.equal(R.cards(null, null, null).find(x => x.tab === 'video').ok, false);
});

test('reading models: only those the Read documents page can use (NuExtract), not every OCR-named model', () => {
    const llm = [...LLM, { id: 'GLM-OCR-Q8_0.gguf', label: 'GLM-OCR', kind: 'chat', vision: true, where: [local] },
                 { id: 'RolmOCR.i1-IQ4_XS.gguf', label: 'RolmOCR', kind: 'chat', vision: true, where: [local] }];
    const c = Object.fromEntries(R.cards(llm, IMG, AUDIO).map(x => [x.tab, x]));
    assert.deepEqual(c.docs.facts, ['1 reading model']);
});
