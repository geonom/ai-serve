// node --test tests/models.test.mjs -- the chat model list built from GET /v1/llm/models (static/models.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

createRequire(import.meta.url)('../static/i18n.js');   // tx() -- English in node
const M = createRequire(import.meta.url)('../static/models.js');
const row = (id, over = {}) => ({ id, label: id, quant: null, kind: 'chat', vision: false, thinking: false, mtp: false,
                                  exclusive: false, loaded: false, context_length: 8192, where: [{ place: 'local', loaded: false }], ...over });
const MODELS = [
    row('gemma', { quant: 'Q4_K_M', vision: true, thinking: true, loaded: true, where: [{ place: 'local', loaded: true, ctx_size: 32000 }] }),
    row('strata', { exclusive: true, where: [{ place: 'plugin', plugin: 'strata', name: 'Strata', state: 'stopped', loaded: false }] }),
    row('granite'),
    row('qwen-mac', { where: [{ place: 'device', node: 'macbook', online: true, loaded: false }] }),
    row('tiny-pi', { where: [{ place: 'device', node: 'pi', online: false, loaded: false }] }),
    row('bge-m3', { kind: 'embedding' }), row('YuE', { kind: 'base' }),
];

test('groups: running, Strata, here, devices -- and no embedding or base models', () => {
    const g = M.groups(MODELS);
    assert.deepEqual(g.map(x => [x.label, x.options.map(o => o.value)]), [
        ['Running now', ['gemma']], ['Exclusive (takes all graphics cards)', ['strata']],
        ['On the gateway machine', ['granite']], ['On other machines in the network', ['qwen-mac', 'tiny-pi']]]);
    assert.equal(g[3].options[1].disabled, true);      // only on an offline device
    assert.equal(g[3].options[0].text, 'qwen-mac @ macbook');
    assert.equal(g[0].options[0].text, 'gemma · Q4_K_M (image, thinking)');
});

test('context: the running instance first, else what the model was trained for', () => {
    assert.equal(M.contextFor(MODELS[0]), 32000);
    assert.equal(M.contextFor(MODELS[2]), 8192);
    assert.equal(M.contextFor(undefined), null);
});

test('notes for Strata and device-only models', () => {
    assert.match(M.note(MODELS[1]), /Strata takes all graphics cards/);
    assert.match(M.note(MODELS[3]), /Only on macbook/);
    assert.equal(M.note(MODELS[2]), '');
});
