// node --test tests/imagesteps.test.mjs -- the Steps field follows the model (static/imagesteps.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
req('../static/i18n.js');
const S = req('../static/imagesteps.js');

const IRIS = { default: 4, adjustable: false, max: 4 };
const MOSS = { default: 30, adjustable: true, max: 60 };

test('a fixed schedule locks the field and sends nothing', () => {
    const f = S.field(IRIS);
    assert.equal(f.disabled, true);
    assert.equal(f.value, 4);
    assert.match(f.note, /fixed 4-step schedule/i);
    assert.equal(S.request(IRIS, '24'), null);
});

test('an adjustable model shows its default and caps at its max', () => {
    const f = S.field(MOSS);
    assert.equal(f.disabled, false);
    assert.equal(f.placeholder, 'default 30');
    assert.equal(f.max, 60);
    assert.equal(S.request(MOSS, '16'), 16);
    assert.equal(S.request(MOSS, '99'), 60);
    assert.equal(S.request(MOSS, ''), null);
});

test('an unknown model leaves the field free', () => {
    assert.equal(S.field(undefined).disabled, false);
    assert.equal(S.request(undefined, '12'), 12);
});
