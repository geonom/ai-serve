// node --test tests/editimages.test.mjs -- the Edit mode's extra pictures and output shape (static/editimages.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
req('../static/i18n.js');
const E = req('../static/editimages.js');

test('a model takes picture 1 plus max_images - 1 more', () => {
    assert.equal(E.extraLimit(5), 4);
    assert.equal(E.extraLimit(1), 0);
    assert.equal(E.extraLimit(undefined), 0);   // an older catalog without max_images: one image
});

test('adding pictures keeps their order and stops at the limit', () => {
    assert.deepEqual(E.add(['a'], ['b', 'c'], 4), { list: ['a', 'b', 'c'], dropped: 0 });
    assert.deepEqual(E.add(['a', 'b', 'c'], ['d', 'e'], 4), { list: ['a', 'b', 'c', 'd'], dropped: 1 });
    assert.deepEqual(E.add([], ['x'], 0), { list: [], dropped: 1 });
});

test('each extra picture gets the number the prompt can use', () => {
    assert.deepEqual(E.labels(2), ['Picture 2', 'Picture 3']);
    // three or four more are put side by side into one picture 2 (image-serve), left to right
    assert.deepEqual(E.labels(4), ['Picture 2 (1st from left)', 'Picture 2 (2nd from left)',
                                   'Picture 2 (3rd from left)', 'Picture 2 (4th from left)']);
});

test('the hint says how to refer to the pictures', () => {
    assert.equal(E.hint(0), '');
    assert.match(E.hint(1), /picture 2/);
    assert.match(E.hint(2), /picture 2.*picture 3/);
    assert.match(E.hint(3), /side by side.*picture 2/);
});

test('the request carries the extra pictures and the shape only when set', () => {
    assert.deepEqual(E.request(['b', 'c'], '16:9'), { extra_images_base64: ['b', 'c'], aspect_ratio: '16:9' });
    assert.deepEqual(E.request([], ''), {});
    assert.deepEqual(E.request(['b'], ''), { extra_images_base64: ['b'] });
});

test('only shapes the model lists are offered', () => {
    assert.deepEqual(E.aspects(['1:1', '16:9']), ['', '1:1', '16:9']);   // '' = like picture 1
    assert.deepEqual(E.aspects(undefined), ['']);
});
