// node --test tests/pictureuse.test.mjs -- what a finished picture can be used for next (static/pictureuse.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
req('../static/i18n.js');
const P = req('../static/pictureuse.js');

const ALL = { edit: true, style: true, video: true, upscale: true, cutout: true };

test('every next step the gateway offers, in a fixed order', () => {
    assert.deepEqual(P.actions(ALL).map(a => a.id), ['edit', 'edit-extra', 'style', 'video', 'upscale', 'cutout']);
    assert.ok(P.actions(ALL).every(a => typeof a.label === 'string' && a.label.length > 0));
});

test('steps the gateway cannot do right now are left out', () => {
    assert.deepEqual(P.actions({ edit: true }).map(a => a.id), ['edit', 'edit-extra']);
    assert.deepEqual(P.actions({ video: true, cutout: true }).map(a => a.id), ['video', 'cutout']);
    assert.deepEqual(P.actions({}), []);
    assert.deepEqual(P.actions(undefined), []);
});

test('a picture is never offered back to the step that just made it from the same picture', () => {
    // enlarging an enlarged picture again is allowed (2x then 2x), cutting out a cut-out is pointless
    assert.ok(!P.actions(ALL, 'cutout').some(a => a.id === 'cutout'));
    assert.ok(P.actions(ALL, 'upscale').some(a => a.id === 'upscale'));
});

test('enlarging stays within the 4096 x 4096 output the gateway allows', () => {
    assert.deepEqual(P.upscaleFactors(1024, 1024), [2, 4]);
    assert.deepEqual(P.upscaleFactors(1360, 768), [2, 4]);       // 4x = 5440 x 3072 = 16.71 MP, just inside
    assert.deepEqual(P.upscaleFactors(1536, 1024), [2]);         // 4x = 6144 x 4096 = 25.2 MP
    assert.deepEqual(P.upscaleFactors(2048, 2048), [2]);
    assert.deepEqual(P.upscaleFactors(3000, 2000), []);
    assert.deepEqual(P.upscaleFactors(0, 0), [2, 4]);             // size unknown: let the gateway decide
});

test('kept in "My images" under the step that made it, with only known fields', () => {
    assert.deepEqual(P.galleryMeta('edit', { prompt: 'make it night', model: 'qwen-edit-2511', label: 'Qwen', seed: 3, seconds: 41.2, extra: 'x' }),
                     { source: 'edit', prompt: 'make it night', model: 'qwen-edit-2511', label: 'Qwen', seed: 3, seconds: 41.2 });
    assert.equal(P.galleryMeta('style', {}).source, 'style');
    assert.equal(P.galleryMeta('upscale', { model: 'realesrgan-x4' }).source, 'upscale');
    assert.equal(P.galleryMeta('cutout', {}).source, 'cutout');
    assert.equal(P.galleryMeta('generate', {}).source, 'bild');
    assert.deepEqual(P.galleryMeta('edit', { prompt: '', seed: null }), { source: 'edit' });   // empties dropped
});

test('base64 out of a data URL, and a data URL for raw base64', () => {
    assert.equal(P.b64('data:image/png;base64,QUJD'), 'QUJD');
    assert.equal(P.b64('QUJD'), 'QUJD');
    assert.equal(P.dataUrl('QUJD'), 'data:image/png;base64,QUJD');
    assert.equal(P.dataUrl('data:image/webp;base64,QUJD'), 'data:image/webp;base64,QUJD');
});
