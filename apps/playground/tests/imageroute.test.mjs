// node --test tests/imageroute.test.mjs -- which way a Generate model goes (static/imageroute.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const R = createRequire(import.meta.url)('../static/imageroute.js');

test('main-ComfyUI checkpoints go the ComfyUI way with their file name (LoRA, resolution, negative prompt)', () => {
    assert.deepEqual(R.generateRoute({ id: 'z-image-turbo', provider: 'comfyui', aliases: ['z_image_turbo-Q6_K.gguf'] }),
                     { value: 'z_image_turbo-Q6_K.gguf', gateway: false });
});

test('ComfyUI models without a checkpoint file (MOSS) go through the gateway by id -- not "undefined"', () => {
    // 2026-11-27: all six MOSS rows had value="undefined", ComfyUI answered 400 and the page showed nothing
    assert.deepEqual(R.generateRoute({ id: 'moss-turbo-4', provider: 'comfyui', aliases: [] }), { value: 'moss-turbo-4', gateway: true });
    assert.deepEqual(R.generateRoute({ id: 'moss', provider: 'comfyui' }), { value: 'moss', gateway: true });
});

test('other plugins go through the gateway by id', () => {
    assert.deepEqual(R.generateRoute({ id: 'iris-turbo', provider: 'sd-serve', aliases: ['iris', 'iris-3b-turbo'] }),
                     { value: 'iris-turbo', gateway: true });
});
