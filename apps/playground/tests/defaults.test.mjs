// Defaults a first visitor gets (2026-11-27, owner: thinking off by default, as in the gateway).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../static/index.html', import.meta.url), 'utf8');

test('chat starts with thinking off', () => {
    const box = html.match(/<input[^>]*id="gateway-thinking"[^>]*>/);
    assert.ok(box, 'the thinking checkbox exists');
    assert.doesNotMatch(box[0], /\bchecked\b/);
});
