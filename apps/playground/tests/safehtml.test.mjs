// node --test tests/safehtml.test.mjs -- text that goes into HTML strings (static/safehtml.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const escapeHtml = createRequire(import.meta.url)('../static/safehtml.js');

test('markup in model answers or names is shown, never run', () => {
    // 2026-11-27: the chat's final bubble put the answer and its reasoning into innerHTML as they came
    assert.equal(escapeHtml('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
    assert.equal(escapeHtml("a & b's"), 'a &amp; b&#39;s');
});

test('plain text and non-strings', () => {
    assert.equal(escapeHtml('Sofia'), 'Sofia');
    assert.equal(escapeHtml(null), '');
    assert.equal(escapeHtml(42), '42');
});
