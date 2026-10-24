// node --test tests/help.test.mjs -- help per page (static/help.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';

const { HELP, HelpRules } = createRequire(import.meta.url)('../static/help.js');

test('every page of the navigation has help with first steps', () => {
    const tabs = [...readFileSync(new URL('../static/index.html', import.meta.url), 'utf8').matchAll(/data-tab="([^"]+)"/g)].map(m => m[1]);
    assert.ok(tabs.length >= 9);
    for (const t of tabs) {
        assert.ok(HELP[t], `no help for ${t}`);
        assert.ok(HELP[t].title && HELP[t].intro && HELP[t].steps.length >= 3, t);
    }
});

test('opens by itself once per page', () => {
    assert.equal(HelpRules.autoOpen('texts', []), true);
    assert.equal(HelpRules.autoOpen('texts', ['texts']), false);
    assert.equal(HelpRules.autoOpen('nope', []), false);
    assert.deepEqual(HelpRules.markSeen('texts', ['docs']), ['docs', 'texts']);
    assert.deepEqual(HelpRules.markSeen('docs', ['docs']), ['docs']);
});
