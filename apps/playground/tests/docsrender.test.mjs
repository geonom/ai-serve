// node --test tests/docsrender.test.mjs -- a read document shown as a document (static/docsrender.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const render = createRequire(import.meta.url)('../static/docsrender.js');

test("NuExtract's tables become tables (2026-11-27: shown as raw <table><tr><td> text)", () => {
    const html = render('<table><tr><th>Item</th><th>Price</th></tr><tr><td>Coffee</td><td>3.50</td></tr></table>');
    assert.equal(html, '<table><tr><th>Item</th><th>Price</th></tr><tr><td>Coffee</td><td>3.50</td></tr></table>');
});

test('nothing but bare known tags survives: no attributes, no scripts, no unknown tags', () => {
    // (refused tags are plain text, so they sit in a paragraph)
    assert.equal(render('<td onclick="x()">a</td>'), '<p>&lt;td onclick=&quot;x()&quot;&gt;a</td></p>');
    assert.equal(render('<script>alert(1)</script>'), '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
    assert.equal(render('<img src=x onerror=alert(1)>'), '<p>&lt;img src=x onerror=alert(1)&gt;</p>');
    assert.equal(render('<a href="javascript:x">y</a>'), '<p>&lt;a href=&quot;javascript:x&quot;&gt;y&lt;/a&gt;</p>');
});

test('a little markdown: headings, bold, list items, paragraphs', () => {
    assert.equal(render('# Invoice 4711'), '<h3>Invoice 4711</h3>');
    assert.equal(render('## To'), '<h4>To</h4>');
    assert.equal(render('Total: **42.50 EUR**'), '<p>Total: <strong>42.50 EUR</strong></p>');
    assert.equal(render('- one\n- two'), '<ul><li>one</li><li>two</li></ul>');
    assert.equal(render('a\nb\n\nc'), '<p>a<br>b</p><p>c</p>');
    assert.equal(render('**<b>x</b>**'), '<p><strong><b>x</b></strong></p>');
});

test('page breaks become a line', () => {
    assert.equal(render('a\n\n<!-- PAGE BREAK -->\n\nb'), '<p>a</p><hr><p>b</p>');
});
