// node --test tests/apierror.test.mjs -- readable error texts from gateway answers (static/apierror.js)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const req = createRequire(import.meta.url);
const errorText = req('../static/apierror.js');

test('a plain detail is shown as it is', () => {
    assert.equal(errorText({ detail: 'No room on GPU 0' }, 507), 'No room on GPU 0');
});

test("FastAPI's field errors become one readable line, not [object Object]", () => {
    const body = { detail: [{ loc: ['body', 'scale'], msg: 'Input should be 2 or 4', type: 'literal_error' },
                            { loc: ['body', 'image'], msg: 'Field required' }] };
    assert.equal(errorText(body, 422), 'scale: Input should be 2 or 4; image: Field required');
});

test('OpenAI-style error objects and nested details', () => {
    assert.equal(errorText({ error: { message: 'model not found' } }, 400), 'model not found');
    assert.equal(errorText({ detail: { message: 'busy' } }, 409), 'busy');
});

test('nothing usable -> the HTTP status', () => {
    assert.equal(errorText({}, 502), 'HTTP error 502');
    assert.equal(errorText(null, 500), 'HTTP error 500');
});

const { friendly } = req('../static/apierror.js');

test('"no room on the graphics card" becomes a plain "busy, try again" with the details kept apart', () => {
    const f = friendly(507, 'GPU 1: needs 18271 MB, 6427 MB free, even after unloading every idle user only 8775 MB');
    assert.match(f.text, /busy/i);
    assert.match(f.text, /minute/i);
    assert.ok(!/MB/.test(f.text));
    assert.match(f.details, /18271 MB/);
    assert.equal(f.retry, true);
});

test('a part that is starting or down, a busy one, the daily quota, a bad request', () => {
    assert.equal(friendly(503, 'image-serve is not reachable').retry, true);
    assert.match(friendly(503, 'x').text, /moment/i);
    assert.match(friendly(409, 'busy').text, /busy/i);
    assert.equal(friendly(429, 'Today\'s 60 images are used up.').text, 'Today\'s 60 images are used up.');   // already plain
    assert.equal(friendly(400, 'Please describe the picture.').text, 'Please describe the picture.');
    assert.equal(friendly(400, 'Please describe the picture.').retry, false);
});

test('a server error says so without the stack', () => {
    const f = friendly(500, 'Traceback (most recent call last): ...');
    assert.match(f.text, /went wrong/i);
    assert.match(f.details, /Traceback/);
});
