// node --test sdk/door/door-session.test.mjs
// The door's login can run out while a page is open. Instead of reloading (which loses typed text, the
// reading position, ticked boxes -- owner, 2026-10-20), a same-origin fetch that fails because of it waits
// while the login is renewed in the background, then is sent again. No page reload, no lost state.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createDoorSession } = require('./door-session.js');

const ORIGIN = 'https://storytime.apps.example.org';
const ok = (body = 'ok') => ({ status: 200, type: 'basic', ok: true, body });
const redirected = () => ({ status: 0, type: 'opaqueredirect', ok: false });

function setup({ session = true, silentWorks = true }) {
    const state = { session, frames: 0, popups: 0, banner: null, calls: [] };
    const rawFetch = async (input, init = {}) => {
        const url = typeof input === 'string' ? input : input.url;
        state.calls.push(url);
        if (init.redirect === 'manual') return state.session ? ok() : redirected();   // the probe
        if (!state.session) throw new TypeError('NetworkError when attempting to fetch resource.');
        return ok(url);
    };
    const ds = createDoorSession({
        fetch: rawFetch, origin: ORIGIN, sleep: () => new Promise(r => setTimeout(r, 1)),
        frame: () => { state.frames += 1; if (silentWorks) state.session = true; return () => {}; },
        popup: () => { state.popups += 1; state.session = true; return { close() {} }; },
        banner: (onClick) => { state.banner = onClick; return () => { state.banner = 'closed'; }; },
    });
    return { ds, state };
}

test('a working session passes straight through', async () => {
    const { ds, state } = setup({});
    const r = await ds.fetch('/api/stories');
    assert.equal(r.body, '/api/stories');
    assert.equal(state.frames, 0);
});

test('a lost session is renewed silently and the request sent again -- no reload', async () => {
    const { ds, state } = setup({ session: false });
    const r = await ds.fetch('/api/stories', { method: 'POST', body: '{"text":"Kapitel 3"}' });
    assert.equal(r.body, '/api/stories');
    assert.equal(state.frames, 1);
    assert.equal(state.popups, 0);
});

test('parallel requests share one renewal', async () => {
    const { ds, state } = setup({ session: false });
    await Promise.all([ds.fetch('/a'), ds.fetch('/b'), ds.fetch('/c')]);
    assert.equal(state.frames, 1);
});

test('when Nextcloud wants a password: a banner, a popup on click, then the request goes on', async () => {
    const { ds, state } = setup({ session: false, silentWorks: false });
    const pending = ds.fetch('/api/save', { method: 'POST', body: 'x' });
    for (let i = 0; i < 50 && typeof state.banner !== 'function'; i++) await new Promise(r => setTimeout(r, 1));
    assert.equal(typeof state.banner, 'function');
    state.banner();                         // the person clicks "Anmelden"
    const r = await pending;
    assert.equal(r.body, '/api/save');
    assert.equal(state.popups, 1);
    assert.equal(state.banner, 'closed');
});

test('a real network error is not mistaken for a lost login', async () => {
    const ds = createDoorSession({
        fetch: async () => { throw new TypeError('offline'); }, origin: ORIGIN, sleep: async () => {},
        frame: () => { throw new Error('must not renew'); }, popup: () => null, banner: () => () => {},
    });
    await assert.rejects(ds.fetch('/api/x'), /offline/);
});

test('other origins are left alone', async () => {
    const seen = [];
    const ds = createDoorSession({
        fetch: async (u) => { seen.push(u); throw new TypeError('cors'); }, origin: ORIGIN, sleep: async () => {},
        frame: () => { throw new Error('must not renew'); }, popup: () => null, banner: () => () => {},
    });
    await assert.rejects(ds.fetch('https://example.org/x'), /cors/);
    assert.deepEqual(seen, ['https://example.org/x']);
});

test('a 401 from the door (Accept: application/json) is handled like a redirect', async () => {
    let session = false;
    const ds = createDoorSession({
        fetch: async (u, init = {}) => {
            if (init.redirect === 'manual') return session ? ok() : { status: 401, type: 'basic', ok: false };
            return session ? ok(u) : { status: 401, type: 'basic', ok: false };
        },
        origin: ORIGIN, sleep: async () => {},
        frame: () => { session = true; return () => {}; }, popup: () => null, banner: () => () => {},
    });
    const r = await ds.fetch(`${ORIGIN}/sw/quota`, { headers: { Accept: 'application/json' } });
    assert.equal(r.status, 200);
});

test("if the login can't be renewed, the request fails with its own error (no blind retry)", async () => {
    let calls = 0;
    const ds = createDoorSession({
        fetch: async (u, init = {}) => {
            if (init.redirect === 'manual') return redirected();
            calls += 1;
            throw new TypeError('NetworkError');
        },
        origin: ORIGIN, sleep: async () => {},
        frame: () => () => {}, popup: () => null, banner: () => () => {},
    });
    await assert.rejects(ds.fetch('/api/x'), /NetworkError/);
    assert.equal(calls, 1);
});
