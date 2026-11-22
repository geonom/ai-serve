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

// ---- 2026-11-22: a login that broke at Nextcloud's host (429 Too Many Requests) was made worse by us: every tab
// renewed at once, a failed renewal was tried again on the next background poll, and the door was asked once a
// second for 30 minutes. Now: nothing is checked unless a request fails; one renewal for all tabs; a failed
// silent renewal waits 5 minutes; background reads never wait for the person; the door is only asked while the
// login window is open.

function shared() {
    let held = Promise.resolve();
    const data = new Map();
    const listeners = [];
    return {
        lock: (name, fn) => { const run = held.then(fn); held = run.catch(() => {}); return run; },
        store: { get: (k) => data.get(k) ?? null, set: (k, v) => data.set(k, v) },
        signal: { send: (m) => listeners.forEach((l) => l(m)), listen: (l) => listeners.push(l) },
    };
}

function tab(world, common, extra = {}) {
    const state = { frames: 0, popups: 0, banner: null, probes: 0, sent: 0 };
    const ds = createDoorSession({
        fetch: async (input, init = {}) => {
            if (init.redirect === 'manual') { state.probes += 1; return world.session ? ok() : redirected(); }
            state.sent += 1;
            return world.session ? ok(typeof input === 'string' ? input : input.url) : redirected();
        },
        origin: ORIGIN, sleep: () => new Promise((r) => setTimeout(r, 1)), now: () => world.now,
        frame: () => { state.frames += 1; world.silentAttempts = (world.silentAttempts || 0) + 1;
                       if (world.silentWorks) world.session = true; return () => {}; },
        popup: () => { state.popups += 1; const w = { closed: false, close() { w.closed = true; } };
                       world.window = w; return w; },
        banner: (onClick) => { state.banner = onClick; return () => { state.banner = 'closed'; }; },
        ...common, ...extra,
    });
    return { ds, state };
}

test('two tabs share one renewal -- the second one finds the session back and asks nobody', async () => {
    const world = { session: false, silentWorks: true, now: 0 };
    const common = shared();
    const a = tab(world, common), b = tab(world, common);
    await Promise.all([a.ds.fetch('/a', { method: 'POST' }), b.ds.fetch('/b', { method: 'POST' })]);
    assert.equal(world.silentAttempts, 1);
});

test('a failed silent renewal is not tried again for 5 minutes, in no tab', async () => {
    const world = { session: false, silentWorks: false, now: 0 };
    const common = shared();
    const a = tab(world, common), b = tab(world, common);
    await a.ds.fetch('/poll');                          // a background read: fails, banner shown
    assert.equal(world.silentAttempts, 1);
    world.now = 60_000;
    await b.ds.fetch('/poll');                          // another tab a minute later: no new Nextcloud login
    assert.equal(world.silentAttempts, 1);
    assert.equal(typeof b.state.banner, 'function');
    const c = tab(world, common);
    world.now = 5 * 60_000 + 1;
    await c.ds.fetch('/poll');                          // after 5 minutes one quiet try again
    assert.equal(world.silentAttempts, 2);
});

test('while signed out, background reads fail at once and ask nothing more', async () => {
    const world = { session: false, silentWorks: false, now: 0 };
    const t = tab(world, shared());
    const first = await t.ds.fetch('/api/status');
    assert.equal(first.type, 'opaqueredirect');
    const probes = t.state.probes;
    for (let i = 0; i < 20; i++) await t.ds.fetch('/api/status');   // the page keeps polling
    assert.equal(world.silentAttempts, 1);
    assert.ok(t.state.probes - probes <= 20, 'at most the one cheap check per failed request');
    assert.equal(t.state.popups, 0);
});

test('a send waits for the person, and the door is only asked while the login window is open', async () => {
    const world = { session: false, silentWorks: false, now: 0 };
    const t = tab(world, shared());
    const pending = t.ds.fetch('/api/save', { method: 'POST', body: 'Kapitel 3' });
    for (let i = 0; i < 50 && typeof t.state.banner !== 'function'; i++) await new Promise((r) => setTimeout(r, 1));
    await new Promise((r) => setTimeout(r, 20));
    const before = t.state.probes;
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(t.state.probes, before, 'no checks while nobody is logging in');
    t.state.banner();                                   // "Anmelden"
    world.window.closed = true;                         // closed without logging in
    await new Promise((r) => setTimeout(r, 30));
    const afterClose = t.state.probes;
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(t.state.probes, afterClose, 'checks stop when the window is closed');
    t.state.banner();                                   // second try, this time with a login
    world.session = true;
    const r = await pending;
    assert.equal(r.body, '/api/save');
    assert.equal(t.state.banner, 'closed');
    assert.equal(world.window.closed, true);
});

test("a login in another tab releases this tab's waiting send", async () => {
    const world = { session: false, silentWorks: false, now: 0 };
    const common = shared();
    const a = tab(world, common), b = tab(world, common);
    const pending = b.ds.fetch('/api/save', { method: 'POST' });
    for (let i = 0; i < 50 && typeof b.state.banner !== 'function'; i++) await new Promise((r) => setTimeout(r, 1));
    a.ds.fetch('/x', { method: 'POST' });
    for (let i = 0; i < 50 && typeof a.state.banner !== 'function'; i++) await new Promise((r) => setTimeout(r, 1));
    a.state.banner();
    world.session = true;                               // logged in through tab a's window
    const r = await pending;
    assert.equal(r.body, '/api/save');
    assert.equal(b.state.popups, 0);
});
