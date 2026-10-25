// The production server for ai-serve's Vite apps (prodServer.js). Security: it may hand out ONLY files of the
// built dist/ folder plus what the app's own handlers answer -- never data/, server/, package.json, dotfiles,
// Vite's dev paths (/@fs/, ?raw) or anything a symlink points to outside dist/. Everything else -> 404.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createProdServer } from './prodServer.js';

const project = mkdtempSync(join(tmpdir(), 'prod-'));
const dist = join(project, 'dist');
mkdirSync(join(dist, 'assets'), { recursive: true });
mkdirSync(join(dist, 'audio', 'fr'), { recursive: true });
mkdirSync(join(project, 'data', 'users'), { recursive: true });
writeFileSync(join(dist, 'index.html'), '<!doctype html><title>app</title>');
writeFileSync(join(dist, 'assets', 'index-abc123.js'), 'console.log(1)');
writeFileSync(join(dist, 'audio', 'fr', 'bonjour.opus'), Buffer.alloc(1000, 7));
writeFileSync(join(dist, '.hidden'), 'secret');
writeFileSync(join(project, 'data', 'users', 'state.json'), '{"secret":true}');
writeFileSync(join(project, 'package.json'), '{}');
symlinkSync(join(project, 'data'), join(dist, 'leak'));

let server;
let base;
beforeAll(async () => {
  const api = (req, res, next) => {
    if (req.url.split('?')[0] !== '/api/state') return next();
    res.setHeader('Content-Type', 'application/json');
    res.end('{"ok":true}');
  };
  const boom = (req, res, next) => (req.url === '/api/boom' ? Promise.reject(new Error('/home/x/secret path')) : next());
  server = createProdServer({ root: dist, handlers: [api, boom] });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});
afterAll(() => server.close());

const get = (path, init) => fetch(base + path, init);

describe('what is served', () => {
  it('serves the page, built assets and public files', async () => {
    let r = await get('/');
    expect(r.status).toBe(200);
    expect(r.headers.get('content-type')).toMatch(/text\/html/);
    expect(await r.text()).toContain('<title>app</title>');
    r = await get('/assets/index-abc123.js');
    expect(r.status).toBe(200);
    expect(r.headers.get('content-type')).toMatch(/javascript/);
    expect(r.headers.get('cache-control')).toBe('no-cache');
    const again = await get('/assets/index-abc123.js', { headers: { 'If-Modified-Since': r.headers.get('last-modified') } });
    expect(again.status).toBe(304);
    expect((await get('/index.html')).status).toBe(200);
    expect((await get('/assets/index-abc123.js?v=2')).status).toBe(200);
  });

  it("passes the app's own routes to its handlers", async () => {
    const r = await get('/api/state');
    expect(r.status).toBe(200);
    expect(await r.json()).toEqual({ ok: true });
  });

  it('answers handler errors with a plain 500 that leaks nothing', async () => {
    const r = await get('/api/boom');
    expect(r.status).toBe(500);
    expect(await r.text()).not.toContain('secret');
  });

  it('sets safety headers', async () => {
    const r = await get('/');
    expect(r.headers.get('x-content-type-options')).toBe('nosniff');
    expect(r.headers.get('referrer-policy')).toBe('same-origin');
  });
});

describe('what is never served', () => {
  it.each([
    '/data/users/state.json', '/../data/users/state.json', '/%2e%2e/data/users/state.json',
    '/..%2fdata%2fusers%2fstate.json', '/package.json', '/../package.json', '/.hidden', '/%2ehidden',
    '/leak/users/state.json', '/@fs' + project + '/data/users/state.json', '/src/main.js',
    '/server/state.js', '/node_modules/.vite/deps/x.js', '/assets/', '/audio', '/x%00.html', '/a\\..\\package.json',
    '/%E0%A4%A', '/nope.js',
  ])('%s -> 404', async (path) => {
    const r = await get(path);
    expect(r.status).toBe(404);
    expect(await r.text()).not.toContain('secret');
  });

  it('refuses methods other than GET/HEAD for files', async () => {
    expect((await get('/index.html', { method: 'POST' })).status).toBe(405);
    const head = await get('/index.html', { method: 'HEAD' });
    expect(head.status).toBe(200);
    expect(await head.text()).toBe('');
  });
});

describe('audio needs ranges (Safari plays media only with Range support)', () => {
  it('answers a byte range with 206', async () => {
    const r = await get('/audio/fr/bonjour.opus', { headers: { Range: 'bytes=0-99' } });
    expect(r.status).toBe(206);
    expect(r.headers.get('content-range')).toBe('bytes 0-99/1000');
    expect((await r.arrayBuffer()).byteLength).toBe(100);
    expect(r.headers.get('accept-ranges')).toBe('bytes');
  });

  it('answers an open-ended and a suffix range', async () => {
    let r = await get('/audio/fr/bonjour.opus', { headers: { Range: 'bytes=900-' } });
    expect(r.headers.get('content-range')).toBe('bytes 900-999/1000');
    r = await get('/audio/fr/bonjour.opus', { headers: { Range: 'bytes=-10' } });
    expect(r.headers.get('content-range')).toBe('bytes 990-999/1000');
  });

  it('refuses an unsatisfiable range with 416', async () => {
    const r = await get('/audio/fr/bonjour.opus', { headers: { Range: 'bytes=5000-6000' } });
    expect(r.status).toBe(416);
  });
});
