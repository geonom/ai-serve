// Production server for ai-serve's Vite apps (2026-10-25, owner: apps behind the door must not run on the Vite
// dev server -- it serves any file under the project root, and its fs.deny guard was bypassed several times in
// 2025, CVE-2025-30208 and after). This one hands out ONLY regular files inside the built `root` (dist/) --
// realpath-checked, so a symlink can't lead out; no dotfiles, no directory listings -- plus whatever the app's
// own connect-style handlers `(req, res, next)` answer (its /api). Everything else is 404. Node built-ins only;
// copy this file into an app's server/ like servePolicy.js. Tests: prodServer.test.js.
import { createReadStream, realpathSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, sep } from 'node:path';

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.gif': 'image/gif', '.ico': 'image/x-icon', '.avif': 'image/avif', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.ttf': 'font/ttf', '.otf': 'font/otf', '.opus': 'audio/ogg', '.ogg': 'audio/ogg', '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav', '.m4a': 'audio/mp4', '.mp4': 'video/mp4', '.webm': 'video/webm', '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm', '.map': 'application/json; charset=utf-8',
};

function plain(res, status, text) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end(text + '\n');
}

/** URL path -> absolute file inside root, or null. */
export function resolveFile(root, rawUrl, { index = 'index.html' } = {}) {
  let path;
  try {
    path = decodeURIComponent((rawUrl || '/').split('?')[0].split('#')[0]);
  } catch {
    return null;
  }
  if (!path.startsWith('/') || path.includes('\\') || path.includes('\0')) return null;
  const segs = path.split('/').filter(Boolean);
  if (segs.some((s) => s === '..' || s.startsWith('.') || s.startsWith('@'))) return null;
  if (path === '/') segs.push(index);
  try {
    const realRoot = realpathSync(root);
    const file = realpathSync(join(realRoot, ...segs));
    if (!file.startsWith(realRoot + sep)) return null;
    return statSync(file).isFile() ? file : null;
  } catch {
    return null;
  }
}

function sendFile(req, res, file) {
  const size = statSync(file).size;
  res.setHeader('Content-Type', TYPES[extname(file).toLowerCase()] || 'application/octet-stream');
  res.setHeader('Accept-Ranges', 'bytes');
  // always revalidated (cheap: 304 when unchanged) -- guessing "hashed, cache forever" from a file name would
  // also catch public files and keep old versions in browsers
  const mtime = statSync(file).mtime;
  mtime.setMilliseconds(0);
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Last-Modified', mtime.toUTCString());
  const since = Date.parse(req.headers['if-modified-since'] || '');
  if (!req.headers.range && !Number.isNaN(since) && mtime.getTime() <= since) {
    res.statusCode = 304;
    return res.end();
  }
  let start = 0;
  let end = size - 1;
  const range = req.headers.range;
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (m && (m[1] || m[2])) {
      if (m[1]) {
        start = Number(m[1]);
        end = m[2] ? Math.min(Number(m[2]), size - 1) : size - 1;
      } else {
        start = Math.max(size - Number(m[2]), 0);
      }
      if (start > end || start >= size) {
        res.setHeader('Content-Range', `bytes */${size}`);
        return plain(res, 416, 'Range not satisfiable');
      }
      res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    }
  }
  if (res.statusCode !== 206) res.statusCode = 200;
  res.setHeader('Content-Length', String(end - start + 1));
  if (req.method === 'HEAD' || size === 0) return res.end();
  createReadStream(file, { start, end }).on('error', () => res.destroy()).pipe(res);
}

/**
 * @param {{root: string, handlers?: Function[], log?: (msg: string) => void}} opts
 *   root: the built folder (dist/); handlers: the app's connect-style middlewares, tried in order first.
 */
export function createProdServer({ root, handlers = [], log = (m) => console.error(m) }) {
  return createServer((req, res) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'same-origin');
    const fail = (e) => {
      log(`[prodServer] ${req.method} ${req.url}: ${e?.stack || e}`);
      if (!res.headersSent) plain(res, 500, 'Internal error');
      else res.destroy();
    };
    let i = 0;
    const next = (err) => {
      if (err) return fail(err);
      const h = handlers[i++];
      if (h) {
        try {
          const r = h(req, res, next);
          if (r && typeof r.catch === 'function') r.catch(fail);
        } catch (e) {
          fail(e);
        }
        return;
      }
      const file = resolveFile(root, req.url);
      if (!file) return plain(res, 404, 'Not found');
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        res.setHeader('Allow', 'GET, HEAD');
        return plain(res, 405, 'Method not allowed');
      }
      sendFile(req, res, file);
    };
    next();
  });
}
