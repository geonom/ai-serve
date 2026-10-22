// What the Vite dev server may hand out -- an allowlist in front of Vite's own middlewares, because Vite
// serves ANY file under the project root (found 2026-10-17: data/users/*/state.json and
// scripts/deploy.env were downloadable). Allowed: the page, src/, files in public/, Vite's client and
// pre-bundled deps, virtual modules, /api/. Everything else -> 404. Tests: server/servePolicy.test.js.
// The same file is used by every Vite app of ai-serve-ui (pv-calc, TrioLingo, Kristallhelden).
import { lstatSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';

const PREFIXES = ['/src/', '/api/', '/@vite/client', '/@vite/env', '/@react-refresh',
                  '/node_modules/.vite/deps/', '/node_modules/vite/dist/client/', '/@id/__x00__'];

export function isAllowed(rawUrl, { publicDir, pages = ['/', '/index.html'], extraPrefixes = [] }) {
  let path;
  try {
    path = decodeURIComponent((rawUrl || '/').split('?')[0].split('#')[0]);
  } catch {
    return false;
  }
  if (!path.startsWith('/') || path.includes('\\') || path.includes('\0')) return false;
  if (path.split('/').some((seg) => seg === '..' || (seg.startsWith('.') && seg !== '.vite'))) return false;
  if (pages.includes(path)) return true;
  if ([...PREFIXES, ...extraPrefixes].some((p) => path.startsWith(p))) {
    return !(path.startsWith('/node_modules/') && !path.startsWith('/node_modules/.vite/deps/')
             && !path.startsWith('/node_modules/vite/dist/client/'));
  }
  if (!publicDir) return false;
  const base = resolve(publicDir);
  const file = resolve(join(base, path));
  if (!file.startsWith(base + sep)) return false;
  try {
    const st = lstatSync(file);
    return st.isFile() && !st.isSymbolicLink();
  } catch {
    return false;
  }
}

export function servePolicyPlugin(opts) {
  const deny = (req, res, next) => {
    if (isAllowed(req.url, opts)) return next();
    res.statusCode = 404;
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.end('Nicht gefunden\n');
  };
  return {
    name: 'llama-serve-serve-policy',
    enforce: 'pre',
    configureServer(server) { server.middlewares.use(deny); },
    configurePreviewServer(server) { server.middlewares.use(deny); },
  };
}
