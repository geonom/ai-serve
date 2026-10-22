// What the Vite dev server may hand out (server/servePolicy.js). Security: Vite serves ANY file under
// the project root by default -- found 2026-10-17: data/users/*/state.json (everyone's progress) and
// scripts/deploy.env (FTP credentials) were downloadable. Allowlist instead: the page, src/, public/,
// Vite's own client and pre-bundled deps, /api/. Everything else -> 404.
import { describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { isAllowed } from './servePolicy.js';

const root = mkdtempSync(join(tmpdir(), 'policy-'));
const pub = join(root, 'public');
mkdirSync(join(pub, 'audio'), { recursive: true });
writeFileSync(join(pub, 'sw.js'), '');
writeFileSync(join(pub, 'audio', 'a.opus'), '');
writeFileSync(join(root, 'secret.txt'), 'x');
symlinkSync(join(root, 'secret.txt'), join(pub, 'link.txt'));
const allowed = (p) => isAllowed(p, { publicDir: pub });

describe('serve policy', () => {
  it('serves the app', () => {
    for (const p of ['/', '/index.html', '/src/main.js', '/src/views/Learn.js?import', '/sw.js', '/audio/a.opus',
                     '/@vite/client', '/@vite/env', '/@react-refresh', '/node_modules/.vite/deps/react.js?v=1',
                     '/@id/__x00__vite/preload-helper', '/api/state', '/index.html?html-proxy&index=0.js']) {
      expect(allowed(p), p).toBe(true);
    }
  });

  it('refuses everything else', () => {
    for (const p of ['/data/users/abc/state.json', '/scripts/deploy.env', '/.env', '/.git/config', '/server/state.js',
                     '/package.json', '/vite.config.js', '/@fs/etc/passwd', '/@fs' + root + '/secret.txt',
                     '/@id/' + root + '/secret.txt', '/src/../data/x.json', '/src/%2e%2e/data/x.json',
                     '/src/..%2fdata/x.json', '/node_modules/jose/package.json', '/link.txt', '/secret.txt',
                     '/src/.hidden', '/missing.js', '/audio/../../secret.txt', '/src\\..\\x']) {
      expect(allowed(p), p).toBe(false);
    }
  });
});
