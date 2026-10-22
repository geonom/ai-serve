/* door-session.js -- keep an app working when its login at the door (Pomerium, apps.example.org) runs out.
 *
 * Include it first in any app behind the door:  <script src="door-session.js"></script>
 * It wraps window.fetch for same-origin requests. When one fails because the door no longer knows the
 * person (the door redirects to its login -> fetch throws, or answers 401 to `Accept: application/json`),
 * the login is renewed in the background and the request is sent again. The page is never reloaded, so
 * typed text, the reading position and ticked boxes stay (owner, 2026-10-20).
 *
 * Renewal: first silently -- a hidden frame (no scripts) walks through the door's login; while the person
 * is still signed in at Nextcloud that needs no input. If that doesn't bring the session back (Nextcloud
 * wants a password), a small notice offers "Anmelden": a click opens the login in a small window, which is
 * closed again once the session is back. Meanwhile the waiting requests just wait. One renewal at a time.
 * A real network error (offline) is passed on unchanged; other origins are left alone.
 * Not covered: XMLHttpRequest, EventSource, WebSocket. Tests: node --test sdk/door/door-session.test.mjs
 */
(function (root) {
    'use strict';
    // A path behind the door that no app serves and no service worker caches: while signed in the app
    // answers it (404 is fine), otherwise the door redirects to its login (or 401 for JSON).
    const PROBE = '/.door-probe';
    const RENEW = '/.door-renew';       // walked through the login in a frame / window; the app's answer doesn't matter

    function createDoorSession(d) {
        const raw = d.fetch;
        const sleep = d.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
        let renewing = null;

        function sameOrigin(input) {
            try {
                const url = new URL(typeof input === 'string' ? input : input.url, d.origin + '/');
                return url.origin === d.origin;
            } catch (e) { return false; }
        }

        // true = the door no longer knows us; false = signed in; null = can't tell (offline)
        async function lost() {
            try {
                const r = await raw(d.origin + PROBE, { redirect: 'manual', credentials: 'same-origin', cache: 'no-store',
                                                     headers: { Accept: 'application/json' } });
                return r.type === 'opaqueredirect' || r.status === 401 || r.status === 0;
            } catch (e) { return null; }
        }

        async function until(test, tries, ms) {
            for (let i = 0; i < tries; i++) {
                if (await test()) return true;
                await sleep(ms);
            }
            return false;
        }

        async function silent() {
            const remove = d.frame(d.origin + RENEW + '?t=' + Date.now());
            try { return await until(async () => (await lost()) === false, 12, 1000); }
            finally { try { remove && remove(); } catch (e) { /* gone */ } }
        }

        async function interactive() {
            let win = null;
            const close = d.banner(() => { win = d.popup(d.origin + RENEW + '?t=' + Date.now()); });
            try { return await until(async () => (await lost()) === false, 1800, 1000); }   // up to 30 min
            finally {
                try { close && close(); } catch (e) { /* gone */ }
                try { win && win.close(); } catch (e) { /* gone */ }
            }
        }

        function renew() {
            if (!renewing) {
                renewing = (async () => (await silent()) || (await interactive()))()
                    .finally(() => { renewing = null; });
            }
            return renewing;
        }

        async function wrapped(input, init) {
            if (!sameOrigin(input)) return raw(input, init);
            const again = (typeof Request !== 'undefined' && input instanceof Request) ? input.clone() : input;
            let r;
            try {
                r = await raw(input, init);
            } catch (err) {
                if ((await lost()) !== true) throw err;       // offline or a real error: not ours to handle
                if (!(await renew())) throw err;              // couldn't renew: no blind retry
                return raw(again, init);
            }
            if ((r.status === 401 || r.type === 'opaqueredirect') && (await lost()) === true) {
                if (!(await renew())) return r;
                return raw(again, init);
            }
            return r;
        }

        return { fetch: wrapped, renew, lost };
    }

    function install(win) {
        if (!win || !win.fetch || win.__doorSession) return;
        const doc = win.document;
        const ds = createDoorSession({
            fetch: win.fetch.bind(win),
            origin: win.location.origin,
            frame: (url) => {
                const f = doc.createElement('iframe');
                f.setAttribute('sandbox', 'allow-same-origin allow-forms');   // cookies yes, the app's scripts no
                f.setAttribute('aria-hidden', 'true');
                f.style.cssText = 'position:absolute;width:1px;height:1px;border:0;left:-9999px;top:-9999px;';
                f.src = url;
                doc.body.appendChild(f);
                return () => f.remove();
            },
            popup: (url) => win.open(url, 'door-login', 'width=480,height=640'),
            banner: (onClick) => {
                const bar = doc.createElement('div');
                bar.setAttribute('role', 'status');
                bar.style.cssText = 'position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:2147483647;'
                    + 'background:#1f2430;color:#fff;padding:10px 14px;border-radius:10px;font:14px system-ui,sans-serif;'
                    + 'box-shadow:0 4px 16px rgba(0,0,0,.3);display:flex;gap:10px;align-items:center;max-width:calc(100% - 32px);';
                const text = doc.createElement('span');
                text.textContent = 'Deine Anmeldung ist abgelaufen – deine Eingaben bleiben erhalten.';
                const btn = doc.createElement('button');
                btn.textContent = 'Anmelden';
                btn.style.cssText = 'background:#3550D8;color:#fff;border:0;border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer;';
                btn.addEventListener('click', onClick);
                bar.append(text, btn);
                doc.body.appendChild(bar);
                return () => bar.remove();
            },
        });
        win.__doorSession = ds;
        win.fetch = ds.fetch;
    }

    if (typeof module !== 'undefined' && module.exports) module.exports = { createDoorSession, install };
    if (typeof window !== 'undefined') install(window);
})(typeof globalThis !== 'undefined' ? globalThis : this);
