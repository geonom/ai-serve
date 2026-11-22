/* door-session.js -- keep an app working when its login at the door (Pomerium, apps.example.org) runs out.
 *
 * Include it first in any app behind the door:  <script src="door-session.js"></script>
 * It wraps window.fetch for same-origin requests. When one fails because the door no longer knows the
 * person (the door redirects to its login -> fetch throws, or answers 401 to `Accept: application/json`),
 * the login is renewed in the background and the request is sent again. The page is never reloaded, so
 * typed text, the reading position and ticked boxes stay (owner, 2026-10-20).
 *
 * Nothing is checked on a timer: only a request that fails asks whether the login is gone (owner, 2026-11-22).
 * Renewal: first silently -- a hidden frame (no scripts) walks through the door's login; while the person is still
 * signed in at Nextcloud that needs no input. One renewal for all tabs (a lock shared by the tabs; the next tab
 * first checks whether the session is back), and a failed silent try isn't repeated for 5 minutes in any tab --
 * each try is a Nextcloud login, and Nextcloud's host answers too many with 429 (2026-11-22: tabs renewing at
 * once and a retry on every background poll kept the login broken). Then the page is "signed out": a small
 * notice offers "Anmelden"; a send (POST, PUT ...) waits for it, a background read (GET) fails at once and asks
 * nothing more. A click opens the login in a small window; the door is asked every 2 s only while that window is
 * open. A login in another tab releases the waiting sends here too (BroadcastChannel).
 * A real network error (offline) is passed on unchanged; other origins are left alone.
 * Not covered: XMLHttpRequest, EventSource, WebSocket. Tests: node --test sdk/door/door-session.test.mjs
 */
(function (root) {
    'use strict';
    // A path behind the door that no app serves and no service worker caches: while signed in the app
    // answers it (404 is fine), otherwise the door redirects to its login (or 401 for JSON).
    const PROBE = '/.door-probe';
    const RENEW = '/.door-renew';       // walked through the login in a frame / window; the app's answer doesn't matter

    const SILENT_GAP_MS = 5 * 60 * 1000;      // a failed quiet renewal waits this long, across tabs
    const WINDOW_CHECK_MS = 2000;               // how often the door is asked while the login window is open
    const WINDOW_CHECKS = 300;                  // ... for at most 10 minutes
    const LAST_TRY = 'door-session-silent-at';

    function createDoorSession(d) {
        const raw = d.fetch;
        const sleep = d.sleep || ((ms) => new Promise((r) => setTimeout(r, ms)));
        const now = d.now || (() => Date.now());
        const lock = d.lock || ((name, fn) => fn());
        const memory = new Map();
        const store = d.store || { get: (k) => memory.get(k) ?? null, set: (k, v) => memory.set(k, v) };
        const signal = d.signal || null;
        let renewing = null;
        let signedOut = null;       // {promise, resolve, close, checking} while the person has to log in again

        function sameOrigin(input) {
            try {
                const url = new URL(typeof input === 'string' ? input : input.url, d.origin + '/');
                return url.origin === d.origin;
            } catch (e) { return false; }
        }

        function isSend(input, init) {
            const m = ((init && init.method) || (typeof input === 'object' && input && input.method) || 'GET');
            return !/^(GET|HEAD)$/i.test(m);
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
            const remove = d.frame(d.origin + RENEW + '?t=' + now());
            try { return await until(async () => (await lost()) === false, 12, 1000); }
            finally { try { remove && remove(); } catch (e) { /* gone */ } }
        }

        function back() {           // the session is back: close the notice, release the waiting sends
            const s = signedOut;
            if (!s) return;
            signedOut = null;
            try { s.close && s.close(); } catch (e) { /* gone */ }
            s.resolve(true);
        }

        async function openLogin() {
            const s = signedOut;
            if (!s || s.checking) return;
            s.checking = true;
            const win = d.popup(d.origin + RENEW + '?t=' + now());
            try {
                for (let i = 0; i < WINDOW_CHECKS && signedOut === s; i++) {
                    await sleep(WINDOW_CHECK_MS);
                    if (signedOut !== s) return;                     // another tab's login released us
                    if ((await lost()) === false) break;
                    if (win && win.closed) {                         // closed without a login: stop asking
                        if ((await lost()) !== false) return;
                        break;
                    }
                }
                if (signedOut === s && (await lost()) === false) {
                    if (signal) try { signal.send('back'); } catch (e) { /* no channel */ }
                    back();
                }
            } finally {
                s.checking = false;
                if (signedOut !== s) try { win && !win.closed && win.close(); } catch (e) { /* gone */ }
            }
        }

        function waitForPerson() {
            if (!signedOut) {
                let resolve;
                const promise = new Promise((r) => { resolve = r; });
                signedOut = { promise, resolve, close: null, checking: false };
                signedOut.close = d.banner(() => { openLogin(); });
            }
            return signedOut.promise;
        }

        // one quiet renewal for all tabs; false = the person has to log in
        function renewSilently() {
            if (signedOut) return Promise.resolve(false);
            if (!renewing) {
                renewing = Promise.resolve(lock('door-session-renew', async () => {
                    if ((await lost()) === false) return true;       // another tab renewed meanwhile
                    const last = store.get(LAST_TRY);
                    if (last !== null && last !== undefined && now() - Number(last) < SILENT_GAP_MS) {
                        return false;                                // tried lately: don't ask Nextcloud again
                    }
                    store.set(LAST_TRY, String(now()));
                    const ok = await silent();
                    if (ok && signal) try { signal.send('back'); } catch (e) { /* no channel */ }
                    return ok;
                })).finally(() => { renewing = null; });
            }
            return renewing;
        }

        async function renew(wait) {
            if (await renewSilently()) return true;
            const person = waitForPerson();
            return wait ? person : false;
        }

        if (signal) {
            try {
                signal.listen((m) => {
                    if (m === 'back' && signedOut) lost().then((l) => { if (l === false) back(); });
                });
            } catch (e) { /* no channel */ }
        }

        async function wrapped(input, init) {
            if (!sameOrigin(input)) return raw(input, init);
            const send = isSend(input, init);
            const again = (typeof Request !== 'undefined' && input instanceof Request) ? input.clone() : input;
            let r;
            try {
                r = await raw(input, init);
            } catch (err) {
                if (signedOut && !send) throw err;                    // known: signed out, nothing to ask
                if ((await lost()) !== true) throw err;              // offline or a real error: not ours to handle
                if (!(await renew(send))) throw err;                 // couldn't renew: no blind retry
                return raw(again, init);
            }
            if (r.status === 401 || r.type === 'opaqueredirect') {
                if (signedOut && !send) return r;
                if ((await lost()) === true) {
                    if (!(await renew(send))) return r;
                    return raw(again, init);
                }
            }
            return r;
        }

        return { fetch: wrapped, renew: () => renew(true), lost };
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
            lock: (win.navigator && win.navigator.locks)
                ? (name, fn) => win.navigator.locks.request(name, fn) : undefined,
            store: {
                get: (k) => { try { return win.localStorage.getItem(k); } catch (e) { return null; } },
                set: (k, v) => { try { win.localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
            },
            signal: (typeof win.BroadcastChannel === 'function') ? (() => {
                const ch = new win.BroadcastChannel('door-session');
                return { send: (m) => ch.postMessage(m), listen: (l) => { ch.onmessage = (e) => l(e.data); } };
            })() : undefined,
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
