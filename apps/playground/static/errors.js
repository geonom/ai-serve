// Script errors must be seen and traceable (2026-11-22: sending a chat did nothing on the owner's Mac -- no
// message, no request reached the server; the reason was only in that browser's console). Loaded first, so it
// also catches errors while the other scripts load: shows a note and tells the Playground's server, which logs it.
(function () {
    function note(msg) {
        if (typeof window.showToast === 'function') return window.showToast(msg, 'error');
        let box = document.getElementById('sw-toasts');
        if (!box) {
            box = document.createElement('div');
            box.id = 'sw-toasts';
            box.setAttribute('role', 'status');
            (document.body || document.documentElement).appendChild(box);
        }
        const t = document.createElement('div');
        t.className = 'sw-toast error';
        t.textContent = msg;
        t.addEventListener('click', () => t.remove());
        box.appendChild(t);
        setTimeout(() => t.remove(), 15000);
    }

    let sent = 0;
    function report(message, where, stack) {
        const page = (document.querySelector('.nav-item.active, [aria-current="page"]') || {}).textContent || '';
        note((window.tx ? window.tx('Something went wrong in this page: {err}', { err: message })
                        : 'Something went wrong in this page: ' + message));
        // a page stuck in "sending" gets its controls back
        ['chat-send-btn', 'chat-user-input'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = false; });
        if (sent++ >= 20) return;   // one broken loop must not flood the log
        try {
            fetch('/sw/client-error', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, keepalive: true,
                body: JSON.stringify({ message: String(message).slice(0, 500), where: String(where || '').slice(0, 200),
                                       page: page.trim().slice(0, 40), stack: String(stack || '').slice(0, 1200) }),
            }).catch(() => {});
        } catch (e) { /* nothing more to do */ }
    }

    window.addEventListener('error', e => {
        if (!e.message) return;   // a failed <img>/<script> load, not a script error
        report(e.message, `${(e.filename || '').split('/').pop()}:${e.lineno || 0}:${e.colno || 0}`, e.error && e.error.stack);
    });
    window.addEventListener('unhandledrejection', e => {
        const r = e.reason || {};
        report(r.message || String(r), 'promise', r.stack);
    });
})();
