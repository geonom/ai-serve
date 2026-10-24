// Welcome (owner, 2026-10-22): what the gateway is, how to use it as an API (with this machine's real address), and
// how to start and open the Playground. Opens by itself on the first visit in a browser, then from the sidebar's
// "Welcome & help". Uses el()/api()/serviceAction()/serviceOpenControl()/renderServiceAutostart() from services.js.
(function () {
    const KEY = 'ai-serve-welcome-seen';
    const dlg = document.getElementById('welcome-dialog');
    if (!dlg) return;
    const $ = (id) => document.getElementById(id);
    const seen = () => { try { return localStorage.getItem(KEY) === '1'; } catch (e) { return false; } };
    const markSeen = () => { try { localStorage.setItem(KEY, '1'); } catch (e) { /* once per visit then */ } };

    function fillApi() {
        const base = `${window.location.origin}/v1`;
        $('welcome-base').textContent = base;
        $('welcome-curl').textContent =
`curl ${base}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'`;
        $('welcome-python').textContent =
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="not-needed-here")
reply = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)`;
    }

    async function renderPlayground() {
        const box = $('welcome-playground');
        let s = null;
        try { s = ((await api('/api/services')).services || []).find(x => x.id === 'playground'); } catch (e) { s = null; }
        if (!s) {
            box.replaceChildren(el('p', { class: 'muted', text: tx('The Playground isn\'t installed here (it comes with ai-serve in apps/playground).') }));
            return;
        }
        const running = s.instances && s.instances.length > 0;
        const parts = [el('p', {}, el('span', { class: `pill ${running ? 'pill-ok' : 'pill-off'}` }, el('span', { class: 'pill-dot' }),
                                      el('span', { text: running ? tx('running') : tx('stopped') })))];
        const actions = el('div', { class: 'app-actions' });
        if (running) {
            const open = serviceOpenControl(s);
            if (open) actions.append(open);
        } else {
            actions.append(el('button', { class: 'btn btn-primary btn-sm', text: tx('Start the Playground'), onclick: async (e) => {
                e.target.disabled = true;
                e.target.textContent = tx('Starting…');
                await serviceAction('playground', 'start');
                for (let i = 0; i < 20; i++) {             // until it answers, then offer "Open"
                    await new Promise(r => setTimeout(r, 1000));
                    const now = ((await api('/api/services').catch(() => ({}))).services || []).find(x => x.id === 'playground');
                    if (now && now.instances && now.instances.length) break;
                }
                renderPlayground();
            } }));
        }
        actions.append(renderServiceAutostart(s));
        parts.push(actions);
        box.replaceChildren(...parts);
    }

    function open() {
        fillApi();
        renderPlayground();
        if (!dlg.open) dlg.showModal();
        markSeen();
    }

    $('welcome-btn')?.addEventListener('click', open);
    dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });   // click outside closes
    dlg.querySelectorAll('[data-goto]').forEach(b => b.addEventListener('click', () => dlg.close()));
    if (!seen()) setTimeout(open, 400);
})();
