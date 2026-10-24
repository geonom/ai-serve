"""Screenshots for the help per page (static/help/<tab>.jpg), each page in a meaningful state.

Runs headless Chromium (Playwright) against the running Playground on this machine (loopback = the owner).
Uses the GPU: a chat answer, an image, a layout image, speech, a read document. Re-run after a page changes:
    python3 scripts/help_screenshots.py [tab ...]
Neutral example content only -- these pictures are shown to everyone who opens the help. The owner's own
gallery and comparison history must never end up in them (found 2026-10-20: the first run showed both), so
the per-person lists are filtered in the browser to entries whose prompt is in EXAMPLE_PROMPTS.
"""
import io
import os
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

BASE = os.environ.get("PLAYGROUND_URL", "http://127.0.0.1:8037/")
OUT = os.path.join(os.path.dirname(__file__), "..", "static", "help")
VIEWPORT = {"width": 1440, "height": 900}
WIDTH = 1200   # stored width
EXAMPLE_PROMPTS = ("a red fox", "a cozy bakery storefront")   # lowercase prefixes of prompts that may be shown


def _example(entry) -> bool:
    return str(entry.get("prompt") or "").strip().lower().startswith(EXAMPLE_PROMPTS)


def only_examples(route):
    """GET /sw/gallery and /sw/compare: pass through only the neutral example entries."""
    if route.request.method != "GET":
        return route.continue_()
    resp = route.fetch()
    data = resp.json()
    for key in ("images", "comparisons"):
        if isinstance(data.get(key), list):
            data[key] = [e for e in data[key] if _example(e)]
    route.fulfill(response=resp, json=data)


def save(page, tab):
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)
    png = page.screenshot()
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
    img.save(os.path.join(OUT, f"{tab}.jpg"), quality=82, optimize=True)
    print("saved", tab)


def sample_document() -> str:
    """A made-up invoice to read (no real data)."""
    img = Image.new("RGB", (1240, 1600), "white")
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
        f = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        big = f = ImageFont.load_default()
    d.text((90, 90), "Rechnung Nr. 2026-0417", font=big, fill="black")
    d.text((90, 170), "Beispiel GmbH · Musterweg 1 · 12345 Musterstadt", font=f, fill="#333")
    d.text((90, 230), "Datum: 14.10.2026", font=f, fill="black")
    rows = [("Position", "Menge", "Preis"), ("Gartenschlauch 20 m", "1", "34,90 €"),
            ("Gießkanne 10 l", "2", "25,80 €"), ("Blumenerde 40 l", "3", "26,97 €")]
    y = 330
    for i, (a, b, c) in enumerate(rows):
        d.text((90, y), a, font=big if i == 0 else f, fill="black")
        d.text((760, y), b, font=f, fill="black")
        d.text((960, y), c, font=f, fill="black")
        d.line((90, y + 50, 1150, y + 50), fill="#bbb", width=2)
        y += 80
    d.text((760, y + 30), "Summe: 87,67 €", font=big, fill="black")
    path = os.path.join(tempfile.gettempdir(), "playground-help-rechnung.png")
    img.save(path)
    return path


def open_tab(page, tab):
    page.click(f'.tab-link[data-tab="{tab}"]')
    page.wait_for_timeout(800)


def shoot(page, tab):
    open_tab(page, tab)
    if tab == "overview":
        page.wait_for_function("document.querySelectorAll('#ov-cards .ov-card').length > 0", timeout=120_000)
        page.wait_for_timeout(500)
    elif tab == "gateway":
        page.fill("#chat-user-input", "Erklär in drei kurzen Sätzen, warum der Himmel blau ist.")
        page.click("#chat-send-btn")
        page.wait_for_function("document.querySelectorAll('#chat-messages .chat-message, #chat-messages .message').length >= 2"
                               " && !document.getElementById('chat-send-btn').disabled", timeout=300_000)
        page.wait_for_timeout(1500)
    elif tab == "texts":
        page.wait_for_function("document.getElementById('emb-model').options.length > 0", timeout=60_000)
        page.click("#emb-run")
        page.click("#rr-run")
        page.wait_for_function("/Fertig/.test(document.getElementById('emb-status').textContent)"
                               " && /Fertig/.test(document.getElementById('rr-status').textContent)", timeout=300_000)
    elif tab == "docs":
        page.set_input_files("#ocr-file", sample_document())
        page.click("#ocr-run-btn")
        page.wait_for_function("/Fertig|Fehlgeschlagen/.test(document.getElementById('ocr-status').textContent)", timeout=600_000)
    elif tab == "image-gen":
        page.fill("#comfy-prompt", "a red fox sitting in fresh snow at golden hour, soft light, photograph")
        page.click("#comfy-generate-btn")
        page.wait_for_function("document.getElementById('comfy-result-img').style.display !== 'none'"
                               " && document.getElementById('comfy-result-img').naturalWidth > 0", timeout=600_000)
        page.wait_for_timeout(1000)
    elif tab == "layout":
        page.fill("#lay-prompt", "A cozy bakery storefront on a sunny morning, watercolor illustration")
        page.evaluate("""() => {
            lay.elements = [
              {box: [0.18, 0.06, 0.82, 0.24], kind: 'text', caption: 'wooden shop sign', text: 'Bäckerei Sonnenschein',
               usePalette: false, palette: ''},
              {box: [0.06, 0.52, 0.42, 0.96], kind: 'character', caption: 'a baker holding a tray of bread rolls',
               text: '', usePalette: false, palette: ''},
            ];
            lay.selected = -1;
            layRender();
        }""")
        page.select_option("#lay-variant", "moss-turbo-4-bf16")
        page.click("#lay-run-btn")
        page.wait_for_function("/fertig|Fertig|Fehl/.test(document.getElementById('lay-status').textContent)", timeout=900_000)
        page.wait_for_timeout(1000)
    elif tab == "compare":
        page.wait_for_selector("#cmp-history button, #cmp-history a, #cmp-history .cmp-past", timeout=30_000)
        page.click("#cmp-history >> text=fox", timeout=10_000)
        page.wait_for_timeout(3000)
    elif tab == "mine":
        page.wait_for_timeout(4000)
    elif tab == "speech":
        page.wait_for_function("document.getElementById('tts-form').children.length > 2", timeout=120_000)
        page.fill("#tts-text", "Guten Morgen! Heute probieren wir aus, wie verschiedene Stimmen klingen.")
        page.click("#tts-generate-btn")
        page.wait_for_function("/Fertig|Fehlgeschlagen/.test(document.getElementById('tts-status').textContent)", timeout=600_000)
    save(page, tab)


def main(tabs):
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.route("**/sw/gallery", only_examples)
        page.route("**/sw/gallery?*", only_examples)
        page.route("**/sw/compare", only_examples)
        page.add_init_script("localStorage.setItem('spielwiese-help-seen', JSON.stringify("
                             "['overview','gateway','texts','docs','image-gen','layout','compare','mine','speech','quota']));"
                             "localStorage.setItem('spielwiese-theme', 'light');")
        page.goto(BASE)
        page.wait_for_timeout(2500)
        for tab in tabs:
            try:
                shoot(page, tab)
            except Exception as e:   # one page failing shouldn't stop the others
                print("FAILED", tab, type(e).__name__, str(e)[:300])
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["texts", "speech", "gateway", "docs", "image-gen", "layout", "compare", "mine"])
