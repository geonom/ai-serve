// Help per page (owner, 2026-10-20): what a page can do, a screenshot of it in use, and first steps. Opens by
// itself the first time someone opens a page (remembered in their browser; if the browser keeps nothing, once
// per visit), and again from the "?" in the bar. Screenshots: static/help/<tab>.jpg.
// Pure part (tests: node --test tests/help.test.mjs): which pages have help, and when it opens by itself.

const HELP = {
    overview: {
        title: 'Willkommen auf der Spielwiese',
        intro: 'Die Spielwiese ist ein Ort zum Ausprobieren: Chat mit Sprachmodellen, Bilder erzeugen und bearbeiten, '
             + 'Texte vorlesen und abschreiben, Dokumente lesen – alles läuft auf privaten Rechnern, die dem Netzwerk '
             + 'hinzugefügt wurden.',
        caption: 'Die Übersicht zeigt, was das Gateway gerade anbietet.',
        steps: [
            'Auf der Übersicht eine Karte anklicken – oder oben in der Leiste eine Seite wählen.',
            'Jede Seite erklärt sich beim ersten Öffnen selbst; das „?“ oben rechts zeigt die Erklärung wieder.',
            'Beim ersten Mal lädt ein Modell ein paar Sekunden bis Minuten – danach geht es schneller.',
        ],
        tip: 'Ausgegraute Karten kann das Gateway gerade nicht – z. B. weil ein Dienst nicht läuft.',
    },
    gateway: {
        title: 'Chat',
        intro: 'Mit Sprachmodellen schreiben – mit allen, die das Gateway erreicht: auf den Rechnern im Netzwerk '
             + 'und weiteren Modellservern wie Strata. Das gewählte Modell wird beim ersten Schreiben automatisch geladen.',
        caption: 'Ein Gespräch mit Gemma 4; links Modell und Einstellungen.',
        steps: [
            'Links unter „Modell“ eines wählen – oder „Auto“ lassen, dann antwortet das gerade laufende.',
            'Unten eine Nachricht schreiben und „Senden“ drücken.',
            'Modelle mit „Bild“ verstehen auch Bilder: unter „Bild oder Ton anhängen“ eines dazulegen.',
            '„Denken“-Modelle überlegen erst; der Gedankengang lässt sich unter der Antwort aufklappen.',
        ],
        tip: 'Für Antworten in fester Form (JSON) ein Schema eintragen – „Schema bauen“ hilft dabei.',
    },
    texts: {
        title: 'Texte vergleichen',
        intro: 'Wie ähnlich sich Texte in ihrer Bedeutung sind – und welche Texte am besten zu einer Frage passen. '
             + 'Das ist die Grundlage jeder Suche „nach Sinn statt nach Wörtern“.',
        caption: 'Hund und Welpe liegen nah beieinander, Börse und Aktien auch – über Kreuz kaum.',
        steps: [
            'Oben links Texte eintragen, einen pro Zeile, und „Vergleichen“ drücken.',
            'Rechts zeigt die Tabelle, wie nah sich je zwei Texte sind (1 = gleiche Bedeutung).',
            'Unten links eine Frage und mehrere Texte eintragen und „Sortieren“ drücken.',
            'Das Rerank-Modell bewertet jeden Text zusammen mit der Frage – der passendste steht oben.',
        ],
        tip: 'Funktioniert sprachübergreifend: ein deutscher und ein englischer Satz mit gleicher Bedeutung liegen nah beieinander.',
    },
    docs: {
        title: 'Dokumente lesen',
        intro: 'Ein Foto oder PDF abschreiben lassen – als sauberer Text mit Überschriften und Tabellen, oder gezielt '
             + 'einzelne Angaben herausziehen (z. B. Betrag und Datum einer Rechnung).',
        caption: 'Eine Seite als Markdown abgeschrieben.',
        steps: [
            'Eine Datei wählen: ein Bild oder ein PDF (bis 50 Seiten, jede Seite wird einzeln gelesen).',
            'Unter „Art“ wählen: als Markdown abschreiben, freier Inhalt, oder „Felder nach Vorlage“.',
            'Für Felder eine Vorlage angeben, z. B. {"betrag": "number", "datum": "string"}.',
            '„Lesen“ drücken; das Ergebnis lässt sich rechts kopieren.',
        ],
        tip: 'Stehen dieselben Zahlen in einer Tabelle und im Fließtext, im Feld „Hinweise“ sagen, woher sie kommen sollen.',
    },
    'image-gen': {
        title: 'Bild erstellen',
        intro: 'Bilder aus einer Beschreibung erzeugen, ein vorhandenes Bild per Anweisung verändern, oder den Stil '
             + 'eines Bildes auf ein anderes übertragen.',
        caption: 'Ein Bild mit Z-Image Turbo, dem schnellen Standardmodell.',
        steps: [
            'Oben die Art wählen: „Erzeugen“, „Bearbeiten“ (ein Bild verändern) oder „Stil übertragen“.',
            'Ein Modell wählen – die Beschreibung darunter sagt, wofür es gut ist.',
            'Beschreiben, was zu sehen sein soll (Englisch klappt meist am besten), Seitenverhältnis wählen, „Bild erzeugen“.',
            'Jedes Bild landet zusätzlich unter „Meine Bilder“.',
        ],
        tip: 'Gleicher Seed + gleicher Prompt = gleiches Bild. So lassen sich einzelne Einstellungen gezielt ausprobieren.',
    },
    layout: {
        title: 'Bild mit Positionen',
        intro: 'Festlegen, was wo im Bild sein soll: Rahmen aufziehen und beschreiben – auch Schrift, die genau so im '
             + 'Bild erscheint (MOSS-Image).',
        caption: 'Zwei Rahmen: ein Schild mit Text oben, eine Figur unten links.',
        steps: [
            'Unter „Ganzes Bild“ beschreiben, was das Bild insgesamt zeigt.',
            'Auf der Fläche einen Rahmen aufziehen und eintragen, was hinein soll – bei „Text“ den genauen Wortlaut.',
            'Weitere Rahmen hinzufügen; „Alle Rahmen löschen“ fängt neu an.',
            'Format und Variante wählen (Turbo ist schneller) und „Bild erzeugen“ drücken.',
        ],
        tip: '„API-Aufruf“ zeigt, wie dasselbe Bild aus einem eigenen Programm angefordert wird.',
    },
    compare: {
        title: 'Modelle vergleichen',
        intro: 'Denselben Prompt mit demselben Seed durch mehrere Bildmodelle schicken – auch mit verschiedenen '
             + 'Schrittzahlen – und die Ergebnisse direkt nebeneinander oder übereinander vergleichen.',
        caption: 'Ein Fuchs im Schnee mit drei Modellen.',
        steps: [
            'Einen Prompt eintragen und ein Seitenverhältnis wählen.',
            'Modelle anhaken; bei Bedarf mehrere Schrittzahlen oder Seeds angeben.',
            '„Vergleich starten“ – Modelle auf verschiedenen Grafikkarten rechnen gleichzeitig.',
            'Im Ergebnis zwischen den Bildern wechseln oder mit dem Schieber überblenden.',
        ],
        tip: 'Der Vergleich läuft auf dem Server weiter, auch wenn du die Seite schließt; frühere stehen unten.',
    },
    mine: {
        title: 'Meine Bilder',
        intro: 'Alles, was du auf der Spielwiese erzeugt hast – mit Prompt, Modell und Seed, nur für dich sichtbar.',
        caption: 'Die Galerie, nach Modell filterbar.',
        steps: [
            'Oben nach Modell filtern oder alle zeigen.',
            'Ein Bild anklicken, um es groß zu sehen – mit Prompt, Seed und Dauer.',
            'Mit Prompt und Seed lässt sich ein Bild unter „Bild erstellen“ genau wieder erzeugen.',
        ],
        tip: '',
    },
    speech: {
        title: 'Sprache',
        intro: 'Text vorlesen lassen – mit eingebauten Stimmen, beschriebenen Stimmen oder einer geklonten Stimme aus '
             + 'einer Aufnahme – und Aufnahmen abschreiben, auf Wunsch mit getrennten Sprechern.',
        caption: '„Sprechen lassen“ mit CosyVoice 3; links passen sich die Felder der gewählten Engine an.',
        steps: [
            'Oben „Sprechen lassen“ oder „Abschreiben“ wählen.',
            'Eine Engine und die Art wählen – das Formular zeigt, was diese braucht (Aufnahme, Stil, Sprecher …).',
            'Zum Sprechen den Text rechts eintragen und „Sprechen“ drücken; das Ergebnis lässt sich herunterladen.',
            'Zum Abschreiben eine Aufnahme wählen oder hochladen; das Ergebnis gibt es auch als Untertitel (SRT).',
        ],
        tip: 'Regler („anpassen“) greifen nur, wenn angehakt – sonst gilt die Voreinstellung der Engine.',
    },
    quota: {
        title: 'Kontingente',
        intro: 'Wie viel Rechenzeit und wie viele Bilder Gäste pro Tag nutzen dürfen. Du selbst bist nicht begrenzt.',
        caption: '',
        steps: [
            'Oben die Standardwerte für alle Gäste eintragen; leer = unbegrenzt.',
            'Darunter einzelnen Personen eigene Werte geben.',
            '„Speichern“ – die Zähler beginnen jeden Tag um Mitternacht neu.',
        ],
        tip: '',
    },
};

const HelpRules = (() => {
    const KEY = 'spielwiese-help-seen';
    // Open by itself only on a page with help that this browser hasn't seen yet.
    function autoOpen(tab, seen) {
        return !!HELP[tab] && !seen.includes(tab);
    }
    function markSeen(tab, seen) {
        return seen.includes(tab) ? seen : [...seen, tab];
    }
    return { KEY, autoOpen, markSeen, pages: () => Object.keys(HELP) };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = { HELP, HelpRules };

if (typeof document !== 'undefined' && document.getElementById('help-dialog')) (() => {
    const dlg = document.getElementById('help-dialog');
    let memory = [];   // when the browser keeps nothing: once per visit
    const readSeen = () => { try { return JSON.parse(localStorage.getItem(HelpRules.KEY) || '[]'); } catch (e) { return memory; } };
    const writeSeen = list => { memory = list; try { localStorage.setItem(HelpRules.KEY, JSON.stringify(list)); } catch (e) { /* kept in memory */ } };

    function show(tab) {
        const h = HELP[tab];
        if (!h) return;
        document.getElementById('help-title').textContent = h.title;
        document.getElementById('help-intro').textContent = h.intro;
        const fig = document.getElementById('help-figure'), img = document.getElementById('help-img');
        fig.hidden = !h.caption;
        if (h.caption) {
            img.src = `help/${tab}.jpg?v=2`;
            img.alt = h.caption;
            img.onerror = () => { fig.hidden = true; };
            document.getElementById('help-caption').textContent = h.caption;
        }
        const ol = document.getElementById('help-steps');
        ol.replaceChildren(...h.steps.map(s => { const li = document.createElement('li'); li.textContent = s; return li; }));
        document.getElementById('help-tip').textContent = h.tip ? `Tipp: ${h.tip}` : '';
        if (!dlg.open) dlg.showModal();
        writeSeen(HelpRules.markSeen(tab, readSeen()));
    }

    const current = () => (document.querySelector('.tab-link.active') || {}).dataset?.tab;
    const maybe = tab => { if (HelpRules.autoOpen(tab, readSeen())) show(tab); };
    document.getElementById('help-btn').addEventListener('click', () => show(current()));
    document.querySelectorAll('.tab-link[data-tab]').forEach(l => l.addEventListener('click', () => maybe(l.dataset.tab)));
    dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });   // click outside closes
    // the page the app opened on (app.js restores the last one before this runs)
    setTimeout(() => maybe(current()), 300);
})();
