// node --test tests/talk.test.mjs -- "Speech > Talk" (static/talk.js): session settings per language, audio conversion
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

createRequire(import.meta.url)('../static/i18n.js');
const R = createRequire(import.meta.url)('../static/talk.js');

test('English keeps the model card prompt and lets the model speak itself', () => {
    const ev = R.sessionUpdate('en', 'male');
    assert.equal(ev.session.instructions, 'Respond with interleaved text and audio.');
    assert.equal(ev.session.audio, undefined);
});

test('another language asks the gateway to speak it; the language itself is in the guidance', () => {
    const ev = R.sessionUpdate('de', 'male');
    assert.equal(ev.session.instructions, 'Respond with interleaved text and audio.');
    assert.deepEqual(ev.session.audio, { output: { language: 'de', voice: 'male' } });
    assert.equal(R.sessionUpdate('fr', 'robot').session.audio.output.voice, 'female');
});

test('only speech models are offered', () => {
    const models = R.speechModels({ models: [{ id: 'a.gguf', speech: true }, { id: 'b.gguf', speech: false }, { id: 'c.gguf' }] });
    assert.deepEqual(models.map(m => m.id), ['a.gguf']);
    assert.deepEqual(R.speechModels(null), []);
});

test('48 kHz microphone samples become 16 kHz averages', () => {
    const out = R.downsample(Float32Array.from([0, 0.3, 0.6, 1, 1, 1, 0.5]), 48000);
    assert.equal(out.length, 2);
    assert.ok(Math.abs(out[0] - 0.3) < 1e-6 && out[1] === 1);
    assert.deepEqual(Array.from(R.downsample([0.5, 0.25], 16000)), [0.5, 0.25]);
});

test('PCM16 base64 there and back', () => {
    const b64 = R.toPcm16Base64(Float32Array.from([0, 0.5, -1, 2]));
    assert.equal(Buffer.from(b64, 'base64').readInt16LE(2), 16384);
    const back = R.pcm16Base64ToFloat(b64);
    assert.ok(Math.abs(back[1] - 0.5) < 1e-3 && back[2] === -32767 / 32768 && back[3] === 32767 / 32768);
});

test('the sentence splits into words and choices', () => {
    assert.deepEqual(R.sentencePieces('Talk with {model} in {lang}{voice}.'),
        ['Talk with ', { pick: 'model' }, ' in ', { pick: 'lang' }, { pick: 'voice' }, '.']);
});

test('tools go into every session, English keeps the exact system prompt', () => {
    const en = R.sessionUpdate('en', 'female');
    assert.equal(en.session.instructions, 'Respond with interleaved text and audio.');
    assert.deepEqual(en.session.tools.map(t => t.name), ['write_text', 'continue_text', 'change_text', 'look_up']);
    assert.equal(R.sessionUpdate('de', 'female').session.tools.length, 4);
    assert.deepEqual(R.sessionUpdate('en', 'female', false).session.tools, []);
    assert.match(R.guidance('de'), /Always answer in German/);
    assert.doesNotMatch(R.guidance('en'), /Always answer in/);
});

test('writers: chat models only, Gemma 4 12B first, no speech or exclusive ones', () => {
    const w = R.writerModels({ models: [
        { id: 'qwen.gguf', kind: 'chat' }, { id: 'gemma-4-12b-it.gguf', kind: 'chat' }, { id: 'omni.gguf', kind: 'chat', speech: true },
        { id: 'bge.gguf', kind: 'embedding' }, { id: 'strata', kind: 'chat', exclusive: true }, { id: 'gemma-3-4b.gguf', kind: 'chat' }] });
    assert.deepEqual(w.map(m => m.id), ['gemma-4-12b-it.gguf', 'gemma-3-4b.gguf', 'qwen.gguf']);
    const loaded = R.writerModels({ models: [{ id: 'gemma-4-12b-it.gguf', kind: 'chat' }, { id: 'qwen.gguf', kind: 'chat', loaded: true }] });
    assert.equal(loaded[0].id, 'gemma-4-12b-it.gguf');   // a loaded model is usually someone's running job
    const twins = R.writerModels({ models: [{ id: 'gemma-4-12b-it-qat-q4_0.gguf', kind: 'chat' },
                                            { id: 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf', kind: 'chat', loaded: true }] });
    assert.equal(twins[0].id, 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf');   // the same model already running
});

test('tool calls streamed as text are not shown', () => {
    assert.equal(R.visibleAnswer('Gern. <tool_call>\n<function=write_text>'), 'Gern.');
    assert.equal(R.visibleAnswer('Gern. <tool'), 'Gern. ');
    assert.equal(R.visibleAnswer('3 < 4 ist wahr.'), '3 < 4 ist wahr.');
});

test('the writer: continue adds paragraphs, change and write replace the text', () => {
    assert.equal(R.applyWriter('continue', 'One.\n\nTwo.', ' Three. '), 'One.\n\nTwo.\n\nThree.');
    assert.equal(R.applyWriter('continue', '', 'Three.'), 'Three.');
    assert.equal(R.applyWriter('change', 'Old.', 'New.'), 'New.');
    const m = R.writerMessages('continue', 'One.', '', 'de');
    assert.match(m[0].content, /Write in German/);
    assert.match(m[1].content, /One\.[\s\S]*Continue it: one more paragraph/);
    assert.match(R.writerMessages('change', 'One.', 'red sails', 'en')[1].content, /Change it: red sails\. Output the whole revised text/);
});

test('what Omni hears about the page', () => {
    assert.equal(R.textNote('', '  '), '(Note: the text page is empty now.)');
    const n = R.textNote('Mara', 'First.\n\nSecond one is here.');
    assert.match(n, /"Mara" now has 2 paragraphs\. The last one begins: "Second one is here\."/);
    assert.deepEqual(R.paragraphs('a\n\n\n b \n\nc'), ['a', 'b', 'c']);
});

test('Wikipedia: one search call in the conversation language, a summary out', () => {
    const url = R.wikiUrl('de', 'Ada Lovelace');
    assert.ok(url.startsWith('https://de.wikipedia.org/w/api.php?') && url.includes('gsrsearch=Ada%20Lovelace') && url.includes('origin=*'));
    assert.ok(R.wikiUrl('../x', 'a').startsWith('https://en.wikipedia.org/'));
    assert.deepEqual(R.wikiSummary({ query: { pages: { 1: { title: 'Ada Lovelace', extract: ' Mathematician. ' } } } }),
        { title: 'Ada Lovelace', text: 'Mathematician.' });
    assert.equal(R.wikiSummary({}), null);
});

test('turns: voice starts a turn, a pause ends it, Omni from the speakers needs more', () => {
    const t = new R.Turns({ startS: 0.2, endS: 0.5 });
    const step = 0.1;
    const run = (level, n, playing = false) => { const out = []; for (let i = 0; i < n; i++) out.push(t.feed(level, step, playing)); return out.filter(Boolean); };
    assert.deepEqual(run(0.002, 10), []);                 // room noise
    assert.deepEqual(run(0.1, 3), ['start']);
    assert.deepEqual(run(0.1, 5), []);                    // still talking
    assert.deepEqual(run(0.002, 6), ['end']);
    assert.deepEqual(run(0.02, 5, true), []);             // quiet echo of Omni while it plays
    assert.deepEqual(run(0.2, 3, true), ['start']);       // the user speaks up over it
    assert.ok(Math.abs(R.rms([0.5, -0.5]) - 0.5) < 1e-9);
});

test('turns: real microphone blocks (128 samples) with gaps between syllables', () => {
    const t = new R.Turns();
    const block = 128 / 48000;                       // ~2.7 ms, what the AudioWorklet delivers
    const events = [];
    const feed = (level, seconds) => { for (let s = 0; s < seconds; s += block) { const e = t.feed(level, block); if (e) events.push(e); } };
    feed(0.001, 1.5);                                // a quiet room
    for (let i = 0; i < 12; i++) { feed(0.15, 0.08); feed(0.003, 0.03); }   // syllables with short gaps
    assert.deepEqual(events, ['start']);
    feed(0.001, 1.2);                                // the pause after the sentence
    assert.deepEqual(events, ['start', 'end']);
    feed(0.002, 0.004); feed(0.3, 0.004);            // a click is no turn
    feed(0.001, 0.5);
    assert.deepEqual(events, ['start', 'end']);
});

test('fixed confirmations only where Omni won\'t confirm by itself', () => {
    assert.equal(R.confirmation('write', 'en'), null);
    assert.equal(R.confirmation('change', 'de'), 'In Ordnung, ich ändere das.');
    assert.equal(R.confirmation('write', 'sv'), null);          // no sentence of ours: Omni tries itself
    assert.match(R.TOOL_DONE, /one short sentence/);
});

test('an example sentence only in English', () => {
    assert.match(R.guidance('en'), /like "I'll write it now\."/);
    assert.doesNotMatch(R.guidance('de'), /like "/);       // any example was just said back in German
    assert.doesNotMatch(R.guidance('sv'), /like "/);
});
