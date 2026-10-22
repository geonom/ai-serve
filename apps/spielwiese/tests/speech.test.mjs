// node --test tests/speech.test.mjs -- the Speech tab's rules (static/speech.js): it builds requests from
// whatever the engines describe in GET /v1/audio/models, so these use made-up engines on purpose.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

createRequire(import.meta.url)('../static/i18n.js');   // tx() -- English in node
const R = createRequire(import.meta.url)('../static/speech.js');

const ENGINE = { id: 'future', fields: [
    { name: 'language', type: 'language', label: 'Sprache' },
    { name: 'diarize', type: 'bool', label: 'Sprecher trennen' },
    { name: 'diarize_backend', type: 'choice', label: 'mit', show_if: { diarize: true } },
    { name: 'num_speakers', type: 'int', label: 'Anzahl', show_if: { diarize: true } },
    { name: 'hotwords', type: 'list', label: 'Wörter' },
] };

test('stt: only visible, filled fields are sent, typed', () => {
    const audio = { audio_base64: 'QQ==' };
    assert.deepEqual(R.sttPayload(ENGINE, { diarize: false, diarize_backend: 'x', num_speakers: '3', language: '' }, audio),
        { engine: 'future', audio_base64: 'QQ==', diarize: false });
    assert.deepEqual(R.sttPayload(ENGINE, { diarize: true, diarize_backend: 'x', num_speakers: '3', hotwords: 'Munin\n\n Anna ' }, audio),
        { engine: 'future', audio_base64: 'QQ==', diarize: true, diarize_backend: 'x', num_speakers: 3, hotwords: ['Munin', 'Anna'] });
});

const MODE = { variant: 'v1', label: 'x', inputs: { reference: 'required', ref_text: 'optional', instruct: 'optional' },
               params: [{ name: 'top_p', min: 0, max: 1 }] };

test('tts: what is still missing', () => {
    assert.deepEqual(R.missingInputs(MODE, {}), ['a recording']);
    assert.deepEqual(R.missingInputs(MODE, { reference: true }), []);
    assert.deepEqual(R.missingInputs({ inputs: { speaker: 'required', instruct: 'required' } }, { speaker: true }), ['a style']);
});

test('tts: the body carries only the inputs the mode takes and only changed params', () => {
    const body = R.ttsPayload({ id: 'future' }, MODE, { text: 'Hallo', language: 'de', refVoicePath: '/v/a.wav',
        refText: 'hi', speaker: 'Vivian', instruct: 'ruhig', params: { top_p: 0.5, other: 1 } });
    assert.deepEqual(body, { engine: 'future', text: 'Hallo', language: 'de', variant: 'v1', ref_voice_path: '/v/a.wav',
        ref_text: 'hi', instruct: 'ruhig', params: { top_p: 0.5 } });
    const bare = R.ttsPayload({ id: 'e' }, { variant: null, inputs: {} }, { text: 'x', params: {} });
    assert.deepEqual(bare, { engine: 'e', text: 'x', language: 'en' });   // no language chosen: the UI's (English in node)
});

test('tags: one per group, a second click clears it', () => {
    const g = [['male', 'female'], ['child', 'elderly']];
    let s = R.toggleTag(g, [], 'male');
    s = R.toggleTag(g, s, 'child');
    s = R.toggleTag(g, s, 'female');
    assert.deepEqual(s, ['child', 'female']);
    assert.deepEqual(R.toggleTag(g, s, 'female'), ['child']);
});

test('srt and readable text', () => {
    const segs = [{ start: 0, end: 1.5, text: ' Hallo ', speaker: 'SPEAKER_00' }, { start: 61.25, end: 62, text: 'Ja', speaker: 'SPEAKER_01' }];
    assert.equal(R.toSrt(segs), '1\n00:00:00,000 --> 00:00:01,500\nSPEAKER_00: Hallo\n\n2\n00:01:01,250 --> 00:01:02,000\nSPEAKER_01: Ja\n');
    assert.equal(R.readable({ segments: segs }), '[0:00] SPEAKER_00: Hallo\n[1:01] SPEAKER_01: Ja');
    assert.equal(R.readable({ transcript: 'Nur Text', segments: [{ text: 'Nur Text' }] }), 'Nur Text');
});
