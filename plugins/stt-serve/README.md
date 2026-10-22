# stt-serve

Local speech-to-text server: WhisperX transcription with word alignment, speaker diarization (pyannote or Nemotron-3-Diarization), speaker matching across recordings, and live (chunked) sessions.

```bash
./setup.sh && ./run.sh                     # 127.0.0.1:8060
curl -s localhost:8060/v1/transcribe -H 'Content-Type: application/json' \
     -d "{\"audio_base64\": \"$(base64 -w0 clip.wav)\", \"language\": \"de\", \"diarize\": true}"
```

Details: `CLAUDE.md`, API docs at `/docs`.
