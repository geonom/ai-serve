# stt-serve

Local speech-to-text server: WhisperX transcription with word alignment, speaker diarization (pyannote or Nemotron-3-Diarization), speaker matching across recordings, and live (chunked) sessions.

```bash
./setup.sh && ./run.sh                     # 127.0.0.1:8060
curl -s localhost:8060/v1/transcribe -H 'Content-Type: application/json' \
     -d "{\"audio_base64\": \"$(base64 -w0 clip.wav)\", \"language\": \"de\", \"diarize\": true}"
```

Details: `CLAUDE.md`, API docs at `/docs`.

**Status:** runs on the developer's Linux machine with NVIDIA (WhisperX, pyannote, Nemotron, MOSS-Transcribe, live
sessions). Not tried: a clean setup through ai-serve's installer, a Mac, CPU only. Nemotron and MOSS-Transcribe need
an NVIDIA GPU; pyannote's diarization model is gated: accept its terms on Hugging Face
(pyannote/speaker-diarization-community-1) and set `HF_TOKEN` (with ai-serve: in `ai-serve.local.env`).
