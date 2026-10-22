# tts-serve

Local TTS server: OmniVoice, CosyVoice 3, Chatterbox Multilingual, VoxCPM2 and Qwen3-TTS behind one API, each engine in its own provider folder with its own venv. Workers start on demand and unload when idle.

```bash
./run.sh                                   # 127.0.0.1:8050
curl -s localhost:8050/v1/engines
curl -s localhost:8050/v1/generate -H 'Content-Type: application/json' \
     -d '{"engine": "chatterbox", "text": "Hallo!", "language": "de", "response_format": "wav"}' -o out.wav
```

Set up an engine: `providers/<id>/setup.sh`. Details: `CLAUDE.md`.
