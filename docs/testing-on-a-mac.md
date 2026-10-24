# Testing ai-serve on a Mac (Apple Silicon)

What still needs a real Mac (see the Status table in the README). Each step says what to run and what to send back.
Use a fresh copy so nothing from an earlier install helps it along:

    git clone https://github.com/geonom/ai-serve ~/ai-serve-test && cd ~/ai-serve-test

Keep the terminal output of each step. If something fails, the plugin's log is in `plugins/<name>/logs/` and the
gateway's in `logs/`.

## 1. Install with a language model, chat with an image

    ./install.sh --offers llm-gemma-e2b --yes
    AI_SERVE_PORT=8031 ./ai-serve start

Open `http://localhost:8031`, then the Playground (Apps & Services → Playground → Open), Chat, attach any photo and
ask what is in it. **Send back:** how long the install took, the answer, and:

    curl -s localhost:8031/api/version | python3 -m json.tool

From that, `gpus[0].memory_source` matters most: `ioreg_gpu` means the GPU's own memory figure was read (what we
want), `psutil_system_ram` means it fell back to the whole system's RAM. Also note `gpu_limit_mb` and `free`.

## 2. Speech and transcription offers

    ./ai-serve setup --offers tts-chatterbox,stt-whisper --yes

Then in the Playground: Speech → speak a sentence with Chatterbox (it uses MLX on a Mac), and transcribe a short
recording with Whisper. **Send back:** setup time, whether both worked, and roughly how long each took.

## 3. Images with Iris

    ./ai-serve setup --offers image-iris --yes

Playground → Create images → Iris-3B Turbo, any prompt. **Send back:** the time per image, and whether Compute & models →
"Who uses the GPUs" showed Iris while it was loaded.

## 4. ComfyUI (image-serve) without NVIDIA

    ./ai-serve setup --offers image-comfyui --yes

This is the least certain part: the Qwen-Image-Edit text encoder is an fp8 file made for CUDA. Try, in the
Playground (Create images): a Z-Image picture and an edit of it with Qwen-Image-Edit 2511. Then download `birefnet`
and `moss-gguf` in the dashboard (Compute & models → Downloads), cut out the Z-Image picture

    curl -s localhost:8031/v1/images/edits -F image=@picture.png -F task=remove_background \
      | python3 -c "import json,sys,base64; open('cutout.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

and make one MOSS-Image picture. **Send back:** what worked, the time per image, and the
first error message of anything that failed.

## 5. Tidy up

    ./ai-serve stop 2>/dev/null; rm -rf ~/ai-serve-test

The models are inside that folder, so this frees the disk space again.
