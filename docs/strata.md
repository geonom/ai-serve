# Strata as a language model in ai-serve

[Strata](https://github.com/Niko1221/Strata) (a separate open-source project, its own license) runs one very large
mixture-of-experts model – 125 billion parameters – on a gaming PC: the most-used experts sit on the graphics card,
all of them in RAM, the rest is read from the SSD. ai-serve can offer it as a model next to its own llama.cpp ones
through a small adapter, `extras/strata-plugin`.

## Is it for your machine?

**Made for PCs with a dedicated graphics card** (NVIDIA RTX 20–50, recent AMD Radeon; 12 GB of VRAM or more) and
**32 GB of RAM or more** (64 GB runs every size). That's where its main trick pays off: a small, fast VRAM pool for
the busy experts in front of a large, slower RAM pool for all the others. Strata's own README has the details and
the list of cards.

**Apple Silicon:** Strata also has a Metal engine for M1–M5 Macs (macOS 15+), new and measured by its authors on two
Macs (M5 Max 128 GB, M4 32 GB – see Strata's `docs/MAC.md`). With unified memory the VRAM-vs-RAM split above doesn't
exist, so that advantage is gone. What remains: the model itself, its speculative decoding, and on 32–48 GB Macs the
experts streamed from the SSD. Smaller Macs (e.g. 24 GB) are below what it needs. On a Mac whose memory holds the
whole model, llama.cpp could run a GGUF of it as well – which of the two is faster there hasn't been compared.

**Tried with ai-serve:** only on a Linux PC with two NVIDIA cards (RTX 3090 + 4060 Ti, 96 GB RAM). The adapter
with Strata's Mac engine has not been tried.

**While Strata runs it owns every GPU of the machine:** ai-serve unloads idle models before it starts and refuses
other GPU work (507) until Strata is stopped. Loading takes 4–10 minutes. Strata is never unloaded automatically –
stop it in the dashboard (or set `STRATA_KEEP_ALIVE`).

## Set it up

1. **Install Strata** by its own instructions (its installer on a PC, `./setup.sh` on a Mac) and check that it
   answers on its own (`http://127.0.0.1:8080/v1/models`). Then stop it – ai-serve will start it when needed.
2. **Move the adapter** where ai-serve looks for plugins:

   ```bash
   mv extras/strata-plugin plugins/
   cd plugins/strata-plugin && ./setup.sh
   ```

3. **Point it at Strata** in `plugins/strata-plugin/.env`:

   ```bash
   STRATA_DIR=/path/to/Strata
   STRATA_CONFIG=/path/to/Strata/strata-iq3_s.json   # the strata-<model>.json Strata's setup wrote
   ```

   More settings (port, start timeout, keep-alive, extra model names): `.env.example`.
4. `./ai-serve restart`

The model now appears in the dashboard and the Spielwiese (marked as using every GPU), and requests with
`"model": "strata"` go to it. The first one starts Strata and waits until it's loaded.

## Without the adapter

Strata speaks the OpenAI API itself, so it can also be added as a plain server: *Compute & models → More model
servers*, port 8080. Then ai-serve neither starts nor stops it and doesn't know it holds the GPUs – fine if you start
it by hand and don't load other models at the same time, otherwise they compete for VRAM.
