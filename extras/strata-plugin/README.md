# strata-plugin

Makes [Strata](https://github.com/Niko1221/Strata) available as a language-model plugin of **ai-serve**. You only
need this if you run Strata. It is a large MoE model that takes every GPU of a machine.

1. Install Strata itself, following its own instructions.
2. Move this folder into ai-serve's `plugins/` (it ships in `extras/` so it doesn't show up for people without
   Strata), or put it under another `SERVICE_ROOTS` entry.
3. Run `./setup.sh`, then set `STRATA_DIR` in `.env`.

ai-serve then lists the model as `strata`. Step by step, and whether it suits your machine (made for dedicated
GPUs; on Apple Silicon its VRAM/RAM split brings nothing, the adapter untried there): ai-serve's `docs/strata.md`. It starts Strata on the first request: all GPUs are reserved for it,
and loading takes several minutes. You can stop Strata from the dashboard.

License: AGPL-3.0 (like ai-serve). Strata itself has its own license.
