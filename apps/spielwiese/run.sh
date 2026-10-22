#!/bin/sh
# Start the Spielwiese (stdlib only, no venv). Port from SPIELWIESE_PORT (default 8037).
cd "$(dirname "$0")" && exec python3 -m spielwiese.server
