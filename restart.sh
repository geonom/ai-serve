#!/bin/bash
# Signal the running launcher to restart the server.
# Any member of the 'users' group can run this — no sudo needed.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
touch "$DIR/restart.flag"
echo "Restart flag set. Server will restart in ~2 seconds."
