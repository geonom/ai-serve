#!/bin/sh
# Start the Playground. Port from PLAYGROUND_PORT (default 8037; the old SPIELWIESE_* names still work).
# It needs Python 3.9+ and PyJWT (for sign-ins through the door): PLAYGROUND_PYTHON if set, else ai-serve's own
# environment when the Playground lives in ai-serve/apps/ (it has PyJWT), else ./.venv, else python3.
cd "$(dirname "$0")" || exit 1
for py in "${PLAYGROUND_PYTHON:-$SPIELWIESE_PYTHON}" ../../venv/bin/python ./.venv/bin/python python3; do
    [ -n "$py" ] || continue
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import jwt" >/dev/null 2>&1; then
        exec "$py" -m playground.server
    fi
done
echo "Playground: no Python with PyJWT found. Install it, e.g. 'python3 -m pip install \"PyJWT[crypto]\"'," >&2
echo "or set PLAYGROUND_PYTHON to a Python that has it." >&2
exit 1
