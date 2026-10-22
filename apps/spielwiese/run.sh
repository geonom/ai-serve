#!/bin/sh
# Start the Spielwiese. Port from SPIELWIESE_PORT (default 8037).
# It needs Python 3.9+ and PyJWT (for sign-ins through the door): SPIELWIESE_PYTHON if set, else ai-serve's own
# environment when the Spielwiese lives in ai-serve/apps/ (it has PyJWT), else ./.venv, else python3.
cd "$(dirname "$0")" || exit 1
for py in "$SPIELWIESE_PYTHON" ../../venv/bin/python ./.venv/bin/python python3; do
    [ -n "$py" ] || continue
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import jwt" >/dev/null 2>&1; then
        exec "$py" -m spielwiese.server
    fi
done
echo "Spielwiese: no Python with PyJWT found. Install it, e.g. 'python3 -m pip install \"PyJWT[crypto]\"'," >&2
echo "or set SPIELWIESE_PYTHON to a Python that has it." >&2
exit 1
