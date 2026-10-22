#!/bin/bash
# AI-Serve UI Launcher

# Settings: ai-serve.local.env (read by backend/config.py; see ai-serve.example.env)

# Identify where the script is located
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # real path, also when started through the old llama-serve-ui link

echo "--- AI-Serve UI Launcher ---"
echo "Project Directory: $INSTALL_DIR"
echo "------------------------------"

# Setup and activate virtual environment
VENV_DIR="$INSTALL_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install backend requirements if present (uv if available -- its venvs have no pip -- else pip)
if [ -f "$INSTALL_DIR/backend/requirements.txt" ]; then
    echo "Installing backend requirements..."
    if command -v uv >/dev/null; then
        uv pip install -q --python "$VENV_DIR/bin/python" -r "$INSTALL_DIR/backend/requirements.txt"
    else
        python -m pip install -q -r "$INSTALL_DIR/backend/requirements.txt"
    fi
fi

# Group-writable umask so logs/ and benchmarks/ are accessible to all group members.
umask 002

FLAG="$INSTALL_DIR/restart.flag"
KILLFLAG="$INSTALL_DIR/kill.flag"

while true; do
    rm -f "$FLAG"
    rm -f "$KILLFLAG"
    echo "[launcher] Starting server..."
    PYTHONPATH="$INSTALL_DIR" python "$INSTALL_DIR/backend/main.py" &
    SERVER_PID=$!

    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep 2
        if [ -f "$FLAG" ]; then
            echo "[launcher] Restart flag detected — stopping server (PID $SERVER_PID)..."
            kill "$SERVER_PID"
            break
        fi
        if [ -f "$KILLFLAG" ]; then
            echo "[launcher] Kill flag detected — stopping server (PID $SERVER_PID)..."
            kill "$SERVER_PID"
            break
        fi
    done

    wait "$SERVER_PID"

    if [ -f "$KILLFLAG" ]; then
        break
    fi

    if [ ! -f "$FLAG" ]; then
        echo "[launcher] Server exited without restart flag — shutting down."
        break
    fi

    echo "[launcher] Restarting..."
done
