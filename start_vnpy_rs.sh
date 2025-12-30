#!/usr/bin/env bash

PORT=8000
echo "Checking status for start up..."

# 1. Check for any remaining core.main:app processes (orphans)
ORPHAN_PIDS=$(pgrep -f "core.main:app")
if [ -n "$ORPHAN_PIDS" ]; then
    echo "Found orphan uvicorn processes: $ORPHAN_PIDS"
    echo "Cleaning up..."
    echo "$ORPHAN_PIDS" | xargs kill -9 2>/dev/null
fi

# 2. Check if port 8000 is occupied
PID=$(lsof -t -i :$PORT)
if [ -n "$PID" ]; then
    echo "Port $PORT is currently in use by PID $PID."
    echo "Attempting to release port $PORT..."
    kill -9 $PID
    echo "Port $PORT released."
else
    echo "Port $PORT is available."
fi

if [[ "$1" == "-b" ]]; then
    echo "Starting frontend build..."
    cd /home/airst/Workspace/vnpy/core/web_ui && npm run build
    if [ $? -eq 0 ]; then
        echo "Frontend build successful."
    else
        echo "Frontend build failed. Aborting."
        exit 1
    fi
fi

cd /home/airst/Workspace/vnpy

echo "Starting vn.py application..."

# cat /dev/null > web_ui.log

/home/airst/Workspace/.venv/bin/python /home/airst/Workspace/vnpy/core/main.py