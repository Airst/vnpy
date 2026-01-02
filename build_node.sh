#!/usr/bin/env bash
echo "Starting frontend build..."

cd /home/airst/Workspace/vnpy/core/web_ui && npm run build
if [ $? -eq 0 ]; then
    echo "Frontend build successful."
else
    echo "Frontend build failed. Aborting."
    exit 1
fi