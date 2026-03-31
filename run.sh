#!/bin/bash
# SentiScope — start everything with one command
echo "Starting SentiScope backend on http://localhost:8000 ..."
echo "Open index.html in your browser (just double-click it)"
echo ""
uvicorn api:app --reload --port 8000 --host 0.0.0.0