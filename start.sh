#!/usr/bin/env bash
# One container, two processes: dashboard ($PORT, default 7860 on Hugging Face Spaces) in the background,
# backend (8000) in the foreground. The terminal shows the backend (uvicorn) logs; the dashboard's output goes
# to .data/streamlit.log. BACKEND_AUTOSTART=false stops the dashboard from starting a second backend.
set -e
cd "$(dirname "$0")"
mkdir -p .data
BACKEND_AUTOSTART=false streamlit run ui/app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
  --server.headless true > .data/streamlit.log 2>&1 &
echo "Dashboard starting on port ${PORT:-7860} (logs: .data/streamlit.log)"
exec uvicorn api.main:app_factory --factory --host 0.0.0.0 --port 8000
