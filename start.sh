#!/usr/bin/env bash
# One container, two processes: backend (8000) + dashboard ($PORT, default 7860 on Hugging Face Spaces)
set -e
uvicorn api.main:app_factory --factory --host 0.0.0.0 --port 8000 &
exec streamlit run ui/app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
