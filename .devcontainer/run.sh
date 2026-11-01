#!/usr/bin/env bash
# Starts backend (8000) + dashboard (7860) when the Codespace opens.
cd "$(dirname "$0")/.."
token="${OPERATOR_TOKEN:-$(grep -E '^OPERATOR_TOKEN=' .env 2>/dev/null | cut -d= -f2-)}"
if [ "${#token}" -lt 8 ]; then
  echo "OPERATOR_TOKEN is missing or shorter than 8 characters."
  echo "Add it as a Codespaces secret (GitHub Settings -> Codespaces -> Secrets) and rebuild,"
  echo "or set it in .env, then run: bash .devcontainer/run.sh"
  exit 1
fi
if curl -s -o /dev/null http://127.0.0.1:8000/api/health; then
  echo "GridSentinel is already running (dashboard on port 7860)."
  exit 0
fi
exec bash start.sh
