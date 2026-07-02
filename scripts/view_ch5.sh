#!/usr/bin/env bash
# Launch the Chapter 5 clevis optimization viewer (default port 8090).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m viewers.chapter5_clevis_optimization.server "$@"
