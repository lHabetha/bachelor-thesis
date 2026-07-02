#!/usr/bin/env bash
# Launch the Chapter 6 overlap-repair viewer (default port 8091).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m viewers.chapter6_overlap_repair.server "$@"
