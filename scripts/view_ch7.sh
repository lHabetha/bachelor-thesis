#!/usr/bin/env bash
# Launch the Chapter 7 AeroForge overlap-repair viewer (default port 8092).
# On-demand 3D rendering needs the bachelor-thesis env + local AeroForge checkout;
# metrics-only inspection works with plain Python 3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m viewers.chapter7_aeroforge_repair.server "$@"
