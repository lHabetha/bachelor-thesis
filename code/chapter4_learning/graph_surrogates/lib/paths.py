"""Path constants for Chapter 4 graph-surrogate experiments (public release)."""

from __future__ import annotations

import sys
from pathlib import Path

_GRAPH_ROOT = Path(__file__).resolve().parents[1]
_CH4_ROOT = _GRAPH_ROOT.parent
if str(_CH4_ROOT) not in sys.path:
    sys.path.insert(0, str(_CH4_ROOT))

from release_paths import (  # noqa: E402
    FIGURES_DIR as _RELEASE_FIGURES,
    GRAPH_FROZEN,
    GRAPH_RUNS_ROOT,
    HOLDOUT_PATH,
    SEED_SETS_ROOT,
    pool_parquet_path,
    seed_sets_root_for_pool,
)

TASK27_ROOT = _GRAPH_ROOT
CONFIG_DIR = TASK27_ROOT / "configs"
DATA_DIR = GRAPH_RUNS_ROOT / "data"
ARCH_SCREEN_DIR = DATA_DIR / "arch_screen"
FIXED_GRID_DIR = DATA_DIR / "fixed_grid"
FIXED_LABEL_SET_DIR = GRAPH_FROZEN
MODELS_DIR = GRAPH_RUNS_ROOT / "models"
REPORTS_DIR = GRAPH_RUNS_ROOT / "reports"
FIGURES_DIR = _RELEASE_FIGURES

TASK22_DENSE = pool_parquet_path("dense50k_v1")
TASK22_SEEDS = seed_sets_root_for_pool("dense50k_v1")
HOLDOUT = HOLDOUT_PATH

# Full Chapter 4 trajectory replay is not shipped; fixed-label manifest is frozen.
TASK22_TRAJ = GRAPH_RUNS_ROOT / "task22_trajectories"
TASK22_BASELINE = GRAPH_RUNS_ROOT / "task22_baseline"

DEFAULT_MODEL_ID = "row1_pure_hybrid_B1000_T2500_best"
