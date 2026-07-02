"""Central paths for Chapter 4 tree ensembles (public release)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_TREE_ROOT = Path(__file__).resolve().parents[1]
_CH4_ROOT = _TREE_ROOT.parent
if str(_CH4_ROOT) not in sys.path:
    sys.path.insert(0, str(_CH4_ROOT))

from release_paths import (  # noqa: E402
    FIGURES_DIR as _RELEASE_FIGURES,
    HOLDOUT_PATH as _HOLDOUT,
    SEED_SETS_ROOT,
    TREE_FROZEN,
    TREE_RUNS_ROOT,
    pool_parquet_path,
    seed_sets_root_for_pool,
)

ROOT = _TREE_ROOT
CONFIG_DIR = ROOT / "configs"
PROTOCOL_PATH = CONFIG_DIR / "protocol_v2_labelblind.json"

DATA_DIR = TREE_RUNS_ROOT / "data"
SEED_SETS_DIR = seed_sets_root_for_pool("dense50k_v1")
RANDOM_BASELINE_DIR = DATA_DIR / "random_baseline"
ACTIVE_GRID_DIR = DATA_DIR / "active_grid_labelblind"
RESULTS_DIR = TREE_FROZEN
MODELS_DIR = TREE_RUNS_ROOT / "models"
REPORTS_DIR = TREE_RUNS_ROOT / "reports_labelblind"
FIGURES_DIR = _RELEASE_FIGURES

POOL_PATH = pool_parquet_path("dense50k_v1")
HOLDOUT_PATH = _HOLDOUT
MLP_LIB_ROOT = _CH4_ROOT / "mlp_active_learning"


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        RANDOM_BASELINE_DIR / "results",
        RANDOM_BASELINE_DIR / "checkpoints",
        ACTIVE_GRID_DIR / "results",
        ACTIVE_GRID_DIR / "checkpoints",
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
