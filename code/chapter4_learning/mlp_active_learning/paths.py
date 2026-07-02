"""Path helpers for MLP active-learning scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_MLP_ROOT = Path(__file__).resolve().parent
_CH4_ROOT = _MLP_ROOT.parent
if str(_CH4_ROOT) not in sys.path:
    sys.path.insert(0, str(_CH4_ROOT))

from release_paths import (  # noqa: E402
    DATASETS_ROOT,
    FIGURES_DIR,
    HOLDOUT_PATH,
    MLP_FROZEN_10K,
    MLP_FROZEN_22K,
    MLP_FROZEN_50K,
    MLP_RUNS_ROOT,
    SEED_SETS_ROOT,
    pool_key,
    pool_parquet_path,
    seed_set_path,
)

MLP_ROOT = _MLP_ROOT
CONFIG_DIR = MLP_ROOT / "configs"

DEFAULT_POOL_ID = "dense50k_v1"
DEFAULT_EXPERIMENT_ID = "dense50k_v2_labelblind"
DEFAULT_BASELINE_ID = "dense50k_v1"


def runs_data_dir() -> Path:
    return MLP_RUNS_ROOT / "data"


def trajectories_dir(experiment_id: str = DEFAULT_EXPERIMENT_ID) -> Path:
    return runs_data_dir() / "trajectories" / experiment_id


def baseline_dir(baseline_id: str = DEFAULT_BASELINE_ID) -> Path:
    return runs_data_dir() / "baseline" / baseline_id


def aggregate_results_dir(experiment_id: str) -> Path:
    return runs_data_dir() / "results" / experiment_id


def frozen_results_dir(experiment_id: str) -> Path:
    if experiment_id == "dense50k_v2_labelblind":
        return MLP_FROZEN_50K
    if experiment_id == "dense10k_v1_labelblind":
        return MLP_FROZEN_10K
    return aggregate_results_dir(experiment_id)


def pool_output_dir(pool_id: str) -> Path:
    from release_paths import pool_key

    return MLP_RUNS_ROOT / "pools" / pool_key(pool_id)
