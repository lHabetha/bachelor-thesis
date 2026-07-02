"""Shared path constants for the public Chapter 4 learning release."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CODE_ROOT = REPO_ROOT / "code"
CHAPTER3_ROOT = CODE_ROOT / "chapter3_clevis_setup"
DATASETS_ROOT = REPO_ROOT / "datasets" / "chapter4_clevis_pools"
RESULTS_ROOT = REPO_ROOT / "results" / "chapter4_label_efficiency"

HOLDOUT_PATH = DATASETS_ROOT / "holdout" / "holdout.parquet"
SEED_SETS_ROOT = DATASETS_ROOT / "seed_sets"

# Pool IDs -> dataset folder names
POOL_ID_TO_KEY: dict[str, str] = {
    "dense50k_v1": "pool_50k",
    "dense10k_v1": "pool_10k",
    "pool_22k": "pool_22k",
}

# Seed-set folder per pool ID (see datasets/chapter4_clevis_pools/seed_sets/README.md)
SEED_SET_DIR_BY_POOL_ID: dict[str, str] = {
    "dense50k_v1": "dense50k_v1",
    "dense10k_v1": "dense10k_v1",
    "pool_22k": "pool_22k",
}

# Frozen thesis summaries
MLP_FROZEN_50K = RESULTS_ROOT / "mlp_dense50k_labelblind"
MLP_FROZEN_10K = RESULTS_ROOT / "mlp_dense10k_labelblind"
MLP_FROZEN_22K = RESULTS_ROOT / "mlp_pool_22k"
GRAPH_FROZEN = RESULTS_ROOT / "task27_graphs"
TREE_FROZEN = RESULTS_ROOT / "task30_trees"
FIGURES_DIR = RESULTS_ROOT / "figures"

# Writable runtime outputs for reruns
MLP_RUNS_ROOT = RESULTS_ROOT / "mlp_runs"
GRAPH_RUNS_ROOT = RESULTS_ROOT / "graph_runs"
TREE_RUNS_ROOT = RESULTS_ROOT / "tree_runs"


def pool_key(pool_id: str) -> str:
    return POOL_ID_TO_KEY.get(pool_id, pool_id)


def seed_set_dir(pool_id: str) -> str:
    return SEED_SET_DIR_BY_POOL_ID.get(pool_id, pool_key(pool_id))


def seed_sets_root_for_pool(pool_id: str) -> Path:
    return SEED_SETS_ROOT / seed_set_dir(pool_id)


def pool_parquet_path(pool_id: str) -> Path:
    return DATASETS_ROOT / pool_key(pool_id) / "candidate_pool.parquet"


def seed_set_path(pool_id: str, seed_split: int, base_size: int) -> Path:
    return (
        seed_sets_root_for_pool(pool_id)
        / f"R{seed_split:03d}"
        / f"base_prefix_{base_size}.json"
    )


def ensure_chapter3_importable() -> None:
    import sys

    code_root = str(CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
