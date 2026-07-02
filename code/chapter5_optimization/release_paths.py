"""Shared path constants for the public Chapter 5 optimization release."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CODE_ROOT = REPO_ROOT / "code"
CHAPTER3_ROOT = CODE_ROOT / "chapter3_clevis_setup"
CHAPTER4_ROOT = CODE_ROOT / "chapter4_learning"

BENCHMARK_DIR = REPO_ROOT / "datasets" / "chapter5_blocked200"
BENCHMARK_ID = "blocked_200_v1"
BENCHMARK_PARQUET = BENCHMARK_DIR / f"{BENCHMARK_ID}.parquet"
BENCHMARK_JSONL = BENCHMARK_DIR / f"{BENCHMARK_ID}.jsonl"
BENCHMARK_NORMALIZATION = BENCHMARK_DIR / "normalization.json"
CHECKPOINTS_DIR = REPO_ROOT / "results" / "chapter5_optimization" / "checkpoints"
RESULTS_ROOT = REPO_ROOT / "results" / "chapter5_optimization"
RUNS_DIR = RESULTS_ROOT / "runs"
COMPARISONS_DIR = RESULTS_ROOT / "comparisons"
CONSTRAINT_STUDIES_DIR = RESULTS_ROOT / "constraint_studies"
ANALYSIS_DIR = RESULTS_ROOT / "analysis"
OPTIMIZERS_DIR = PACKAGE_ROOT / "optimizers"
SCHEMAS_DIR = PACKAGE_ROOT / "schemas"

DEFAULT_MLP_MODEL_ID = "row1_uncertainty_disagreement_B1000_T2500_best"
DEFAULT_MLP_CHECKPOINT = CHECKPOINTS_DIR / DEFAULT_MLP_MODEL_ID

# Chapter 4 dataset paths (export_model / benchmark leakage checks)
CH4_POOL = REPO_ROOT / "datasets" / "chapter4_clevis_pools" / "pool_50k" / "candidate_pool.parquet"
CH4_HOLDOUT = REPO_ROOT / "datasets" / "chapter4_clevis_pools" / "holdout" / "holdout.parquet"
CH4_SEED_SETS = REPO_ROOT / "datasets" / "chapter4_clevis_pools" / "seed_sets" / "dense50k_v1"
CH4_TRAJECTORIES = (
    RESULTS_ROOT.parent
    / "chapter4_label_efficiency"
    / "mlp_runs"
    / "data"
    / "trajectories"
    / "dense50k_v2_labelblind"
)


def ensure_chapter3_importable() -> None:
    import sys

    code_root = str(CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def ensure_mlp_lib_importable() -> None:
    import sys

    mlp_root = str(CHAPTER4_ROOT / "mlp_active_learning")
    ch4_root = str(CHAPTER4_ROOT)
    for path in (mlp_root, ch4_root):
        if path not in sys.path:
            sys.path.insert(0, path)
