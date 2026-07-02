"""Shared path constants for the public Chapter 6 overlap release."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CODE_ROOT = REPO_ROOT / "code"
CHAPTER3_ROOT = CODE_ROOT / "chapter3_clevis_setup"
CHAPTER5_ROOT = CODE_ROOT / "chapter5_optimization"

CONFIG_DIR = PACKAGE_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "datasets" / "chapter6_overlap_clevis"
POOL_100K_DIR = DATA_ROOT / "pool_100k"
HOLDOUT_5K_DIR = DATA_ROOT / "holdout_5k"
LABELED_DIR = DATA_ROOT / "labeled"
CALIBRATION_DIR = DATA_ROOT / "calibration"
CH64_BENCHMARK_DIR = DATA_ROOT / "benchmark_strict_repair_ch64_v1"
CH65_BENCHMARK_DIR = DATA_ROOT / "benchmark_strict_blocked_v2"

RESULTS_ROOT = REPO_ROOT / "results" / "chapter6_overlap"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
COMPARISONS_DIR = RESULTS_ROOT / "comparisons"
REPAIR_CH64_DIR = RESULTS_ROOT / "repair_ch64"
REPAIR_CH64_TIMED_DIR = RESULTS_ROOT / "repair_ch64_timed"
ZIGZAG_CH65_DIR = RESULTS_ROOT / "zigzag_ch65"
REPORTS_DIR = RESULTS_ROOT / "reports"
RUNS_DIR = RESULTS_ROOT / "runs"
VIEWER_RUNS_DIR = REPO_ROOT / "viewers" / "chapter6_overlap_repair" / "data"
GEOMETRY_CACHE_DIR = RESULTS_ROOT / "geometry_cache"

POOL_100K_CSV = POOL_100K_DIR / "pool.csv"
HOLDOUT_5K_CSV = HOLDOUT_5K_DIR / "pool.csv"
ACQUIRED_RANDOM_15K = LABELED_DIR / "acquired_random_15k.csv"
HOLDOUT_LABELED_5K = LABELED_DIR / "holdout_labeled_5k.csv"
CALIBRATION_ROWS = CALIBRATION_DIR / "calibration_rows.csv"
CH64_STARTS_CSV = CH64_BENCHMARK_DIR / "starts.csv"
CH65_STARTS_CSV = CH65_BENCHMARK_DIR / "starts.csv"

V3_MODEL_DIR = CHECKPOINTS_DIR / "overlap_regressor_regression_v3_selected"
MULTITASK_MODEL_DIR = CHECKPOINTS_DIR / "overlap_regressor_multitask_v1_selected"
ASM_MODEL_DIR = REPO_ROOT / "results" / "chapter5_optimization" / "checkpoints" / "row1_uncertainty_disagreement_B1000_T2500_best"
ASM_NORMALIZATION_JSON = REPO_ROOT / "datasets" / "chapter5_blocked200" / "normalization.json"


def ensure_chapter3_importable() -> None:
    import sys
    code_root = str(CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def ensure_chapter5_importable() -> None:
    import sys
    for path in (str(CODE_ROOT), str(CHAPTER5_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
