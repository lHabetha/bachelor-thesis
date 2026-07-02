"""Shared path constants for the public Chapter 7 AeroForge release."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CODE_ROOT = REPO_ROOT / "code"

DATA_ROOT = REPO_ROOT / "datasets" / "chapter7_aeroforge_adv"
ADV_DIR = DATA_ROOT / "advs"
LABELS_CLOUD = DATA_ROOT / "labels_cloud" / "labels.jsonl"
LABELS_CLOUD_DIR = DATA_ROOT / "labels_cloud"
LABELS_SIM = DATA_ROOT / "labels_sim" / "labels.jsonl"
LABELS_SIM_DIR = DATA_ROOT / "labels_sim"
BENCHMARK_JSON = DATA_ROOT / "benchmark_overlap_ranked100_v2" / "overlap_ranked100_v2.json"
NORMALIZATION_JSON = DATA_ROOT / "normalization_bounds.json"

RESULTS_ROOT = REPO_ROOT / "results" / "chapter7_aeroforge"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
REGISTRY_JSON = CHECKPOINTS_DIR / "registry.json"
RUNS_DIR = RESULTS_ROOT / "runs"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
COMPARISONS_DIR = RESULTS_ROOT / "comparisons"
MODEL_GRID_DIR = COMPARISONS_DIR / "model_grid_100k_v1"
PERFORMANCE_MLP_DIR = RESULTS_ROOT / "performance_mlp_v1"
PERFORMANCE_MLP_CKPT = CHECKPOINTS_DIR / "performance_mlp_90k.pt"
VIEWER_RUNS_DIR = REPO_ROOT / "viewers" / "chapter7_aeroforge_repair" / "data"

WORKSPACE_ROOT = REPO_ROOT.parent
AEROFORGE_ROOT = Path(os.environ.get("AEROFORGE_ROOT", str(WORKSPACE_ROOT / "aeroforge")))


def ensure_code_importable() -> None:
    import sys

    code_root = str(CODE_ROOT)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def subprocess_env() -> dict[str, str]:
    """Environment for `-m chapter7_aeroforge.*` subprocesses from ``REPO_ROOT``."""
    import os

    env = os.environ.copy()
    code_root = str(CODE_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = code_root if not existing else f"{code_root}{os.pathsep}{existing}"
    return env
