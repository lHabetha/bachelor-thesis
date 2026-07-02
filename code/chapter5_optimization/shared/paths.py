"""Canonical paths for the public Chapter 5 optimization workbench."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

_SPEC = importlib.util.spec_from_file_location("_chapter5_release_paths", _PKG / "release_paths.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load Chapter 5 release_paths from {_PKG / 'release_paths.py'}")
_RELEASE_PATHS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RELEASE_PATHS)

TASK26_ROOT = _RELEASE_PATHS.PACKAGE_ROOT
PROJECT_ROOT = _RELEASE_PATHS.REPO_ROOT
BENCHMARK_DIR = _RELEASE_PATHS.BENCHMARK_DIR
BENCHMARK_ID = _RELEASE_PATHS.BENCHMARK_ID
BENCHMARK_JSONL = _RELEASE_PATHS.BENCHMARK_JSONL
BENCHMARK_NORMALIZATION = _RELEASE_PATHS.BENCHMARK_NORMALIZATION
MODELS_DIR = _RELEASE_PATHS.CHECKPOINTS_DIR
OPTIMIZERS_DIR = _RELEASE_PATHS.OPTIMIZERS_DIR
RUNS_DIR = _RELEASE_PATHS.RUNS_DIR
CONSTRAINT_STUDIES_DIR = _RELEASE_PATHS.CONSTRAINT_STUDIES_DIR
SCHEMAS_DIR = _RELEASE_PATHS.SCHEMAS_DIR
COMPARISONS_DIR = _RELEASE_PATHS.COMPARISONS_DIR
ANALYSIS_DIR = _RELEASE_PATHS.ANALYSIS_DIR

TASK22_HOLDOUT = _RELEASE_PATHS.CH4_HOLDOUT
DENSE50K_POOL = _RELEASE_PATHS.CH4_POOL
DENSE50K_SEEDS = _RELEASE_PATHS.CH4_SEED_SETS
DENSE50K_TRAJ = _RELEASE_PATHS.CH4_TRAJECTORIES
ensure_chapter3_importable = _RELEASE_PATHS.ensure_chapter3_importable
ensure_mlp_lib_importable = _RELEASE_PATHS.ensure_mlp_lib_importable
