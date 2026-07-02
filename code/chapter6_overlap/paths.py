"""Compatibility path shim for Chapter 6 modules.

Modules import this shim so their defaults resolve inside ``bachelor-thesis/``.
"""

from __future__ import annotations

from .release_paths import (  # noqa: F401
    CONFIG_DIR,
    DATA_ROOT as DATA_DIR,
    FIGURES_DIR,
    GEOMETRY_CACHE_DIR,
    PACKAGE_ROOT as CHAPTER6_ROOT,
    REPORTS_DIR,
    RESULTS_ROOT,
    RUNS_DIR,
    TABLES_DIR,
    CHECKPOINTS_DIR as MODELS_DIR,
    REPO_ROOT,
    CHAPTER3_ROOT,
)

CLEVIS_DIR = CHAPTER3_ROOT
PIPELINE_DIR = CHAPTER3_ROOT
