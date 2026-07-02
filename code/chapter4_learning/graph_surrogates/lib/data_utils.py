"""Data loading and fixed-label-set reconstruction for Chapter 4."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import (
    FIXED_LABEL_SET_DIR,
    HOLDOUT,
    TASK22_BASELINE,
    TASK22_DENSE,
    TASK22_SEEDS,
    TASK22_TRAJ,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MLP_LIB = _REPO_ROOT / "mlp_active_learning"
if str(_MLP_LIB) not in sys.path:
    sys.path.insert(0, str(_MLP_LIB))

from lib.formula_oracle import FEATURE_NAMES  # noqa: E402

FEATURE_LIST = list(FEATURE_NAMES)
BASE_SIZES = [0, 250, 500, 750, 1000, 1500, 2000, 2500]
T_VALUES = list(range(250, 3501, 250))
ROWS = ["row1_pure_hybrid", "row2_hybrid_diverse_half"]


@dataclass(frozen=True)
class DatasetBundle:
    pool: pd.DataFrame
    holdout: pd.DataFrame
    holdout_X: np.ndarray
    holdout_y: np.ndarray


def set_single_thread() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def load_bundle() -> DatasetBundle:
    pool = pd.read_parquet(TASK22_DENSE)
    holdout = pd.read_parquet(HOLDOUT)
    holdout_X = holdout[FEATURE_LIST].to_numpy(dtype=np.float64)
    holdout_y = holdout["label"].to_numpy(dtype=np.int64)
    return DatasetBundle(pool=pool, holdout=holdout, holdout_X=holdout_X, holdout_y=holdout_y)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def base_ids(base_size: int, seed_split: int) -> list[str]:
    path = TASK22_SEEDS / f"R{seed_split:03d}" / f"base_prefix_{base_size}.json"
    data = _read_json(path)
    return [str(x) for x in data.get("param_ids", [])]


def _manifest_selected_ids(row: str, base_size: int, seed_split: int, total_labels: int) -> list[str] | None:
    """Use the shipped fixed-label manifest when full Chapter 4 trajectories are absent."""
    manifest_path = FIXED_LABEL_SET_DIR / "fixed_label_sets_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    for item in manifest.get("sets", []):
        if (
            item.get("row") == row
            and int(item.get("base_size", -1)) == int(base_size)
            and int(item.get("seed_split", -1)) == int(seed_split)
            and int(item.get("total_labels", -1)) == int(total_labels)
        ):
            return [str(x) for x in item.get("param_ids", [])]
    return None


def selected_ids(row: str, base_size: int, seed_split: int, total_labels: int) -> list[str]:
    """Return the exact Chapter 4 train IDs for one AL row/B/R/T point."""
    if row not in ROWS:
        raise ValueError(f"Unknown AL row: {row}")
    if total_labels < base_size:
        raise ValueError(f"T={total_labels} must be >= B={base_size}")
    ids = base_ids(base_size, seed_split)
    n_rounds = (total_labels - base_size + 249) // 250
    traj_dir = TASK22_TRAJ / row / f"B{base_size}_R{seed_split:03d}"
    for round_idx in range(1, n_rounds + 1):
        path = traj_dir / f"selections_round_{round_idx:02d}.json"
        if not path.exists():
            manifest_ids = _manifest_selected_ids(row, base_size, seed_split, total_labels)
            if manifest_ids is not None:
                return manifest_ids[:total_labels]
            raise FileNotFoundError(f"Missing Chapter 4 selection file: {path}")
        data = _read_json(path)
        ids.extend(str(x) for x in data.get("param_ids", []))
    return ids[:total_labels]


def baseline_ids(total_labels: int, replicate: int, n_pool: int) -> list[int]:
    """Reconstruct the deterministic random baseline indices from Chapter 4."""
    rng = np.random.default_rng(77000 + total_labels * 100 + replicate)
    idx = rng.choice(n_pool, size=min(total_labels, n_pool), replace=False)
    return [int(i) for i in idx]


def training_arrays(
    pool: pd.DataFrame,
    *,
    row: str,
    base_size: int | None = None,
    seed_split: int | None = None,
    total_labels: int,
    replicate: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build train arrays for an AL or baseline run."""
    if row == "baseline_random":
        if replicate is None:
            raise ValueError("baseline_random requires replicate")
        idx = baseline_ids(total_labels, replicate, len(pool))
        subset = pool.iloc[idx]
        ids = [str(x) for x in subset["param_id"].tolist()]
    else:
        if base_size is None or seed_split is None:
            raise ValueError("AL rows require base_size and seed_split")
        ids = selected_ids(row, base_size, seed_split, total_labels)
        subset = pool[pool["param_id"].isin(ids)]
        # Preserve the selected order for deterministic validation splits.
        order = {pid: i for i, pid in enumerate(ids)}
        subset = subset.assign(_order=subset["param_id"].map(order)).sort_values("_order")
    X = subset[FEATURE_LIST].to_numpy(dtype=np.float64)
    y = subset["label"].to_numpy(dtype=np.int64)
    return X, y, ids


def valid_total_labels(base_size: int) -> list[int]:
    return [t for t in T_VALUES if t >= max(250, base_size)]


def screen_jobs() -> list[dict]:
    jobs: list[dict] = []
    for row in ROWS:
        for base_size in [0, 1000, 2500]:
            for total_labels in [250, 500, 1000, 2000, 3500]:
                if total_labels < max(250, base_size):
                    continue
                for seed_split in range(1, 9):
                    jobs.append(
                        {
                            "row": row,
                            "base_size": base_size,
                            "seed_split": seed_split,
                            "total_labels": total_labels,
                            "replicate": None,
                        }
                    )
    for total_labels in [250, 500, 1000, 2000, 3500]:
        for replicate in range(1, 6):
            jobs.append(
                {
                    "row": "baseline_random",
                    "base_size": 0,
                    "seed_split": None,
                    "total_labels": total_labels,
                    "replicate": replicate,
                }
            )
    return jobs


def full_grid_jobs() -> list[dict]:
    jobs: list[dict] = []
    for row in ROWS:
        for base_size in BASE_SIZES:
            for total_labels in valid_total_labels(base_size):
                for seed_split in range(1, 9):
                    jobs.append(
                        {
                            "row": row,
                            "base_size": base_size,
                            "seed_split": seed_split,
                            "total_labels": total_labels,
                            "replicate": None,
                        }
                    )
    for total_labels in T_VALUES:
        for replicate in range(1, 6):
            jobs.append(
                {
                    "row": "baseline_random",
                    "base_size": 0,
                    "seed_split": None,
                    "total_labels": total_labels,
                    "replicate": replicate,
                }
            )
    return jobs


def write_fixed_label_manifest(jobs: Iterable[dict], pool: pd.DataFrame) -> Path:
    FIXED_LABEL_SET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        _, _, ids = training_arrays(
            pool,
            row=job["row"],
            base_size=job.get("base_size"),
            seed_split=job.get("seed_split"),
            total_labels=int(job["total_labels"]),
            replicate=job.get("replicate"),
        )
        rows.append({**job, "n_ids": len(ids), "param_ids": ids})
    path = FIXED_LABEL_SET_DIR / "fixed_label_sets_manifest.json"
    path.write_text(json.dumps({"sets": rows}, indent=2))
    return path
