"""Load quick-sim labels and build raw-ADV → performance-vector training data."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chapter7_aeroforge.ml.features import FeatureSpec, encode_adv_dataframe

from chapter7_aeroforge.release_paths import LABELS_SIM

DEFAULT_SIM_LABELS_PATH = LABELS_SIM


@dataclass(frozen=True)
class SimTargetSpec:
    name: str
    path: tuple[str, ...]
    lo: float
    hi: float


# Curated quick-sim scalars for the performance surrogate (§19.14).
SIM_TARGET_SPECS: tuple[SimTargetSpec, ...] = (
    SimTargetSpec("CD0", ("uav_performance", "aero_simple", "CD0"), 0.008, 0.08),
    SimTargetSpec(
        "L_D_analytic",
        ("uav_performance", "aero_simple", "theoretical_L_over_D_max"),
        0.5,
        60.0,
    ),
    SimTargetSpec("CD0_wing", ("uav_performance", "aero_simple", "CD0_wing"), 0.005, 0.04),
    SimTargetSpec("CD0_tail", ("uav_performance", "aero_simple", "CD0_tail"), 0.0, 0.02),
    SimTargetSpec(
        "CD0_fuselage",
        ("uav_performance", "aero_simple", "CD0_fuselage"),
        0.001,
        0.04,
    ),
    SimTargetSpec(
        "L_D_vlm",
        ("uav_performance", "asb_vlm", "L_over_D_from_vlm"),
        0.5,
        40.0,
    ),
    SimTargetSpec("CL_vlm", ("uav_performance", "asb_vlm", "CL_from_vlm"), -0.8, 1.5),
    SimTargetSpec("Cm_vlm", ("uav_performance", "asb_vlm", "Cm_from_vlm"), -2.0, 2.0),
)

SIM_TARGET_NAMES: tuple[str, ...] = tuple(spec.name for spec in SIM_TARGET_SPECS)


@dataclass
class SimDataset:
    sample_idx: np.ndarray
    x_raw: np.ndarray
    y: np.ndarray
    spec: FeatureSpec
    target_names: tuple[str, ...]
    family: list[str]


def _nested_get(obj: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)) and np.isfinite(float(cur)):
        return float(cur)
    return None


def _extract_targets(quick_report: dict[str, Any]) -> np.ndarray:
    vals = []
    for spec in SIM_TARGET_SPECS:
        v = _nested_get(quick_report, spec.path)
        vals.append(np.nan if v is None else v)
    return np.array(vals, dtype=np.float64)


def _passes_row_filter(
    row: dict[str, Any],
    y: np.ndarray,
) -> bool:
    if not row.get("ok") or not row.get("vlm_ok"):
        return False
    if not np.all(np.isfinite(y)):
        return False
    for spec, value in zip(SIM_TARGET_SPECS, y):
        if not (spec.lo <= value <= spec.hi):
            return False
    return True


def load_sim_dataset(
    labels_path: Path | None = None,
) -> SimDataset:
    """Load filtered quick-sim rows: raw ADV (39) → 8 performance scalars."""
    labels_path = Path(labels_path or DEFAULT_SIM_LABELS_PATH)
    kept_rows: list[dict] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            y = _extract_targets(row.get("quick_report", {}))
            if not _passes_row_filter(row, y):
                continue
            kept_rows.append(row)

    if not kept_rows:
        raise RuntimeError(f"No rows passed the sim filter in {labels_path}")

    adv_rows = [row["adv"] for row in kept_rows]
    df = pd.DataFrame(adv_rows)
    x_raw, spec = encode_adv_dataframe(df, engineered=False)
    y = np.vstack([_extract_targets(row.get("quick_report", {})) for row in kept_rows])
    sample_idx = np.array([int(row["sample_idx"]) for row in kept_rows], dtype=np.int64)
    family = [str(row.get("family", "?")) for row in kept_rows]

    return SimDataset(
        sample_idx=sample_idx,
        x_raw=x_raw,
        y=y,
        spec=spec,
        target_names=SIM_TARGET_NAMES,
        family=family,
    )
