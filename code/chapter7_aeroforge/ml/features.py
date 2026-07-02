"""Load labeled ADV rows and build a basic feature matrix for overlap regression."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chapter7_aeroforge.release_paths import LABELS_CLOUD

DEFAULT_LABELS_PATH = LABELS_CLOUD

# Keys always null in generated dataset.
_DROP_KEYS = {
    "root_csv_filepath",
    "tip_csv_filepath",
    "hstab_root_csv_filepath",
    "hstab_tip_csv_filepath",
    "vstab_root_csv_filepath",
    "vstab_tip_csv_filepath",
}

_CATEGORICAL = {
    "design_name",
    "tail_type",
    "root_naca_code",
    "tip_naca_code",
    "airfoil_source",
    "hstab_airfoil_source",
    "vstab_airfoil_source",
    "hstab_root_naca_code",
    "hstab_tip_naca_code",
    "vstab_root_naca_code",
    "vstab_tip_naca_code",
}

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "tail_wing_dx",
    "wing_x",
    "tail_x",
    "wing_mean_chord",
    "hstab_mean_chord",
    "effective_dx",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Column layout for the encoded feature matrix."""

    numeric_cols: tuple[str, ...]
    one_hot_cols: tuple[str, ...]
    feature_names: tuple[str, ...]


@dataclass
class LabeledDataset:
    """In-memory labeled dataset ready for train/test splits."""

    sample_idx: np.ndarray
    x_raw: np.ndarray
    y_mm3: np.ndarray
    is_overlap: np.ndarray
    spec: FeatureSpec
    family: list[str]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append spacing/chord proxy columns derived from raw ADV numerics."""
    out = df.copy()
    wing_position = out["wing_position"].astype(float)
    tail_position = out["tail_position"].astype(float)
    length = out["length"].astype(float)
    wingspan = out["wingspan"].astype(float)
    aspect_ratio = out["aspect_ratio"].astype(float)
    hstab_semispan = out["hstab_semispan"].astype(float)
    hstab_aspect_ratio = out["hstab_aspect_ratio"].astype(float)

    tail_wing_dx = (tail_position - wing_position) * length
    wing_mean_chord = wingspan / aspect_ratio.replace(0.0, np.nan)
    hstab_mean_chord = (2.0 * hstab_semispan) / hstab_aspect_ratio.replace(0.0, np.nan)

    out["tail_wing_dx"] = tail_wing_dx
    out["wing_x"] = wing_position * length
    out["tail_x"] = tail_position * length
    out["wing_mean_chord"] = wing_mean_chord
    out["hstab_mean_chord"] = hstab_mean_chord
    out["effective_dx"] = tail_wing_dx - 0.5 * wing_mean_chord - 0.5 * hstab_mean_chord
    return out


def load_labeled_dataset(
    labels_path: Path | None = None,
    *,
    engineered: bool = False,
) -> LabeledDataset:
    """Load ok rows from labels.jsonl and return raw feature matrix + overlap targets."""
    labels_path = Path(labels_path or DEFAULT_LABELS_PATH)
    rows: list[dict] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No ok rows found in {labels_path}")

    adv_rows = [row["adv"] for row in rows]
    df = pd.DataFrame(adv_rows)
    for key in _DROP_KEYS:
        if key in df.columns:
            df = df.drop(columns=[key])

    # Drop constant columns (e.g. airfoil_source == naca_4 everywhere).
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    if engineered:
        df = add_engineered_features(df)

    numeric_cols = [c for c in df.columns if c not in _CATEGORICAL]
    cat_cols = [c for c in df.columns if c in _CATEGORICAL]

    x_num = df[numeric_cols].astype(float).to_numpy(dtype=np.float64)
    one_hot_parts: list[np.ndarray] = []
    one_hot_names: list[str] = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, dtype=float)
        one_hot_parts.append(dummies.to_numpy(dtype=np.float64))
        one_hot_names.extend(dummies.columns.tolist())

    if one_hot_parts:
        x_cat = np.hstack(one_hot_parts)
        x_raw = np.hstack([x_num, x_cat])
    else:
        x_raw = x_num

    feature_names = tuple(numeric_cols) + tuple(one_hot_names)
    spec = FeatureSpec(
        numeric_cols=tuple(numeric_cols),
        one_hot_cols=tuple(one_hot_names),
        feature_names=feature_names,
    )

    y_mm3 = np.array([float(r.get("overlap_mm3_cut_each", 0.0)) for r in rows], dtype=np.float64)
    is_overlap = np.array([bool(r.get("is_overlap", y > 1.0)) for r, y in zip(rows, y_mm3)])
    sample_idx = np.array([int(r["sample_idx"]) for r in rows], dtype=np.int64)
    family = [str(r.get("family", "?")) for r in rows]

    return LabeledDataset(
        sample_idx=sample_idx,
        x_raw=x_raw,
        y_mm3=y_mm3,
        is_overlap=is_overlap,
        spec=spec,
        family=family,
    )


def encode_adv_dataframe(df: pd.DataFrame, *, engineered: bool = False) -> tuple[np.ndarray, FeatureSpec]:
    """Build the encoded ADV feature matrix (same layout as overlap surrogates)."""
    work = df.copy()
    for key in _DROP_KEYS:
        if key in work.columns:
            work = work.drop(columns=[key])

    constant_cols = [c for c in work.columns if work[c].nunique(dropna=False) <= 1]
    if constant_cols:
        work = work.drop(columns=constant_cols)

    if engineered:
        work = add_engineered_features(work)

    numeric_cols = [c for c in work.columns if c not in _CATEGORICAL]
    cat_cols = [c for c in work.columns if c in _CATEGORICAL]

    x_num = work[numeric_cols].astype(float).to_numpy(dtype=np.float64)
    one_hot_parts: list[np.ndarray] = []
    one_hot_names: list[str] = []
    for col in cat_cols:
        dummies = pd.get_dummies(work[col].astype(str), prefix=col, dtype=float)
        one_hot_parts.append(dummies.to_numpy(dtype=np.float64))
        one_hot_names.extend(dummies.columns.tolist())

    if one_hot_parts:
        x_cat = np.hstack(one_hot_parts)
        x_raw = np.hstack([x_num, x_cat])
    else:
        x_raw = x_num

    feature_names = tuple(numeric_cols) + tuple(one_hot_names)
    spec = FeatureSpec(
        numeric_cols=tuple(numeric_cols),
        one_hot_cols=tuple(one_hot_names),
        feature_names=feature_names,
    )
    return x_raw, spec


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score using train statistics only."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_train_z = (x_train - mean) / scale
    x_test_z = (x_test - mean) / scale
    return x_train_z, x_test_z, mean, scale
