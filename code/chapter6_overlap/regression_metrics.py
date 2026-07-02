"""Regression metrics and continuous-label helpers for Chapter 6."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .label_cache import DEFAULT_THRESHOLD_NORM, LabelCache, params_from_row

PAIR_COLUMNS = (
    "bracket__main_pin",
    "bracket__splint",
    "main_pin__splint",
)


def continuous_label_from_payload(payload: dict) -> dict:
    label = payload["label"]
    total_volume = max(float(label["total_part_volume_analytic"]), 1e-12)
    pair_norms = {f"pair_norm_{name}": 0.0 for name in PAIR_COLUMNS}
    pair_volumes = {f"pair_volume_{name}": 0.0 for name in PAIR_COLUMNS}
    for pair in label["pairs"]:
        name = pair["pair"]
        volume = float(pair["volume"])
        pair_volumes[f"pair_volume_{name}"] = volume
        pair_norms[f"pair_norm_{name}"] = volume / total_volume
    dominant = max(PAIR_COLUMNS, key=lambda name: pair_norms[f"pair_norm_{name}"])
    if float(label["total_overlap_norm"]) <= DEFAULT_THRESHOLD_NORM:
        dominant = "clean"
    return {
        "param_hash": payload["param_hash"],
        "total_overlap_norm": float(label["total_overlap_norm"]),
        "total_overlap_volume": float(label["total_overlap_volume"]),
        "total_part_volume_analytic": total_volume,
        "overlap_binary": int(bool(label["overlap_binary"])),
        "dominant_pair": dominant,
        "label_wall_time_s": float(label.get("wall_time_s", 0.0)),
        **pair_volumes,
        **pair_norms,
    }


def continuous_labels(rows: list[dict], cache: LabelCache) -> list[dict]:
    out = []
    for row in rows:
        out.append(continuous_label_from_payload(cache.label(params_from_row(row))))
    return out


def transformed_target(y: np.ndarray, *, threshold_norm: float = DEFAULT_THRESHOLD_NORM) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0) / threshold_norm)


def inverse_transformed_target(y_log: np.ndarray, *, threshold_norm: float = DEFAULT_THRESHOLD_NORM) -> np.ndarray:
    return np.expm1(np.maximum(y_log, 0.0)) * threshold_norm


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def magnitude_bin(value: float, *, threshold_norm: float = DEFAULT_THRESHOLD_NORM) -> str:
    if value <= threshold_norm:
        return "clean_or_below_threshold"
    if value <= 2e-4:
        return "tiny"
    if value <= 1e-3:
        return "small"
    if value <= 5e-3:
        return "moderate"
    return "large"


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pair_true: np.ndarray | None = None,
    pair_pred: np.ndarray | None = None,
    dominant_pairs: list[str] | None = None,
    threshold_norm: float = DEFAULT_THRESHOLD_NORM,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    y_true_log = transformed_target(y_true, threshold_norm=threshold_norm)
    y_pred_log = transformed_target(y_pred, threshold_norm=threshold_norm)
    out: dict[str, float | int] = {
        "n": int(len(y_true)),
        "mae_norm": float(abs_err.mean()),
        "rmse_norm": float(np.sqrt(np.mean(err**2))),
        "median_ae_norm": float(np.median(abs_err)),
        "mae_log": float(np.abs(y_pred_log - y_true_log).mean()),
        "rmse_log": float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))),
        "pearson_norm": _corr(y_true, y_pred),
        "spearman_norm": _corr(_rankdata(y_true), _rankdata(y_pred)),
        "near_threshold_mae_norm": 0.0,
    }
    near = (y_true <= 5.0 * threshold_norm) | (y_pred <= 5.0 * threshold_norm)
    if near.any():
        out["near_threshold_mae_norm"] = float(abs_err[near].mean())

    for bin_name in ("clean_or_below_threshold", "tiny", "small", "moderate", "large"):
        mask = np.array([magnitude_bin(v, threshold_norm=threshold_norm) == bin_name for v in y_true])
        out[f"n_bin_{bin_name}"] = int(mask.sum())
        out[f"mae_bin_{bin_name}"] = float(abs_err[mask].mean()) if mask.any() else 0.0

    if pair_true is not None and pair_pred is not None:
        pair_abs = np.abs(np.asarray(pair_pred, dtype=float) - np.asarray(pair_true, dtype=float))
        for i, name in enumerate(PAIR_COLUMNS):
            out[f"mae_pair_{name}"] = float(pair_abs[:, i].mean())

    if dominant_pairs is not None:
        by_pair: dict[str, list[float]] = defaultdict(list)
        for pair, value in zip(dominant_pairs, abs_err.tolist()):
            by_pair[pair].append(float(value))
        for pair, values in by_pair.items():
            out[f"mae_dominant_{pair}"] = float(np.mean(values))
            out[f"n_dominant_{pair}"] = int(len(values))

    return out
