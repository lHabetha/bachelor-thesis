"""Metrics for Chapter 4."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def eval_binary(y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 1e-7, 1.0 - 1e-7)
    pred = (prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "recall_blocked": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "recall_assemblable": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_blocked": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "precision_assemblable": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "pos_rate_true": float(np.mean(y_true)),
        "pos_rate_pred": float(np.mean(pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, prob))
    except ValueError:
        out["brier"] = float("nan")
    out["ece_10bin"] = calibration_ece(y_true, prob, n_bins=10)
    return out


def calibration_ece(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)
