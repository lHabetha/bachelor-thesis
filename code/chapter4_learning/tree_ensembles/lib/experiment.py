"""Shared fit/evaluate helpers for Chapter 4 jobs."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import features, labels
from .metrics import eval_binary
from .models import fit_model


def fit_eval_record(
    *,
    model_id: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    seed: int,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    X_train = features(train_df)
    y_train = labels(train_df)
    X_hold = features(holdout_df)
    y_hold = labels(holdout_df)
    fit = fit_model(model_id, X_train, y_train, seed=seed, protocol=protocol)
    t0 = time.perf_counter()
    prob = fit.predict_proba(X_hold)
    predict_wall_s = time.perf_counter() - t0
    metrics = eval_binary(y_hold, prob)
    metrics.update(
        {
            "fit_wall_s": float(fit.fit_wall_s),
            "predict_wall_s": float(predict_wall_s),
            "train_size": int(len(train_df)),
            "train_pos_rate": float(np.mean(y_train)) if len(y_train) else float("nan"),
            "model_metadata": fit.metadata,
        }
    )
    return metrics, fit
