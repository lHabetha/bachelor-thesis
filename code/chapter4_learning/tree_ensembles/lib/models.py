"""Model registry and wrappers for Chapter 4."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .paths import MLP_LIB_ROOT

if str(MLP_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(MLP_LIB_ROOT))

from lib.formula_oracle import FEATURE_NAMES  # noqa: E402
from lib.model_mlp64 import fit_mlp, train_ensemble  # noqa: E402


@dataclass
class FitResult:
    model_id: str
    model: Any
    fit_wall_s: float
    metadata: dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba_task30"):
            return self.model.predict_proba_task30(X)
        return self.model.predict_proba(X)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "uncertainty"):
            return self.model.uncertainty(X)
        return np.zeros(len(X), dtype=np.float64)


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.probability, dtype=np.float64)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.minimum(p, 1.0 - p)


class MLPTask30Wrapper:
    def __init__(self, fitted):
        self.fitted = fitted

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fitted.predict_proba(X)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.minimum(p, 1.0 - p)


class TreeWrapper:
    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float64)
        return proba.reshape(-1).astype(np.float64)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "estimators_"):
            preds: list[np.ndarray] = []
            estimators = np.ravel(self.estimator.estimators_)
            for tree in estimators:
                if hasattr(tree, "predict_proba"):
                    p = tree.predict_proba(X)
                    if p.ndim == 2 and p.shape[1] > 1:
                        preds.append(p[:, 1])
            if preds:
                return np.std(np.stack(preds, axis=0), axis=0).astype(np.float64)
        p = self.predict_proba(X)
        return np.minimum(p, 1.0 - p)


def _class_weights(y: np.ndarray) -> dict[int, float]:
    n_pos = max(float(np.sum(y == 1)), 1.0)
    n_neg = max(float(np.sum(y == 0)), 1.0)
    n = n_pos + n_neg
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def fit_model(model_id: str, X: np.ndarray, y: np.ndarray, *, seed: int, protocol: dict[str, Any]) -> FitResult:
    """Fit one model and return a uniform Chapter 4 wrapper."""
    y = y.astype(int)
    t0 = time.perf_counter()
    if len(np.unique(y)) < 2:
        model = ConstantProbabilityModel(float(np.mean(y)))
        return FitResult(model_id, model, time.perf_counter() - t0, {"constant_model": True})

    cfg = protocol["models"][model_id]
    kind = cfg["kind"]

    if kind == "mlp":
        fitted = fit_mlp(X, y, seed=seed)
        return FitResult(
            model_id,
            MLPTask30Wrapper(fitted),
            time.perf_counter() - t0,
            {"kind": kind, "best_epoch": fitted.best_epoch, "best_val_loss": fitted.best_val_loss},
        )

    if kind == "sklearn_random_forest":
        from sklearn.ensemble import RandomForestClassifier

        est = RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 160)),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=_class_weights(y),
            n_jobs=1,
            random_state=seed,
        )
    elif kind == "sklearn_extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        est = ExtraTreesClassifier(
            n_estimators=int(cfg.get("n_estimators", 160)),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=_class_weights(y),
            n_jobs=1,
            random_state=seed,
        )
    elif kind == "sklearn_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        sample_weight = np.array([_class_weights(y)[int(v)] for v in y], dtype=np.float64)
        est = HistGradientBoostingClassifier(
            max_iter=int(cfg.get("max_iter", 180)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_leaf_nodes=int(cfg.get("max_leaf_nodes", 31)),
            l2_regularization=float(cfg.get("l2_regularization", 0.001)),
            random_state=seed,
        )
        est.fit(X, y, sample_weight=sample_weight)
        return FitResult(model_id, TreeWrapper(est), time.perf_counter() - t0, {"kind": kind})
    elif kind == "xgboost":
        from xgboost import XGBClassifier

        scale_pos_weight = max(float(np.sum(y == 0)), 1.0) / max(float(np.sum(y == 1)), 1.0)
        est = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 220)),
            max_depth=int(cfg.get("max_depth", 3)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            subsample=float(cfg.get("subsample", 0.9)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model kind for {model_id}: {kind}")

    est.fit(X, y)
    return FitResult(model_id, TreeWrapper(est), time.perf_counter() - t0, {"kind": kind})


def fit_acquisition_model(model_id: str, X: np.ndarray, y: np.ndarray, *, seed: int, protocol: dict[str, Any]) -> FitResult:
    """Fit the model used to score an active-learning round."""
    if model_id != "mlp64":
        return fit_model(model_id, X, y, seed=seed, protocol=protocol)
    t0 = time.perf_counter()
    ensemble = train_ensemble(X, y, base_seed=seed)

    class MLPEnsembleWrapper:
        def __init__(self, members):
            self.members = members

        def predict_proba(self, X_in: np.ndarray) -> np.ndarray:
            probs = np.stack([m.predict_proba(X_in) for m in self.members], axis=0)
            return probs.mean(axis=0)

        def uncertainty(self, X_in: np.ndarray) -> np.ndarray:
            probs = np.stack([m.predict_proba(X_in) for m in self.members], axis=0)
            return probs.std(axis=0)

    return FitResult(
        model_id,
        MLPEnsembleWrapper(ensemble),
        time.perf_counter() - t0,
        {"kind": "mlp_ensemble", "n_members": len(ensemble)},
    )


def export_tree_artifact(fit: FitResult, out_dir: Path, metrics: dict[str, Any], provenance: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit.model, out_dir / "model.joblib")
    card = {
        "task30_tree_surrogate": fit.model_id != "mlp64",
        "model_id": provenance.get("model_id", fit.model_id),
        "architecture": fit.model_id,
        "feature_names": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "metrics": metrics,
        "provenance": provenance,
        "metadata": fit.metadata,
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
