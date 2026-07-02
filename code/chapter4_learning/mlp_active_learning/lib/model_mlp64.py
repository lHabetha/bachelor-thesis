"""Frozen MLP 64->32 model for Chapter 4 (same as Chapter 4 winning config).

Architecture: 13 -> Linear(64) -> ReLU -> Dropout(0.1) -> Linear(32) -> ReLU -> Dropout(0.1) -> Linear(1)
Loss: BCEWithLogitsLoss with pos_weight = n_neg / n_pos
Optimizer: Adam lr=3e-3, weight_decay=1e-3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .formula_oracle import FEATURE_NAMES, N_FEATURES

# ─── Standardizer ────────────────────────────────────────────────────────


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std < 1e-9] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, Xz: np.ndarray) -> np.ndarray:
        return Xz * self.std + self.mean


# ─── Model ───────────────────────────────────────────────────────────────


class MLP64(nn.Module):
    """MLP 64->32 binary classifier (logit output)."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── Training ────────────────────────────────────────────────────────────

TRAIN_LR = 3e-3
TRAIN_WD = 1e-3
TRAIN_EPOCHS = 160
TRAIN_PATIENCE = 22
TRAIN_BATCH = 32
ENSEMBLE_SIZE = 5


@dataclass
class FittedModel:
    scaler: Standardizer
    model: MLP64
    best_epoch: int
    best_val_loss: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xz = self.scaler.transform(X)
        with torch.no_grad():
            xt = torch.as_tensor(Xz, dtype=torch.float32)
            return torch.sigmoid(self.model(xt)).cpu().numpy()


def _iter_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, rng: np.random.Generator
):
    idx = np.arange(len(y))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch = idx[start : start + batch_size]
        yield x[batch], y[batch]


def fit_mlp(
    X: np.ndarray, y: np.ndarray, *, seed: int, val_frac: float = 0.15
) -> FittedModel:
    """Train one MLP with early stopping on a stratified val split."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    y = y.astype(np.int64)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Guard: need at least 2 samples total to do any training
    if len(y) < 2:
        model = MLP64()
        scaler = Standardizer(mean=np.zeros(X.shape[1]), std=np.ones(X.shape[1]))
        return FittedModel(scaler=scaler, model=model, best_epoch=0, best_val_loss=float("inf"))

    n_val_pos = max(1, int(round(val_frac * len(pos_idx)))) if len(pos_idx) > 1 else 0
    n_val_neg = max(1, int(round(val_frac * len(neg_idx)))) if len(neg_idx) > 1 else 0
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])

    # If train is empty, use all data for both (degenerate but safe)
    if len(train_idx) == 0:
        train_idx = np.arange(len(y))
        val_idx = np.arange(len(y))
    elif len(val_idx) == 0:
        val_idx = train_idx[:max(1, len(train_idx) // 5)]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    scaler = Standardizer.fit(X[train_idx])
    x_train = scaler.transform(X[train_idx])
    x_val = scaler.transform(X[val_idx])
    y_train = y[train_idx]
    y_val = y[val_idx]

    model = MLP64()
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WD)

    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pw = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    xtr = torch.as_tensor(x_train, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.float32)
    xva = torch.as_tensor(x_val, dtype=torch.float32)
    yva = torch.as_tensor(y_val, dtype=torch.float32)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for xb, yb in _iter_batches(xtr, ytr, TRAIN_BATCH, rng):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva), yva).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= TRAIN_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("MLP training never improved")
    model.load_state_dict(best_state)
    model.eval()
    return FittedModel(scaler=scaler, model=model, best_epoch=best_epoch, best_val_loss=best_loss)


def train_ensemble(
    X: np.ndarray, y: np.ndarray, *, base_seed: int
) -> list[FittedModel]:
    """Train ENSEMBLE_SIZE members with different seeds."""
    return [fit_mlp(X, y, seed=base_seed + 1009 * i) for i in range(ENSEMBLE_SIZE)]


def ensemble_predict(ensemble: list[FittedModel], X: np.ndarray) -> np.ndarray:
    """Mean probability across ensemble members. Shape (N,)."""
    probs = np.stack([m.predict_proba(X) for m in ensemble], axis=0)
    return probs.mean(axis=0)


def ensemble_std(ensemble: list[FittedModel], X: np.ndarray) -> np.ndarray:
    """Std of probabilities across ensemble. Shape (N,)."""
    probs = np.stack([m.predict_proba(X) for m in ensemble], axis=0)
    return probs.std(axis=0)


# ─── Metrics ─────────────────────────────────────────────────────────────

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


def eval_holdout(
    y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute standard holdout metrics."""
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
        "train_pos_rate": float(np.mean(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, prob))
    except ValueError:
        out["brier"] = float("nan")
    return out
