"""Training, evaluation, and artifact utilities for Chapter 4 graph surrogates."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from torch import nn

from .data_utils import FEATURE_LIST, set_single_thread
from .graph_models import Standardizer, create_model, parameter_count


@dataclass
class FittedSurrogate:
    architecture: str
    model: nn.Module
    scaler: Standardizer
    best_epoch: int
    best_val_loss: float
    metadata: dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xz = self.scaler.transform(X)
        with torch.no_grad():
            xt = torch.as_tensor(Xz, dtype=torch.float32)
            return torch.sigmoid(self.model(xt)).cpu().numpy()

    def gradient(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        self.model.eval()
        xz = self.scaler.transform(x.reshape(1, -1))
        xt = torch.as_tensor(xz, dtype=torch.float32)
        xt.requires_grad_(True)
        prob = torch.sigmoid(self.model(xt))
        prob.backward()
        grad_z = xt.grad.detach().cpu().numpy().reshape(-1)
        grad_raw = grad_z / self.scaler.std
        return grad_raw.astype(np.float64), float(prob.item())


def calibration_ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if not np.any(mask):
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def eval_binary(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
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
        "brier": float(brier_score_loss(y_true, prob)),
        "ece_10bin": calibration_ece(y_true, prob, n_bins=10),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def _split_indices(y: np.ndarray, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val_pos = max(1, int(round(val_frac * len(pos)))) if len(pos) > 1 else 0
    n_val_neg = max(1, int(round(val_frac * len(neg)))) if len(neg) > 1 else 0
    val = np.concatenate([pos[:n_val_pos], neg[:n_val_neg]])
    train = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])
    if len(train) == 0:
        train = np.arange(len(y))
        val = np.arange(len(y))
    elif len(val) == 0:
        val = train[: max(1, len(train) // 5)]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def fit_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    architecture: str,
    seed: int,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 160,
    patience: int = 22,
    val_frac: float = 0.15,
) -> FittedSurrogate:
    set_single_thread()
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    y = y.astype(np.int64)

    train_idx, val_idx = _split_indices(y, seed, val_frac)
    scaler = Standardizer.fit(X[train_idx])
    xtr = torch.as_tensor(scaler.transform(X[train_idx]), dtype=torch.float32)
    ytr = torch.as_tensor(y[train_idx], dtype=torch.float32)
    xva = torch.as_tensor(scaler.transform(X[val_idx]), dtype=torch.float32)
    yva = torch.as_tensor(y[val_idx], dtype=torch.float32)

    model = create_model(architecture)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_pos = float(np.sum(y[train_idx] == 1))
    n_neg = float(np.sum(y[train_idx] == 0))
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xtr[batch]), ytr[batch])
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
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{architecture} training never improved")
    model.load_state_dict(best_state)
    model.eval()
    return FittedSurrogate(
        architecture=architecture,
        model=model,
        scaler=scaler,
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        metadata={
            "seed": seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "pos_weight": float(pos_weight.item()),
            "parameter_count": parameter_count(model),
            "fit_wall_s": float(time.perf_counter() - t0),
        },
    )


def save_artifact(
    fitted: FittedSurrogate,
    out_dir: Path,
    *,
    extra_card: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fitted.model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "standardizer.npz", mean=fitted.scaler.mean, std=fitted.scaler.std)
    card: dict[str, Any] = {
        "model_id": out_dir.name,
        "architecture": fitted.architecture,
        "task27_graph_surrogate": True,
        "feature_names": FEATURE_LIST,
        "n_features": len(FEATURE_LIST),
        "best_epoch": fitted.best_epoch,
        "best_val_loss": fitted.best_val_loss,
        **fitted.metadata,
    }
    if extra_card:
        card.update(extra_card)
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))


def load_artifact(model_dir: Path) -> FittedSurrogate:
    card = json.loads((model_dir / "model_card.json").read_text())
    architecture = card["architecture"]
    model = create_model(architecture)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    npz = np.load(model_dir / "standardizer.npz")
    scaler = Standardizer(mean=npz["mean"], std=npz["std"])
    model.eval()
    return FittedSurrogate(
        architecture=architecture,
        model=model,
        scaler=scaler,
        best_epoch=int(card.get("best_epoch", 0)),
        best_val_loss=float(card.get("best_val_loss", 0.0)),
        metadata=card,
    )
