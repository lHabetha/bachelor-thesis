"""Small PyTorch overlap surrogate models for Chapter 6."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Standardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (x - self.mean_) / self.scale_


class OverlapMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 120,
    lr: float = 2e-3,
) -> tuple[OverlapMLP, Standardizer]:
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    y = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    model = OverlapMLP(x.shape[1])
    pos = float(y.sum().item())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model, std


def train_classifier_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 120,
    lr: float = 2e-3,
) -> tuple[OverlapMLP, Standardizer, dict]:
    model, std = train_classifier(x_train, y_train, seed=seed, epochs=epochs, lr=lr)
    prob = predict_proba(model, std, x_val)
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float((pred == y_val).mean()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
    }
    return model, std, metrics


def predict_proba(model: OverlapMLP, std: Standardizer, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = torch.tensor(std.transform(x), dtype=torch.float32)
        return torch.sigmoid(model(z)).cpu().numpy()


def train_overlap_regressor(
    x_train: np.ndarray,
    y_overlap_norm: np.ndarray,
    *,
    threshold_norm: float,
    hidden: tuple[int, ...] = (64, 32),
    seed: int = 0,
    epochs: int = 180,
    lr: float = 2e-3,
) -> tuple[OverlapMLP, Standardizer]:
    """Train a magnitude surrogate on log-scaled normalized overlap."""
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    y_scaled = np.log1p(np.maximum(y_overlap_norm, 0.0) / threshold_norm)
    y = torch.tensor(y_scaled.astype(np.float32), dtype=torch.float32)
    model = OverlapMLP(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model, std


def predict_overlap_norm(
    model: OverlapMLP,
    std: Standardizer,
    x: np.ndarray,
    *,
    threshold_norm: float,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = torch.tensor(std.transform(x), dtype=torch.float32)
        pred = torch.clamp(model(z), min=0.0).cpu().numpy()
    return np.expm1(pred) * threshold_norm
