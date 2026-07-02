"""Overlap regression models and metrics for Chapter 7 ADV architecture screen."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from torch import nn

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

DEFAULT_TAU_MM3 = 1.0
# Gate-augmented regression loss (§19.10): L = SmoothL1(z) + λ·BCE(logit(z), overlap).
DEFAULT_GATE_LAMBDA = 0.2
DEFAULT_GATE_LAMBDA_STRONG = 0.3
DEFAULT_GATE_LOGIT_SCALE = 10.0

# 100k grid variant ids (all mlp_256_128_64_32).
VARIANT_RAW_LOG_HUBER = "raw_log_huber"
VARIANT_ENG_LOG_HUBER = "eng_log_huber"
VARIANT_ENG_GATE_AUG_STRONG = "eng_gate_aug_strong"
VARIANT_ENG_MULTITASK_GATE_STRONG = "eng_multitask_gate_strong"

GRID_VARIANTS: dict[str, dict] = {
    VARIANT_RAW_LOG_HUBER: {
        "engineered": False,
        "loss_variant": "log_huber",
        "multitask": False,
    },
    VARIANT_ENG_LOG_HUBER: {
        "engineered": True,
        "loss_variant": "log_huber",
        "multitask": False,
    },
    VARIANT_ENG_GATE_AUG_STRONG: {
        "engineered": True,
        "loss_variant": "gate_augmented",
        "multitask": False,
        "gate_lambda": DEFAULT_GATE_LAMBDA_STRONG,
        "gate_logit_scale": DEFAULT_GATE_LOGIT_SCALE,
    },
    VARIANT_ENG_MULTITASK_GATE_STRONG: {
        "engineered": True,
        "loss_variant": "multitask_gate",
        "multitask": True,
        "gate_lambda": DEFAULT_GATE_LAMBDA_STRONG,
    },
}


def overlap_threshold_z(*, tau_mm3: float = DEFAULT_TAU_MM3) -> float:
    """z value corresponding to y = tau (overlap/clean boundary in log target space)."""
    return float(np.log10(2.0))  # z = log10(1 + tau/tau) = log10(2)

MLP_ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "mlp_32": (32,),
    "mlp_64_32": (64, 32),
    "mlp_128_64": (128, 64),
    "mlp_128_64_32": (128, 64, 32),
    "mlp_256_128_64": (256, 128, 64),
    "mlp_256_128_64_32": (256, 128, 64, 32),
}

TREE_ARCHITECTURES = ("rf", "xgb", "lgbm")


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


class PerformanceMLP(nn.Module):
    """Vector-valued MLP for quick-sim performance regression."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple[int, ...] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TargetStandardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        self.mean_ = y.mean(axis=0)
        self.scale_ = y.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (y - self.mean_) / self.scale_

    def inverse_transform(self, y_z: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return y_z * self.scale_ + self.mean_


class OverlapMLPMultitask(nn.Module):
    """Shared trunk with overlap regression head + binary overlap head."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.reg_head = nn.Linear(prev, 1)
        self.bin_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        z = self.reg_head(h).squeeze(-1)
        logit = self.bin_head(h).squeeze(-1)
        return z, logit

    def forward_z(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


def transformed_target(y_mm3: np.ndarray, *, tau_mm3: float = DEFAULT_TAU_MM3) -> np.ndarray:
    """z = log10(1 + y / tau)."""
    y = np.maximum(np.asarray(y_mm3, dtype=float), 0.0)
    return np.log10(1.0 + y / tau_mm3)


def inverse_transformed_target(z: np.ndarray, *, tau_mm3: float = DEFAULT_TAU_MM3) -> np.ndarray:
    """Invert z back to overlap mm3."""
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    return (np.power(10.0, z) - 1.0) * tau_mm3


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tpr = float((y_pred & y_true).sum() / max(y_true.sum(), 1))
    tnr = float((~y_pred & ~y_true).sum() / max((~y_true).sum(), 1))
    return 0.5 * (tpr + tnr)


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float((y_pred & y_true).sum())
    fp = float((y_pred & ~y_true).sum())
    fn = float((~y_pred & y_true).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def regression_metrics(
    y_true_mm3: np.ndarray,
    y_pred_mm3: np.ndarray,
    *,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> dict[str, float | int]:
    y_true = np.maximum(np.asarray(y_true_mm3, dtype=float), 0.0)
    y_pred = np.maximum(np.asarray(y_pred_mm3, dtype=float), 0.0)
    err = y_pred - y_true
    abs_err = np.abs(err)

    z_true = transformed_target(y_true, tau_mm3=tau_mm3)
    z_pred = transformed_target(y_pred, tau_mm3=tau_mm3)
    mae_log = float(np.abs(z_pred - z_true).mean())

    overlap_true = y_true > tau_mm3
    overlap_pred = y_pred > tau_mm3

    out: dict[str, float | int] = {
        "n": int(len(y_true)),
        "mae_log": mae_log,
        "typical_factor": float(10.0 ** mae_log),
        "rmse_log": float(np.sqrt(np.mean((z_pred - z_true) ** 2))),
        "mae_mm3": float(abs_err.mean()),
        "rmse_mm3": float(np.sqrt(np.mean(err**2))),
        "median_ae_mm3": float(np.median(abs_err)),
        "pearson_mm3": _corr(y_true, y_pred),
        "spearman_mm3": _corr(_rankdata(y_true), _rankdata(y_pred)),
        "gate_balanced_acc": _balanced_accuracy(overlap_true, overlap_pred),
        "gate_f1": _f1(overlap_true, overlap_pred),
        "cond_mae_log": 0.0,
        "cond_n": 0,
    }

    if overlap_true.any():
        cond = np.abs(z_pred[overlap_true] - z_true[overlap_true])
        out["cond_mae_log"] = float(cond.mean())
        out["cond_n"] = int(overlap_true.sum())

    near = overlap_true | overlap_pred
    if near.any():
        out["near_threshold_mae_mm3"] = float(abs_err[near].mean())
    else:
        out["near_threshold_mae_mm3"] = 0.0

    return out


def train_mlp(
    x_train: np.ndarray,
    y_train_mm3: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int = 42,
    epochs: int = 180,
    lr: float = 2e-3,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> tuple[OverlapMLP, Standardizer]:
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    z = transformed_target(y_train_mm3, tau_mm3=tau_mm3)
    y = torch.tensor(z.astype(np.float32), dtype=torch.float32)

    model = OverlapMLP(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model, std


def train_performance_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: tuple[int, ...] = (256, 128, 64, 32),
    seed: int = 42,
    epochs: int = 180,
    lr: float = 2e-3,
) -> tuple[PerformanceMLP, Standardizer, TargetStandardizer]:
    """Train vector performance MLP with SmoothL1 on z-scored targets."""
    torch.manual_seed(seed)
    x_std = Standardizer().fit(x_train)
    y_std = TargetStandardizer().fit(y_train)
    x = torch.tensor(x_std.transform(x_train), dtype=torch.float32)
    y = torch.tensor(y_std.transform(y_train), dtype=torch.float32)

    model = PerformanceMLP(x.shape[1], y.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    return model, x_std, y_std


def predict_performance_mlp(
    model: PerformanceMLP,
    x_std: Standardizer,
    y_std: TargetStandardizer,
    x: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_z = model(torch.tensor(x_std.transform(x), dtype=torch.float32)).cpu().numpy()
    return y_std.inverse_transform(pred_z)


def performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Per-target and mean MAE in physical units."""
    out: dict[str, float] = {}
    maes: list[float] = []
    for j, name in enumerate(target_names):
        mae = float(np.mean(np.abs(y_true[:, j] - y_pred[:, j])))
        out[f"mae_{name}"] = mae
        maes.append(mae)
    out["mae_mean"] = float(np.mean(maes))
    return out


def train_mlp_gate_augmented(
    x_train: np.ndarray,
    y_train_mm3: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int = 42,
    epochs: int = 180,
    lr: float = 2e-3,
    tau_mm3: float = DEFAULT_TAU_MM3,
    gate_lambda: float = DEFAULT_GATE_LAMBDA,
    gate_logit_scale: float = DEFAULT_GATE_LOGIT_SCALE,
) -> tuple[OverlapMLP, Standardizer]:
    """Train overlap MLP with log-Huber regression + soft overlap/clean gate BCE."""
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    z_np = transformed_target(y_train_mm3, tau_mm3=tau_mm3)
    z = torch.tensor(z_np.astype(np.float32), dtype=torch.float32)

    overlap = (np.asarray(y_train_mm3, dtype=float) > tau_mm3)
    n_pos = max(int(overlap.sum()), 1)
    n_neg = max(int((~overlap).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    z_thresh = overlap_threshold_z(tau_mm3=tau_mm3)
    g = torch.tensor(overlap.astype(np.float32), dtype=torch.float32)

    model = OverlapMLP(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    reg_loss_fn = nn.SmoothL1Loss()
    gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_pred = model(x)
        reg_loss = reg_loss_fn(z_pred, z)
        logit = gate_logit_scale * (z_pred - z_thresh)
        gate_loss = gate_loss_fn(logit, g)
        loss = reg_loss + gate_lambda * gate_loss
        loss.backward()
        opt.step()
    return model, std


def predict_mlp(
    model: OverlapMLP,
    std: Standardizer,
    x: np.ndarray,
    *,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = torch.tensor(std.transform(x), dtype=torch.float32)
        pred_z = torch.clamp(model(z), min=0.0).cpu().numpy()
    return inverse_transformed_target(pred_z, tau_mm3=tau_mm3)


def train_mlp_multitask_gate(
    x_train: np.ndarray,
    y_train_mm3: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int = 42,
    epochs: int = 180,
    lr: float = 2e-3,
    tau_mm3: float = DEFAULT_TAU_MM3,
    gate_lambda: float = DEFAULT_GATE_LAMBDA_STRONG,
) -> tuple[OverlapMLPMultitask, Standardizer]:
    """Train multitask MLP: SmoothL1(z) + λ·BCE(binary_logit, is_overlap)."""
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    z_np = transformed_target(y_train_mm3, tau_mm3=tau_mm3)
    z = torch.tensor(z_np.astype(np.float32), dtype=torch.float32)

    overlap = (np.asarray(y_train_mm3, dtype=float) > tau_mm3)
    n_pos = max(int(overlap.sum()), 1)
    n_neg = max(int((~overlap).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    g = torch.tensor(overlap.astype(np.float32), dtype=torch.float32)

    model = OverlapMLPMultitask(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    reg_loss_fn = nn.SmoothL1Loss()
    gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        z_pred, bin_logit = model(x)
        reg_loss = reg_loss_fn(z_pred, z)
        gate_loss = gate_loss_fn(bin_logit, g)
        loss = reg_loss + gate_lambda * gate_loss
        loss.backward()
        opt.step()
    return model, std


def predict_mlp_multitask(
    model: OverlapMLPMultitask,
    std: Standardizer,
    x: np.ndarray,
    *,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (overlap_mm3 from reg head, P(overlap) from binary head)."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(std.transform(x), dtype=torch.float32)
        z_pred, bin_logit = model(xt)
        pred_z = torch.clamp(z_pred, min=0.0).cpu().numpy()
        prob = torch.sigmoid(bin_logit).cpu().numpy()
    return (
        inverse_transformed_target(pred_z, tau_mm3=tau_mm3),
        prob,
    )


def binary_head_metrics(y_true_mm3: np.ndarray, prob_overlap: np.ndarray, *, tau_mm3: float = DEFAULT_TAU_MM3) -> dict[str, float]:
    """Classification metrics for an explicit binary overlap head."""
    overlap_true = np.asarray(y_true_mm3, dtype=float) > tau_mm3
    overlap_pred = np.asarray(prob_overlap, dtype=float) >= 0.5
    return {
        "gate_f1_binary": _f1(overlap_true, overlap_pred),
        "gate_balanced_acc_binary": _balanced_accuracy(overlap_true, overlap_pred),
    }


def make_tree_model(name: str, *, seed: int = 42) -> Any:
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=1,
        )
    if name == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        return xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if name == "lgbm":
        if lgb is not None:
            return lgb.LGBMRegressor(
                n_estimators=400,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
        return HistGradientBoostingRegressor(
            max_iter=400,
            max_leaf_nodes=31,
            learning_rate=0.05,
            random_state=seed,
        )
    raise ValueError(f"Unknown tree architecture: {name}")


def train_tree(
    x_train: np.ndarray,
    y_train_mm3: np.ndarray,
    *,
    name: str,
    seed: int = 42,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> Any:
    model = make_tree_model(name, seed=seed)
    z = transformed_target(y_train_mm3, tau_mm3=tau_mm3)
    model.fit(x_train, z)
    return model


def predict_tree(
    model: Any,
    x: np.ndarray,
    *,
    tau_mm3: float = DEFAULT_TAU_MM3,
) -> np.ndarray:
    pred_z = np.maximum(model.predict(x), 0.0)
    return inverse_transformed_target(pred_z, tau_mm3=tau_mm3)


def architecture_catalog() -> list[tuple[str, str, tuple[int, ...] | str]]:
    """Return (arch_id, family, config) for all 9 architectures."""
    catalog: list[tuple[str, str, tuple[int, ...] | str]] = []
    for arch_id, hidden in MLP_ARCHITECTURES.items():
        catalog.append((arch_id, "mlp", hidden))
    for arch_id in TREE_ARCHITECTURES:
        catalog.append((arch_id, "tree", arch_id))
    return catalog
