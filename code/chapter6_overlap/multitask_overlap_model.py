"""Chapter 6 #5e — multitask overlap surrogate (volume + binary heads).

A shared MLP trunk (the selected Chapter 6 backbone ``(256,128,64,32)``) feeds two
heads:

- ``head_vol``: scalar regression on log-scaled normalized overlap, identical target
  and SmoothL1 loss to the single-head :func:`models.train_overlap_regressor`. This
  head keeps the Chapter 6.3 ``MAE_log`` parity with the deployed v3 regressor.
- ``head_bin``: scalar logit for a binary "any overlap above ``tau0``" classifier
  trained with ``BCEWithLogits``. It yields a magnitude-invariant ``P(overlap)`` used
  for active-learning acquisition (:mod:`multitask_al_probe`) and as a no-overlap stop
  signal (``tau_bin``) for the later strict-repair stage (#5d).

Joint loss: ``L = L_vol + lambda_bin * L_bin``.

Running this module as ``__main__`` trains the random-15k baseline, sweeps
``lambda_bin`` for volume ``MAE_log`` parity, reports both heads' holdout metrics, and
exports the selected checkpoint to ``models/overlap_regressor_multitask_v1_selected/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

from ._public_helpers import FEATURES, _features, _read_csv
from .label_cache import DEFAULT_THRESHOLD_NORM
from .models import Standardizer
from .paths import MODELS_DIR, RUNS_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import PAIR_COLUMNS, regression_metrics

TAU_BIN_CANDIDATES = (0.05, 0.1, 0.2)


class OverlapMultitaskMLP(nn.Module):
    """Shared trunk with a log-overlap regression head and a binary-overlap head."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128, 64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.head_vol = nn.Linear(prev, 1)
        self.head_bin = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_vol(h).squeeze(-1), self.head_bin(h).squeeze(-1)


def train_multitask_overlap_model(
    x_train: np.ndarray,
    y_overlap_norm: np.ndarray,
    *,
    threshold_norm: float,
    hidden: tuple[int, ...] = (256, 128, 64, 32),
    seed: int = 0,
    epochs: int = 180,
    lr: float = 2e-3,
    lambda_bin: float = 1.0,
    bin_threshold: float | None = None,
    pos_weight: float | str | None = "auto",
) -> tuple[OverlapMultitaskMLP, Standardizer]:
    """Jointly train the volume and binary heads on a shared trunk.

    ``y_vol = log1p(max(y, 0) / threshold_norm)`` (SmoothL1) and
    ``y_bin = 1[y_overlap_norm > bin_threshold]`` (BCEWithLogits). ``bin_threshold``
    defaults to ``threshold_norm`` (``tau0 = 5e-5``), the same scale that defines a
    "clean" design in Chapter 6.3.
    """
    if bin_threshold is None:
        bin_threshold = threshold_norm
    torch.manual_seed(seed)
    std = Standardizer().fit(x_train)
    x = torch.tensor(std.transform(x_train), dtype=torch.float32)
    y_vol_np = np.log1p(np.maximum(y_overlap_norm, 0.0) / threshold_norm)
    y_vol = torch.tensor(y_vol_np.astype(np.float32), dtype=torch.float32)
    y_bin_np = (np.asarray(y_overlap_norm, dtype=np.float64) > bin_threshold).astype(np.float32)
    y_bin = torch.tensor(y_bin_np, dtype=torch.float32)

    model = OverlapMultitaskMLP(x.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    vol_loss_fn = nn.SmoothL1Loss()

    if pos_weight == "auto":
        pos = float(y_bin.sum().item())
        neg = float(len(y_bin) - pos)
        pw: torch.Tensor | None = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    elif pos_weight is None:
        pw = None
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
    bin_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        vol_pred, bin_logit = model(x)
        loss = vol_loss_fn(vol_pred, y_vol) + lambda_bin * bin_loss_fn(bin_logit, y_bin)
        loss.backward()
        opt.step()
    return model, std


def predict_multitask(
    model: OverlapMultitaskMLP,
    std: Standardizer,
    x: np.ndarray,
    *,
    threshold_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(overlap_norm, p_overlap)`` for ``x``."""
    model.eval()
    with torch.no_grad():
        z = torch.tensor(std.transform(x), dtype=torch.float32)
        vol_log, bin_logit = model(z)
        vol = torch.clamp(vol_log, min=0.0).cpu().numpy()
        prob = torch.sigmoid(bin_logit).cpu().numpy()
    return np.expm1(vol) * threshold_norm, prob


def predict_overlap_norm_multitask(
    model: OverlapMultitaskMLP,
    std: Standardizer,
    x: np.ndarray,
    *,
    threshold_norm: float,
) -> np.ndarray:
    """Volume-head only convenience wrapper (drop-in for ``predict_overlap_norm``)."""
    return predict_multitask(model, std, x, threshold_norm=threshold_norm)[0]


def predict_p_overlap(
    model: OverlapMultitaskMLP,
    std: Standardizer,
    x: np.ndarray,
) -> np.ndarray:
    """Binary-head probability of any overlap above ``tau0``."""
    model.eval()
    with torch.no_grad():
        z = torch.tensor(std.transform(x), dtype=torch.float32)
        _, bin_logit = model(z)
        return torch.sigmoid(bin_logit).cpu().numpy()


def save_multitask(model_dir: Path, model: OverlapMultitaskMLP, std: Standardizer, meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model_state.pt")
    np.savez(
        model_dir / "standardizer.npz",
        mean=std.mean_,
        scale=std.scale_,
        features=np.array(FEATURES),
    )
    (model_dir / "architecture.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_multitask_overlap_model(model_dir: Path) -> tuple[OverlapMultitaskMLP, Standardizer, dict]:
    """Load a multitask checkpoint exported by :func:`save_multitask`."""
    arch = json.loads((model_dir / "architecture.json").read_text(encoding="utf-8"))
    hidden = tuple(int(v) for v in arch["hidden"])
    model = OverlapMultitaskMLP(len(FEATURES), hidden=hidden)
    model.load_state_dict(torch.load(model_dir / "model_state.pt", map_location="cpu"))
    model.eval()
    data = np.load(model_dir / "standardizer.npz", allow_pickle=True)
    std = Standardizer()
    std.mean_ = data["mean"].astype(np.float32)
    std.scale_ = data["scale"].astype(np.float32)
    return model, std, arch


def binary_metrics(y_true_bin: np.ndarray, prob: np.ndarray, *, prefix: str = "bin_") -> dict:
    """Holdout binary-head metrics (sklearn; the project already depends on it)."""
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    pred = (np.asarray(prob) >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, pred, average="binary", zero_division=0
    )
    out = {
        f"{prefix}accuracy": float((pred == y_true_bin).mean()),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true_bin, pred)),
        f"{prefix}precision": float(precision),
        f"{prefix}recall": float(recall),
        f"{prefix}f1": float(f1),
        f"{prefix}positive_rate_true": float(np.mean(y_true_bin)),
    }
    if len(set(y_true_bin.tolist())) > 1:
        out[f"{prefix}auc"] = float(roc_auc_score(y_true_bin, prob))
    else:
        out[f"{prefix}auc"] = float("nan")
    return out


# --------------------------------------------------------------------------------------
# Random-15k baseline + lambda_bin sweep + export.
# --------------------------------------------------------------------------------------


def _target(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["total_overlap_norm"]) for r in rows], dtype=np.float32)


def _pair_targets(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[float(r[f"pair_norm_{name}"]) for name in PAIR_COLUMNS] for r in rows],
        dtype=np.float32,
    )


def _dominant_pairs(rows: list[dict]) -> list[str]:
    return [str(r["dominant_pair"]) for r in rows]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    run_dir = RUNS_DIR / "models" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_csv(args.train_csv)[: args.budget]
    holdout_rows = _read_csv(args.holdout_labeled_csv)
    train_x = _features(train_rows)
    train_y = _target(train_rows)
    holdout_x = _features(holdout_rows)
    holdout_y = _target(holdout_rows)
    holdout_pair = _pair_targets(holdout_rows)
    dominant = _dominant_pairs(holdout_rows)

    bin_threshold = float(args.bin_threshold) if args.bin_threshold is not None else DEFAULT_THRESHOLD_NORM
    holdout_y_bin = (holdout_y > bin_threshold).astype(int)
    hidden = tuple(int(v) for v in args.hidden.split(","))
    lambda_bins = [float(v) for v in args.lambda_bins.split(",") if v.strip()]

    rows: list[dict] = []
    states: dict[float, tuple] = {}
    for idx, lam in enumerate(lambda_bins):
        model, std = train_multitask_overlap_model(
            train_x,
            train_y,
            threshold_norm=DEFAULT_THRESHOLD_NORM,
            hidden=hidden,
            seed=args.seed + idx,
            epochs=args.epochs,
            lambda_bin=lam,
            bin_threshold=bin_threshold,
        )
        pred_vol, pred_p = predict_multitask(model, std, holdout_x, threshold_norm=DEFAULT_THRESHOLD_NORM)
        metrics = regression_metrics(holdout_y, pred_vol, pair_true=holdout_pair, dominant_pairs=dominant)
        bmetrics = binary_metrics(holdout_y_bin, pred_p)
        row = {
            "lambda_bin": lam,
            "hidden": "-".join(str(v) for v in hidden),
            "budget": int(len(train_rows)),
            "bin_threshold_norm": bin_threshold,
            "label_source": "random",
            "train_overlap_positive_rate": float(np.mean(train_y > bin_threshold)),
            **metrics,
            **bmetrics,
        }
        rows.append(row)
        states[lam] = (model.state_dict(), std, hidden, lam)
        auc = row.get("bin_auc", float("nan"))
        print(
            f"[lambda_bin={lam:>4}] mae_log={row['mae_log']:.5f} mae_norm={row['mae_norm']:.6g} "
            f"bin_f1={row['bin_f1']:.4f} bin_auc={auc:.4f}"
        )

    control = next((r for r in rows if r["lambda_bin"] == 0.0), None)
    feasible = [r for r in rows if r["lambda_bin"] > 0.0]
    if not feasible:
        raise SystemExit("need at least one lambda_bin > 0 to train a usable binary head")
    # Prioritize volume MAE_log parity (thesis requirement); ties broken by binary AUC.
    selected = min(feasible, key=lambda r: (round(float(r["mae_log"]), 4), -float(r.get("bin_auc") or 0.0)))
    sel_lambda = float(selected["lambda_bin"])
    parity_vs_control = (
        float(selected["mae_log"]) - float(control["mae_log"]) if control is not None else float("nan")
    )

    sel_dir = MODELS_DIR / args.selected_dir
    state_dict, std, sel_hidden, _ = states[sel_lambda]
    meta = {
        "architecture": f"multitask_{'_'.join(str(v) for v in sel_hidden)}",
        "hidden": list(sel_hidden),
        "multitask": True,
        "heads": {
            "head_vol": "log1p(total_overlap_norm / tau0) regression (SmoothL1)",
            "head_bin": "P(overlap > tau0) logit (BCEWithLogits)",
        },
        "budget": int(args.budget),
        "lambda_bin": sel_lambda,
        "bin_threshold_norm": bin_threshold,
        "threshold_norm": DEFAULT_THRESHOLD_NORM,
        "tau_bin_candidates": list(TAU_BIN_CANDIDATES),
        "selected_by": "volume_mae_log_parity",
        "label_source": "random",
    }
    model = OverlapMultitaskMLP(len(FEATURES), hidden=sel_hidden)
    model.load_state_dict(state_dict)
    save_multitask(sel_dir, model, std, meta)

    auc_sel = selected.get("bin_auc", float("nan"))
    (sel_dir / "model_card.md").write_text(
        "# Overlap Regressor — Chapter 6 #5e multitask (v1)\n\n"
        f"- Architecture: `{meta['architecture']}` hidden `{tuple(sel_hidden)}` (shared trunk + two heads)\n"
        f"- Training labels: `{args.budget}` random-labelled rows from `{Path(args.train_csv).name}`\n"
        f"- Joint loss: `L_vol (SmoothL1 on log1p(y/{DEFAULT_THRESHOLD_NORM})) + lambda_bin * L_bin (BCEWithLogits)`\n"
        f"- Selected `lambda_bin`: `{sel_lambda}` (by volume MAE_log parity; ties by binary AUC)\n\n"
        "## Volume head (repair gradients / MAE_log)\n\n"
        f"- Holdout `mae_log`: `{selected['mae_log']:.5f}`, `mae_norm`: `{selected['mae_norm']:.6g}`\n"
        f"- Parity vs lambda_bin=0 volume-only control: `{parity_vs_control:+.5f}` MAE_log\n"
        "- Same role as the deployed single-head regressor: predicted overlap / log-overlap for repair descent.\n\n"
        "## Binary head (P(overlap), repair stop)\n\n"
        f"- Binary label rule: `total_overlap_norm > {bin_threshold}` (= tau0 clean threshold).\n"
        f"- Holdout binary F1: `{selected['bin_f1']:.4f}`, AUC: `{auc_sel:.4f}`, "
        f"accuracy: `{selected['bin_accuracy']:.4f}`, positive rate: `{selected['bin_positive_rate_true']:.3f}`\n"
        f"- Repair stop criterion (#5d): declare \"no overlap\" when `P(overlap) < tau_bin`, "
        f"sweep `tau_bin in {{{', '.join(str(t) for t in TAU_BIN_CANDIDATES)}}}` (analogous to the Chapter 5 assemblability tau).\n\n"
        "## Deployment note\n\n"
        "- Loadable via `multitask_overlap_model.load_multitask_overlap_model`.\n"
        "- `strict_repair` / `zigzag` defaults are NOT switched to this checkpoint yet; integration is tracked in #5d.\n",
        encoding="utf-8",
    )

    _write_csv(run_dir / "lambda_sweep.csv", rows)
    summary = {
        "run_id": args.run_id,
        "train_csv": str(args.train_csv),
        "holdout_labeled_csv": str(args.holdout_labeled_csv),
        "budget": int(args.budget),
        "epochs": args.epochs,
        "hidden": list(hidden),
        "bin_threshold_norm": bin_threshold,
        "lambda_bins": lambda_bins,
        "selected_lambda_bin": sel_lambda,
        "selected_mae_log": float(selected["mae_log"]),
        "selected_bin_f1": float(selected["bin_f1"]),
        "selected_bin_auc": (None if not np.isfinite(auc_sel) else float(auc_sel)),
        "parity_mae_log_vs_control": (None if not np.isfinite(parity_vs_control) else float(parity_vs_control)),
        "control_lambda0_mae_log": (None if control is None else float(control["mae_log"])),
        "tau_bin_candidates": list(TAU_BIN_CANDIDATES),
        "selected_model_dir": str(sel_dir),
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        f"# Multitask overlap baseline ({args.run_id})\n\n"
        f"Random-{args.budget} multitask training (volume + binary heads) on the selected "
        f"`{'-'.join(str(v) for v in hidden)}` backbone, with a `lambda_bin` sweep over "
        f"`{lambda_bins}`.\n\n"
        f"- Selected `lambda_bin = {sel_lambda}`; holdout volume `mae_log = {selected['mae_log']:.5f}` "
        f"(parity vs lambda_bin=0 control: `{parity_vs_control:+.5f}`).\n"
        f"- Binary head holdout F1 `{selected['bin_f1']:.4f}`, AUC `{auc_sel:.4f}`.\n"
        f"- Exported checkpoint: `{sel_dir}`.\n\n"
        "See `lambda_sweep.csv` and `summary.json` for the full sweep.\n",
        encoding="utf-8",
    )
    print(
        f"selected lambda_bin={sel_lambda} mae_log={selected['mae_log']:.5f} "
        f"(control lambda0={None if control is None else round(float(control['mae_log']), 5)}) "
        f"bin_f1={selected['bin_f1']:.4f} bin_auc={auc_sel:.4f}"
    )
    print(sel_dir)
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="multitask_overlap_v1")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ACQUIRED_RANDOM_15K,
    )
    parser.add_argument(
        "--holdout-labeled-csv",
        type=Path,
        default=HOLDOUT_LABELED_5K,
    )
    parser.add_argument("--budget", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=360540)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--hidden", default="256,128,64,32")
    parser.add_argument("--lambda-bins", default="0.0,0.25,0.5,1.0,2.0")
    parser.add_argument("--bin-threshold", type=float, default=None)
    parser.add_argument("--selected-dir", default="overlap_regressor_multitask_v1_selected")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
