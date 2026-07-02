"""Compare MLP architectures for Chapter 6 overlap-volume regression."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ._public_helpers import FEATURES, _features, _read_csv
from .label_cache import DEFAULT_THRESHOLD_NORM
from .models import predict_overlap_norm, train_overlap_regressor
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

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "small": (64, 32),
    "medium": (128, 64, 32),
    "wide_deep": (256, 128, 64, 32),
}


def _parse_budgets(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _target(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["total_overlap_norm"]) for r in rows], dtype=np.float32)


def _pair_targets(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[f"pair_norm_{name}"]) for name in PAIR_COLUMNS] for r in rows], dtype=np.float32)


def _dominant_pairs(rows: list[dict]) -> list[str]:
    return [str(r["dominant_pair"]) for r in rows]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_standardizer(path: Path, std, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=std.mean_, scale=std.scale_, features=np.array(features))


def run(args: argparse.Namespace) -> Path:
    run_dir = RUNS_DIR / "models" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    acquired = _read_csv(args.acquired_csv)
    holdout = _read_csv(args.holdout_labeled_csv)
    budgets = _parse_budgets(args.budgets)
    holdout_x = _features(holdout)
    holdout_y = _target(holdout)
    holdout_pair = _pair_targets(holdout)
    dominant = _dominant_pairs(holdout)
    rows = []
    best: dict | None = None
    saved_models = []
    for arch_name, hidden in ARCHITECTURES.items():
        for budget in budgets:
            train_rows = acquired[:budget]
            train_x = _features(train_rows)
            train_y = _target(train_rows)
            model, std = train_overlap_regressor(
                train_x,
                train_y,
                threshold_norm=DEFAULT_THRESHOLD_NORM,
                hidden=hidden,
                seed=args.seed + budget + len(hidden),
                epochs=args.epochs,
            )
            pred = predict_overlap_norm(model, std, holdout_x, threshold_norm=DEFAULT_THRESHOLD_NORM)
            metrics = regression_metrics(
                holdout_y,
                pred,
                pair_true=holdout_pair,
                dominant_pairs=dominant,
            )
            row = {
                "architecture": arch_name,
                "hidden": "-".join(str(v) for v in hidden),
                "budget": int(budget),
                "train_overlap_mean": float(train_y.mean()),
                "train_overlap_median": float(np.median(train_y)),
                "train_overlap_positive_rate": float(np.mean(train_y > DEFAULT_THRESHOLD_NORM)),
                **metrics,
            }
            rows.append(row)
            if best is None or float(row["mae_norm"]) < float(best["mae_norm"]):
                model_dir = MODELS_DIR / "overlap_regressor_regression_v3_selected"
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / "model_state.pt")
                _save_standardizer(model_dir / "standardizer.npz", std, FEATURES)
                (model_dir / "architecture.json").write_text(
                    json.dumps({"architecture": arch_name, "hidden": hidden, "budget": budget}, indent=2),
                    encoding="utf-8",
                )
                best = row
    for arch_name, hidden in ARCHITECTURES.items():
        train_rows = acquired[: max(budgets)]
        train_x = _features(train_rows)
        train_y = _target(train_rows)
        model, std = train_overlap_regressor(
            train_x,
            train_y,
            threshold_norm=DEFAULT_THRESHOLD_NORM,
            hidden=hidden,
            seed=args.seed + 5000 + len(hidden),
            epochs=args.epochs,
        )
        model_dir = MODELS_DIR / f"overlap_regressor_{arch_name}_v2"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model_state.pt")
        _save_standardizer(model_dir / "standardizer.npz", std, FEATURES)
        (model_dir / "model_card.md").write_text(
            f"# Overlap Regressor {arch_name} v2\n\n"
            f"- Hidden layers: `{hidden}`\n"
            f"- Training labels: `{max(budgets)}` from `{args.acquired_csv}`\n"
            f"- Target: continuous `overlap_total_norm`, trained on log-scaled overlap.\n"
            f"- Evaluation metrics: see `{run_dir / 'architecture_metrics.csv'}`.\n",
            encoding="utf-8",
        )
        saved_models.append(str(model_dir))

    _write_csv(run_dir / "architecture_metrics.csv", rows)
    summary = {
        "run_id": args.run_id,
        "acquired_csv": str(args.acquired_csv),
        "holdout_labeled_csv": str(args.holdout_labeled_csv),
        "budgets": budgets,
        "epochs": args.epochs,
        "architectures": {k: list(v) for k, v in ARCHITECTURES.items()},
        "best_by_mae_norm": best,
        "saved_models": saved_models,
        "selected_model_dir": str(MODELS_DIR / "overlap_regressor_regression_v3_selected"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir, summary)
    print(run_dir)
    return run_dir


def _write_report(run_dir: Path, summary: dict) -> None:
    best = summary["best_by_mae_norm"]
    lines = [
        "# Chapter 6 Regression MLP Architecture Comparison",
        "",
        "Three MLP regressors are compared on continuous normalized overlap volume. No classification metrics are used.",
        "",
        "## Best Architecture By MAE",
        "",
        "| Architecture | Budget | Hidden | MAE norm | RMSE norm | Spearman | Near-threshold MAE |",
        "|---|---:|---|---:|---:|---:|---:|",
        f"| {best['architecture']} | {best['budget']} | {best['hidden']} | {best['mae_norm']:.6g} | "
        f"{best['rmse_norm']:.6g} | {best['spearman_norm']:.4f} | {best['near_threshold_mae_norm']:.6g} |",
        "",
        "Artifacts:",
        "",
        "- `architecture_metrics.csv`",
        f"- selected model: `{summary['selected_model_dir']}`",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="mlp_arch_regression_v2")
    parser.add_argument(
        "--acquired-csv",
        type=Path,
        default=RUNS_DIR / "active_learning" / "al_regression_v2_15k" / "acquired_random_15k.csv",
    )
    parser.add_argument(
        "--holdout-labeled-csv",
        type=Path,
        default=HOLDOUT_LABELED_5K,
    )
    parser.add_argument("--budgets", default="3000,5000,7500,10000,12500,15000")
    parser.add_argument("--seed", type=int, default=252540)
    parser.add_argument("--epochs", type=int, default=140)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
