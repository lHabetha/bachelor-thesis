"""Regression-first active learning for Chapter 6 continuous overlap volume."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ._public_helpers import FEATURES, _features, _read_csv
from .label_cache import DEFAULT_THRESHOLD_NORM, LabelCache, params_from_row
from .models import predict_overlap_norm, train_overlap_regressor
from .paths import DATA_DIR, RUNS_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import PAIR_COLUMNS, continuous_label_from_payload, regression_metrics


def _parse_budgets(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _label_rows(rows: list[dict], cache: LabelCache) -> list[dict]:
    labeled = []
    for row in rows:
        payload = cache.label(params_from_row(row))
        labeled.append({**row, **continuous_label_from_payload(payload)})
    return labeled


def _target(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["total_overlap_norm"]) for r in rows], dtype=np.float32)


def _pair_targets(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[f"pair_norm_{name}"]) for name in PAIR_COLUMNS] for r in rows], dtype=np.float32)


def _dominant_pairs(rows: list[dict]) -> list[str]:
    return [str(r["dominant_pair"]) for r in rows]


def _train_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int,
    epochs: int,
) -> np.ndarray:
    model, std = train_overlap_regressor(
        train_x,
        train_y,
        threshold_norm=DEFAULT_THRESHOLD_NORM,
        hidden=hidden,
        seed=seed,
        epochs=epochs,
    )
    return predict_overlap_norm(model, std, eval_x, threshold_norm=DEFAULT_THRESHOLD_NORM)


def _ensemble_variance(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_x: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int,
    epochs: int,
    ensemble_size: int,
) -> np.ndarray:
    preds = []
    for member in range(ensemble_size):
        preds.append(
            _train_predict(
                train_x,
                train_y,
                candidate_x,
                hidden=hidden,
                seed=seed + 1009 * member,
                epochs=epochs,
            )
        )
    return np.var(np.vstack(preds), axis=0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_acquired(
    run_dir: Path,
    name: str,
    pool_rows: list[dict],
    selected: np.ndarray,
    cache: LabelCache,
) -> None:
    rows = []
    for order, idx in enumerate(selected.tolist()):
        row = dict(pool_rows[int(idx)])
        payload = cache.label(params_from_row(row))
        rows.append(
            {
                "acquisition_order": order,
                "pool_index": int(idx),
                **continuous_label_from_payload(payload),
                **row,
            }
        )
    _write_csv(run_dir / f"acquired_{name}.csv", rows)


def _evaluate(
    strategy: str,
    budget: int,
    selected: np.ndarray,
    pool_rows: list[dict],
    pool_x: np.ndarray,
    holdout_x: np.ndarray,
    holdout_labeled: list[dict],
    *,
    hidden: tuple[int, ...],
    seed: int,
    epochs: int,
) -> dict:
    train_rows = _label_rows([pool_rows[int(i)] for i in selected], LabelCache())
    pred = _train_predict(
        pool_x[selected],
        _target(train_rows),
        holdout_x,
        hidden=hidden,
        seed=seed,
        epochs=epochs,
    )
    metrics = regression_metrics(
        _target(holdout_labeled),
        pred,
        pair_true=_pair_targets(holdout_labeled),
        dominant_pairs=_dominant_pairs(holdout_labeled),
    )
    return {
        "strategy": strategy,
        "budget": int(budget),
        "n_labels": int(len(selected)),
        "train_overlap_mean": float(_target(train_rows).mean()),
        "train_overlap_median": float(np.median(_target(train_rows))),
        "train_overlap_positive_rate": float(np.mean(_target(train_rows) > DEFAULT_THRESHOLD_NORM)),
        **metrics,
    }


def run(args: argparse.Namespace) -> Path:
    run_dir = RUNS_DIR / "active_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pool_rows = _read_csv(args.pool_csv)
    holdout_rows = _read_csv(args.holdout_csv)[: args.holdout_n]
    pool_x = _features(pool_rows)
    holdout_x = _features(holdout_rows)
    cache = LabelCache()
    t0 = time.perf_counter()
    holdout_labeled = _label_rows(holdout_rows, cache)
    budgets = _parse_budgets(args.budgets)
    hidden = tuple(int(v) for v in args.hidden.split(",") if v)
    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(pool_rows))

    random_order = rng.choice(all_indices, size=max(budgets), replace=False)
    active_selected = random_order[: args.base].copy()
    remaining_mask = np.ones(len(pool_rows), dtype=bool)
    remaining_mask[active_selected] = False

    logs: list[dict] = []
    for budget in budgets:
        random_selected = random_order[:budget]
        logs.append(
            _evaluate(
                "random",
                budget,
                random_selected,
                pool_rows,
                pool_x,
                holdout_x,
                holdout_labeled,
                hidden=hidden,
                seed=args.seed + budget,
                epochs=args.epochs,
            )
        )

        while len(active_selected) < budget:
            k = min(args.batch, budget - len(active_selected))
            train_rows = _label_rows([pool_rows[int(i)] for i in active_selected], cache)
            candidates = np.flatnonzero(remaining_mask)
            variances = _ensemble_variance(
                pool_x[active_selected],
                _target(train_rows),
                pool_x[candidates],
                hidden=hidden,
                seed=args.seed + len(active_selected),
                epochs=args.ensemble_epochs,
                ensemble_size=args.ensemble_size,
            )
            new_idx = candidates[np.argsort(-variances)[:k]]
            active_selected = np.concatenate([active_selected, new_idx])
            remaining_mask[new_idx] = False

        logs.append(
            _evaluate(
                "regression_uncertainty",
                budget,
                active_selected,
                pool_rows,
                pool_x,
                holdout_x,
                holdout_labeled,
                hidden=hidden,
                seed=args.seed + 17 + budget,
                epochs=args.epochs,
            )
        )

    _write_csv(run_dir / "regression_learning_log.csv", logs)
    _write_acquired(run_dir, "random", pool_rows, random_order[: max(budgets)], cache)
    _write_acquired(run_dir, "regression_uncertainty", pool_rows, active_selected, cache)
    _write_csv(run_dir / "holdout_labeled.csv", holdout_labeled)

    best_active = min(
        [r for r in logs if r["strategy"] == "regression_uncertainty"],
        key=lambda r: float(r["mae_norm"]),
    )
    best_random = min([r for r in logs if r["strategy"] == "random"], key=lambda r: float(r["mae_norm"]))
    summary = {
        "run_id": args.run_id,
        "pool_csv": str(args.pool_csv),
        "holdout_csv": str(args.holdout_csv),
        "holdout_n": args.holdout_n,
        "base": args.base,
        "batch": args.batch,
        "budgets": budgets,
        "max_total": max(budgets),
        "hidden": hidden,
        "epochs": args.epochs,
        "ensemble_epochs": args.ensemble_epochs,
        "ensemble_size": args.ensemble_size,
        "elapsed_s": time.perf_counter() - t0,
        "best_active_by_mae": best_active,
        "best_random_by_mae": best_random,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir, summary)
    print(run_dir)
    return run_dir


def _write_report(run_dir: Path, summary: dict) -> None:
    lines = [
        "# Chapter 6 Regression Active Learning",
        "",
        "This run treats overlap as a continuous regression target. No BAC, accuracy, ROC-AUC, or recall metrics are used.",
        "",
        "## Protocol",
        "",
        f"- Candidate pool: `{summary['pool_csv']}`",
        f"- Holdout pool: `{summary['holdout_csv']}`",
        f"- Holdout labels: `{summary['holdout_n']}`",
        f"- Random base labels: `{summary['base']}`",
        f"- Budgets: `{summary['budgets']}`",
        f"- Acquisition: regression ensemble uncertainty on transformed overlap magnitude.",
        "",
        "## Best MAE Rows",
        "",
        "| Strategy | Labels | MAE norm | RMSE norm | Spearman | Near-threshold MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("best_random_by_mae", "best_active_by_mae"):
        row = summary[key]
        lines.append(
            f"| {row['strategy']} | {row['n_labels']} | {row['mae_norm']:.6g} | "
            f"{row['rmse_norm']:.6g} | {row['spearman_norm']:.4f} | "
            f"{row['near_threshold_mae_norm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `regression_learning_log.csv`",
            "- `acquired_random.csv`",
            "- `acquired_regression_uncertainty.csv`",
            "- `holdout_labeled.csv`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="al_regression_v2_15k")
    parser.add_argument("--pool-csv", type=Path, default=POOL_100K_CSV)
    parser.add_argument("--holdout-csv", type=Path, default=HOLDOUT_5K_CSV)
    parser.add_argument("--holdout-n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=252530)
    parser.add_argument("--base", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=2500)
    parser.add_argument("--budgets", default="3000,5000,7500,10000,12500,15000")
    parser.add_argument("--hidden", default="128,64,32")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--ensemble-epochs", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
