"""Chapter 6 quick probe: active-learning strategy variants for overlap regression.

This is a deliberately smaller follow-up to ``task36_al_compare.py``. It keeps the
selected Chapter 6 architecture and ``MAE_log`` metric fixed, then asks whether the
negative active-learning result is specific to raw overlap-variance acquisition.
All strategies use only model predictions on the unlabeled pool; true labels are
queried only after an index has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ._public_helpers import _features, _read_csv  # noqa: E402
from .label_cache import DEFAULT_THRESHOLD_NORM, LabelCache, params_from_row  # noqa: E402
from .models import predict_overlap_norm, train_overlap_regressor  # noqa: E402
from .paths import DATA_DIR, FIGURES_DIR, MODELS_DIR, RUNS_DIR, TABLES_DIR  # noqa: E402
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import (  # noqa: E402
    PAIR_COLUMNS,
    continuous_label_from_payload,
    magnitude_bin,
    regression_metrics,
    transformed_target,
)

Strategy = Literal[
    "raw_variance",
    "log_variance",
    "near_threshold_log_variance",
    "stratified_log_variance",
]

STRATEGIES: tuple[Strategy, ...] = (
    "raw_variance",
    "log_variance",
    "near_threshold_log_variance",
    "stratified_log_variance",
)

STRATEGY_LABELS = {
    "raw_variance": "raw variance",
    "log_variance": "log variance",
    "near_threshold_log_variance": "near-threshold log variance",
    "stratified_log_variance": "stratified log variance",
}

SERIES_STYLE = {
    "random": ("random", "#e45756"),
    "active_base3000": ("active (base 3000)", "#4c78a8"),
    "active_base5000": ("active (base 5000)", "#54a24b"),
    "active_base10000": ("active (base 10000)", "#b279a2"),
}

METRIC_NAMES = (
    "mae_log",
    "mae_norm",
    "rmse_log",
    "near_threshold_mae_norm",
    "mae_bin_clean_or_below_threshold",
    "mae_bin_tiny",
    "mae_bin_small",
    "mae_bin_large",
    "spearman_norm",
)


@dataclass(frozen=True)
class Job:
    kind: str
    seed: int
    strategy: Strategy | None


class MemLabels:
    """Process-local memoized continuous labels keyed by pool index."""

    def __init__(self, pool_rows: list[dict]) -> None:
        self.pool_rows = pool_rows
        self.cache = LabelCache()
        self._mem: dict[int, dict] = {}

    def get(self, idx: int) -> dict:
        hit = self._mem.get(idx)
        if hit is None:
            payload = self.cache.label(params_from_row(self.pool_rows[idx]))
            hit = continuous_label_from_payload(payload)
            self._mem[idx] = hit
        return hit

    def targets(self, indices: np.ndarray) -> np.ndarray:
        return np.array([float(self.get(int(i))["total_overlap_norm"]) for i in indices], dtype=np.float32)


def _holdout_arrays(holdout_rows: list[dict], cache: LabelCache):
    labeled = [continuous_label_from_payload(cache.label(params_from_row(r))) for r in holdout_rows]
    y = np.array([float(r["total_overlap_norm"]) for r in labeled], dtype=np.float32)
    pair = np.array(
        [[float(r[f"pair_norm_{name}"]) for name in PAIR_COLUMNS] for r in labeled],
        dtype=np.float32,
    )
    dominant = [str(r["dominant_pair"]) for r in labeled]
    return y, pair, dominant


def _ensemble_predictions(
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
        model, std = train_overlap_regressor(
            train_x,
            train_y,
            threshold_norm=DEFAULT_THRESHOLD_NORM,
            hidden=hidden,
            seed=seed + 1009 * member,
            epochs=epochs,
        )
        preds.append(predict_overlap_norm(model, std, candidate_x, threshold_norm=DEFAULT_THRESHOLD_NORM))
    return np.vstack(preds)


def _select_by_strategy(
    strategy: Strategy,
    candidates: np.ndarray,
    preds: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    pred_log = transformed_target(preds, threshold_norm=DEFAULT_THRESHOLD_NORM)
    mean_pred = preds.mean(axis=0)
    mean_log = pred_log.mean(axis=0)
    raw_var = preds.var(axis=0)
    log_var = pred_log.var(axis=0)

    if strategy == "raw_variance":
        scores = raw_var
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "log_variance":
        scores = log_var
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "near_threshold_log_variance":
        # Highest around roughly 2*tau0, broad enough to include clean/tiny/small
        # predicted overlaps without collapsing to exactly zero.
        center = math.log1p(2.0)
        sigma = 0.9
        weight = np.exp(-0.5 * ((mean_log - center) / sigma) ** 2)
        scores = log_var * weight
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "stratified_log_variance":
        scores = log_var
        n_bins = min(5, max(1, len(candidates)))
        order = np.argsort(mean_log)
        bins = np.array_split(order, n_bins)
        base_quota = k // n_bins
        remainder = k % n_bins
        chosen: list[int] = []
        for bin_id, bin_local in enumerate(bins):
            if len(bin_local) == 0:
                continue
            quota = base_quota + (1 if bin_id < remainder else 0)
            quota = min(quota, len(bin_local))
            if quota > 0:
                chosen.extend(bin_local[np.argsort(-scores[bin_local])[:quota]].tolist())
        if len(chosen) < k:
            used = set(chosen)
            for idx in np.argsort(-scores):
                if int(idx) not in used:
                    chosen.append(int(idx))
                if len(chosen) >= k:
                    break
        chosen_local = np.array(chosen[:k], dtype=int)
    else:  # pragma: no cover
        raise ValueError(f"unknown strategy: {strategy}")

    selected_scores = scores[chosen_local]
    return candidates[chosen_local], {
        "selected_score_mean": float(np.mean(selected_scores)),
        "selected_pred_norm_mean": float(np.mean(mean_pred[chosen_local])),
        "selected_pred_log_mean": float(np.mean(mean_log[chosen_local])),
        "candidate_pred_norm_mean": float(np.mean(mean_pred)),
        "candidate_pred_log_mean": float(np.mean(mean_log)),
    }


def _eval(
    *,
    strategy: str,
    series: str,
    budget: int,
    seed: int,
    selected: np.ndarray,
    pool_x: np.ndarray,
    labels: MemLabels,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    holdout_pair: np.ndarray,
    holdout_dominant: list[str],
    hidden: tuple[int, ...],
    epochs: int,
    eval_seed: int,
) -> dict:
    train_y = labels.targets(selected)
    model, std = train_overlap_regressor(
        pool_x[selected],
        train_y,
        threshold_norm=DEFAULT_THRESHOLD_NORM,
        hidden=hidden,
        seed=eval_seed,
        epochs=epochs,
    )
    pred = predict_overlap_norm(model, std, holdout_x, threshold_norm=DEFAULT_THRESHOLD_NORM)
    metrics = regression_metrics(holdout_y, pred, pair_true=holdout_pair, dominant_pairs=holdout_dominant)
    bins = [magnitude_bin(float(v)) for v in train_y]
    row = {
        "strategy": strategy,
        "series": series,
        "seed": int(seed),
        "budget": int(budget),
        "n_labels": int(len(selected)),
        "train_overlap_positive_rate": float(np.mean(train_y > DEFAULT_THRESHOLD_NORM)),
    }
    for bin_name in ("clean_or_below_threshold", "tiny", "small", "moderate", "large"):
        row[f"train_frac_{bin_name}"] = float(np.mean([b == bin_name for b in bins]))
    row.update(metrics)
    return row


def _run_random_job(args_dict: dict, seed: int) -> list[dict]:
    ctx = _load_context(args_dict)
    rng = np.random.default_rng(seed)
    random_order = rng.choice(np.arange(ctx["pool_x"].shape[0]), size=max(ctx["budgets"]), replace=False)
    rows = []
    for budget in ctx["budgets"]:
        rows.append(
            _eval(
                strategy="random",
                series="random",
                budget=budget,
                seed=seed,
                selected=random_order[:budget],
                pool_x=ctx["pool_x"],
                labels=ctx["labels"],
                holdout_x=ctx["holdout_x"],
                holdout_y=ctx["holdout_y"],
                holdout_pair=ctx["holdout_pair"],
                holdout_dominant=ctx["holdout_dominant"],
                hidden=ctx["hidden"],
                epochs=ctx["epochs"],
                eval_seed=seed * 1000 + budget // 100,
            )
        )
    return rows


def _run_strategy_job(args_dict: dict, seed: int, strategy: Strategy) -> list[dict]:
    ctx = _load_context(args_dict)
    rng = np.random.default_rng(seed)
    n_pool = ctx["pool_x"].shape[0]
    random_order = rng.choice(np.arange(n_pool), size=max(ctx["budgets"]), replace=False)
    rows = []
    acquisition_logs = []

    for base in ctx["bases"]:
        series = f"active_base{base}"
        active = random_order[:base].copy()
        remaining = np.ones(n_pool, dtype=bool)
        remaining[active] = False
        for budget in [b for b in ctx["budgets"] if b >= base]:
            while len(active) < budget:
                k = min(budget - len(active), int(ctx["batch"]))
                candidates = np.flatnonzero(remaining)
                preds = _ensemble_predictions(
                    ctx["pool_x"][active],
                    ctx["labels"].targets(active),
                    ctx["pool_x"][candidates],
                    hidden=ctx["hidden"],
                    seed=seed + len(active),
                    epochs=ctx["ensemble_epochs"],
                    ensemble_size=ctx["ensemble_size"],
                )
                new_idx, score_info = _select_by_strategy(strategy, candidates, preds, k)
                acquisition_logs.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "series": series,
                        "from_labels": int(len(active)),
                        "to_labels": int(len(active) + len(new_idx)),
                        **score_info,
                    }
                )
                active = np.concatenate([active, new_idx])
                remaining[new_idx] = False

            rows.append(
                _eval(
                    strategy=strategy,
                    series=series,
                    budget=budget,
                    seed=seed,
                    selected=active,
                    pool_x=ctx["pool_x"],
                    labels=ctx["labels"],
                    holdout_x=ctx["holdout_x"],
                    holdout_y=ctx["holdout_y"],
                    holdout_pair=ctx["holdout_pair"],
                    holdout_dominant=ctx["holdout_dominant"],
                    hidden=ctx["hidden"],
                    epochs=ctx["epochs"],
                    eval_seed=seed * 1000 + budget // 100,
                )
            )
    for row in rows:
        row["acquisition_events"] = len(acquisition_logs)
    return rows


def _load_context(args_dict: dict) -> dict:
    # Keep each worker from using all cores internally; parallelism is at job level.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    selected_dir = MODELS_DIR / args_dict["selected_dir"]
    arch_json = selected_dir / "architecture.json"
    if arch_json.exists():
        hidden = tuple(int(v) for v in json.loads(arch_json.read_text())["hidden"])
    else:
        hidden = tuple(int(v) for v in args_dict["hidden"].split(","))

    pool_rows = _read_csv(Path(args_dict["pool_csv"]))
    holdout_rows = _read_csv(Path(args_dict["holdout_csv"]))[: int(args_dict["holdout_n"])]
    labels = MemLabels(pool_rows)
    holdout_y, holdout_pair, holdout_dominant = _holdout_arrays(holdout_rows, labels.cache)
    return {
        "pool_x": _features(pool_rows),
        "holdout_x": _features(holdout_rows),
        "labels": labels,
        "holdout_y": holdout_y,
        "holdout_pair": holdout_pair,
        "holdout_dominant": holdout_dominant,
        "hidden": hidden,
        "budgets": [int(b) for b in args_dict["budgets"].split(",") if b.strip()],
        "bases": [int(b) for b in args_dict["bases"].split(",") if b.strip()],
        "batch": int(args_dict["batch"]),
        "epochs": int(args_dict["epochs"]),
        "ensemble_epochs": int(args_dict["ensemble_epochs"]),
        "ensemble_size": int(args_dict["ensemble_size"]),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    keys = sorted({(r["strategy"], r["series"], int(r["budget"])) for r in rows})
    for strategy, series, budget in keys:
        matches = [r for r in rows if r["strategy"] == strategy and r["series"] == series and int(r["budget"]) == budget]
        entry = {"strategy": strategy, "series": series, "budget": budget, "n_seeds": len(matches)}
        metric_keys = sorted({k for r in matches for k in r if isinstance(r.get(k), (int, float))})
        for key in metric_keys:
            if key in {"seed", "budget", "n_labels"}:
                continue
            vals = [float(r[key]) for r in matches if r.get(key) not in ("", None)]
            if vals:
                entry[f"{key}_mean"] = float(np.mean(vals))
                entry[f"{key}_std"] = float(np.std(vals))
        out.append(entry)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_strategy(agg: list[dict], strategy: Strategy, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    rows_by_series: dict[str, list[dict]] = {}
    for series in SERIES_STYLE:
        if series == "random":
            rows = [r for r in agg if r["strategy"] == "random" and r["series"] == "random"]
        else:
            rows = [r for r in agg if r["strategy"] == strategy and r["series"] == series]
        rows_by_series[series] = sorted(rows, key=lambda r: int(r["budget"]))

    for series, rows in rows_by_series.items():
        if not rows:
            continue
        label, color = SERIES_STYLE[series]
        xs = [int(r["budget"]) for r in rows]
        means = np.array([float(r["mae_log_mean"]) for r in rows])
        stds = np.array([float(r["mae_log_std"]) for r in rows])
        ax.plot(xs, means, marker="o", ms=4, color=color, label=label)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_title(STRATEGY_LABELS[strategy])
    ax.set_xlabel("Total training labels")
    ax.set_ylabel("Holdout log-scaled overlap error (MAE$_{\\log}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(FIGURES_DIR / out_png.name, dpi=180)
    plt.close(fig)


def _plot_combined(agg: list[dict], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True, sharey=True)
    for ax, strategy in zip(axes.flat, STRATEGIES):
        for series in SERIES_STYLE:
            if series == "random":
                rows = [r for r in agg if r["strategy"] == "random" and r["series"] == "random"]
            else:
                rows = [r for r in agg if r["strategy"] == strategy and r["series"] == series]
            rows = sorted(rows, key=lambda r: int(r["budget"]))
            if not rows:
                continue
            label, color = SERIES_STYLE[series]
            xs = [int(r["budget"]) for r in rows]
            means = np.array([float(r["mae_log_mean"]) for r in rows])
            stds = np.array([float(r["mae_log_std"]) for r in rows])
            ax.plot(xs, means, marker="o", ms=3, color=color, label=label)
            ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)
        ax.set_title(STRATEGY_LABELS[strategy], fontsize=10)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Total training labels")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAE$_{\\log}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(FIGURES_DIR / out_png.name, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    out_dir = RUNS_DIR / "active_learning" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    bases = [int(b) for b in args.bases.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    args_dict = {
        "selected_dir": args.selected_dir,
        "hidden": args.hidden,
        "pool_csv": str(args.pool_csv),
        "holdout_csv": str(args.holdout_csv),
        "holdout_n": args.holdout_n,
        "budgets": args.budgets,
        "bases": args.bases,
        "batch": args.batch,
        "epochs": args.epochs,
        "ensemble_epochs": args.ensemble_epochs,
        "ensemble_size": args.ensemble_size,
    }

    jobs = [Job("random", seed, None) for seed in seeds]
    jobs.extend(Job("strategy", seed, strategy) for strategy in STRATEGIES for seed in seeds)

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    print(f"running {len(jobs)} jobs with workers={args.workers}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for job in jobs:
            if job.kind == "random":
                fut = ex.submit(_run_random_job, args_dict, job.seed)
            else:
                assert job.strategy is not None
                fut = ex.submit(_run_strategy_job, args_dict, job.seed, job.strategy)
            futures[fut] = job
        for fut in as_completed(futures):
            job = futures[fut]
            rows = fut.result()
            all_rows.extend(rows)
            label = job.strategy or "random"
            print(f"done {job.kind}:{label}:seed{job.seed} rows={len(rows)} elapsed={time.perf_counter() - t0:.0f}s")

    _write_csv(out_dir / "al_strategy_probe_per_seed.csv", all_rows)
    agg = _aggregate(all_rows)
    _write_csv(out_dir / "al_strategy_probe_agg.csv", agg)
    _write_csv(TABLES_DIR / "task36_al_strategy_probe_agg.csv", agg)

    plot_paths = []
    for strategy in STRATEGIES:
        path = out_dir / f"task36_al_strategy_probe_{strategy}.png"
        _plot_strategy(agg, strategy, path)
        plot_paths.append(str(path))
    combined = out_dir / "task36_al_strategy_probe_combined.png"
    _plot_combined(agg, combined)

    final_budget = max(budgets)
    random_final = [
        r for r in agg if r["strategy"] == "random" and r["series"] == "random" and int(r["budget"]) == final_budget
    ][0]
    final_rows = [r for r in agg if int(r["budget"]) == final_budget and r["series"] != "random"]
    best_active = min(final_rows, key=lambda r: float(r["mae_log_mean"]))
    summary = {
        "run_id": args.run_id,
        "seeds": seeds,
        "budgets": budgets,
        "bases": bases,
        "strategies": list(STRATEGIES),
        "headline_metric": "mae_log",
        "random_final_mae_log_mean": random_final["mae_log_mean"],
        "random_final_mae_log_std": random_final["mae_log_std"],
        "best_active_final": best_active,
        "active_beats_random_at_final": float(best_active["mae_log_mean"]) < float(random_final["mae_log_mean"]),
        "elapsed_s": time.perf_counter() - t0,
        "plots": plot_paths,
        "combined_plot": str(combined),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    print(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="task36_al_strategy_probe_v1")
    parser.add_argument("--pool-csv", type=Path, default=POOL_100K_CSV)
    parser.add_argument("--holdout-csv", type=Path, default=HOLDOUT_5K_CSV)
    parser.add_argument("--holdout-n", type=int, default=5000)
    parser.add_argument("--selected-dir", default="overlap_regressor_regression_v3_selected")
    parser.add_argument("--hidden", default="128,64,32")
    parser.add_argument("--seeds", default="360101,360202,360303")
    parser.add_argument("--budgets", default="2000,3000,5000,7500,10000,12500,15000")
    parser.add_argument("--bases", default="3000,5000,10000")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--ensemble-epochs", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
