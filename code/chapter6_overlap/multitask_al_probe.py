"""Chapter 6 #5e — binary-head active-learning probe for the multitask overlap model.

Clone of ``task36_al_strategy_probe`` that swaps the single-head regressor for the
multitask model (volume + binary heads) and drives acquisition from the binary head's
ensemble uncertainty rather than from volume variance. The volume head still supplies
the headline ``MAE_log`` on the holdout; the binary head supplies a magnitude-invariant
``P(overlap)`` so the acquisition score does not scale with overlap volume (the failure
mode of raw volume variance).

Strategies (3-member ensemble of multitask nets for acquisition, single net for eval):

- ``binary_variance``: ensemble variance of ``sigmoid(P(overlap))`` (epistemic
  disagreement; magnitude-invariant) — primary.
- ``binary_entropy``: predictive entropy of the ensemble-MEAN ``P(overlap)`` (total
  predictive uncertainty; peaks where the mean is ``p ~= 0.5``).
- ``binary_uncertainty``: mean over ensemble members of each member's OWN binary entropy
  (aleatoric / "the models are individually closest to ``p ~= 0.5``"). Differs from
  ``binary_entropy`` wherever the members disagree: it rewards consensus-boundary points
  and down-weights split-confident ones. Added Chapter 6 (2026-06-25), replacing
  ``binary_variance_stratified`` in the reported thesis trio.
- ``binary_variance_stratified``: binary variance with even draws across predicted-``p``
  bins (retired from the thesis trio; branch retained for repo history only).

Plus a passive ``random`` baseline. The near-threshold log-variance acquisition of
Chapter 6 is intentionally NOT run here; it is removed from the thesis.

Re-running one strategy: pass ``--recompute binary_uncertainty --reuse-per-seed
<v1 per_seed csv>`` to compute only that strategy and splice the untouched ``random`` /
``binary_variance`` / ``binary_entropy`` rows back from a previous run unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from .multitask_overlap_model import (  # noqa: E402
    binary_metrics,
    predict_multitask,
    predict_p_overlap,
    train_multitask_overlap_model,
)
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
)

Strategy = Literal[
    "binary_variance",
    "binary_entropy",
    "binary_variance_stratified",
    "binary_uncertainty",
]

# Reported thesis trio (Chapter 6, 2026-06-25): binary_variance_stratified retired in
# favour of binary_uncertainty (mean per-member proximity to p=0.5). The stratified
# branch stays in _select_by_strategy for repo history but is no longer in this tuple,
# so it is neither recomputed nor plotted.
STRATEGIES: tuple[Strategy, ...] = (
    "binary_variance",
    "binary_entropy",
    "binary_uncertainty",
)

STRATEGY_LABELS = {
    "binary_variance": "binary variance",
    "binary_entropy": "binary entropy",
    "binary_variance_stratified": "binary variance (stratified)",
    "binary_uncertainty": "binary uncertainty",
}

SERIES_STYLE = {
    "random": ("random", "#e45756"),
    "active_base3000": ("active (base 3000)", "#4c78a8"),
    "active_base5000": ("active (base 5000)", "#54a24b"),
    "active_base10000": ("active (base 10000)", "#b279a2"),
}

_EPS = 1e-7


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


def _ensemble_p_overlap(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_x: np.ndarray,
    *,
    hidden: tuple[int, ...],
    seed: int,
    epochs: int,
    ensemble_size: int,
    lambda_bin: float,
) -> np.ndarray:
    """Return an ``(ensemble_size, n_candidates)`` matrix of binary-head probabilities."""
    preds = []
    for member in range(ensemble_size):
        model, std = train_multitask_overlap_model(
            train_x,
            train_y,
            threshold_norm=DEFAULT_THRESHOLD_NORM,
            hidden=hidden,
            seed=seed + 1009 * member,
            epochs=epochs,
            lambda_bin=lambda_bin,
            bin_threshold=DEFAULT_THRESHOLD_NORM,
        )
        preds.append(predict_p_overlap(model, std, candidate_x))
    return np.vstack(preds)


def _select_by_strategy(
    strategy: Strategy,
    candidates: np.ndarray,
    probs: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    mean_p = probs.mean(axis=0)
    var_p = probs.var(axis=0)
    clipped = np.clip(mean_p, _EPS, 1.0 - _EPS)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    # Per-member (aleatoric) uncertainty: mean over ensemble members of each member's own
    # binary entropy. Maximised when the individual models sit at p~=0.5; differs from
    # `entropy` (entropy of the mean) wherever the members disagree.
    clipped_members = np.clip(probs, _EPS, 1.0 - _EPS)
    member_entropy = -(clipped_members * np.log(clipped_members) + (1.0 - clipped_members) * np.log(1.0 - clipped_members))
    mean_member_entropy = member_entropy.mean(axis=0)

    if strategy == "binary_variance":
        scores = var_p
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "binary_entropy":
        scores = entropy
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "binary_uncertainty":
        scores = mean_member_entropy
        chosen_local = np.argsort(-scores)[:k]
    elif strategy == "binary_variance_stratified":
        scores = var_p
        n_bins = min(5, max(1, len(candidates)))
        order = np.argsort(mean_p)
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
        "selected_p_mean": float(np.mean(mean_p[chosen_local])),
        "selected_var_mean": float(np.mean(var_p[chosen_local])),
        "selected_entropy_mean": float(np.mean(entropy[chosen_local])),
        "candidate_p_mean": float(np.mean(mean_p)),
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
    holdout_y_bin: np.ndarray,
    holdout_pair: np.ndarray,
    holdout_dominant: list[str],
    hidden: tuple[int, ...],
    epochs: int,
    lambda_bin: float,
    eval_seed: int,
) -> dict:
    train_y = labels.targets(selected)
    model, std = train_multitask_overlap_model(
        pool_x[selected],
        train_y,
        threshold_norm=DEFAULT_THRESHOLD_NORM,
        hidden=hidden,
        seed=eval_seed,
        epochs=epochs,
        lambda_bin=lambda_bin,
        bin_threshold=DEFAULT_THRESHOLD_NORM,
    )
    pred_vol, pred_p = predict_multitask(model, std, holdout_x, threshold_norm=DEFAULT_THRESHOLD_NORM)
    metrics = regression_metrics(holdout_y, pred_vol, pair_true=holdout_pair, dominant_pairs=holdout_dominant)
    bmetrics = binary_metrics(holdout_y_bin, pred_p)
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
    row.update(bmetrics)
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
                holdout_y_bin=ctx["holdout_y_bin"],
                holdout_pair=ctx["holdout_pair"],
                holdout_dominant=ctx["holdout_dominant"],
                hidden=ctx["hidden"],
                epochs=ctx["epochs"],
                lambda_bin=ctx["lambda_bin"],
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
                probs = _ensemble_p_overlap(
                    ctx["pool_x"][active],
                    ctx["labels"].targets(active),
                    ctx["pool_x"][candidates],
                    hidden=ctx["hidden"],
                    seed=seed + len(active),
                    epochs=ctx["ensemble_epochs"],
                    ensemble_size=ctx["ensemble_size"],
                    lambda_bin=ctx["lambda_bin"],
                )
                new_idx, score_info = _select_by_strategy(strategy, candidates, probs, k)
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
                    holdout_y_bin=ctx["holdout_y_bin"],
                    holdout_pair=ctx["holdout_pair"],
                    holdout_dominant=ctx["holdout_dominant"],
                    hidden=ctx["hidden"],
                    epochs=ctx["epochs"],
                    lambda_bin=ctx["lambda_bin"],
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
    lambda_bin = float(args_dict["lambda_bin"])
    if arch_json.exists():
        arch = json.loads(arch_json.read_text())
        hidden = tuple(int(v) for v in arch["hidden"])
        if "lambda_bin" in arch:
            lambda_bin = float(arch["lambda_bin"])
    else:
        hidden = tuple(int(v) for v in args_dict["hidden"].split(","))

    pool_rows = _read_csv(Path(args_dict["pool_csv"]))
    holdout_rows = _read_csv(Path(args_dict["holdout_csv"]))[: int(args_dict["holdout_n"])]
    labels = MemLabels(pool_rows)
    holdout_y, holdout_pair, holdout_dominant = _holdout_arrays(holdout_rows, labels.cache)
    holdout_y_bin = (holdout_y > DEFAULT_THRESHOLD_NORM).astype(int)
    return {
        "pool_x": _features(pool_rows),
        "holdout_x": _features(holdout_rows),
        "labels": labels,
        "holdout_y": holdout_y,
        "holdout_y_bin": holdout_y_bin,
        "holdout_pair": holdout_pair,
        "holdout_dominant": holdout_dominant,
        "hidden": hidden,
        "lambda_bin": lambda_bin,
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
            vals = [float(r[key]) for r in matches if r.get(key) not in ("", None) and np.isfinite(float(r[key]))]
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


def _read_per_seed(path: Path) -> list[dict]:
    """Load a per-seed CSV back, coercing numeric columns to float so ``_aggregate`` (which
    keys metric columns off ``isinstance(.., (int, float))``) treats them as numbers. Keeps
    ``strategy``/``series`` as strings and drops empty cells."""
    out: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for key, val in raw.items():
                if key in ("strategy", "series"):
                    row[key] = val
                elif val in ("", None):
                    continue
                else:
                    try:
                        row[key] = float(val)
                    except ValueError:
                        row[key] = val
            out.append(row)
    return out


def _plot_strategy(agg: list[dict], strategy: Strategy, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
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
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(13.5, 4.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
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
        ax.set_xlabel("Total training labels")
        ax.grid(True, alpha=0.25)
    axes.flat[0].set_ylabel("MAE$_{\\log}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
        "lambda_bin": args.lambda_bin,
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

    recompute = [s.strip() for s in (getattr(args, "recompute", "") or "").split(",") if s.strip()]
    if not recompute:
        recompute = ["random", *STRATEGIES]

    jobs = [Job("random", seed, None) for seed in seeds if "random" in recompute]
    jobs.extend(
        Job("strategy", seed, strategy)
        for strategy in STRATEGIES
        if strategy in recompute
        for seed in seeds
    )

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    print(f"running {len(jobs)} jobs with workers={args.workers} (recompute={recompute})")
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

    reuse_path = getattr(args, "reuse_per_seed", "") or ""
    if reuse_path:
        keep = ({"random"} | set(STRATEGIES)) - set(recompute)
        reused = [r for r in _read_per_seed(Path(reuse_path)) if r.get("strategy") in keep]
        print(f"reusing {len(reused)} per-seed rows from {reuse_path} for strategies {sorted(keep)}")
        all_rows.extend(reused)

    _write_csv(out_dir / "multitask_al_probe_per_seed.csv", all_rows)
    agg = _aggregate(all_rows)
    _write_csv(out_dir / "multitask_al_probe_agg.csv", agg)
    _write_csv(TABLES_DIR / "multitask_al_probe_agg.csv", agg)

    plot_paths = []
    for strategy in STRATEGIES:
        path = out_dir / f"multitask_al_probe_{strategy}.png"
        _plot_strategy(agg, strategy, path)
        plot_paths.append(str(path))
    combined = out_dir / "multitask_al_probe_combined.png"
    _plot_combined(agg, combined)

    final_budget = max(budgets)
    random_final = [
        r for r in agg if r["strategy"] == "random" and r["series"] == "random" and int(r["budget"]) == final_budget
    ][0]
    final_rows = [r for r in agg if int(r["budget"]) == final_budget and r["series"] != "random"]
    best_active = min(final_rows, key=lambda r: float(r["mae_log_mean"]))
    summary = {
        "run_id": args.run_id,
        "model": "multitask_overlap (volume + binary heads)",
        "acquisition": "binary head ensemble uncertainty",
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
    parser.add_argument("--run-id", default="multitask_al_probe_v1")
    parser.add_argument("--pool-csv", type=Path, default=POOL_100K_CSV)
    parser.add_argument("--holdout-csv", type=Path, default=HOLDOUT_5K_CSV)
    parser.add_argument("--holdout-n", type=int, default=5000)
    parser.add_argument("--selected-dir", default="overlap_regressor_multitask_v1_selected")
    parser.add_argument("--hidden", default="256,128,64,32")
    parser.add_argument("--lambda-bin", type=float, default=1.0)
    parser.add_argument("--seeds", default="360101,360202,360303")
    parser.add_argument("--budgets", default="2000,3000,5000,7500,10000,12500,15000")
    parser.add_argument("--bases", default="3000,5000,10000")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--ensemble-epochs", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--recompute",
        default="",
        help="comma list of strategies (and/or 'random') to actually compute; empty = all",
    )
    parser.add_argument(
        "--reuse-per-seed",
        default="",
        help="existing per-seed CSV; rows for kept strategies not in --recompute are spliced in unchanged",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
