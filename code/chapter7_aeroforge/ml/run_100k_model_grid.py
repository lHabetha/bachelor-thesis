#!/usr/bin/env python3
"""Resumable 100k-label MLP comparison grid (4 variants, 10-fold CV, 3 seeds)."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


from chapter7_aeroforge.ml.features import (  # noqa: E402
    ENGINEERED_FEATURE_NAMES,
    load_labeled_dataset,
    standardize_train_test,
)
from chapter7_aeroforge.ml.models import (  # noqa: E402
    DEFAULT_GATE_LOGIT_SCALE,
    DEFAULT_TAU_MM3,
    GRID_VARIANTS,
    MLP_ARCHITECTURES,
    binary_head_metrics,
    predict_mlp,
    predict_mlp_multitask,
    regression_metrics,
    train_mlp,
    train_mlp_gate_augmented,
    train_mlp_multitask_gate,
)

from chapter7_aeroforge.release_paths import LABELS_CLOUD, MODEL_GRID_DIR

DEFAULT_LABELS_PATH = LABELS_CLOUD
DEFAULT_OUT_DIR = MODEL_GRID_DIR
DEFAULT_ARCH_ID = "mlp_256_128_64_32"
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_N_FOLDS = 10
DEFAULT_BUDGETS = (
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    60000,
    70000,
    80000,
    90000,
    100000,
)


@dataclass(frozen=True)
class JobSpec:
    variant_id: str
    arch_id: str
    hidden: tuple[int, ...]
    budget: int
    seed: int
    fold: int
    n_folds: int
    epochs: int
    labels_path: str
    engineered: bool
    loss_variant: str
    multitask: bool
    gate_lambda: float
    gate_logit_scale: float

    def job_key(self) -> str:
        return "|".join(
            [
                self.variant_id,
                str(self.budget),
                str(self.seed),
                str(self.fold),
                self.labels_path,
                str(self.epochs),
                self.loss_variant,
                str(self.gate_lambda),
                str(self.gate_logit_scale),
            ]
        )


def _mp():
    import multiprocessing as mp

    return mp.get_context("spawn")


def _pin_worker_threads() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


def _worker_init() -> None:
    _pin_worker_threads()


def _fold_split(n: int, seed: int, fold: int, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.arange(n, dtype=np.int64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(order))
    train_pos, test_pos = folds[fold]
    return order[train_pos], order[test_pos]


def _run_job(job: JobSpec) -> dict:
    t0 = time.perf_counter()

    dataset = load_labeled_dataset(Path(job.labels_path), engineered=job.engineered)
    n = min(job.budget, len(dataset.x_raw))
    if n < job.n_folds * 2:
        raise RuntimeError(f"Budget {job.budget} too small for {job.n_folds}-fold CV")

    train_idx, test_idx = _fold_split(n, job.seed, job.fold, job.n_folds)

    x_train_raw = dataset.x_raw[train_idx]
    x_test_raw = dataset.x_raw[test_idx]
    y_train = dataset.y_mm3[train_idx]
    y_test = dataset.y_mm3[test_idx]

    x_train, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)
    train_seed = job.seed * 1000 + job.fold

    extra_metrics: dict[str, float] = {}
    if job.multitask:
        model, std = train_mlp_multitask_gate(
            x_train,
            y_train,
            hidden=job.hidden,
            seed=train_seed,
            epochs=job.epochs,
            tau_mm3=DEFAULT_TAU_MM3,
            gate_lambda=job.gate_lambda,
        )
        y_pred, prob = predict_mlp_multitask(model, std, x_test, tau_mm3=DEFAULT_TAU_MM3)
        extra_metrics = binary_head_metrics(y_test, prob, tau_mm3=DEFAULT_TAU_MM3)
    elif job.loss_variant == "gate_augmented":
        model, std = train_mlp_gate_augmented(
            x_train,
            y_train,
            hidden=job.hidden,
            seed=train_seed,
            epochs=job.epochs,
            tau_mm3=DEFAULT_TAU_MM3,
            gate_lambda=job.gate_lambda,
            gate_logit_scale=job.gate_logit_scale,
        )
        y_pred = predict_mlp(model, std, x_test, tau_mm3=DEFAULT_TAU_MM3)
    else:
        model, std = train_mlp(
            x_train,
            y_train,
            hidden=job.hidden,
            seed=train_seed,
            epochs=job.epochs,
            tau_mm3=DEFAULT_TAU_MM3,
        )
        y_pred = predict_mlp(model, std, x_test, tau_mm3=DEFAULT_TAU_MM3)

    metrics = regression_metrics(y_test, y_pred, tau_mm3=DEFAULT_TAU_MM3)
    wall_s = time.perf_counter() - t0

    row = {
        "job_key": job.job_key(),
        "variant_id": job.variant_id,
        "arch_id": job.arch_id,
        "family": "mlp",
        "config": json.dumps(list(job.hidden)),
        "budget": job.budget,
        "n_total": n,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seed": job.seed,
        "fold": job.fold,
        "n_folds": job.n_folds,
        "engineered": job.engineered,
        "loss_variant": job.loss_variant,
        "multitask": job.multitask,
        "gate_lambda": job.gate_lambda,
        "gate_logit_scale": job.gate_logit_scale,
        "n_features": int(dataset.x_raw.shape[1]),
        "epochs": job.epochs,
        "wall_s": wall_s,
        **{k: metrics[k] for k in metrics},
        **extra_metrics,
    }
    return row


def _load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                keys.add(str(row["job_key"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _load_all_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_results(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant_id"]), int(row["budget"]))].append(row)

    agg_rows: list[dict] = []
    for (variant_id, budget), group in sorted(groups.items()):
        mae_logs = np.array([float(r["mae_log"]) for r in group], dtype=float)
        factors = np.array([float(r["typical_factor"]) for r in group], dtype=float)
        gate_f1 = np.array([float(r["gate_f1"]) for r in group], dtype=float)
        spearman = np.array([float(r["spearman_mm3"]) for r in group], dtype=float)
        cond_mae = np.array([float(r["cond_mae_log"]) for r in group], dtype=float)
        cond_factor = np.power(10.0, cond_mae)

        agg: dict = {
            "variant_id": variant_id,
            "arch_id": group[0]["arch_id"],
            "budget": budget,
            "n_runs": len(group),
            "engineered": group[0]["engineered"],
            "loss_variant": group[0]["loss_variant"],
            "multitask": group[0]["multitask"],
            "mae_log_mean": float(mae_logs.mean()),
            "mae_log_std": float(mae_logs.std(ddof=1) if len(group) > 1 else 0.0),
            "typical_factor_mean": float(factors.mean()),
            "typical_factor_std": float(factors.std(ddof=1) if len(group) > 1 else 0.0),
            "gate_f1_mean": float(gate_f1.mean()),
            "gate_f1_std": float(gate_f1.std(ddof=1) if len(group) > 1 else 0.0),
            "spearman_mm3_mean": float(spearman.mean()),
            "spearman_mm3_std": float(spearman.std(ddof=1) if len(group) > 1 else 0.0),
            "cond_mae_log_mean": float(cond_mae.mean()),
            "cond_mae_log_std": float(cond_mae.std(ddof=1) if len(group) > 1 else 0.0),
            "cond_typical_factor_mean": float(cond_factor.mean()),
            "cond_typical_factor_std": float(cond_factor.std(ddof=1) if len(group) > 1 else 0.0),
        }
        if group[0].get("multitask"):
            bin_f1 = np.array([float(r.get("gate_f1_binary", 0.0)) for r in group], dtype=float)
            agg["gate_f1_binary_mean"] = float(bin_f1.mean())
            agg["gate_f1_binary_std"] = float(bin_f1.std(ddof=1) if len(group) > 1 else 0.0)
        agg_rows.append(agg)
    return agg_rows


def _write_progress(out_dir: Path, *, total: int, completed: int, failed: int, pending: int) -> None:
    payload = {
        "total_jobs": total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "progress.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_jobs(
    *,
    variants: list[str],
    budgets: list[int],
    seeds: list[int],
    folds: list[int],
    n_folds: int,
    epochs: int,
    labels_path: Path,
    arch_id: str,
) -> list[JobSpec]:
    hidden = MLP_ARCHITECTURES[arch_id]
    jobs: list[JobSpec] = []
    for variant_id in variants:
        spec = GRID_VARIANTS[variant_id]
        for budget in budgets:
            for seed in seeds:
                for fold in folds:
                    jobs.append(
                        JobSpec(
                            variant_id=variant_id,
                            arch_id=arch_id,
                            hidden=hidden,
                            budget=budget,
                            seed=seed,
                            fold=fold,
                            n_folds=n_folds,
                            epochs=epochs,
                            labels_path=str(labels_path.resolve()),
                            engineered=bool(spec["engineered"]),
                            loss_variant=str(spec["loss_variant"]),
                            multitask=bool(spec.get("multitask", False)),
                            gate_lambda=float(spec.get("gate_lambda", 0.0)),
                            gate_logit_scale=float(
                                spec.get("gate_logit_scale", DEFAULT_GATE_LOGIT_SCALE)
                            ),
                        )
                    )
    return jobs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="100k MLP comparison grid (resumable)")
    ap.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--arch", type=str, default=DEFAULT_ARCH_ID)
    ap.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(GRID_VARIANTS.keys()),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: budget=5000, seed=42, fold=0, all 4 variants.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.arch not in MLP_ARCHITECTURES:
        raise SystemExit(f"Unknown MLP arch: {args.arch}")
    for vid in args.variants:
        if vid not in GRID_VARIANTS:
            raise SystemExit(f"Unknown variant: {vid}")

    seeds = list(args.seeds)
    budgets = list(args.budgets)
    if args.smoke:
        budgets = [5000]
        seeds = [seeds[0]]
        folds = [0]
    else:
        folds = list(range(args.n_folds))

    jobs = _make_jobs(
        variants=list(args.variants),
        budgets=budgets,
        seeds=seeds,
        folds=folds,
        n_folds=args.n_folds,
        epochs=args.epochs,
        labels_path=args.labels_path,
        arch_id=args.arch,
    )

    results_jsonl = args.out_dir / ("results_smoke.jsonl" if args.smoke else "results.jsonl")
    failures_jsonl = args.out_dir / ("failures_smoke.jsonl" if args.smoke else "failures.jsonl")
    results_csv = args.out_dir / ("results_smoke.csv" if args.smoke else "results.csv")
    agg_csv = args.out_dir / ("results_agg_smoke.csv" if args.smoke else "results_agg.csv")

    done_keys = _load_completed_keys(results_jsonl)
    pending = [j for j in jobs if j.job_key() not in done_keys]

    print(f"100k model grid: {len(jobs)} total, {len(done_keys)} done, {len(pending)} pending")
    print(f"Labels: {args.labels_path}")
    print(f"Variants: {args.variants}")
    print(f"Architecture: {args.arch}")
    print(f"Budgets: {budgets}")
    print(f"Seeds: {seeds}, folds: {args.n_folds}")
    print(f"Output: {args.out_dir}")

    completed = len(done_keys)
    failed = 0
    t0 = time.perf_counter()

    _write_progress(
        args.out_dir,
        total=len(jobs),
        completed=completed,
        failed=failed,
        pending=len(pending),
    )

    if not pending:
        print("All jobs already complete.")
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            mp_context=_mp(),
        ) as pool:
            futures = {pool.submit(_run_job, job): job for job in pending}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    row = fut.result()
                    _append_jsonl(results_jsonl, row)
                    completed += 1
                    all_rows = _load_all_jsonl(results_jsonl)
                    _write_csv(all_rows, results_csv)
                    agg_rows = _aggregate_results(all_rows)
                    _write_csv(agg_rows, agg_csv)
                    _write_progress(
                        args.out_dir,
                        total=len(jobs),
                        completed=completed,
                        failed=failed,
                        pending=len(jobs) - completed - failed,
                    )
                    print(
                        f"  done {job.variant_id} budget={job.budget:6d} "
                        f"seed={job.seed} fold={job.fold} "
                        f"mae_log={row['mae_log']:.4f} wall={row['wall_s']:.1f}s "
                        f"[{completed}/{len(jobs)}]"
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    fail_row = {"job_key": job.job_key(), "error": str(exc), "job": job.__dict__}
                    _append_jsonl(failures_jsonl, fail_row)
                    _write_progress(
                        args.out_dir,
                        total=len(jobs),
                        completed=completed,
                        failed=failed,
                        pending=len(jobs) - completed - failed,
                    )
                    print(
                        f"  FAIL {job.variant_id} budget={job.budget} "
                        f"seed={job.seed} fold={job.fold}: {exc}"
                    )

    all_rows = _load_all_jsonl(results_jsonl)
    all_rows.sort(
        key=lambda r: (
            str(r["variant_id"]),
            int(r["budget"]),
            int(r["seed"]),
            int(r["fold"]),
        )
    )
    _write_csv(all_rows, results_csv)
    agg_rows = _aggregate_results(all_rows)
    _write_csv(agg_rows, agg_csv)

    meta = {
        "n_jobs_total": len(jobs),
        "n_completed": len(all_rows),
        "n_failed": failed,
        "workers": args.workers,
        "seeds": seeds,
        "n_folds": args.n_folds,
        "budgets": budgets,
        "variants": list(args.variants),
        "architecture": args.arch,
        "engineered_features": list(ENGINEERED_FEATURE_NAMES),
        "gate_lambda_strong": 0.3,
        "epochs": args.epochs,
        "labels_path": str(args.labels_path.resolve()),
        "learning": "random_passive",
        "wall_s_total": time.perf_counter() - t0,
        "results_jsonl": str(results_jsonl.resolve()),
        "results_csv": str(results_csv.resolve()),
        "results_agg_csv": str(agg_csv.resolve()),
    }
    meta_name = "run_meta_smoke.json" if args.smoke else "run_meta.json"
    (args.out_dir / meta_name).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    _write_progress(
        args.out_dir,
        total=len(jobs),
        completed=len(all_rows),
        failed=failed,
        pending=max(0, len(jobs) - len(all_rows) - failed),
    )

    print(f"\nSaved {len(all_rows)} rows to {results_csv}")
    print(f"Saved {len(agg_rows)} aggregated rows to {agg_csv}")
    print(f"Session wall: {meta['wall_s_total']:.1f}s")


if __name__ == "__main__":
    main()
