"""Parallel random baseline for Chapter 4 dense-pool (50k) rerun.

For each total-label budget T in {250, 500, ..., 3500}, draws `replicates`
random subsets from the 50k pool, trains a model, and evaluates on holdout.

Uses subprocess-based parallelism to avoid MPS fork crashes on macOS.

Usage:
    python run_baseline_random.py --replicates 5 --workers 12
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from lib.io import atomic_json_write

T_MAX = 3500
STEP_SIZE = 250
T_VALUES = list(range(STEP_SIZE, T_MAX + 1, STEP_SIZE))

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import (  # noqa: E402
    HOLDOUT_PATH,
    baseline_dir,
    pool_parquet_path,
)

_DEFAULT_DENSE_DATA_ID = "dense50k_v1"
_DEFAULT_BASELINE_ID = "dense50k_v1"


def _run_one_baseline(args: tuple[int, int, str]) -> dict[str, Any]:
    """Worker: train a random baseline at a given T and replicate."""
    t_budget, replicate, pool_path_str = args

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from lib.formula_oracle import FEATURE_NAMES
    from lib.io import read_parquet, set_single_thread
    from lib.model_mlp64 import eval_holdout, fit_mlp

    set_single_thread()

    try:
        pool_records = read_parquet(Path(pool_path_str))
        holdout_records = read_parquet(HOLDOUT_PATH)

        holdout_X = np.array(
            [[float(r[f]) for f in FEATURE_NAMES] for r in holdout_records], dtype=np.float64
        )
        holdout_y = np.array([int(r["label"]) for r in holdout_records], dtype=np.int64)

        rng = np.random.default_rng(77000 + t_budget * 100 + replicate)
        idx = rng.choice(len(pool_records), size=min(t_budget, len(pool_records)), replace=False)
        subset = [pool_records[i] for i in idx]

        train_X = np.array(
            [[float(r[f]) for f in FEATURE_NAMES] for r in subset], dtype=np.float64
        )
        train_y = np.array([int(r["label"]) for r in subset], dtype=np.int64)

        eval_seed = 77100 + t_budget + replicate * 7
        fitted = fit_mlp(train_X, train_y, seed=eval_seed)
        prob = fitted.predict_proba(holdout_X)
        metrics = eval_holdout(holdout_y, prob)
        metrics.update({
            "row": "baseline_random",
            "total_labels": t_budget,
            "replicate": replicate,
            "status": "ok",
        })
        return metrics
    except Exception as exc:
        return {
            "row": "baseline_random",
            "total_labels": t_budget,
            "replicate": replicate,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dense-data-id", default=_DEFAULT_DENSE_DATA_ID)
    parser.add_argument("--baseline-id", default=_DEFAULT_BASELINE_ID)
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially (safe on macOS if fork crashes)")
    args = parser.parse_args()

    pool_path = pool_parquet_path(args.dense_data_id)
    baseline_dir_path = baseline_dir(args.baseline_id)
    baseline_dir_path.mkdir(parents=True, exist_ok=True)

    all_jobs: list[tuple[int, int, str]] = []
    for t in T_VALUES:
        for rep in range(1, args.replicates + 1):
            out_path = baseline_dir_path / f"T{t:04d}_rep{rep}.json"
            if out_path.exists():
                continue
            all_jobs.append((t, rep, str(pool_path)))

    print(f"[baseline] {len(all_jobs)} baseline jobs to run", flush=True)

    if not all_jobs:
        print("[baseline] All baseline runs already exist.", flush=True)
        return

    t0 = time.perf_counter()
    completed = 0
    errors = 0

    if args.sequential or args.workers <= 1:
        for job in all_jobs:
            result = _run_one_baseline(job)
            out_path = baseline_dir_path / f"T{job[0]:04d}_rep{job[1]}.json"
            atomic_json_write(out_path, result)
            if result.get("status") == "ok":
                completed += 1
            else:
                errors += 1
                print(f"[baseline] ERROR T={job[0]} rep={job[1]}: "
                      f"{result.get('error', '')[:150]}", flush=True)
            if (completed + errors) % 10 == 0:
                print(f"[baseline] Progress: {completed + errors}/{len(all_jobs)}", flush=True)
    else:
        mp_ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as executor:
            futures = {executor.submit(_run_one_baseline, job): job for job in all_jobs}

            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    out_path = baseline_dir_path / f"T{job[0]:04d}_rep{job[1]}.json"
                    atomic_json_write(out_path, result)
                    if result.get("status") == "ok":
                        completed += 1
                    else:
                        errors += 1
                        print(f"[baseline] ERROR T={job[0]} rep={job[1]}: "
                              f"{result.get('error', '')[:150]}", flush=True)
                except Exception as exc:
                    errors += 1
                    print(f"[baseline] EXCEPTION T={job[0]} rep={job[1]}: {exc}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[baseline] DONE. {completed} ok, {errors} errors in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
