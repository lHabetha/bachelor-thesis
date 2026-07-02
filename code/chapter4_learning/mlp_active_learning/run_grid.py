"""Parallel grid scheduler for Chapter 4 dense-pool label-blind trajectories.

Enumerates all (row, B, R) jobs and executes them in parallel using
ProcessPoolExecutor. Supports resume by checking checkpoint state.

Usage:
    python run_grid.py --workers 12 --rows all --seed-splits 8
    python run_grid.py --workers 12 --resume
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

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from lib.io import read_checkpoint

BASE_SIZES = [0, 250, 500, 750, 1000, 1500, 2000, 2500]
ROW_IDS = ["row1_uncertainty_disagreement", "row2_diverse_uncertainty"]
T_MAX = 3500
EXPERIMENT_ID = "dense50k_v2_labelblind"
DENSE_DATA_ID = "dense50k_v1"

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import runs_data_dir  # noqa: E402

_DATA_DIR = runs_data_dir()


def _parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _is_job_complete(
    data_dir: Path,
    experiment_id: str,
    row: str,
    base_size: int,
    seed_split: int,
) -> bool:
    """Check if a trajectory job has already reached T_MAX."""
    traj_dir = data_dir / "trajectories" / experiment_id / row / f"B{base_size}_R{seed_split:03d}"
    cp = read_checkpoint(traj_dir)
    if cp is None:
        return False
    return int(cp.get("total_labels", 0)) >= T_MAX


def _run_one_job(args: tuple[str, int, int, str, str, str]) -> dict[str, Any]:
    """Worker function: run a single trajectory job."""
    row, base_size, seed_split, data_dir_str, experiment_id, dense_data_id = args

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import torch
    torch.set_num_threads(1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from run_trajectory import run_trajectory

    try:
        result = run_trajectory(
            row=row,
            base_size=base_size,
            seed_split=seed_split,
            data_dir=Path(data_dir_str),
            experiment_id=experiment_id,
            dense_data_id=dense_data_id,
        )
        return result
    except Exception as exc:
        return {
            "row": row,
            "base_size": base_size,
            "seed_split": seed_split,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rows", default="all", help="Comma-separated row IDs or 'all'")
    parser.add_argument("--seed-splits", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--dense-data-id", default=DENSE_DATA_ID)
    parser.add_argument(
        "--base-sizes",
        default=",".join(str(v) for v in BASE_SIZES),
        help="Comma-separated base sizes",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed jobs, continue partial ones")
    args = parser.parse_args()

    if args.rows == "all":
        rows = ROW_IDS
    else:
        rows = [r.strip() for r in args.rows.split(",")]
    base_sizes = _parse_int_list(args.base_sizes)

    all_jobs: list[tuple[str, int, int, str, str, str]] = []
    skipped = 0
    for row in rows:
        for base_size in base_sizes:
            for split_r in range(1, args.seed_splits + 1):
                if args.resume and _is_job_complete(
                    args.data_dir, args.experiment_id, row, base_size, split_r
                ):
                    skipped += 1
                    continue
                all_jobs.append((
                    row, base_size, split_r, str(args.data_dir),
                    args.experiment_id, args.dense_data_id,
                ))

    total = len(all_jobs) + skipped
    print(f"[run_grid] {len(all_jobs)} jobs to run, {skipped} already complete "
          f"(total grid: {total})", flush=True)
    print(f"[run_grid] Using {args.workers} workers on {os.cpu_count()} available CPUs",
          flush=True)

    if not all_jobs:
        print("[run_grid] Nothing to do.", flush=True)
        return

    t0 = time.perf_counter()
    completed = 0
    errors = 0

    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(_run_one_job, job): job for job in all_jobs}

        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                status = result.get("status", "unknown")
                if status == "ok":
                    completed += 1
                else:
                    errors += 1
                    print(f"[run_grid] ERROR {job[0]} B{job[1]} R{job[2]}: "
                          f"{result.get('error', 'unknown')[:200]}", flush=True)
            except Exception as exc:
                errors += 1
                print(f"[run_grid] EXCEPTION {job[0]} B{job[1]} R{job[2]}: {exc}",
                      flush=True)

            elapsed = time.perf_counter() - t0
            done_total = completed + errors
            if done_total > 0:
                eta = elapsed / done_total * (len(all_jobs) - done_total)
            else:
                eta = 0
            print(f"[run_grid] Progress: {done_total}/{len(all_jobs)} "
                  f"(ok={completed}, err={errors}) "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n[run_grid] DONE. {completed} ok, {errors} errors in {elapsed:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
