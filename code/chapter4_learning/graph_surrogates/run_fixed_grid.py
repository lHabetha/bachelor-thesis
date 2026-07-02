"""Run the full fixed-label Chapter 4 grid for MLP control and best graph model."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .lib.data_utils import full_grid_jobs, load_bundle, set_single_thread, training_arrays
from .lib.graph_models import MODEL_REGISTRY
from .lib.paths import ARCH_SCREEN_DIR, FIXED_GRID_DIR
from .lib.train_eval import eval_binary, fit_surrogate, save_artifact
from .run_arch_screen import _job_key

GRAPH_ARCHES = ["part_graph_mpnn_v1", "constraint_graph_mpnn_v1", "edge_pool_graph_v1"]


def _load_best_graph() -> str:
    summary_path = ARCH_SCREEN_DIR / "architecture_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        best = data.get("best_graph_architecture")
        if best:
            return str(best)
    results_dir = ARCH_SCREEN_DIR / "results"
    scores: dict[str, list[float]] = {a: [] for a in GRAPH_ARCHES}
    for path in results_dir.glob("*.json"):
        rec = json.loads(path.read_text())
        arch = rec.get("architecture")
        if rec.get("status") == "ok" and arch in scores:
            scores[arch].append(float(rec.get("balanced_accuracy", 0.0)))
    if not any(scores.values()):
        return "constraint_graph_mpnn_v1"
    return max(scores, key=lambda a: float(np.mean(scores[a])) if scores[a] else -1.0)


def _run_one(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    architecture, job = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    set_single_thread()
    try:
        bundle = load_bundle()
        X, y, ids = training_arrays(
            bundle.pool,
            row=job["row"],
            base_size=job.get("base_size"),
            seed_split=job.get("seed_split"),
            total_labels=int(job["total_labels"]),
            replicate=job.get("replicate"),
        )
        seed = 37000 + hash(_job_key(architecture, job)) % 1_000_000
        fitted = fit_surrogate(X, y, architecture=architecture, seed=seed)
        prob = fitted.predict_proba(bundle.holdout_X)
        metrics = eval_binary(bundle.holdout_y, prob)
        key = _job_key(architecture, job)
        out_dir = FIXED_GRID_DIR / "checkpoints" / key
        save_artifact(
            fitted,
            out_dir,
            extra_card={
                "stage": "full_fixed_grid",
                "train_ids_count": len(ids),
                "job": job,
                "holdout_metrics": metrics,
            },
        )
        return {
            "status": "ok",
            "architecture": architecture,
            "checkpoint_path": str(out_dir.relative_to(FIXED_GRID_DIR.parents[1])),
            "train_pos_rate": float(np.mean(y)),
            "n_train": int(len(y)),
            **job,
            **metrics,
            **fitted.metadata,
        }
    except Exception as exc:
        return {
            "status": "error",
            "architecture": architecture,
            **job,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--architectures", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.architectures == "auto":
        best_graph = _load_best_graph()
        architectures = ["mlp64_control", best_graph]
    else:
        architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    for arch in architectures:
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch}")

    FIXED_GRID_DIR.mkdir(parents=True, exist_ok=True)
    (FIXED_GRID_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    results_dir = FIXED_GRID_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = full_grid_jobs()
    all_jobs = []
    for arch in architectures:
        for job in jobs:
            out_path = results_dir / f"{_job_key(arch, job)}.json"
            if args.resume and out_path.exists():
                continue
            all_jobs.append((arch, job))
    print(f"[fixed_grid] architectures={architectures}")
    print(f"[fixed_grid] {len(all_jobs)} jobs to run with {args.workers} workers")

    if not all_jobs:
        return
    mp_ctx = multiprocessing.get_context("forkserver")
    ok = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as ex:
        futures = {ex.submit(_run_one, j): j for j in all_jobs}
        for fut in as_completed(futures):
            arch, job = futures[fut]
            result = fut.result()
            out_path = results_dir / f"{_job_key(arch, job)}.json"
            out_path.write_text(json.dumps(result, indent=2))
            if result.get("status") == "ok":
                ok += 1
            else:
                errors += 1
                print(f"[fixed_grid] ERROR {_job_key(arch, job)}: {result.get('error','')[:200]}")
            done = ok + errors
            if done % 50 == 0 or done == len(all_jobs):
                print(f"[fixed_grid] progress {done}/{len(all_jobs)} ok={ok} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
