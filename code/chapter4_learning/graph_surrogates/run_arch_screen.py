"""Run the Chapter 4 architecture screen on fixed Chapter 4 label sets."""

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

from .lib.data_utils import load_bundle, screen_jobs, set_single_thread, training_arrays, write_fixed_label_manifest
from .lib.graph_models import MODEL_REGISTRY
from .lib.paths import ARCH_SCREEN_DIR
from .lib.train_eval import eval_binary, fit_surrogate, save_artifact

ARCHITECTURES = [
    "mlp64_control",
    "wide_mlp_control",
    "part_graph_mpnn_v1",
    "constraint_graph_mpnn_v1",
    "edge_pool_graph_v1",
]


def _job_key(architecture: str, job: dict[str, Any]) -> str:
    row = str(job["row"])
    t = int(job["total_labels"])
    if row == "baseline_random":
        return f"{architecture}__baseline_T{t:04d}_rep{int(job['replicate'])}"
    return (
        f"{architecture}__{row}_B{int(job['base_size']):04d}"
        f"_R{int(job['seed_split']):03d}_T{t:04d}"
    )


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
        seed = 27000 + hash(_job_key(architecture, job)) % 1_000_000
        fitted = fit_surrogate(X, y, architecture=architecture, seed=seed)
        prob = fitted.predict_proba(bundle.holdout_X)
        metrics = eval_binary(bundle.holdout_y, prob)

        key = _job_key(architecture, job)
        out_dir = ARCH_SCREEN_DIR / "checkpoints" / key
        save_artifact(
            fitted,
            out_dir,
            extra_card={
                "stage": "architecture_screen",
                "train_ids_count": len(ids),
                "job": job,
                "holdout_metrics": metrics,
            },
        )
        result = {
            "status": "ok",
            "architecture": architecture,
            "checkpoint_path": str(out_dir.relative_to(ARCH_SCREEN_DIR.parents[1])),
            "train_pos_rate": float(np.mean(y)),
            "n_train": int(len(y)),
            **job,
            **metrics,
            **fitted.metadata,
        }
        return result
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
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    for arch in architectures:
        if arch not in MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch}")

    ARCH_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    (ARCH_SCREEN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    jobs = screen_jobs()
    manifest = write_fixed_label_manifest(jobs, bundle.pool)
    print(f"[arch_screen] fixed-label manifest: {manifest}")

    all_jobs = []
    for arch in architectures:
        for job in jobs:
            out_path = ARCH_SCREEN_DIR / "results" / f"{_job_key(arch, job)}.json"
            if args.resume and out_path.exists():
                continue
            all_jobs.append((arch, job))

    results_dir = ARCH_SCREEN_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[arch_screen] {len(all_jobs)} jobs to run with {args.workers} workers")

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
                print(f"[arch_screen] ERROR {_job_key(arch, job)}: {result.get('error','')[:200]}")
            done = ok + errors
            if done % 25 == 0 or done == len(all_jobs):
                print(f"[arch_screen] progress {done}/{len(all_jobs)} ok={ok} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
