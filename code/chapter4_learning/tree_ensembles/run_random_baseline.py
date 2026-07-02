"""Run Chapter 4 passive random-label baselines."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path
from typing import Any

import numpy as np

from .lib.data_utils import enabled_model_ids, job_key, load_bundle, records_for_ids, set_single_thread, write_json
from .lib.experiment import fit_eval_record
from .lib.paths import RANDOM_BASELINE_DIR, ensure_dirs, load_protocol

_BUNDLE = None
_PROTOCOL: dict[str, Any] | None = None


def _init_worker(protocol: dict[str, Any]) -> None:
    global _BUNDLE, _PROTOCOL
    set_single_thread()
    _PROTOCOL = protocol
    _BUNDLE = load_bundle()


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    assert _BUNDLE is not None
    assert _PROTOCOL is not None
    out_path = Path(job["out_path"])
    if out_path.exists() and not job["force"]:
        return {"status": "skipped", "out_path": str(out_path)}
    rng = np.random.default_rng(job["sample_seed"])
    pool_ids = _BUNDLE.pool["param_id"].astype(str).to_numpy()
    selected = rng.choice(pool_ids, size=job["total_labels"], replace=False).astype(str).tolist()
    train_df = records_for_ids(_BUNDLE.pool, selected)
    metrics, _fit = fit_eval_record(
        model_id=job["model_id"],
        train_df=train_df,
        holdout_df=_BUNDLE.holdout,
        seed=job["model_seed"],
        protocol=_PROTOCOL,
    )
    record = {
        "schema_version": "task30_random_v1",
        "row": "random_baseline",
        "model_id": job["model_id"],
        "seed_split": job["seed_split"],
        "replicate": job["seed_split"],
        "base_size": 0,
        "total_labels": job["total_labels"],
        "sample_seed": job["sample_seed"],
        "model_seed": job["model_seed"],
        "param_ids": selected,
        "metrics": metrics,
    }
    write_json(out_path, record)
    return {"status": "ok", "out_path": str(out_path), "bac": metrics["balanced_accuracy"]}


def _jobs(protocol: dict[str, Any], *, smoke: bool, force: bool) -> list[dict[str, Any]]:
    model_ids = enabled_model_ids(protocol)
    if smoke:
        model_ids = model_ids[:2]
        t_values = [100, 300]
        splits = [1]
    else:
        t_values = [int(v) for v in protocol["labels"]["t_values"]]
        splits = list(range(1, int(protocol["seeds"]["n_splits"]) + 1))
    jobs: list[dict[str, Any]] = []
    for model_id in model_ids:
        for t in t_values:
            for split in splits:
                key = job_key(model_id, "random", f"T{t:04d}", f"R{split:03d}")
                jobs.append(
                    {
                        "model_id": model_id,
                        "total_labels": t,
                        "seed_split": split,
                        "sample_seed": int(protocol["seeds"]["random_baseline_seed"]) + t * 100 + split,
                        "model_seed": int(protocol["seeds"]["model_seed"]) + split * 10000 + t,
                        "out_path": str(RANDOM_BASELINE_DIR / "results" / f"{key}.json"),
                        "force": force,
                    }
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    protocol = load_protocol()
    workers = args.workers or int(protocol["execution"]["workers"])
    jobs = _jobs(protocol, smoke=args.smoke, force=args.force)
    print(f"[task30 random] jobs={len(jobs)} workers={workers}")
    ok = 0
    skipped = 0
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(protocol,)) as ex:
        for res in ex.map(_run_job, jobs):
            ok += res["status"] == "ok"
            skipped += res["status"] == "skipped"
            if (ok + skipped) % 25 == 0 or res["status"] == "ok":
                print(f"[task30 random] ok={ok} skipped={skipped} latest={res['status']} {res['out_path']}", flush=True)
    print(f"[task30 random] complete ok={ok} skipped={skipped}")


if __name__ == "__main__":
    main()
