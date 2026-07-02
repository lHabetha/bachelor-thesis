"""Run Chapter 4 label-blind active-learning trajectories."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from pathlib import Path
from typing import Any

import numpy as np

from .lib.acquisition import select_diverse_uncertainty, select_uncertainty_disagreement
from .lib.data_utils import (
    enabled_model_ids,
    features,
    job_key,
    load_bundle,
    load_seed_ids,
    records_for_ids,
    set_single_thread,
    write_json,
)
from .lib.experiment import fit_eval_record
from .lib.models import fit_acquisition_model
from .lib.paths import ACTIVE_GRID_DIR, ensure_dirs, load_protocol

_BUNDLE = None
_PROTOCOL: dict[str, Any] | None = None


def _init_worker(protocol: dict[str, Any]) -> None:
    global _BUNDLE, _PROTOCOL
    set_single_thread()
    _PROTOCOL = protocol
    _BUNDLE = load_bundle()


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_round(
    *,
    model_id: str,
    seed: int,
    selected_ids: list[str],
    row: str,
    base_size: int,
    seed_split: int,
    round_idx: int,
) -> dict[str, Any]:
    assert _BUNDLE is not None
    assert _PROTOCOL is not None
    train_df = records_for_ids(_BUNDLE.pool, selected_ids)
    metrics, _fit = fit_eval_record(
        model_id=model_id,
        train_df=train_df,
        holdout_df=_BUNDLE.holdout,
        seed=seed,
        protocol=_PROTOCOL,
    )
    return {
        "row": row,
        "model_id": model_id,
        "base_size": base_size,
        "seed_split": seed_split,
        "round_idx": round_idx,
        "total_labels": len(selected_ids),
        "model_seed": seed,
        "metrics": metrics,
    }


def _select_next(
    *,
    model_id: str,
    selected_ids: list[str],
    row: str,
    round_idx: int,
    seed_split: int,
) -> tuple[list[str], dict[str, Any]]:
    assert _BUNDLE is not None
    assert _PROTOCOL is not None
    used = set(selected_ids)
    remaining = _BUNDLE.pool[~_BUNDLE.pool["param_id"].astype(str).isin(used)].copy()
    train_df = records_for_ids(_BUNDLE.pool, selected_ids)
    acq_seed = int(_PROTOCOL["seeds"]["model_seed"]) + seed_split * 10000 + round_idx * 101
    acq_fit = fit_acquisition_model(
        model_id,
        features(train_df),
        train_df["label"].to_numpy(dtype=int),
        seed=acq_seed,
        protocol=_PROTOCOL,
    )
    X_rem = features(remaining)
    p_mean = acq_fit.predict_proba(X_rem)
    p_uncert = acq_fit.uncertainty(X_rem)
    if row == "al_uncertainty_disagreement":
        idx = select_uncertainty_disagreement(X_rem, p_mean, p_uncert, 100)
        meta = {
            "selection_policy": "uncertainty_disagreement",
            "n_uncertainty_disagreement": len(idx),
            "n_diverse_uncertainty": 0,
            "acquisition_inputs": ["p_mean", "p_uncertainty"],
        }
    elif row == "al_diverse_uncertainty":
        idx = select_diverse_uncertainty(X_rem, p_mean, 100)
        meta = {
            "selection_policy": "diverse_uncertainty",
            "n_uncertainty_disagreement": 0,
            "n_diverse_uncertainty": len(idx),
            "acquisition_inputs": ["p_mean", "raw_params"],
        }
    else:
        raise ValueError(f"Unknown row: {row}")
    ids = remaining.iloc[idx]["param_id"].astype(str).tolist()
    meta.update(
        {
            "round_idx": round_idx,
            "n_selected": len(ids),
            "acquisition_fit_wall_s": acq_fit.fit_wall_s,
            "acquisition_metadata": acq_fit.metadata,
            "param_ids": ids,
        }
    )
    return ids, meta


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    assert _BUNDLE is not None
    assert _PROTOCOL is not None
    out_path = Path(job["out_path"])
    existing = None if job["force"] else _load_existing(out_path)
    if existing and existing.get("complete") and existing.get("final_total_labels") >= int(_PROTOCOL["labels"]["t_max"]):
        return {"status": "skipped", "out_path": str(out_path)}

    base_ids = load_seed_ids(job["base_size"], job["seed_split"])
    if existing and not job["force"]:
        selected_ids = [str(x) for x in existing.get("selected_ids", base_ids)]
        round_records = list(existing.get("round_records", []))
        selections = list(existing.get("selections", []))
        start_round = int(existing.get("last_completed_round", 0)) + 1
    else:
        selected_ids = list(base_ids)
        round_records = []
        selections = []
        start_round = 1
        seed = int(_PROTOCOL["seeds"]["model_seed"]) + job["seed_split"] * 10000 + job["base_size"]
        round_records.append(
            _evaluate_round(
                model_id=job["model_id"],
                seed=seed,
                selected_ids=selected_ids,
                row=job["row"],
                base_size=job["base_size"],
                seed_split=job["seed_split"],
                round_idx=0,
            )
        )
        write_json(
            out_path,
            {
                **job,
                "schema_version": "task30_active_v1",
                "selected_ids": selected_ids,
                "round_records": round_records,
                "selections": selections,
                "last_completed_round": 0,
                "final_total_labels": len(selected_ids),
                "complete": False,
            },
        )

    t_max = int(_PROTOCOL["labels"]["t_max"])
    round_idx = start_round
    while len(selected_ids) < t_max:
        new_ids, selection_meta = _select_next(
            model_id=job["model_id"],
            selected_ids=selected_ids,
            row=job["row"],
            round_idx=round_idx,
            seed_split=job["seed_split"],
        )
        selected_ids.extend(new_ids)
        selections.append(selection_meta)
        seed = int(_PROTOCOL["seeds"]["model_seed"]) + job["seed_split"] * 10000 + job["base_size"] + round_idx
        round_records.append(
            _evaluate_round(
                model_id=job["model_id"],
                seed=seed,
                selected_ids=selected_ids,
                row=job["row"],
                base_size=job["base_size"],
                seed_split=job["seed_split"],
                round_idx=round_idx,
            )
        )
        write_json(
            out_path,
            {
                **job,
                "schema_version": "task30_active_v1",
                "selected_ids": selected_ids,
                "round_records": round_records,
                "selections": selections,
                "last_completed_round": round_idx,
                "final_total_labels": len(selected_ids),
                "complete": len(selected_ids) >= t_max,
            },
        )
        round_idx += 1

    return {"status": "ok", "out_path": str(out_path), "rounds": len(round_records)}


def _jobs(protocol: dict[str, Any], *, smoke: bool, force: bool) -> list[dict[str, Any]]:
    model_ids = enabled_model_ids(protocol)
    rows = ["al_uncertainty_disagreement", "al_diverse_uncertainty"]
    bases = [int(v) for v in protocol["labels"]["base_sizes"]]
    splits = list(range(1, int(protocol["seeds"]["n_splits"]) + 1))
    if smoke:
        model_ids = model_ids[:2]
        rows = rows[:1]
        bases = bases[:1]
        splits = [1]
    jobs: list[dict[str, Any]] = []
    for model_id in model_ids:
        for row in rows:
            for base in bases:
                for split in splits:
                    key = job_key(model_id, row, f"B{base:04d}", f"R{split:03d}")
                    jobs.append(
                        {
                            "model_id": model_id,
                            "row": row,
                            "base_size": base,
                            "seed_split": split,
                            "out_path": str(ACTIVE_GRID_DIR / "results" / f"{key}.json"),
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
    print(f"[task30 active] jobs={len(jobs)} workers={workers}")
    ok = 0
    skipped = 0
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(protocol,)) as ex:
        for res in ex.map(_run_job, jobs):
            ok += res["status"] == "ok"
            skipped += res["status"] == "skipped"
            print(f"[task30 active] ok={ok} skipped={skipped} latest={res['status']} {res['out_path']}", flush=True)
    print(f"[task30 active] complete ok={ok} skipped={skipped}")


if __name__ == "__main__":
    main()
