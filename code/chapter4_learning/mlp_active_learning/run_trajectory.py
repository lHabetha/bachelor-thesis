"""Run a single (row, B, R) label-blind active-learning trajectory for 50k.

Chapter 4 keeps the dense50k_v1 pool/seed data but writes new label-blind
trajectories under dense50k_v2_labelblind.

Usage:
    python run_trajectory.py --row row1_uncertainty_disagreement --base-size 250 --seed-split 1
    python run_trajectory.py --row row2_diverse_uncertainty --base-size 500 --seed-split 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from lib.acquisition import (
    select_diverse_uncertainty,
    select_uncertainty_disagreement,
)
from lib.formula_oracle import (
    FEATURE_NAMES,
)
from lib.io import (
    atomic_json_write,
    read_checkpoint,
    read_json,
    read_parquet,
    set_single_thread,
    write_checkpoint,
)
from lib.model_mlp64 import (
    Standardizer,
    eval_holdout,
    ensemble_predict,
    ensemble_std,
    train_ensemble,
)

# ─── Constants ───────────────────────────────────────────────────────────

T_MAX = 3500
STEP_SIZE = 250
BASE_SIZES = [0, 250, 500, 750, 1000, 1500, 2000, 2500]
ROW_IDS = ("row1_uncertainty_disagreement", "row2_diverse_uncertainty")
DENSE_DATA_ID = "dense50k_v1"
EXPERIMENT_ID = "dense50k_v2_labelblind"

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import (  # noqa: E402
    HOLDOUT_PATH,
    pool_parquet_path,
    runs_data_dir,
    seed_set_path,
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _load_pool(dense_data_id: str) -> list[dict[str, Any]]:
    return read_parquet(pool_parquet_path(dense_data_id))


def _load_holdout() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    records = read_parquet(HOLDOUT_PATH)
    X = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in records], dtype=np.float64)
    y = np.array([int(r["label"]) for r in records], dtype=np.int64)
    return X, y, records


def _load_seed_ids(
    base_size: int,
    seed_split: int,
    dense_data_id: str,
) -> list[str]:
    path = seed_set_path(dense_data_id, seed_split, base_size)
    data = read_json(path)
    if data is None:
        raise FileNotFoundError(f"Seed set not found: {path}")
    return data["param_ids"]


def _records_to_X(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[float(r[f]) for f in FEATURE_NAMES] for r in records], dtype=np.float64
    )


def _records_to_y(records: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(r["label"]) for r in records], dtype=np.int64)


# ─── Main trajectory logic ───────────────────────────────────────────────


def run_trajectory(
    row: str,
    base_size: int,
    seed_split: int,
    data_dir: Path | None = None,
    experiment_id: str = EXPERIMENT_ID,
    dense_data_id: str = DENSE_DATA_ID,
) -> dict[str, Any]:
    """Execute one full (row, B, R) trajectory with resume support."""
    set_single_thread()

    if data_dir is None:
        data_dir = runs_data_dir()
    data_dir = Path(data_dir)

    traj_dir = data_dir / "trajectories" / experiment_id / row / f"B{base_size}_R{seed_split:03d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    pool_records = _load_pool(dense_data_id)
    holdout_X, holdout_y, holdout_records = _load_holdout()
    base_ids = _load_seed_ids(base_size, seed_split, dense_data_id)

    pool_by_id = {str(r["param_id"]): r for r in pool_records}

    checkpoint = read_checkpoint(traj_dir)
    if checkpoint is not None:
        selected_ids = set(checkpoint.get("selected_param_ids", []))
        last_round = int(checkpoint.get("last_completed_round", 0))
        print(f"[trajectory] Resuming {row} B{base_size} R{seed_split} from round {last_round}")
    else:
        selected_ids = set()
        last_round = 0

    base_records = [pool_by_id[pid] for pid in base_ids if pid in pool_by_id]
    acquired_records = [pool_by_id[pid] for pid in selected_ids if pid in pool_by_id]
    total_labels = base_size + len(acquired_records)

    if last_round == 0 and base_size > 0:
        train_X = _records_to_X(base_records)
        train_y = _records_to_y(base_records)
        if len(train_X) > 0:
            _eval_and_save_round(
                traj_dir, 0, train_X, train_y, holdout_X, holdout_y,
                row, base_size, seed_split, total_labels,
            )

    round_idx = last_round
    while total_labels < T_MAX:
        round_idx += 1
        n_to_add = min(STEP_SIZE, T_MAX - total_labels)

        current_records = base_records + acquired_records
        train_X = _records_to_X(current_records)
        train_y = _records_to_y(current_records)

        used_ids = set(base_ids) | selected_ids
        remaining = [r for r in pool_records if str(r["param_id"]) not in used_ids]

        if len(remaining) < n_to_add:
            print(f"[trajectory] Pool exhausted at round {round_idx}, only {len(remaining)} left")
            break

        remaining_X = _records_to_X(remaining)

        if len(current_records) == 0:
            rng_cold = np.random.default_rng(55000 + seed_split * 100 + round_idx)
            selected_local_idx = rng_cold.choice(
                len(remaining), size=min(n_to_add, len(remaining)), replace=False
            ).tolist()
            selection_meta = {
                "selection_policy": "random_cold_start",
                "n_hybrid": 0,
                "n_diverse_uncertainty": 0,
                "n_random": len(selected_local_idx),
            }
        else:
            ensemble_seed = 55000 + seed_split * 100 + round_idx
            ensemble = train_ensemble(train_X, train_y, base_seed=ensemble_seed)

            p_mean = ensemble_predict(ensemble, remaining_X)
            p_std = ensemble_std(ensemble, remaining_X)

            if row == "row1_uncertainty_disagreement":
                selected_local_idx = select_uncertainty_disagreement(
                    remaining_X, p_mean, p_std, n_to_add
                )
                selection_meta = {
                    "selection_policy": "uncertainty_disagreement",
                    "n_uncertainty_disagreement": n_to_add,
                    "n_diverse_uncertainty": 0,
                    "acquisition_inputs": ["p_mean", "p_std"],
                }
            elif row == "row2_diverse_uncertainty":
                selected_local_idx = select_diverse_uncertainty(
                    remaining_X, p_mean, n_to_add
                )
                selection_meta = {
                    "selection_policy": "diverse_uncertainty",
                    "n_uncertainty_disagreement": 0,
                    "n_diverse_uncertainty": len(selected_local_idx),
                    "acquisition_inputs": ["p_mean", "raw_params"],
                }
            else:
                raise ValueError(f"Unknown row: {row}")

        new_ids = [str(remaining[i]["param_id"]) for i in selected_local_idx]
        new_records = [remaining[i] for i in selected_local_idx]

        selected_ids.update(new_ids)
        acquired_records.extend(new_records)
        total_labels = base_size + len(acquired_records)

        round_dir = traj_dir / "rounds" / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        selection_meta["round_idx"] = round_idx
        selection_meta["total_labels"] = total_labels
        selection_meta["param_ids"] = new_ids
        atomic_json_write(
            traj_dir / f"selections_round_{round_idx:02d}.json",
            selection_meta,
        )

        train_X_new = _records_to_X(base_records + acquired_records)
        train_y_new = _records_to_y(base_records + acquired_records)
        _eval_and_save_round(
            traj_dir, round_idx, train_X_new, train_y_new, holdout_X, holdout_y,
            row, base_size, seed_split, total_labels,
        )

        write_checkpoint(traj_dir, {
            "row": row,
            "base_size": base_size,
            "seed_split": seed_split,
            "last_completed_round": round_idx,
            "total_labels": total_labels,
            "selected_param_ids": list(selected_ids),
        })

        print(f"[trajectory] {row} B{base_size} R{seed_split} round {round_idx}: "
              f"T={total_labels}", flush=True)

    return {
        "row": row,
        "base_size": base_size,
        "seed_split": seed_split,
        "total_labels": total_labels,
        "rounds_completed": round_idx,
        "status": "ok",
    }


def _eval_and_save_round(
    traj_dir: Path,
    round_idx: int,
    train_X: np.ndarray,
    train_y: np.ndarray,
    holdout_X: np.ndarray,
    holdout_y: np.ndarray,
    row: str,
    base_size: int,
    seed_split: int,
    total_labels: int,
) -> None:
    """Train a fresh model on current data, evaluate on holdout, save metrics."""
    round_dir = traj_dir / "rounds" / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = round_dir / "metrics.json"
    if metrics_path.exists():
        return

    t0 = time.perf_counter()
    eval_seed = 55100 + seed_split * 10 + round_idx
    from lib.model_mlp64 import fit_mlp
    fitted = fit_mlp(train_X, train_y, seed=eval_seed)
    prob = fitted.predict_proba(holdout_X)
    metrics = eval_holdout(holdout_y, prob)

    metrics.update({
        "row": row,
        "base_size": base_size,
        "seed_split": seed_split,
        "total_labels": total_labels,
        "round_idx": round_idx,
        "wall_s": time.perf_counter() - t0,
        "status": "ok",
    })
    atomic_json_write(metrics_path, metrics)


# ─── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True, choices=list(ROW_IDS))
    parser.add_argument("--base-size", type=int, required=True, choices=BASE_SIZES)
    parser.add_argument("--seed-split", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--dense-data-id", default=DENSE_DATA_ID)
    args = parser.parse_args()

    result = run_trajectory(
        row=args.row,
        base_size=args.base_size,
        seed_split=args.seed_split,
        data_dir=args.data_dir,
        experiment_id=args.experiment_id,
        dense_data_id=args.dense_data_id,
    )
    print(f"[trajectory] Done: {result}")


if __name__ == "__main__":
    main()
