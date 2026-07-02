"""Export the best 50k label-blind B=1000 T=2500 seed model as a persisted artifact.

Usage:
    python -m chapter5_optimization.export_model

The final public release ships the trained checkpoint used by the thesis under
``results/chapter5_optimization/checkpoints/``. Re-exporting it requires Chapter 4
active-learning trajectory selections, which are not bundled unless explicitly
added to the release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from shared.paths import (
    DENSE50K_POOL,
    DENSE50K_SEEDS,
    DENSE50K_TRAJ,
    MODELS_DIR,
    TASK22_HOLDOUT,
)
from shared.model_utils import (
    FEATURE_NAMES,
    fit_mlp,
    eval_holdout,
    save_model_artifact,
)

ROW = "row1_uncertainty_disagreement"
BASE_SIZE = 1000
TOTAL_LABELS = 2500
N_ROUNDS = 6  # 1000 + 6*250 = 2500
SEED_SPLITS = range(1, 9)

MODEL_OUT_DIR = MODELS_DIR / "row1_uncertainty_disagreement_B1000_T2500_best"


def _get_train_ids(seed_split: int) -> list[str]:
    """Reconstruct the training IDs for B=1000, T=2500, given seed split."""
    seed_dir = DENSE50K_SEEDS / f"R{seed_split:03d}"
    base_data = json.loads((seed_dir / "base_prefix_1000.json").read_text())
    base_ids = base_data["param_ids"]

    traj_dir = DENSE50K_TRAJ / ROW / f"B{BASE_SIZE}_R{seed_split:03d}"
    acquired = []
    for r in range(1, N_ROUNDS + 1):
        sel_file = traj_dir / f"selections_round_{r:02d}.json"
        if sel_file.exists():
            sel_data = json.loads(sel_file.read_text())
            acquired.extend(sel_data["param_ids"])
        else:
            raise FileNotFoundError(
                "Cannot re-export the exact Chapter 5 checkpoint because Chapter 4 "
                f"trajectory selection file is not shipped: {sel_file}. Use the "
                "released checkpoint under results/chapter5_optimization/checkpoints/ "
                "or add the trajectory tree intentionally."
            )

    return base_ids + acquired


def export_model() -> None:
    pool = pd.read_parquet(DENSE50K_POOL)
    holdout = pd.read_parquet(TASK22_HOLDOUT)

    holdout_X = holdout[list(FEATURE_NAMES)].values.astype(np.float64)
    holdout_y = holdout["label"].values.astype(int)

    best_bac = -1.0
    best_split = -1
    best_model = None
    results = []

    for split in SEED_SPLITS:
        train_ids = _get_train_ids(split)
        train_mask = pool["param_id"].isin(set(train_ids))
        train_df = pool[train_mask]

        X_train = train_df[list(FEATURE_NAMES)].values.astype(np.float64)
        y_train = train_df["label"].values.astype(int)

        eval_seed = 55100 + split * 10 + N_ROUNDS
        model = fit_mlp(X_train, y_train, seed=eval_seed)

        probs = model.predict_proba(holdout_X)
        metrics = eval_holdout(holdout_y, probs)
        bac = metrics["balanced_accuracy"]
        results.append({"seed_split": split, "bac": bac, "metrics": metrics})
        print(f"  R{split:03d}: BAC={bac:.6f}")

        if bac > best_bac:
            best_bac = bac
            best_split = split
            best_model = model

    print(f"\nBest seed split: R{best_split:03d} with BAC={best_bac:.6f}")

    metadata = {
        "source": "50k row1_uncertainty_disagreement B=1000 T=2500",
        "seed_split": best_split,
        "eval_seed": 55100 + best_split * 10 + N_ROUNDS,
        "holdout_balanced_accuracy": best_bac,
        "holdout_metrics": results[best_split - 1]["metrics"],
        "n_train": TOTAL_LABELS,
        "row": ROW,
        "base_size": BASE_SIZE,
        "total_labels": TOTAL_LABELS,
        "pool_source": str(DENSE50K_POOL),
        "holdout_source": str(TASK22_HOLDOUT),
        "all_splits_bac": {f"R{r['seed_split']:03d}": r["bac"] for r in results},
    }

    save_model_artifact(best_model, MODEL_OUT_DIR, metadata)
    print(f"Saved model artifact to {MODEL_OUT_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Re-export the checkpoint. Requires Chapter 4 trajectory selections, "
            "which are not bundled in the public release."
        ),
    )
    args = parser.parse_args(argv)
    if not args.run:
        parser.print_help()
        print(
            "\nThe thesis checkpoint is already released under "
            "results/chapter5_optimization/checkpoints/. Use --run only after "
            "intentionally adding the missing Chapter 4 trajectory selections."
        )
        return 0
    export_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
