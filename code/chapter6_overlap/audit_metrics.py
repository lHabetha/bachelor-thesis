"""Post-hoc metric summaries for Chapter 6 audit reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._public_helpers import FEATURES, _read_csv
from .label_cache import params_from_row
from .paths import RUNS_DIR, TABLES_DIR


def _active_coordinates(start: dict, final: dict, tol: float = 1e-3) -> int:
    return sum(abs(float(final[k]) - float(start[k])) > tol for k in FEATURES)


def summarize_optimizer(results_json: Path, holdout_csv: Path, out_csv: Path) -> list[dict]:
    results = json.loads(results_json.read_text(encoding="utf-8"))
    holdout = {row["sample_id"]: row for row in _read_csv(holdout_csv)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["method"]].append(row)

    rows = []
    for method, vals in grouped.items():
        active = []
        evals = []
        residual_failures = []
        for row in vals:
            start = holdout[row["sample_id"]]
            active.append(_active_coordinates(start, row["final_params"]))
            if method == "direct_coordinate":
                # Each round evaluates up to 13 parameters * 5 step fractions * 2 directions.
                evals.append(1 + int(row["steps"]) * len(FEATURES) * 5 * 2)
            else:
                # The line-search method records accepted iteration count; each
                # iteration can try up to six verifier-gated step sizes.
                evals.append(1 + int(row["steps"]) * 6)
            if row["success"] is False:
                residual_failures.append(float(row["final_overlap_norm"]))
        rows.append(
            {
                "method": method,
                "n": len(vals),
                "successes": sum(bool(v["success"]) for v in vals),
                "mean_final_overlap_norm": float(np.mean([float(v["final_overlap_norm"]) for v in vals])),
                "mean_distance": float(np.mean([float(v["distance"]) for v in vals])),
                "mean_active_coordinates": float(np.mean(active)),
                "mean_estimated_verifier_evaluations": float(np.mean(evals)),
                "mean_failure_overlap_norm": "" if not residual_failures else float(np.mean(residual_failures)),
                "max_failure_overlap_norm": "" if not residual_failures else float(np.max(residual_failures)),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=RUNS_DIR / "optimization_mlp" / "overlap_repair_al_uncertainty_1750_v1" / "results.json",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=RUNS_DIR.parent.parent / "datasets" / "chapter6_overlap_clevis" / "holdout_5k" / "pool.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=TABLES_DIR / "optimizer_audit_metrics.csv",
    )
    args = parser.parse_args()
    rows = summarize_optimizer(args.results_json, args.holdout_csv, args.out_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
