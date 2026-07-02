"""Aggregate metrics from trajectory rounds and baseline into result CSVs.

Scans all completed trajectory checkpoints and round metrics,
produces aggregate CSVs for plotting.

Usage:
    python aggregate_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from lib.io import read_json

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import baseline_dir, runs_data_dir, trajectories_dir  # noqa: E402

EXPERIMENT_ID = "dense50k_v2_labelblind"
BASELINE_ID = "dense50k_v1"
ROW1 = "row1_uncertainty_disagreement"
ROW2 = "row2_diverse_uncertainty"
ROW_IDS = {ROW1, ROW2}


def collect_trajectory_metrics(traj_dir: Path) -> pd.DataFrame:
    """Scan all round metrics.json files from trajectory directories."""
    rows_list: list[dict[str, Any]] = []

    if not traj_dir.exists():
        return pd.DataFrame()

    for row_dir in sorted(traj_dir.iterdir()):
        if not row_dir.is_dir():
            continue
        row_name = row_dir.name
        if row_name not in ROW_IDS:
            continue
        for traj in sorted(row_dir.iterdir()):
            if not traj.is_dir():
                continue
            rounds_dir = traj / "rounds"
            if not rounds_dir.exists():
                continue
            for round_dir in sorted(rounds_dir.iterdir()):
                metrics_path = round_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                m = read_json(metrics_path)
                if m is None:
                    continue
                m["row"] = m.get("row", row_name)
                rows_list.append(m)

    return pd.DataFrame(rows_list)


def collect_baseline_metrics(baseline_dir: Path) -> pd.DataFrame:
    """Scan all baseline JSON files."""
    rows_list: list[dict[str, Any]] = []

    if not baseline_dir.exists():
        return pd.DataFrame()

    for f in sorted(baseline_dir.glob("T*_rep*.json")):
        m = read_json(f)
        if m is not None and m.get("status") == "ok":
            rows_list.append(m)

    return pd.DataFrame(rows_list)


def compute_paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Row2 - Row1 deltas paired by (base_size, seed_split, total_labels)."""
    r1 = df[df["row"] == ROW1].copy()
    r2 = df[df["row"] == ROW2].copy()

    merge_cols = ["base_size", "seed_split", "total_labels"]
    merged = r1.merge(r2, on=merge_cols, suffixes=("_r1", "_r2"))

    metric_cols = ["balanced_accuracy", "accuracy", "macro_f1", "mcc",
                   "recall_blocked", "recall_assemblable", "roc_auc", "brier"]

    deltas = merged[merge_cols].copy()
    for col in metric_cols:
        c1 = f"{col}_r1"
        c2 = f"{col}_r2"
        if c1 in merged.columns and c2 in merged.columns:
            deltas[f"delta_{col}"] = merged[c2] - merged[c1]

    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--baseline-id", default=BASELINE_ID)
    args = parser.parse_args()

    traj_dir = trajectories_dir(args.experiment_id)
    baseline_dir_path = baseline_dir(args.baseline_id)
    results_dir = runs_data_dir() / "results" / args.experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[aggregate] Collecting trajectory metrics...")
    df_traj = collect_trajectory_metrics(traj_dir)
    if len(df_traj) > 0:
        df_traj.to_csv(results_dir / "trajectory_metrics.csv", index=False)
        print(f"  {len(df_traj)} rows -> trajectory_metrics.csv")
    else:
        print("  WARNING: No trajectory metrics found.")

    print("[aggregate] Collecting baseline metrics...")
    df_base = collect_baseline_metrics(baseline_dir_path)
    if len(df_base) > 0:
        df_base.to_csv(results_dir / "baseline_metrics.csv", index=False)
        print(f"  {len(df_base)} rows -> baseline_metrics.csv")
    else:
        print("  WARNING: No baseline metrics found.")

    if len(df_traj) > 0:
        print("[aggregate] Computing paired deltas (Row2 - Row1)...")
        df_deltas = compute_paired_deltas(df_traj)
        df_deltas.to_csv(results_dir / "paired_deltas.csv", index=False)
        print(f"  {len(df_deltas)} rows -> paired_deltas.csv")

        print("[aggregate] Computing summary stats...")
        summary_rows = []
        for (row, bs, tl), grp in df_traj.groupby(["row", "base_size", "total_labels"]):
            summary_rows.append({
                "row": row,
                "base_size": int(bs),
                "total_labels": int(tl),
                "n_seeds": len(grp),
                "bac_mean": grp["balanced_accuracy"].mean(),
                "bac_std": grp["balanced_accuracy"].std(),
                "bac_min": grp["balanced_accuracy"].min(),
                "bac_max": grp["balanced_accuracy"].max(),
                "acc_mean": grp["accuracy"].mean(),
                "f1_mean": grp["macro_f1"].mean(),
                "mcc_mean": grp["mcc"].mean(),
                "auc_mean": grp["roc_auc"].mean(),
            })
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(results_dir / "summary_stats.csv", index=False)
        print(f"  {len(df_summary)} rows -> summary_stats.csv")

    print("[aggregate] Done.")


if __name__ == "__main__":
    main()
