"""Aggregate Chapter 4 random and active-learning results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .lib.paths import ACTIVE_GRID_DIR, RANDOM_BASELINE_DIR, REPORTS_DIR, RESULTS_DIR, ensure_dirs


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, (dict, list)):
            continue
        out[f"{prefix}{key}"] = val
    return out


def load_random_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((RANDOM_BASELINE_DIR / "results").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "source_file": str(path),
            "row": rec["row"],
            "model_id": rec["model_id"],
            "base_size": int(rec.get("base_size", 0)),
            "seed_split": int(rec["seed_split"]),
            "total_labels": int(rec["total_labels"]),
            "round_idx": 0,
        }
        row.update(_flatten_metrics("", rec["metrics"]))
        rows.append(row)
    return pd.DataFrame(rows)


def load_active_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ACTIVE_GRID_DIR / "results").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        for rr in rec.get("round_records", []):
            row = {
                "source_file": str(path),
                "row": rr["row"],
                "model_id": rr["model_id"],
                "base_size": int(rr["base_size"]),
                "seed_split": int(rr["seed_split"]),
                "total_labels": int(rr["total_labels"]),
                "round_idx": int(rr["round_idx"]),
            }
            row.update(_flatten_metrics("", rr["metrics"]))
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "mcc",
        "recall_blocked",
        "recall_assemblable",
        "roc_auc",
        "brier",
        "ece_10bin",
        "fit_wall_s",
        "predict_wall_s",
        "train_pos_rate",
    ]
    existing = [m for m in metrics if m in df.columns]
    grouped = df.groupby(["row", "model_id", "base_size", "total_labels"], dropna=False)
    agg = grouped[existing].agg(["mean", "std", "count"]).reset_index()
    agg.columns = [
        "_".join([str(c) for c in col if c != ""]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    return agg


def _write_report(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chapter 4 Aggregation Snapshot",
        "",
        f"Per-fit rows: {len(df)}",
        f"Summary rows: {len(summary)}",
        "",
    ]
    if not summary.empty and "balanced_accuracy_mean" in summary.columns:
        final = summary[summary["total_labels"] == summary["total_labels"].max()].copy()
        final = final.sort_values(["row", "balanced_accuracy_mean"], ascending=[True, False])
        lines.append("## Best Final BAC By Row")
        lines.append("")
        lines.append(_markdown_table(final[["row", "model_id", "base_size", "balanced_accuracy_mean", "balanced_accuracy_std", "balanced_accuracy_count"]].head(30)))
        lines.append("")
    (REPORTS_DIR / "aggregation_snapshot.md").write_text("\n".join(lines), encoding="utf-8")
    _write_named_reports(summary)


def _write_named_reports(summary: pd.DataFrame) -> None:
    if summary.empty:
        text = "No Chapter 4 results have been aggregated yet.\n"
        for name in ["predictive_power_report.md", "active_learning_report.md", "thesis_insert_notes.md"]:
            (REPORTS_DIR / name).write_text(text, encoding="utf-8")
        return

    final_t = int(summary["total_labels"].max())
    final = summary[summary["total_labels"] == final_t].copy()
    random = final[final["row"] == "random_baseline"].sort_values("balanced_accuracy_mean", ascending=False)
    active = final[final["row"].isin(["al_uncertainty_disagreement", "al_diverse_uncertainty"])].sort_values(
        "balanced_accuracy_mean", ascending=False
    )
    random_table = random[
        ["model_id", "balanced_accuracy_mean", "balanced_accuracy_std", "fit_wall_s_mean"]
    ]
    random_table_text = _markdown_table(random_table) if not random.empty else "No random-baseline rows yet."
    active_cols = ["row", "model_id", "base_size", "balanced_accuracy_mean", "balanced_accuracy_std", "fit_wall_s_mean"]
    active_table_text = _markdown_table(active[active_cols].head(40)) if not active.empty else "No active-learning rows yet."

    (REPORTS_DIR / "predictive_power_report.md").write_text(
        "\n".join(
            [
                "# Chapter 4 Predictive Power Report",
                "",
                f"Aggregated result count: {len(summary)} grouped rows.",
                f"Current maximum label budget: T={final_t}.",
                "",
                "## Random Baseline At Final Budget",
                "",
                random_table_text,
                "",
                "## Active-Learning Rows At Final Budget",
                "",
                active_table_text,
                "",
                "## Interpretation Placeholder",
                "",
                "Replace this paragraph after the full run with a concise comparison of tree ensembles, MLP64, label efficiency, calibration, and compute cost.",
            ]
        ),
        encoding="utf-8",
    )

    al = summary[summary["row"].isin(["al_uncertainty_disagreement", "al_diverse_uncertainty"])].copy()
    if not al.empty:
        pivot = al.pivot_table(
            index=["model_id", "base_size", "total_labels"],
            columns="row",
            values="balanced_accuracy_mean",
        ).reset_index()
        if {"al_uncertainty_disagreement", "al_diverse_uncertainty"}.issubset(pivot.columns):
            pivot["delta_diverse_minus_uncertainty"] = (
                pivot["al_diverse_uncertainty"] - pivot["al_uncertainty_disagreement"]
            )
            delta_table = _markdown_table(
                pivot.sort_values("delta_diverse_minus_uncertainty", ascending=False).head(30)
            )
        else:
            delta_table = "Both active-learning rows are not available yet."
    else:
        delta_table = "No active-learning rows yet."
    (REPORTS_DIR / "active_learning_report.md").write_text(
        "\n".join(
            [
                "# Chapter 4 Active-Learning Report",
                "",
                "This report compares uncertainty/disagreement acquisition against diverse uncertainty for every Chapter 4 model family.",
                "",
                "## Largest Current Diverse-Minus-Uncertainty Deltas",
                "",
                delta_table,
                "",
                "## Interpretation Placeholder",
                "",
                "After the full run, summarize whether diversity helps trees differently than it helped the MLP in Chapter 4.",
            ]
        ),
        encoding="utf-8",
    )

    (REPORTS_DIR / "thesis_insert_notes.md").write_text(
        "\n".join(
            [
                "# Chapter 4 Thesis Insert Notes",
                "",
                "This file is intentionally written as source material for a future thesis subsection, not as final thesis text.",
                "",
                "## Candidate Claims To Fill After Full Run",
                "",
                "- Whether tree ensembles outperform MLP64 at low label budgets.",
                "- Whether tree ensembles remain competitive at high label budgets.",
                "- Whether active learning is architecture-dependent.",
                "- Whether prediction quality transfers into Chapter 5 non-gradient repair.",
                "- What caveat should be stated about losing differentiable gradients.",
                "",
                "## Current Evidence Tables",
                "",
                "See `predictive_power_report.md`, `active_learning_report.md`, and generated figures in `figures/`.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args
    ensure_dirs()
    random_df = load_random_rows()
    active_df = load_active_rows()
    all_df = pd.concat([random_df, active_df], ignore_index=True)
    random_df.to_csv(RESULTS_DIR / "random_metrics.csv", index=False)
    active_df.to_csv(RESULTS_DIR / "active_metrics.csv", index=False)
    all_df.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)
    summary = _summary(all_df)
    summary.to_csv(RESULTS_DIR / "summary_by_group.csv", index=False)
    _write_report(all_df, summary)
    print(f"[task30 aggregate] rows={len(all_df)} summary={len(summary)}")


if __name__ == "__main__":
    main()
