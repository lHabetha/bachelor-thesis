"""Aggregate Chapter 4 graph-surrogate metrics and write reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .lib.paths import ARCH_SCREEN_DIR, FIGURES_DIR, FIXED_GRID_DIR, REPORTS_DIR

GRAPH_ARCHES = {"part_graph_mpnn_v1", "constraint_graph_mpnn_v1", "edge_pool_graph_v1"}


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    ok = df[df["status"] == "ok"].copy()
    metrics = [
        "balanced_accuracy",
        "recall_blocked",
        "recall_assemblable",
        "roc_auc",
        "mcc",
        "brier",
        "ece_10bin",
        "fit_wall_s",
        "parameter_count",
    ]
    present = [m for m in metrics if m in ok.columns]
    agg = ok.groupby(group_cols)[present].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    return agg.reset_index()


def _write_arch_report(df: pd.DataFrame, summary: pd.DataFrame) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        (REPORTS_DIR / "architecture_screen_report.md").write_text(
            "# Architecture Screen\n\nNo architecture-screen results found yet.\n"
        )
        return {"best_graph_architecture": None}

    mean_by_arch = (
        df[df["status"] == "ok"]
        .groupby("architecture")
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            recall_blocked=("recall_blocked", "mean"),
            ece_10bin=("ece_10bin", "mean"),
            fit_wall_s=("fit_wall_s", "mean"),
            n=("balanced_accuracy", "count"),
        )
        .reset_index()
        .sort_values("balanced_accuracy", ascending=False)
    )
    graph_rows = mean_by_arch[mean_by_arch["architecture"].isin(GRAPH_ARCHES)]
    best_graph = None if graph_rows.empty else str(graph_rows.iloc[0]["architecture"])
    mlp = mean_by_arch[mean_by_arch["architecture"] == "mlp64_control"]
    mlp_bac = float(mlp.iloc[0]["balanced_accuracy"]) if not mlp.empty else float("nan")
    graph_bac = float(graph_rows.iloc[0]["balanced_accuracy"]) if not graph_rows.empty else float("nan")
    delta = graph_bac - mlp_bac

    lines = [
        "# Architecture Screen Report",
        "",
        "This report aggregates fixed-label Chapter 4 architecture-screen runs.",
        "",
        "## Mean Metrics By Architecture",
        "",
        _markdown_table(mean_by_arch),
        "",
        "## Decision",
        "",
    ]
    if best_graph is None:
        lines.append("No graph model completed successfully yet.")
    elif delta > 0.002:
        lines.append(
            f"`{best_graph}` is provisionally selected: mean BAC improves over "
            f"`mlp64_control` by {delta:.4f}."
        )
    else:
        lines.append(
            f"`{best_graph}` is the best graph model, but it does not clearly beat "
            f"`mlp64_control` on mean BAC (delta {delta:.4f}). It may still be "
            "exported for optimizer-smoke diagnostics, but the thesis should treat "
            "this as a negative/neutral architecture result unless Chapter 5 gradients improve."
        )
    report = "\n".join(lines) + "\n"
    (REPORTS_DIR / "architecture_screen_report.md").write_text(report)
    out = {
        "best_graph_architecture": best_graph,
        "mlp64_mean_bac": mlp_bac,
        "best_graph_mean_bac": graph_bac,
        "delta_graph_minus_mlp": delta,
    }
    (ARCH_SCREEN_DIR / "architecture_summary.json").write_text(json.dumps(out, indent=2))
    return out


def _write_fixed_report(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if df.empty:
        (REPORTS_DIR / "fixed_grid_report.md").write_text("# Fixed Grid Report\n\nNo fixed-grid results found yet.\n")
        return
    ok = df[df["status"] == "ok"].copy()
    mean_by_arch = (
        ok.groupby("architecture")
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            recall_blocked=("recall_blocked", "mean"),
            ece_10bin=("ece_10bin", "mean"),
            fit_wall_s=("fit_wall_s", "mean"),
            n=("balanced_accuracy", "count"),
        )
        .reset_index()
        .sort_values("balanced_accuracy", ascending=False)
    )
    lines = [
        "# Full Fixed-Label Grid Report",
        "",
        "Aggregates the full Chapter 4 fixed-label grid.",
        "",
        _markdown_table(mean_by_arch),
        "",
    ]
    (REPORTS_DIR / "fixed_grid_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    arch_df = _load_results(ARCH_SCREEN_DIR / "results")
    arch_summary = _summarize(arch_df, ["architecture", "row", "base_size", "total_labels"])
    if not arch_df.empty:
        ARCH_SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        arch_df.to_csv(ARCH_SCREEN_DIR / "architecture_metrics.csv", index=False)
        arch_summary.to_csv(ARCH_SCREEN_DIR / "architecture_summary_by_group.csv", index=False)
    _write_arch_report(arch_df, arch_summary)

    fixed_df = _load_results(FIXED_GRID_DIR / "results")
    fixed_summary = _summarize(fixed_df, ["architecture", "row", "base_size", "total_labels"])
    if not fixed_df.empty:
        FIXED_GRID_DIR.mkdir(parents=True, exist_ok=True)
        fixed_df.to_csv(FIXED_GRID_DIR / "fixed_grid_metrics.csv", index=False)
        fixed_summary.to_csv(FIXED_GRID_DIR / "fixed_grid_summary_by_group.csv", index=False)
    _write_fixed_report(fixed_df, fixed_summary)

    print("[aggregate] wrote Chapter 4 summaries")


if __name__ == "__main__":
    main()
