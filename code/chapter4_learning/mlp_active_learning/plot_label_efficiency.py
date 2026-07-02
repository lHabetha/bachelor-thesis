"""Generate label-efficiency plots for Chapter 4 dense-pool (50k) rerun.

Reads aggregated results and produces thesis-ready figures.

Usage:
    python plot_label_efficiency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import FIGURES_DIR, frozen_results_dir  # noqa: E402

EXPERIMENT_ID = "dense50k_v2_labelblind"
ROW1 = "row1_uncertainty_disagreement"
ROW2 = "row2_diverse_uncertainty"
ROW_LABELS = {
    ROW1: "Uncertainty + Disagreement",
    ROW2: "Diverse Uncertainty",
}
_RESULTS_DIR = frozen_results_dir(EXPERIMENT_ID)
_FIGURES_DIR = FIGURES_DIR

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj = pd.read_csv(_RESULTS_DIR / "trajectory_metrics.csv")
    summary = pd.read_csv(_RESULTS_DIR / "summary_stats.csv")
    baseline_path = _RESULTS_DIR / "baseline_metrics.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    return traj, summary, baseline


# Reversed-log y-axis (matches plot_combined_4panel_log.py): plots 1-BAC on a
# log scale with inverted axis so the near-1.0 region is visually expanded.
BAC_MIN = 0.70
BAC_MAX = 1.00
ERR_MIN = 1e-3  # log scale cannot display y=0, so this tick is labelled as BAC=1.00.
ERR_MAX = 1.0 - BAC_MIN


def _bac_to_log_error(values: np.ndarray) -> np.ndarray:
    """Convert BAC values to clipped log-error coordinates."""
    clipped_bac = np.clip(values, BAC_MIN, BAC_MAX)
    return np.clip(1.0 - clipped_bac, ERR_MIN, ERR_MAX)


def _format_log_yaxis(ax: plt.Axes) -> None:
    """Show BAC on a reversed-log scale with BAC-valued tick labels."""
    ax.set_yscale("log")
    ax.invert_yaxis()
    error_ticks = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01, ERR_MIN]
    ax.yaxis.set_major_locator(FixedLocator(error_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{1-v:.2f}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(ERR_MAX, ERR_MIN)


def plot_row_bac_curves(summary: pd.DataFrame, baseline: pd.DataFrame, row: str, ax: plt.Axes) -> None:
    """Plot BAC mean ± std for one row across base sizes."""
    row_data = summary[summary["row"] == row].copy()
    row_data = row_data[row_data["base_size"] > 0]
    base_sizes = sorted(row_data["base_size"].unique())

    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(base_sizes)))
    for i, bs in enumerate(base_sizes):
        sub = row_data[row_data["base_size"] == bs].sort_values("total_labels")
        if len(sub) == 0:
            continue
        mean_bac = sub["bac_mean"].values
        std_bac = sub["bac_std"].values
        ax.plot(sub["total_labels"], _bac_to_log_error(mean_bac), "-o", markersize=3,
                color=cmap[i], label=f"B={bs}")
        ax.fill_between(
            sub["total_labels"],
            _bac_to_log_error(mean_bac + std_bac),
            _bac_to_log_error(mean_bac - std_bac),
            alpha=0.12, color=cmap[i],
        )

    if len(baseline) > 0:
        bl_agg = baseline.groupby("total_labels")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
        mean_bac = bl_agg["mean"].values
        std_bac = bl_agg["std"].values
        ax.plot(bl_agg["total_labels"], _bac_to_log_error(mean_bac), "--", color="gray",
                linewidth=1.5, label="Random baseline")
        ax.fill_between(
            bl_agg["total_labels"],
            _bac_to_log_error(mean_bac + std_bac),
            _bac_to_log_error(mean_bac - std_bac),
            alpha=0.1, color="gray",
        )

    row_label = ROW_LABELS.get(row, row)
    ax.set_title(f"{row_label} (50k pool)")
    ax.set_xlabel("Total Labels (T)")
    ax.set_ylabel("Balanced Accuracy")
    _format_log_yaxis(ax)
    ax.legend(ncol=2, loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_combined_side_by_side(summary: pd.DataFrame, baseline: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_row_bac_curves(summary, baseline, ROW1, ax1)
    plot_row_bac_curves(summary, baseline, ROW2, ax2)
    fig.suptitle("Dense-Pool (50k) Label Efficiency — Balanced Accuracy", fontsize=12)
    plt.tight_layout()
    return fig


def plot_paired_delta_heatmap(results_dir: Path) -> plt.Figure | None:
    deltas_path = results_dir / "paired_deltas.csv"
    if not deltas_path.exists():
        return None
    df = pd.read_csv(deltas_path)
    if "delta_balanced_accuracy" not in df.columns:
        return None

    df = df[df["base_size"] > 0].copy()
    pivot = df.groupby(["base_size", "total_labels"])["delta_balanced_accuracy"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([int(r) for r in pivot.index])
    ax.set_xlabel("Total Labels (T)")
    ax.set_ylabel("Base Size (B)")
    ax.set_title("BAC Delta: Row2 − Row1 (50k pool)")
    plt.colorbar(im, ax=ax, label="ΔBAC")
    plt.tight_layout()
    return fig


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not (_RESULTS_DIR / "summary_stats.csv").exists():
        print("[plot] No summary_stats.csv found. Run aggregate_results.py first.")
        return

    traj, summary, baseline = _load_data()
    print(f"[plot] Loaded {len(traj)} trajectory rows, {len(baseline)} baseline rows")

    fig = plot_combined_side_by_side(summary, baseline)
    fig.savefig(_FIGURES_DIR / "bac_curves_combined.png")
    plt.close(fig)
    print(f"[plot] Saved bac_curves_combined.png")

    for row in [ROW1, ROW2]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_row_bac_curves(summary, baseline, row, ax)
        fig.savefig(_FIGURES_DIR / f"bac_curves_{row}.png")
        plt.close(fig)
        print(f"[plot] Saved bac_curves_{row}.png")

    fig = plot_paired_delta_heatmap(_RESULTS_DIR)
    if fig is not None:
        fig.savefig(_FIGURES_DIR / "paired_delta_heatmap.png")
        plt.close(fig)
        print(f"[plot] Saved paired_delta_heatmap.png")

    print("[plot] Done.")


if __name__ == "__main__":
    main()
