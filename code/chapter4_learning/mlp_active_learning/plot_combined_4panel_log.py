"""Generate a 6-panel comparison: 10k, 22k, and 50k pools, both rows, log scale.

Layout (3x2):
  Row 0: 10k pool — both label-blind rows
  Row 1: 22k pool — both label-blind rows
  Row 2: 50k pool — both label-blind rows

Uses the 'reversed-log' scale from the original task22 plotting code:
  - Plots 1-BAC on a log scale, inverts y-axis
  - Tick labels show actual BAC values
  - Visually expands the near-1.0 region

Usage:
    python plot_combined_4panel_log.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import FIGURES_DIR, MLP_FROZEN_22K, frozen_results_dir  # noqa: E402

_RESULTS_10K = frozen_results_dir("dense10k_v1_labelblind")
_RESULTS_50K = frozen_results_dir("dense50k_v2_labelblind")
_FIGURES_DIR = FIGURES_DIR

BASE_SIZES = [250, 500, 750, 1000, 1500, 2000, 2500]

ROW_LABELS = {
    "row1_uncertainty_disagreement": "Uncertainty + Disagreement",
    "row2_diverse_uncertainty": "Diverse Uncertainty",
}

BAC_MIN = 0.70
BAC_MAX = 1.00
ERR_MIN = 1e-3  # log scale cannot display y=0, so this tick is labelled as BAC=1.00.
ERR_MAX = 1.0 - BAC_MIN


def _get_base_colors(n: int = 8):
    cmap = plt.get_cmap("viridis", n)
    return [cmap(i) for i in range(n)]


def _format_log_yaxis(ax):
    """Transform y-axis to show BAC on a 'reversed-log' scale.

    Plots 1-BAC on a log scale and inverts the axis so that:
      - Higher BAC is at the top
      - Spacing near 1.0 is visually expanded (logarithmic stretching)
      - Tick labels show the actual BAC values
    """
    ax.set_yscale("log")
    ax.invert_yaxis()
    error_ticks = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01, ERR_MIN]
    ax.yaxis.set_major_locator(FixedLocator(error_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{1-v:.2f}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(ERR_MAX, ERR_MIN)


def _bac_to_log_error(values: np.ndarray) -> np.ndarray:
    """Convert BAC values to clipped log-error coordinates."""
    clipped_bac = np.clip(values, BAC_MIN, BAC_MAX)
    return np.clip(1.0 - clipped_bac, ERR_MIN, ERR_MAX)


def _load_22k_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    agg_path = MLP_FROZEN_22K / "plot_series_agg.csv"
    base_path = MLP_FROZEN_22K / "baseline_random_summary.csv"
    df_agg = pd.read_csv(agg_path)
    df_baseline = pd.read_csv(base_path) if base_path.exists() else None
    return df_agg, df_baseline


def _load_dense_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    summary_path = results_dir / "summary_stats.csv"
    baseline_path = results_dir / "baseline_metrics.csv"

    df_agg = pd.read_csv(summary_path)
    df_agg = df_agg.rename(columns={"bac_mean": "mean_bac", "bac_std": "std_bac"})

    df_baseline = None
    if baseline_path.exists():
        bl = pd.read_csv(baseline_path)
        bl_agg = bl.groupby("total_labels")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
        bl_agg.columns = ["total_labels", "mean_bac", "std_bac"]
        df_baseline = bl_agg

    return df_agg, df_baseline


def _load_10k_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return _load_dense_data(_RESULTS_10K)


def _load_50k_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return _load_dense_data(_RESULTS_50K)


def _plot_one_panel(
    ax, df_agg: pd.DataFrame, df_baseline: pd.DataFrame | None, row: str
):
    """Plot one panel: BAC curves for all base sizes in log mode."""
    colors = _get_base_colors(len(BASE_SIZES))
    df_row = df_agg[df_agg["row"] == row].copy()

    for i, B in enumerate(BASE_SIZES):
        df_b = df_row[df_row["base_size"] == B].sort_values("total_labels")
        if df_b.empty:
            continue
        T = df_b["total_labels"].values
        mean_bac = df_b["mean_bac"].values
        std_bac = df_b["std_bac"].values

        y = _bac_to_log_error(mean_bac)
        y_lo = _bac_to_log_error(mean_bac + std_bac)
        y_hi = _bac_to_log_error(mean_bac - std_bac)
        ax.plot(T, y, "o-", color=colors[i], lw=1.5, markersize=3, label=f"B={B}")
        ax.fill_between(T, y_lo, y_hi, alpha=0.08, color=colors[i])

    if df_baseline is not None:
        df_b = df_baseline.sort_values("total_labels")
        T = df_b["total_labels"].values
        mean_bac = df_b["mean_bac"].values
        std_bac = df_b["std_bac"].values
        y = _bac_to_log_error(mean_bac)
        y_lo = _bac_to_log_error(mean_bac + std_bac)
        y_hi = _bac_to_log_error(mean_bac - std_bac)
        ax.plot(T, y, "s--", color="grey", lw=2, markersize=4,
                alpha=0.8, label="Random baseline")
        ax.fill_between(T, y_lo, y_hi, alpha=0.12, color="grey")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3700)


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[6panel] Loading 10k data...")
    df_10k, bl_10k = _load_10k_data()
    print("[6panel] Loading 22k data...")
    df_22k, bl_22k = _load_22k_data()
    print("[6panel] Loading 50k data...")
    df_50k, bl_50k = _load_50k_data()

    fig, axes = plt.subplots(3, 2, figsize=(16, 16), sharey=True, sharex=True)

    rows = ["row1_uncertainty_disagreement", "row2_diverse_uncertainty"]
    panel_data = [
        ("10k Pool", df_10k, bl_10k),
        ("22k Pool", df_22k, bl_22k),
        ("50k Pool", df_50k, bl_50k),
    ]

    for row_idx, (pool_label, df_pool, bl_pool) in enumerate(panel_data):
        for col, row in enumerate(rows):
            ax = axes[row_idx, col]
            _plot_one_panel(ax, df_pool, bl_pool, row)
            ax.set_title(f"{pool_label} — {ROW_LABELS[row]}", fontsize=11)
            if row_idx == len(panel_data) - 1:
                ax.set_xlabel("Total Labels (T)", fontsize=11)
            if col == 0:
                ax.set_ylabel("Balanced Accuracy", fontsize=11)
            ax.legend(fontsize=7, ncol=3, loc="lower right")

    # Apply log scale to all panels
    for ax in axes.flatten():
        _format_log_yaxis(ax)

    fig.suptitle(
        "Label Efficiency — 10k vs 22k vs 50k Pool (log scale)",
        fontsize=14, y=0.995
    )
    fig.tight_layout()
    out_path = _FIGURES_DIR / "combined_6panel_log.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[6panel] Saved: {out_path}")


if __name__ == "__main__":
    main()
