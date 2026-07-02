"""Plot active-learning advantage by total and acquired label counts.

The figure compares each active-learning cell against the passive random
baseline at the same total label budget. The y-axis is the number of labels
acquired after the initial base set, A = T - B.

Usage:
    python plot_al_advantage_heatmap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
from paths import FIGURES_DIR, frozen_results_dir  # noqa: E402

EXPERIMENT_ID = "dense50k_v2_labelblind"
_RESULTS_DIR = frozen_results_dir(EXPERIMENT_ID)
_FIGURES_DIR = FIGURES_DIR

ROWS = [
    ("row1_uncertainty_disagreement", "Uncertainty + Disagreement"),
    ("row2_diverse_uncertainty", "Diverse Uncertainty"),
]


def _load_tables() -> tuple[pd.DataFrame, pd.Series]:
    summary = pd.read_csv(_RESULTS_DIR / "summary_stats.csv")
    baseline = pd.read_csv(_RESULTS_DIR / "baseline_metrics.csv")
    baseline_mean = baseline.groupby("total_labels")["balanced_accuracy"].mean()
    return summary, baseline_mean


def _pivot_advantage(summary: pd.DataFrame, baseline_mean: pd.Series, row: str) -> pd.DataFrame:
    row_df = summary[(summary["row"] == row) & (summary["base_size"] > 0)].copy()
    row_df["acquired_labels"] = row_df["total_labels"] - row_df["base_size"]
    row_df = row_df[row_df["acquired_labels"] >= 0].copy()
    row_df["baseline_bac"] = row_df["total_labels"].map(baseline_mean)
    row_df["delta_bac"] = row_df["bac_mean"] - row_df["baseline_bac"]
    pivot = row_df.pivot_table(
        index="acquired_labels",
        columns="total_labels",
        values="delta_bac",
        aggfunc="mean",
    )
    return pivot.sort_index()


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary, baseline_mean = _load_tables()
    pivots = [_pivot_advantage(summary, baseline_mean, row) for row, _ in ROWS]

    all_values = np.concatenate([p.to_numpy(dtype=float).ravel() for p in pivots])
    all_values = all_values[np.isfinite(all_values)]
    vmax = max(0.01, float(np.nanmax(all_values)))

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="lightgrey")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (pivot, (_, label)) in zip(axes, zip(pivots, ROWS)):
        data = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Total Labels (T)")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([int(i) for i in pivot.index], fontsize=7)
        ax.grid(False)

    axes[0].set_ylabel("Acquired Labels (T - B)")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("BAC advantage over random baseline")
    fig.suptitle("Active-Learning Advantage Over Random Sampling (50k pool)", fontsize=12)
    fig.savefig(_FIGURES_DIR / "al_advantage_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {_FIGURES_DIR / 'al_advantage_heatmap.png'}")


if __name__ == "__main__":
    main()
