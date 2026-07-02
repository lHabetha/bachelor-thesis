"""Plot architecture-dependent active-learning row comparison for Chapter 4.

Generates a multi-panel heatmap of diverse-minus-uncertainty BAC deltas and a
summary bar chart of mean deltas per model family.

Usage:
    python plot_al_row_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.paths import FIGURES_DIR, RESULTS_DIR, ensure_dirs

ROW_UNC = "al_uncertainty_disagreement"
ROW_DIV = "al_diverse_uncertainty"
MODEL_ORDER = [
    ("mlp64", "MLP64"),
    ("xgboost", "XGBoost"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
    ("extra_trees", "Extra Trees"),
    ("random_forest", "Random Forest"),
]


def _load_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "summary_by_group.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run aggregate_results.py first: {path}")
    return pd.read_csv(path)


def _delta_pivot(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    sub = df[(df["model_id"] == model_id) & (df["row"].isin([ROW_UNC, ROW_DIV]))].copy()
    wide = sub.pivot_table(
        index=["base_size", "total_labels"],
        columns="row",
        values="balanced_accuracy_mean",
        aggfunc="mean",
    )
    delta = wide[ROW_DIV] - wide[ROW_UNC]
    return delta.unstack("total_labels")


def _best_cells(df: pd.DataFrame, model_id: str) -> tuple[pd.Series, pd.Series]:
    sub = df[(df["model_id"] == model_id) & (df["row"].isin([ROW_UNC, ROW_DIV]))].copy()
    best = {}
    for row in [ROW_UNC, ROW_DIV]:
        row_sub = sub[sub["row"] == row]
        best[row] = row_sub.loc[row_sub["balanced_accuracy_mean"].idxmax()]
    return best[ROW_UNC], best[ROW_DIV]


def main() -> None:
    ensure_dirs()
    df = _load_summary()
    pivots = [_delta_pivot(df, model_id) for model_id, _ in MODEL_ORDER]

    all_values = np.concatenate([p.to_numpy(dtype=float).ravel() for p in pivots])
    all_values = all_values[np.isfinite(all_values)]
    vmax = max(0.01, float(np.nanmax(np.abs(all_values))))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.28)
    top = gs[0].subgridspec(2, 3, hspace=0.25, wspace=0.22)
    bottom = gs[1].subgridspec(1, 1)

    heat_axes = [fig.add_subplot(top[i // 3, i % 3]) for i in range(5)]
    last_im = None
    for ax, (model_id, label), pivot in zip(heat_axes, MODEL_ORDER, pivots):
        data = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        last_im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Total Labels (T)", fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([int(c) for c in pivot.columns], rotation=45, fontsize=6)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([int(r) for r in pivot.index], fontsize=7)
        if ax is heat_axes[0] or ax is heat_axes[3]:
            ax.set_ylabel("Base Size (B)", fontsize=8)

    cbar = fig.colorbar(last_im, ax=heat_axes, shrink=0.85, pad=0.01)
    cbar.set_label("ΔBAC (diverse − uncertainty)")

    ax_bar = fig.add_subplot(bottom[0, 0])
    means = []
    labels = []
    colors = []
    for model_id, label in MODEL_ORDER:
        pivot = _delta_pivot(df, model_id)
        delta = pivot.to_numpy(dtype=float).ravel()
        delta = delta[np.isfinite(delta)]
        means.append(float(np.mean(delta)))
        labels.append(label)
        best_unc, best_div = _best_cells(df, model_id)
        winner = ROW_DIV if best_div["balanced_accuracy_mean"] > best_unc["balanced_accuracy_mean"] else ROW_UNC
        colors.append("#2ca02c" if winner == ROW_DIV else "#1f77b4")

    x = np.arange(len(labels))
    bars = ax_bar.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_bar.axhline(0.0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylabel("Mean ΔBAC\n(diverse − uncertainty)")
    ax_bar.set_title(
        "Summary: green = diverse row wins best cell; blue = uncertainty/disagreement wins best cell",
        fontsize=9,
    )
    ax_bar.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, means):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.001 if val >= 0 else -0.003),
            f"{val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )

    fig.suptitle(
        "Chapter 4 — Architecture-Dependent Active-Learning Row Comparison",
        fontsize=12,
        y=0.98,
    )
    out = FIGURES_DIR / "task30_al_row_delta_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[task30 al-row plot] Saved: {out}")


if __name__ == "__main__":
    main()
