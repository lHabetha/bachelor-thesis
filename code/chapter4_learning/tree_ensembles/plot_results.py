"""Generate Chapter 4 thesis-support plots."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lib.paths import FIGURES_DIR, RESULTS_DIR, ensure_dirs

plt.rcParams.update(
    {
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load() -> pd.DataFrame:
    path = RESULTS_DIR / "summary_by_group.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run aggregate_results.py first: {path}")
    return pd.read_csv(path)


def _plot_random(df: pd.DataFrame) -> None:
    d = df[(df["row"] == "random_baseline") & (df["base_size"] == 0)].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model_id, g in d.groupby("model_id"):
        g = g.sort_values("total_labels")
        y = g["balanced_accuracy_mean"].to_numpy()
        s = g["balanced_accuracy_std"].fillna(0).to_numpy()
        x = g["total_labels"].to_numpy()
        ax.plot(x, y, marker="o", ms=3, lw=1.6, label=model_id)
        ax.fill_between(x, y - s, y + s, alpha=0.12)
    ax.set_title("Random-label baseline on dense 50k pool")
    ax.set_xlabel("Training labels")
    ax.set_ylabel("Holdout balanced accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "random_baseline_bac_linear.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model_id, g in d.groupby("model_id"):
        g = g.sort_values("total_labels")
        err = np.maximum(1.0 - g["balanced_accuracy_mean"].to_numpy(), 1e-4)
        ax.plot(g["total_labels"], err, marker="o", ms=3, lw=1.6, label=model_id)
    ax.set_title("Random-label baseline error view")
    ax.set_xlabel("Training labels")
    ax.set_ylabel("1 - balanced accuracy")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES_DIR / "random_baseline_bac_log_error.png")
    plt.close(fig)


def _plot_active_by_model(df: pd.DataFrame) -> None:
    rows = ["al_uncertainty_disagreement", "al_diverse_uncertainty"]
    d = df[df["row"].isin(rows)].copy()
    if d.empty:
        return
    for model_id, model_df in d.groupby("model_id"):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        for ax, row in zip(axes, rows):
            rd = model_df[model_df["row"] == row]
            for base, g in rd.groupby("base_size"):
                g = g.sort_values("total_labels")
                y = g["balanced_accuracy_mean"].to_numpy()
                s = g["balanced_accuracy_std"].fillna(0).to_numpy()
                x = g["total_labels"].to_numpy()
                ax.plot(x, y, marker="o", ms=2.5, lw=1.3, label=f"B={base}")
                ax.fill_between(x, y - s, y + s, alpha=0.08)
            ax.set_title(row)
            ax.set_xlabel("Training labels")
            ax.grid(alpha=0.3)
            ax.set_ylim(0.45, 1.02)
        axes[0].set_ylabel("Holdout balanced accuracy")
        axes[1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Active-learning BAC: {model_id}")
        fig.savefig(FIGURES_DIR / f"active_bac_linear__{model_id}.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
        for ax, row in zip(axes, rows):
            rd = model_df[model_df["row"] == row]
            for base, g in rd.groupby("base_size"):
                g = g.sort_values("total_labels")
                err = np.maximum(1.0 - g["balanced_accuracy_mean"].to_numpy(), 1e-4)
                ax.plot(g["total_labels"], err, marker="o", ms=2.5, lw=1.3, label=f"B={base}")
            ax.set_title(row)
            ax.set_xlabel("Training labels")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("1 - balanced accuracy")
        axes[1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Active-learning error view: {model_id}")
        fig.savefig(FIGURES_DIR / f"active_bac_log_error__{model_id}.png")
        plt.close(fig)


def _plot_compute(df: pd.DataFrame) -> None:
    if "fit_wall_s_mean" not in df.columns:
        return
    d = df[df["row"] == "random_baseline"].copy()
    if d.empty:
        return
    final_t = d["total_labels"].max()
    d = d[d["total_labels"] == final_t].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(d["fit_wall_s_mean"], d["balanced_accuracy_mean"], s=55)
    for _, row in d.iterrows():
        ax.annotate(row["model_id"], (row["fit_wall_s_mean"], row["balanced_accuracy_mean"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit time at final random budget (s, log)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Compute vs BAC at T={int(final_t)} random labels")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(FIGURES_DIR / "compute_vs_bac_random_final.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args
    ensure_dirs()
    df = _load()
    _plot_random(df)
    _plot_active_by_model(df)
    _plot_compute(df)
    print(f"[task30 plots] wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
