#!/usr/bin/env python3
"""Plot 100k MLP grid: four variant learning curves + leaderboard."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chapter7_aeroforge.release_paths import FIGURES_DIR, MODEL_GRID_DIR

DEFAULT_AGG = MODEL_GRID_DIR / "results_agg.csv"
DEFAULT_OUT = FIGURES_DIR / "model_grid_100k.png"

CURVE_STYLE = {
    "raw_log_huber": ("tab:gray", "--", "s", "Raw ADV + log-Huber"),
    "eng_log_huber": ("tab:green", "-", "o", "Engineered + log-Huber"),
    "eng_gate_aug_strong": ("tab:blue", "-.", "D", "Engineered + gate loss (λ=0.3)"),
    "eng_multitask_gate_strong": ("tab:orange", ":", "^", "Engineered + 2-head multitask (λ=0.3)"),
}

ANNOTATE_YOFFSET = {
    "raw_log_huber": -14,
    "eng_log_huber": 10,
    "eng_gate_aug_strong": 0,
    "eng_multitask_gate_strong": -28,
}


def _load_agg(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_curve(ax: plt.Axes, rows: list[dict], *, variant_id: str, annotate: bool) -> None:
    color, linestyle, marker, label = CURVE_STYLE[variant_id]
    rows = sorted(rows, key=lambda r: int(r["budget"]))
    xs = np.array([int(r["budget"]) for r in rows], dtype=float)
    ys = np.array([float(r["mae_log_mean"]) for r in rows], dtype=float)
    ystd = np.array([float(r["mae_log_std"]) for r in rows], dtype=float)

    ax.plot(xs, ys, marker=marker, color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.fill_between(xs, ys - ystd, ys + ystd, color=color, alpha=0.14, linewidth=0)

    if annotate:
        yoff = ANNOTATE_YOFFSET.get(variant_id, 0)
        for x, y in zip(xs, ys):
            factor = 10.0**y
            ax.annotate(
                f"{factor:.2f}×",
                (x, y),
                textcoords="offset points",
                xytext=(0, yoff),
                ha="center",
                fontsize=6,
                color=color,
            )


def _leaderboard(rows: list[dict], *, budget: int = 100_000) -> dict:
    at_budget = [r for r in rows if int(r["budget"]) == budget]
    if not at_budget:
        at_budget = sorted(rows, key=lambda r: int(r["budget"]))[-4:]

    def best(metric: str, higher: bool = False) -> dict:
        key = f"{metric}_mean"
        ranked = sorted(at_budget, key=lambda r: float(r[key]), reverse=higher)
        top = ranked[0]
        return {"variant_id": top["variant_id"], "value": float(top[key])}

    return {
        "budget": budget,
        "best_mae_log": best("mae_log"),
        "best_gate_f1": best("gate_f1", higher=True),
        "best_spearman": best("spearman_mm3", higher=True),
        "best_cond_mae_log": best("cond_mae_log"),
        "rows_at_budget": at_budget,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot 100k MLP grid curves")
    ap.add_argument("--agg", type=Path, default=DEFAULT_AGG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(CURVE_STYLE.keys()),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_agg(args.agg)
    if not rows:
        raise SystemExit(f"No rows in {args.agg}")

    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(str(r["variant_id"]), []).append(r)

    fig, ax = plt.subplots(figsize=(12, 7))
    for vid in args.variants:
        if vid not in by_variant:
            print(f"Warning: no rows for variant {vid}")
            continue
        _plot_curve(ax, by_variant[vid], variant_id=vid, annotate=True)

    ax.set_xlabel("Total labels used (nested pool, 10-fold CV)")
    ax.set_ylabel("Held-out MAE_log  (shaded = ±1 std over 3 seeds × 10 folds)")
    ax.set_title(
        "100k overlap surrogate grid — mlp_256_128_64_32\n"
        "Headline metric: MAE_log on z = log10(1 + overlap_mm³ / 1 mm³); random passive learning",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved plot to {args.out}")

    board = _leaderboard(rows)
    board_path = args.out.parent / "leaderboard.json"
    board_path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    print(f"Saved leaderboard to {board_path}")
    print(
        f"@ {board['budget']} labels: "
        f"best MAE_log={board['best_mae_log']['variant_id']} "
        f"({board['best_mae_log']['value']:.4f}), "
        f"best gate F1={board['best_gate_f1']['variant_id']} "
        f"({board['best_gate_f1']['value']:.4f}), "
        f"best Spearman={board['best_spearman']['variant_id']} "
        f"({board['best_spearman']['value']:.4f})"
    )


if __name__ == "__main__":
    main()
