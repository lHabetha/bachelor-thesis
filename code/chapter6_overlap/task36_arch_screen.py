"""Chapter 6 Delta 2 — plain MLP architecture screen on random labels.

Trains five feed-forward overlap regressors of increasing capacity on the SAME
random-labelled training set (no active acquisition, no seeds, no ensembling) and
selects the best by log-scaled holdout error (``mae_log``). The winner is exported
as ``overlap_regressor_regression_v3_selected`` for the Chapter 6 active-learning
comparison and strict-repair rerun.

This is the reordered, expanded version of ``regression_architectures.py``:
architecture is chosen FIRST, on random labels only, and the headline metric is
``mae_log`` rather than ``mae_norm``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ._public_helpers import FEATURES, _features, _read_csv  # noqa: E402
from .label_cache import DEFAULT_THRESHOLD_NORM  # noqa: E402
from .models import predict_overlap_norm, train_overlap_regressor  # noqa: E402
from .paths import FIGURES_DIR, MODELS_DIR, RUNS_DIR  # noqa: E402
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import PAIR_COLUMNS, regression_metrics  # noqa: E402

# Five increasing-capacity architectures (Q2 default set).
ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "a_64_32": (64, 32),
    "b_128_64": (128, 64),
    "c_128_64_32": (128, 64, 32),
    "d_256_128_64": (256, 128, 64),
    "e_256_128_64_32": (256, 128, 64, 32),
}


def _target(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["total_overlap_norm"]) for r in rows], dtype=np.float32)


def _pair_targets(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[float(r[f"pair_norm_{name}"]) for name in PAIR_COLUMNS] for r in rows],
        dtype=np.float32,
    )


def _dominant_pairs(rows: list[dict]) -> list[str]:
    return [str(r["dominant_pair"]) for r in rows]


def _param_count(hidden: tuple[int, ...], in_dim: int = len(FEATURES)) -> int:
    total = 0
    prev = in_dim
    for width in hidden:
        total += prev * width + width
        prev = width
    total += prev * 1 + 1
    return total


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_standardizer(path: Path, std) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=std.mean_, scale=std.scale_, features=np.array(FEATURES))


def _plot(rows: list[dict], out_png: Path) -> None:
    names = [r["architecture"] for r in rows]
    mae_log = [float(r["mae_log"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(names, mae_log, color="#4c78a8")
    ax.set_ylabel("Holdout log-scaled overlap error (MAE$_{\\log}$)")
    ax.set_xlabel("MLP architecture (hidden layers)")
    ax.set_title("Architecture screen on random labels")
    ax.grid(True, axis="y", alpha=0.25)
    for rect, val in zip(bars, mae_log):
        ax.annotate(
            f"{val:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(FIGURES_DIR / out_png.name, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    run_dir = RUNS_DIR / "models" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_csv(args.train_csv)[: args.budget]
    holdout_rows = _read_csv(args.holdout_labeled_csv)
    train_x = _features(train_rows)
    train_y = _target(train_rows)
    holdout_x = _features(holdout_rows)
    holdout_y = _target(holdout_rows)
    holdout_pair = _pair_targets(holdout_rows)
    dominant = _dominant_pairs(holdout_rows)

    rows: list[dict] = []
    best: dict | None = None
    best_state: tuple | None = None
    for idx, (arch_name, hidden) in enumerate(ARCHITECTURES.items()):
        model, std = train_overlap_regressor(
            train_x,
            train_y,
            threshold_norm=DEFAULT_THRESHOLD_NORM,
            hidden=hidden,
            seed=args.seed + idx,
            epochs=args.epochs,
        )
        pred = predict_overlap_norm(model, std, holdout_x, threshold_norm=DEFAULT_THRESHOLD_NORM)
        metrics = regression_metrics(
            holdout_y,
            pred,
            pair_true=holdout_pair,
            dominant_pairs=dominant,
        )
        row = {
            "architecture": arch_name,
            "hidden": "-".join(str(v) for v in hidden),
            "param_count": _param_count(hidden),
            "budget": int(len(train_rows)),
            "label_source": "random",
            "train_overlap_mean": float(train_y.mean()),
            "train_overlap_median": float(np.median(train_y)),
            "train_overlap_positive_rate": float(np.mean(train_y > DEFAULT_THRESHOLD_NORM)),
            **metrics,
        }
        rows.append(row)
        print(
            f"[{arch_name:>16}] mae_log={row['mae_log']:.5f} mae_norm={row['mae_norm']:.6g} "
            f"near_thr={row['near_threshold_mae_norm']:.6g}"
        )
        if best is None or float(row["mae_log"]) < float(best["mae_log"]):
            best = row
            best_state = (model.state_dict(), std, hidden, arch_name)

    assert best is not None and best_state is not None
    sel_dir = MODELS_DIR / args.selected_dir
    sel_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state[0], sel_dir / "model_state.pt")
    _save_standardizer(sel_dir / "standardizer.npz", best_state[1])
    (sel_dir / "architecture.json").write_text(
        json.dumps(
            {
                "architecture": best_state[3],
                "hidden": list(best_state[2]),
                "budget": int(args.budget),
                "selected_by": "mae_log",
                "label_source": "random",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sel_dir / "model_card.md").write_text(
        f"# Overlap Regressor — Chapter 6 selected (v3)\n\n"
        f"- Architecture: `{best_state[3]}` hidden `{best_state[2]}`\n"
        f"- Selected by: lowest holdout `mae_log`\n"
        f"- Training labels: `{args.budget}` random-labelled rows from `{args.train_csv.name}`\n"
        f"- Target: `log1p(total_overlap_norm / {DEFAULT_THRESHOLD_NORM})`, SmoothL1 loss\n"
        f"- Holdout `mae_log`: `{best['mae_log']:.5f}`, `mae_norm`: `{best['mae_norm']:.6g}`\n",
        encoding="utf-8",
    )

    _write_csv(run_dir / "architecture_metrics.csv", rows)
    _plot(rows, run_dir / "task25_regression_architecture_comparison_v3.png")

    summary = {
        "run_id": args.run_id,
        "train_csv": str(args.train_csv),
        "holdout_labeled_csv": str(args.holdout_labeled_csv),
        "budget": int(args.budget),
        "epochs": args.epochs,
        "headline_metric": "mae_log",
        "architectures": {k: list(v) for k, v in ARCHITECTURES.items()},
        "best_by_mae_log": best,
        "selected_model_dir": str(sel_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"selected: {best['architecture']} ({best['hidden']}) mae_log={best['mae_log']:.5f}")
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="mlp_arch_regression_v3")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ACQUIRED_RANDOM_15K,
    )
    parser.add_argument(
        "--holdout-labeled-csv",
        type=Path,
        default=HOLDOUT_LABELED_5K,
    )
    parser.add_argument("--budget", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=360540)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--selected-dir", default="overlap_regressor_regression_v3_selected")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
