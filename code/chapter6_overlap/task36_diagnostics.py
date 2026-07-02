"""Chapter 6 diagnostics — parity, magnitude-bin error, holdout distribution.

Uses the selected v3 overlap regressor on the fixed holdout to produce three
insight artifacts accepted in the Chapter 6 plan (thesis use not guaranteed):

1. predicted-vs-true parity plot (log--log) with the y=x line;
2. error-by-magnitude-bin table/figure (normalized MAE and log MAE per bin);
3. holdout overlap-distribution summary table.
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

from ._public_helpers import _features, _read_csv  # noqa: E402
from .label_cache import DEFAULT_THRESHOLD_NORM  # noqa: E402
from .models import predict_overlap_norm  # noqa: E402
from .paths import FIGURES_DIR, MODELS_DIR, RUNS_DIR, TABLES_DIR  # noqa: E402
from .release_paths import V3_MODEL_DIR  # noqa: E402
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
)
from .regression_metrics import magnitude_bin, regression_metrics, transformed_target  # noqa: E402
from ._public_helpers import _load_model  # noqa: E402

BIN_ORDER = ("clean_or_below_threshold", "tiny", "small", "moderate", "large")
BIN_LABELS = {
    "clean_or_below_threshold": "clean",
    "tiny": "tiny",
    "small": "small",
    "moderate": "moderate",
    "large": "large",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parity_plot(y_true: np.ndarray, y_pred: np.ndarray, tau: float, out_png: Path) -> None:
    floor = tau / 10.0
    yt = np.clip(y_true, floor, None)
    yp = np.clip(y_pred, floor, None)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(yt, yp, s=6, alpha=0.25, color="#4c78a8", edgecolors="none")
    lim_lo = floor
    lim_hi = max(yt.max(), yp.max()) * 1.3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#333333", lw=1.0, label="$y=x$")
    ax.axvline(tau, color="#e45756", lw=0.8, ls="--", label=r"$\tau_0=5\times10^{-5}$")
    ax.axhline(tau, color="#e45756", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("True normalized overlap")
    ax.set_ylabel("Predicted normalized overlap")
    ax.set_title("Predicted vs. true overlap (holdout)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(FIGURES_DIR / out_png.name, dpi=180)
    plt.close(fig)


def _bin_plot(bin_rows: list[dict], out_png: Path) -> None:
    names = [BIN_LABELS[r["bin"]] for r in bin_rows]
    mae_log = [float(r["mae_log"]) for r in bin_rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(names, mae_log, color="#54a24b")
    ax.set_ylabel("Log-scaled error (MAE$_{\\log}$)")
    ax.set_xlabel("True-overlap magnitude bin")
    ax.set_title("Holdout error by overlap magnitude")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    fig.savefig(FIGURES_DIR / out_png.name, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    out_dir = RUNS_DIR / "diagnostics" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tau = DEFAULT_THRESHOLD_NORM

    model, std, arch = _load_model(args.model_dir)
    holdout = _read_csv(args.holdout_labeled_csv)
    x = _features(holdout)
    y_true = np.array([float(r["total_overlap_norm"]) for r in holdout], dtype=np.float64)
    part_vol = np.array([float(r["total_part_volume_analytic"]) for r in holdout], dtype=np.float64)
    y_pred = predict_overlap_norm(model, std, x.astype(np.float32), threshold_norm=tau).astype(np.float64)

    overall = regression_metrics(y_true, y_pred)
    _parity_plot(y_true, y_pred, tau, out_dir / "task25_overlap_parity_v3.png")

    yt_log = transformed_target(y_true, threshold_norm=tau)
    yp_log = transformed_target(y_pred, threshold_norm=tau)
    abs_err = np.abs(y_pred - y_true)
    log_err = np.abs(yp_log - yt_log)
    bins = np.array([magnitude_bin(v, threshold_norm=tau) for v in y_true])
    bin_rows = []
    for b in BIN_ORDER:
        m = bins == b
        bin_rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "frac": float(m.mean()),
                "mae_norm": float(abs_err[m].mean()) if m.any() else 0.0,
                "mae_log": float(log_err[m].mean()) if m.any() else 0.0,
                "mean_true_norm": float(y_true[m].mean()) if m.any() else 0.0,
            }
        )
    _write_csv(out_dir / "task36_error_by_bin.csv", bin_rows)
    _write_csv(TABLES_DIR / "task36_error_by_bin.csv", bin_rows)
    _bin_plot(bin_rows, out_dir / "task25_overlap_error_by_bin_v3.png")

    q = lambda p: float(np.percentile(y_true, p))  # noqa: E731
    dist_rows = [
        {"statistic": "n", "normalized": float(len(y_true)), "mm3": ""},
        {"statistic": "mean_part_volume", "normalized": "", "mm3": float(part_vol.mean())},
        {"statistic": "median", "normalized": float(np.median(y_true)), "mm3": float(np.median(y_true) * part_vol.mean())},
        {"statistic": "mean", "normalized": float(y_true.mean()), "mm3": float(y_true.mean() * part_vol.mean())},
        {"statistic": "p90", "normalized": q(90), "mm3": float(q(90) * part_vol.mean())},
        {"statistic": "p99", "normalized": q(99), "mm3": float(q(99) * part_vol.mean())},
        {"statistic": "max", "normalized": float(y_true.max()), "mm3": float(y_true.max() * part_vol.mean())},
        {"statistic": "frac_clean_le_tau0", "normalized": float((y_true <= tau).mean()), "mm3": ""},
        {"statistic": "frac_le_1e-3", "normalized": float((y_true <= 1e-3).mean()), "mm3": ""},
    ]
    _write_csv(out_dir / "task36_holdout_distribution.csv", dist_rows)
    _write_csv(TABLES_DIR / "task36_holdout_distribution.csv", dist_rows)

    summary = {
        "run_id": args.run_id,
        "model_dir": str(args.model_dir),
        "model_architecture": arch,
        "tau0": tau,
        "overall_metrics": {k: overall[k] for k in ("mae_log", "rmse_log", "mae_norm", "rmse_norm", "median_ae_norm", "near_threshold_mae_norm", "spearman_norm")},
        "bins": bin_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall_metrics"], indent=2))
    print("bins:")
    for r in bin_rows:
        print(f"  {r['bin']:>24} n={r['n']:>4} mae_log={r['mae_log']:.4f} mae_norm={r['mae_norm']:.6g}")
    print(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="task36_diagnostics_v1")
    parser.add_argument("--model-dir", type=Path, default=V3_MODEL_DIR)
    parser.add_argument(
        "--holdout-labeled-csv",
        type=Path,
        default=HOLDOUT_LABELED_5K,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
