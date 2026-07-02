"""Aggregate the Chapter 6.4 repair matrix into thesis tables + a figure (#5f).

Globs every per-start ``summary.csv`` under the (possibly sharded) ch64 run tree,
groups by ``method`` label, and reports the Chapter 6.4 metrics: strict success
/50, mean/median % overlap-volume reduction (including failures), the fraction of
starts past 50 % / 90 % reduction, mean verifier calls, mean active coordinates,
and median final overlap. Wall time is taken from an uncontended timing run when
available (Chapter 6 fairness).

Outputs:

- ``tables/ch64_repair_summary_v1.csv``   -- one row per method.
- ``tables/ch64_repair_per_start_v1.csv``  -- every per-start row (with
  ``pct_overlap_reduction``), concatenated across shards.
- ``figures/ch64_overlap_reduction.png``   -- per-start % reduction distribution by
  method (coloured by backend, matching the thesis table) plus a three-method
  breakdown of mean reduction by contact family (MLP-only vs finish-line vs
  hybrid-lite).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .paths import FIGURES_DIR, TABLES_DIR
from .release_paths import REPAIR_CH64_DIR, REPAIR_CH64_TIMED_DIR

# Display order + pretty labels for the methods we expect.
METHOD_ORDER = [
    "mlp__v3__tauvol0",
    "mlp__v3__tauvol1e-06",
    "mlp__v3__tauvol1e-05",
    "mlp__multitask__tauvol0",
    "mlp__multitask__taubin0.05",
    "mlp__multitask__taubin0.1",
    "mlp__multitask__taubin0.2",
    "hybrid_lite__v3__tauvol0",
    "hybrid_lite_reduced_calls__v3__tauvol0",
    "hybrid_lite__multitask__taubin0.1",
    "finish_line__v3__tauvol0",
    "random_direct__none",
    "axis_direct__none",
]

PRETTY = {
    "mlp__v3__tauvol0": "MLP-only (v3)",
    "mlp__v3__tauvol1e-06": "MLP-only (v3, tau_vol=1e-6)",
    "mlp__v3__tauvol1e-05": "MLP-only (v3, tau_vol=1e-5)",
    "mlp__multitask__tauvol0": "MLP-only (multitask)",
    "mlp__multitask__taubin0.05": "MLP-only (multitask, tau_bin=0.05)",
    "mlp__multitask__taubin0.1": "MLP-only (multitask, tau_bin=0.1)",
    "mlp__multitask__taubin0.2": "MLP-only (multitask, tau_bin=0.2)",
    "hybrid_lite__v3__tauvol0": "Hybrid-lite (v3)",
    "hybrid_lite_reduced_calls__v3__tauvol0": "Hybrid-lite reduced-calls (v3)",
    "hybrid_lite__multitask__taubin0.1": "Hybrid-lite (multitask, tau_bin=0.1)",
    "finish_line__v3__tauvol0": "Finish-line (v3)",
    "random_direct__none": "Random dirs + direct",
    "axis_direct__none": "Coord-axis + direct",
}

# Left panel: methods ordered and grouped to match Table 6.3 (single-head MLP block,
# then the no-surrogate controls). The "(v3)" suffix is dropped because the backend is
# shown by colour/grouping instead.
FIGURE_METHODS = [
    "hybrid_lite__v3__tauvol0",
    "hybrid_lite_reduced_calls__v3__tauvol0",
    "finish_line__v3__tauvol0",
    "mlp__v3__tauvol0",
    "random_direct__none",
    "axis_direct__none",
]
FIG_LABELS = {
    "hybrid_lite__v3__tauvol0": "Hybrid-lite",
    "hybrid_lite_reduced_calls__v3__tauvol0": "Hybrid-lite\nreduced-calls",
    "finish_line__v3__tauvol0": "Finish-line",
    "mlp__v3__tauvol0": "MLP-only",
    "random_direct__none": "Random dirs + direct",
    "axis_direct__none": "Coord-axis + direct",
}
CONTROL_METHODS = {"random_direct__none", "axis_direct__none"}
SURROGATE_COLOR = "#4f7cac"
CONTROL_COLOR = "#c97b3a"

# Right panel: mean reduction by contact family for single-head methods:
# learned direction only, path extrapolation, reduced signed-gradient polish, and
# bounded coordinate polish.
PANEL_B_METHODS = [
    ("mlp__v3__tauvol0", "MLP-only", "#c0392b", "o", "-"),
    ("finish_line__v3__tauvol0", "Finish-line", "#e08a1e", "s", "--"),
    ("hybrid_lite_reduced_calls__v3__tauvol0", "Hybrid-lite reduced-calls", "#7b68b6", "D", ":"),
    ("hybrid_lite__v3__tauvol0", "Hybrid-lite", "#2e7d32", "^", "-."),
]
# Families ordered by MLP-only's mean reduction (descending) so its bars step down.
CATEGORY_ORDER = [
    "pin_splint_cross_overlap",
    "splint_head_wall_candidate",
    "pin_head_or_shaft_wall_candidate",
    "bracket_splint_other",
    "other_overlap",
]
CATEGORY_LABELS = {
    "pin_splint_cross_overlap": "pin/splint\ncross-hole",
    "splint_head_wall_candidate": "splint-head/\nwall",
    "pin_head_or_shaft_wall_candidate": "pin-head/\nshaft-wall",
    "bracket_splint_other": "bracket/\nsplint",
    "other_overlap": "other",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truth(value: object) -> bool:
    return str(value).lower() == "true"


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary in sorted(root.rglob("summary.csv")):
        for row in _read_csv(summary):
            if row.get("method"):
                rows.append(row)
    if rows:
        return rows
    for results_path in sorted(root.rglob("results.json")):
        data = json.loads(results_path.read_text(encoding="utf-8"))
        for row in data:
            if row.get("method"):
                rows.append({k: v for k, v in row.items() if k not in ("trajectory", "final_params")})
    return rows


def _wall_by_method(timed_root: Path) -> dict[str, float]:
    by_method: dict[str, list[float]] = {}
    if not timed_root.exists():
        return {}
    for row in _collect_rows(timed_root):
        by_method.setdefault(row["method"], []).append(_f(row, "wall_time_s"))
    return {m: float(np.mean(v)) for m, v in by_method.items() if v}


def _summarize(rows: list[dict], wall_override: dict[str, float]) -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    out = []
    ordered = [m for m in METHOD_ORDER if m in by_method]
    for method in ordered:
        vals = by_method[method]
        red = np.array([_f(v, "pct_overlap_reduction") for v in vals])
        finals = np.array([_f(v, "final_overlap_norm") for v in vals])
        first = vals[0]
        out.append(
            {
                "method": method,
                "method_pretty": PRETTY.get(method, method),
                "model_kind": first.get("model_kind", ""),
                "tau_vol": first.get("tau_vol", ""),
                "tau_bin": first.get("tau_bin", ""),
                "n": len(vals),
                "strict_success": sum(_truth(v.get("success")) for v in vals),
                "mean_pct_overlap_reduction": float(np.mean(red)),
                "median_pct_overlap_reduction": float(np.median(red)),
                "frac_reduction_gt50": float(np.mean(red > 50.0)),
                "frac_reduction_gt90": float(np.mean(red > 90.0)),
                "mean_final_overlap_norm": float(np.mean(finals)),
                "median_final_overlap_norm": float(np.median(finals)),
                "mean_distance": float(np.mean([_f(v, "distance") for v in vals])),
                "mean_verifier_calls": float(np.mean([_f(v, "verifier_calls") for v in vals])),
                "mean_active_coordinates": float(np.mean([_f(v, "active_coordinates") for v in vals])),
                "mean_wall_time_s_contended": float(np.mean([_f(v, "wall_time_s") for v in vals])),
                "mean_wall_time_s": wall_override.get(method, float(np.mean([_f(v, "wall_time_s") for v in vals]))),
                "wall_time_uncontended": method in wall_override,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0]) if all(set(r) == set(rows[0]) for r in rows) else sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: per-start % reduction distribution, ordered + coloured by backend to
    # match Table 6.3 (single-head MLP surrogate vs no-surrogate controls).
    ax = axes[0]
    fig_methods = [m for m in FIGURE_METHODS if m in by_method]
    data = [[_f(v, "pct_overlap_reduction") for v in by_method[m]] for m in fig_methods]
    labels = [FIG_LABELS.get(m, m) for m in fig_methods]
    bp = ax.boxplot(data, vert=True, showmeans=True, widths=0.6, patch_artist=True)
    for patch, m in zip(bp["boxes"], fig_methods):
        patch.set_facecolor(CONTROL_COLOR if m in CONTROL_METHODS else SURROGATE_COLOR)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("0.15")
    for i, m in enumerate(fig_methods, start=1):
        succ = sum(_truth(v.get("success")) for v in by_method[m])
        n = len(by_method[m])
        ax.text(i, 103, f"{succ}/{n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Overlap volume reduction (%)")
    ax.set_ylim(0, 113)
    ax.axhline(100, color="0.7", lw=0.8, ls="--")
    ax.set_title("Per-start reduction by method (strict success / n on top)", fontsize=10)
    ax.legend(
        handles=[
            Patch(facecolor=SURROGATE_COLOR, alpha=0.55, label="Single-head MLP surrogate"),
            Patch(facecolor=CONTROL_COLOR, alpha=0.55, label="Direct-search control"),
        ],
        fontsize=8,
        loc="lower left",
    )

    # Panel B: mean reduction by contact family for learned direction only,
    # path extrapolation, reduced signed-gradient polish, and bounded coordinate
    # polish. MLP-only's bars step down (family-dependent direction quality);
    # hybrid-lite flattens the group toward complete reduction.
    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.25)
    cats = [
        c for c in CATEGORY_ORDER
        if any(v.get("strict_category") == c for v in by_method.get("mlp__v3__tauvol0", []))
    ]
    x = np.arange(len(cats))
    n_methods = len(PANEL_B_METHODS)
    width = 0.8 / n_methods
    for i, (method, label, color, *_style) in enumerate(PANEL_B_METHODS):
        vals = by_method.get(method, [])
        if not vals:
            continue
        means = [
            float(np.mean([_f(v, "pct_overlap_reduction") for v in vals if v.get("strict_category") == c]))
            for c in cats
        ]
        offset = (i - (n_methods - 1) / 2.0) * width
        ax.bar(x + offset, means, width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c.replace("_", " ")) for c in cats], fontsize=8)
    ax.set_ylabel("Mean overlap reduction (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Reduction by contact family", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    rows = _collect_rows(args.runs_root)
    if not rows:
        raise SystemExit(f"no summary.csv rows under {args.runs_root}")
    wall_override = _wall_by_method(args.timed_root)
    method_rows = _summarize(rows, wall_override)
    _write_csv(TABLES_DIR / "ch64_repair_summary_v1.csv", method_rows)
    _write_csv(TABLES_DIR / "ch64_repair_per_start_v1.csv", rows)
    _make_figure(rows, FIGURES_DIR / "ch64_overlap_reduction.png")
    print(f"methods: {len(method_rows)} | per-start rows: {len(rows)} | wall-timed methods: {len(wall_override)}")
    for r in method_rows:
        print(
            f"  {r['method']:<36} succ {r['strict_success']:>2}/{r['n']:<2} "
            f"meanRed {r['mean_pct_overlap_reduction']:6.1f}% medRed {r['median_pct_overlap_reduction']:6.1f}% "
            f"dist {r['mean_distance']:.4f} verif {r['mean_verifier_calls']:7.1f} coords {r['mean_active_coordinates']:4.1f} "
            f"wall {r['mean_wall_time_s']:.3f}s{'*' if r['wall_time_uncontended'] else ''}"
        )
    print(TABLES_DIR / "ch64_repair_summary_v1.csv")
    return TABLES_DIR / "ch64_repair_summary_v1.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPAIR_CH64_DIR)
    parser.add_argument("--timed-root", type=Path, default=REPAIR_CH64_TIMED_DIR)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
