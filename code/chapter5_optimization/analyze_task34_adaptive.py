"""Aggregate Chapter 5 adaptive optimizer results and thesis artifacts."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from shared.paths import (
    ANALYSIS_DIR,
    COMPARISONS_DIR,
    CONSTRAINT_STUDIES_DIR,
    RUNS_DIR,
    TASK26_ROOT,
)


OUT_DIR = COMPARISONS_DIR
ANALYSIS_DIR = ANALYSIS_DIR / "task34_adaptive_multistep"
LOCK_STUDY_DIR = CONSTRAINT_STUDIES_DIR / "parameter_locks_task34_adaptive_v1"
THESIS_IMAGE = COMPARISONS_DIR / "task34_optimizer_tradeoff.png"

MAIN_RUNS = [
    ("Adaptive gradient", "adaptive_gradient", "task34_adaptive_gradient_tau0.60"),
    ("Adaptive gradient", "adaptive_gradient", "task34_adaptive_gradient_tau0.75"),
    ("Adaptive gradient", "adaptive_gradient", "task34_adaptive_gradient_tau0.90"),
    ("Adaptive penalty lambda=0.10", "adaptive_penalty", "task34_adaptive_penalty_lam0_10_tau0.60"),
    ("Adaptive penalty lambda=0.10", "adaptive_penalty", "task34_adaptive_penalty_lam0_10_tau0.75"),
    ("Adaptive penalty lambda=0.10", "adaptive_penalty", "task34_adaptive_penalty_lam0_10_tau0.90"),
]

PENALTY_SWEEP_RUNS = [
    ("Adaptive penalty lambda=0.05", "adaptive_penalty_sweep", "task34_adaptive_penalty_lam0_05_tau0.75"),
    ("Adaptive penalty lambda=0.10", "adaptive_penalty_sweep", "task34_adaptive_penalty_lam0_10_tau0.75"),
    ("Adaptive penalty lambda=0.20", "adaptive_penalty_sweep", "task34_adaptive_penalty_lam0_20_tau0.75"),
    ("Adaptive penalty lambda=0.30", "adaptive_penalty_sweep", "task34_adaptive_penalty_lam0_30_tau0.75"),
    ("Adaptive penalty lambda=0.50", "adaptive_penalty_sweep", "task34_adaptive_penalty_lam0_50_tau0.75"),
]

SPARSE_RUNS = [
    ("Adaptive penalty original L2", "adaptive_sparse", "task34_adaptive_penalty_lam0_10_tau0.75"),
    ("Adaptive penalty L1->L2", "adaptive_sparse", "task34_adaptive_penalty_sparse_l1_l2_tau0.75"),
    ("Adaptive penalty L0->L2", "adaptive_sparse", "task34_adaptive_penalty_sparse_l0_l2_tau0.75"),
    ("Adaptive penalty L0->L1->L2", "adaptive_sparse", "task34_adaptive_penalty_sparse_l0_l1_l2_tau0.75"),
]

BASELINE_RUNS = [
    ("One-shot gradient", "baseline", "one_shot_gradient_bracket_v1_tau0.75_ch5"),
    ("Multi-fixed-step gradient", "baseline", "receding_multiscale_ch5_tau0.75"),
    ("Proximity lambda=0.10", "baseline", "penalized_proximity_tau0.75_labelblind"),
    ("Trust region hybrid", "baseline", "trust_region_tau0.75_labelblind"),
    ("Coordinate-axis bracket", "baseline", "coordinate_axis_tau0.75_labelblind"),
]

LOCK_SCENARIOS = [
    "lock_splint_user",
    "lock_pin_user",
    "lock_main_bracket_user",
    "grad_top_3",
    "grad_bottom_12",
    "grad_bottom_11",
    "grad_bottom_10",
]


def load_stats(run_id: str) -> tuple[dict, dict]:
    run_dir = RUNS_DIR / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stats = json.loads((run_dir / "statistics.json").read_text())
    return manifest, stats


def load_trajectories(run_id: str) -> list[dict]:
    return json.loads((RUNS_DIR / run_id / "trajectories.json").read_text())


def row(label: str, family: str, run_id: str) -> dict:
    manifest, stats = load_stats(run_id)
    sparse = stats.get("sparse_metrics", {})
    return {
        "label": label,
        "family": family,
        "run_id": run_id,
        "optimizer_id": manifest["optimizer_id"],
        "tau": manifest["tau"],
        "constraint_id": manifest.get("constraint_id") or "",
        "verified_ok": stats["oracle_confirmed_count"],
        "surrogate_success": stats["surrogate_success_count"],
        "false_ok": stats["false_success_count"],
        "no_crossing": stats["no_crossing_count"],
        "mean_l2": stats["distances"]["all_mean"],
        "mean_l2_verified": stats["distances"]["oracle_confirmed_mean"],
        "mean_l1": sparse.get("l1_distance_mean", 0.0),
        "active_coords": sparse.get("active_coordinate_count_mean", 0.0),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def subgroup_rows(rows: list[dict]) -> list[dict]:
    out = []
    for base in rows:
        _manifest, stats = load_stats(base["run_id"])
        for subgroup, values in stats.get("subgroups", {}).items():
            out.append({
                "label": base["label"],
                "family": base["family"],
                "run_id": base["run_id"],
                "tau": base["tau"],
                "subgroup": subgroup,
                "count": values.get("count", 0),
                "verified_ok": values.get("oracle_confirmed", 0),
                "surrogate_success": values.get("surrogate_success", 0),
                "mean_l2": values.get("mean_distance", 0.0),
            })
    return out


def step_diagnostic_rows(rows: list[dict]) -> list[dict]:
    out = []
    for base in rows:
        trajectories = load_trajectories(base["run_id"])
        evaluations = [int(t["n_evaluations"]) for t in trajectories]
        magnitudes = [
            float(step["step_magnitude"])
            for traj in trajectories
            for step in traj["steps"]
            if step["step_magnitude"] is not None and float(step["step_magnitude"]) > 0.0
        ]
        valid_steps = [
            step
            for traj in trajectories
            for step in traj["steps"]
            if step["valid"] and step["step_magnitude"] is not None and float(step["step_magnitude"]) > 0.0
        ]
        reasons = [
            reason
            for traj in trajectories
            for step in traj["steps"]
            for reason in step.get("invalid_reasons", [])
        ]
        out.append({
            "label": base["label"],
            "family": base["family"],
            "run_id": base["run_id"],
            "tau": base["tau"],
            "mean_evaluations": statistics.fmean(evaluations) if evaluations else 0.0,
            "median_evaluations": statistics.median(evaluations) if evaluations else 0.0,
            "mean_step_magnitude": statistics.fmean(magnitudes) if magnitudes else 0.0,
            "median_step_magnitude": statistics.median(magnitudes) if magnitudes else 0.0,
            "max_step_magnitude": max(magnitudes) if magnitudes else 0.0,
            "valid_candidate_steps": len(valid_steps),
            "invalid_design_steps": sum(1 for reason in reasons if reason == "invalid_design_space"),
            "backtrack_tag_count": sum(1 for reason in reasons if reason.startswith("adaptive_backtrack_")),
            "bracket_tag_count": sum(1 for reason in reasons if reason == "adaptive_bracket"),
            "sparse_refine_tag_count": sum(1 for reason in reasons if reason == "adaptive_sparse_refine"),
        })
    return out


def markdown_table(rows: list[dict], title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append("| Variant | Tau | Verified OK | False OK | Mean L2 | Mean L1 | Active coords | Run ID |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['tau']:.2f} | {r['verified_ok']}/200 | {r['false_ok']} | "
            f"{r['mean_l2']:.4f} | {r['mean_l1']:.4f} | {r['active_coords']:.2f} | `{r['run_id']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_plot(rows: list[dict]) -> None:
    THESIS_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "#6f6f6f",
        "adaptive_gradient": "#1f77b4",
        "adaptive_penalty": "#d62728",
        "adaptive_penalty_sweep": "#ff7f0e",
    }
    markers = {
        "baseline": "o",
        "adaptive_gradient": "s",
        "adaptive_penalty": "^",
        "adaptive_penalty_sweep": "D",
    }
    plt.figure(figsize=(7.0, 4.6))
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        plt.scatter(
            [r["mean_l2"] for r in subset],
            [r["verified_ok"] for r in subset],
            label=family.replace("_", " "),
            color=colors.get(family, "#333333"),
            marker=markers.get(family, "o"),
            s=58,
            alpha=0.9,
        )
        for r in subset:
            short = r["label"].replace("Adaptive ", "A. ").replace("lambda=", "l=")
            plt.annotate(short, (r["mean_l2"], r["verified_ok"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Mean normalized L2 distance")
    plt.ylabel("Verified OK count")
    plt.title("Chapter 5 optimizer repair-distance trade-off")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(THESIS_IMAGE, dpi=220)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_STUDY_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = [row(*spec) for spec in MAIN_RUNS]
    penalty_rows = [row(*spec) for spec in PENALTY_SWEEP_RUNS]
    sparse_rows = [row(*spec) for spec in SPARSE_RUNS]
    tradeoff_rows = [row(*spec) for spec in BASELINE_RUNS + MAIN_RUNS + PENALTY_SWEEP_RUNS]

    lock_specs = []
    for method_label, run_prefix in [
        ("Adaptive gradient", "task34_adaptive_gradient_tau0.75"),
        ("Adaptive penalty lambda=0.10", "task34_adaptive_penalty_lam0_10_tau0.75"),
    ]:
        for scenario in LOCK_SCENARIOS:
            lock_specs.append((
                f"{method_label} / {scenario}",
                "adaptive_lock",
                f"{run_prefix}_lock_{scenario}",
            ))
    lock_rows = [row(*spec) for spec in lock_specs]
    all_task34_rows = main_rows + penalty_rows + sparse_rows + lock_rows

    write_csv(OUT_DIR / "task34_main_tau_sweep.csv", main_rows)
    write_csv(OUT_DIR / "task34_penalty_sweep.csv", penalty_rows)
    write_csv(OUT_DIR / "task34_sparse_norms.csv", sparse_rows)
    write_csv(OUT_DIR / "task34_tradeoff_points.csv", tradeoff_rows)
    write_csv(OUT_DIR / "task34_subgroup_breakdown.csv", subgroup_rows(all_task34_rows))
    write_csv(ANALYSIS_DIR / "task34_step_diagnostics.csv", step_diagnostic_rows(all_task34_rows))
    write_csv(LOCK_STUDY_DIR / "thesis_ready_summary.csv", lock_rows)

    report = [
        "# Chapter 5 Adaptive Multi-Step Optimizer Analysis",
        "",
        "Selected step-size rule: bold-driver backtracking with validity-aware shrinking, progress-based expansion, and post-crossing bracketing.",
        "",
        markdown_table(main_rows, "Adaptive Tau Sweep"),
        markdown_table(penalty_rows, "Penalty Sweep At Tau 0.75"),
        markdown_table(sparse_rows, "Sparse Adaptive Penalized Selection"),
        markdown_table(lock_rows, "Reduced Adaptive Lock Matrix"),
        "## Tuning Summary",
        "",
        "- Round 1 used a 32-start subgroup-balanced subset.",
        "- Round 2 used all 200 starts through the lightweight tuning harness.",
        "- `bold_backtrack` tied the best full-start verified count at 165/200 with zero false successes and lower mean L2 than the other top-coverage variants.",
        "",
        "## Additional Diagnostics",
        "",
        f"- Subgroup breakdown CSV: `{(OUT_DIR / 'task34_subgroup_breakdown.csv').relative_to(TASK26_ROOT)}`.",
        f"- Step-size and evaluation diagnostics CSV: `{(ANALYSIS_DIR / 'task34_step_diagnostics.csv').relative_to(TASK26_ROOT)}`.",
        "",
        f"Trade-off plot written to `{THESIS_IMAGE.relative_to(PROJECT_ROOT)}`.",
        "",
    ]
    (ANALYSIS_DIR / "task34_analysis_report.md").write_text("\n".join(report))
    (OUT_DIR / "task34_comparison.md").write_text("\n".join(report))
    write_plot(tradeoff_rows)


if __name__ == "__main__":
    main()
