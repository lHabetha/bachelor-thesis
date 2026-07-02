"""Export Chapter 6 consolidated reports, tables, and figures from run artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import DATA_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR, CHAPTER6_ROOT


CALIBRATION_RUN = CHAPTER6_ROOT / "runs" / "calibration" / "analytic_calibration_v2_4pitch"
AL_RUN = CHAPTER6_ROOT / "runs" / "active_learning" / "al_scale_v1_3k"
OPT_RUN = CHAPTER6_ROOT / "runs" / "optimization_mlp" / "overlap_repair_al_uncertainty_1750_v1"
ZIGZAG_RUN = CHAPTER6_ROOT / "runs" / "zigzag" / "zigzag_smoke_v2_calibrated"
POOL_RUNS = {
    "pool_100k": DATA_DIR / "pool_100k",
    "holdout_5k": DATA_DIR / "holdout_5k",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    for path in (REPORTS_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _copy_tables() -> None:
    shutil.copyfile(CALIBRATION_RUN / "analytic_calibration_rows.csv", TABLES_DIR / "calibration_rows.csv")
    shutil.copyfile(AL_RUN / "active_learning_log.csv", TABLES_DIR / "active_learning_log.csv")
    shutil.copyfile(OPT_RUN / "summary.csv", TABLES_DIR / "optimizer_comparison_summary.csv")
    shutil.copyfile(ZIGZAG_RUN / "summary.csv", TABLES_DIR / "zigzag_summary.csv")


def _calibration_report() -> None:
    rows = _read_csv(CALIBRATION_RUN / "analytic_calibration_rows.csv")
    summary = _load_json(CALIBRATION_RUN / "summary.json")
    selected = summary["settings"]["pitch_0.5"]
    pitches = sorted({float(r["pitch_mm"]) for r in rows})
    overlap_rows = [r for r in rows if float(r["boolean_raw"]) > 0]
    selected_overlap_rows = [r for r in overlap_rows if float(r["pitch_mm"]) == 0.5]
    mean_boolean_overlap_nonzero = (
        sum(float(r["boolean_raw"]) for r in selected_overlap_rows) / len(selected_overlap_rows)
    )
    mean_abs_error_nonzero = (
        sum(float(r["abs_error_raw"]) for r in selected_overlap_rows) / len(selected_overlap_rows)
    )

    fig, ax1 = plt.subplots(figsize=(6.2, 3.7))
    labels = [str(p).rstrip("0").rstrip(".") for p in pitches]
    mean_abs = []
    wall = []
    for p in pitches:
        pitch_rows = [r for r in overlap_rows if float(r["pitch_mm"]) == p]
        mean_abs.append(sum(float(r["abs_error_raw"]) for r in pitch_rows) / len(pitch_rows))
        wall.append(sum(float(r["analytic_wall_time_s"]) for r in pitch_rows) / len(pitch_rows))
    x = range(len(pitches))
    ax1.bar([v - 0.18 for v in x], mean_abs, width=0.36, label="mean abs. error (mm$^3$)", color="#4c78a8")
    ax1.set_ylabel("Mean abs. error (mm$^3$)")
    ax1.set_xticks(list(x), labels)
    ax1.set_xlabel("Voxel pitch (mm)")
    ax2 = ax1.twinx()
    ax2.bar([v + 0.18 for v in x], wall, width=0.36, label="analytic label time (s)", color="#f58518")
    ax2.set_ylabel("Mean analytic wall time (s)")
    ax1.set_title("Analytic voxel calibration against mesh booleans")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIGURES_DIR / "calibration_error_speed.png", dpi=180)
    plt.close(fig)

    import statistics as _st

    pitch_table_lines = [
        "## Per-Pitch Comparison (overlapping assemblies)",
        "",
        "| Pitch (mm) | Mean abs. error (mm^3) | Median rel. error | Zero/nonzero agreement | Mean analytic time (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(pitches):
        p_over = [r for r in overlap_rows if float(r["pitch_mm"]) == p]
        p_all = [r for r in rows if float(r["pitch_mm"]) == p]
        rel = [float(r["rel_error_nonzero"]) for r in p_all if r["rel_error_nonzero"] not in ("", "None")]
        mae_p = sum(float(r["abs_error_raw"]) for r in p_over) / len(p_over)
        agree_p = sum(1 for r in p_all if r["boolean_nonzero"] == r["analytic_nonzero"]) / len(p_all)
        t_p = sum(float(r["analytic_wall_time_s"]) for r in p_over) / len(p_over)
        rel_med = _st.median(rel) if rel else float("nan")
        pitch_table_lines.append(
            f"| {str(p).rstrip('0').rstrip('.')} | {mae_p:.3f} | {rel_med:.4f} | {agree_p:.4f} | {t_p:.5f} |"
        )
    pitch_table_lines.append("")

    miss_rows = [
        r for r in rows
        if float(r["pitch_mm"]) == 0.5 and r["boolean_nonzero"] != r["analytic_nonzero"]
    ]
    lines = [
        "# SDF/Voxel Label Calibration Summary",
        "",
        "## Selected Backend",
        "",
        "- Primary Chapter 6 acquisition label: `analytic_voxel_clevis_v1`.",
        "- Selected pitch: `0.5 mm`.",
        "- Acquisition/early-optimizer binary threshold: normalized overlap `5e-5`.",
        "- Strict final-repair threshold used in current thesis-facing repair evidence: normalized overlap `1e-6`, with mesh-boolean verification for accepted examples.",
        "- Reason for pivot: mesh-backed signed-distance evaluation was too slow for active learning on this clevis pipeline.",
        "",
        "## Calibration Result",
        "",
        "| Metric at 0.5 mm pitch | Value |",
        "|---|---:|",
        f"| Samples | {summary['n_samples']} |",
        f"| Zero/nonzero agreement | {selected['zero_nonzero_agreement']:.4f} |",
        f"| Mean boolean overlap, nonzero samples | {mean_boolean_overlap_nonzero:.3f} mm^3 |",
        f"| Mean absolute error, nonzero samples | {mean_abs_error_nonzero:.3f} mm^3 |",
        f"| Mean absolute error | {selected['mean_abs_error_raw']:.3f} mm^3 |",
        f"| Median absolute error | {selected['median_abs_error_raw']:.3f} mm^3 |",
        f"| Max absolute error | {selected['max_abs_error_raw']:.3f} mm^3 |",
        f"| Median relative error on nonzero rows | {selected['median_rel_error_nonzero']:.4f} |",
        f"| Mean analytic wall time | {selected['mean_analytic_wall_time_s']:.5f} s/sample |",
        f"| Mean boolean wall time | {selected['mean_boolean_wall_time_s']:.2f} s/sample |",
        "",
        *pitch_table_lines,
        f"Figure: `figures/calibration_error_speed.png`.",
        "",
        "## Known Limitation",
        "",
    ]
    if miss_rows:
        r = miss_rows[0]
        lines.append(
            "The selected pitch missed one tiny boolean-positive case: "
            f"`{r['sample_id']}` had boolean volume `{float(r['boolean_raw']):.3f} mm^3` "
            f"and analytic volume `{float(r['analytic_raw']):.3f} mm^3` at 0.5 mm pitch. "
            "This is why Chapter 6 uses a small nonzero normalized threshold and keeps mesh booleans as the calibration reference."
        )
    else:
        lines.append("No zero/nonzero disagreements occurred at the selected pitch in this calibration run.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m chapter6_overlap.calibrate_analytic --run-id analytic_calibration_v2_4pitch --n 12 --pitch 1.0,0.75,0.5,0.25 --max-points-per-pair 6000000")
    lines.append("```")
    (REPORTS_DIR / "sdf_label_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _active_learning_report() -> None:
    rows = _read_csv(AL_RUN / "active_learning_log.csv")
    summary = _load_json(AL_RUN / "summary.json")

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for strategy, vals in by_strategy.items():
        vals = sorted(vals, key=lambda r: int(r["n_labels"]))
        ax.plot(
            [int(r["n_labels"]) for r in vals],
            [float(r["balanced_accuracy"]) for r in vals],
            marker="o",
            label=strategy.replace("_", " "),
        )
    ax.set_xlabel("Acquired labels")
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_ylim(0.78, 0.92)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Overlap active learning")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "active_learning_bac_curve.png", dpi=180)
    plt.close(fig)

    label_distribution = []
    for strategy in ("random", "uncertainty", "diverse_uncertainty"):
        acquired = _read_csv(AL_RUN / f"acquired_{strategy}.csv")
        labels = Counter(int(r["overlap_binary"]) for r in acquired)
        label_distribution.append({
            "strategy": strategy,
            "n_acquired": len(acquired),
            "clean": labels[0],
            "overlap": labels[1],
            "overlap_rate": f"{labels[1] / len(acquired):.4f}",
        })
    _write_csv(TABLES_DIR / "active_learning_acquired_distribution.csv", label_distribution)

    lines = [
        "# Active-Learning Overlap Summary",
        "",
        "## Protocol",
        "",
        "- Candidate pool: `datasets/chapter6_overlap_clevis/pool_100k/pool.csv` (parameter-only, no overlap labels).",
        "- Evaluation pool: `datasets/chapter6_overlap_clevis/holdout_5k/pool.csv`.",
        "- Held-out labels used for metrics: 1000.",
        "- Acquisition budget: base `250`, batch `250`, total `3000`.",
        "- Labels were acquired on demand through `LabelCache`; the run writes `acquired_<strategy>.csv` for reproducibility.",
        "",
        "## Best Rows",
        "",
        "| Strategy | Best labels | BAC | Accuracy | Overlap recall | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, row in summary["best_by_strategy"].items():
        lines.append(
            f"| {strategy} | {row['n_labels']} | {row['balanced_accuracy']:.4f} | "
            f"{row['accuracy']:.4f} | {row['recall_overlap']:.4f} | {row['roc_auc']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The overlap boundary is learnable from the 13 raw parameters, but the classification result is not as saturated as the main assemblability experiments. Uncertainty acquisition is the strongest row in this run, reaching BAC `0.9040` at 1750 labels; random reaches BAC `0.8858` only at 3000 labels.",
        "",
        "Figure: `figures/active_learning_bac_curve.png`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m chapter6_overlap.active_learning --run-id al_scale_v1_3k --holdout-n 1000 --base 250 --batch 250 --total 3000 --epochs 120",
        "```",
    ])
    (REPORTS_DIR / "active_learning_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optimizer_report() -> None:
    rows = _read_csv(OPT_RUN / "summary.csv")
    methods = sorted({r["method"] for r in rows})
    stats = []
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        stats.append({
            "method": method,
            "successes": sum(r["success"] == "True" for r in vals),
            "n": len(vals),
            "mean_final_overlap_norm": sum(float(r["final_overlap_norm"]) for r in vals) / len(vals),
            "mean_distance": sum(float(r["distance"]) for r in vals) / len(vals),
        })

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar([s["method"].replace("_", "\n") for s in stats], [s["successes"] for s in stats], color=["#54a24b", "#e45756"])
    ax.set_ylabel("Verified successes out of 20")
    ax.set_ylim(0, 20)
    ax.set_title("Overlap repair comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "optimizer_success_comparison.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Overlap Optimizer Comparison",
        "",
        "## Protocol",
        "",
        "- Matched starts: first 20 overlapping rows from `holdout_5k`.",
        "- Verification: calibrated analytic voxel label (`pitch=0.5 mm`, normalized threshold `5e-5`).",
        "- Learned repair: log-overlap MLP regressor trained from the first 1750 persisted uncertainty acquisitions in `al_scale_v1_3k/acquired_uncertainty.csv`, then verifier-gated line search.",
        "- Direct repair: gradient-free coordinate search over the same 13-D parameter space and the same calibrated verifier. This is the documented fallback direct SDF/voxel baseline, not a fully autograd differentiable SDF implementation.",
        "",
        "| Method | Successes | Mean final overlap norm | Mean normalized distance |",
        "|---|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s['method']} | {s['successes']}/{s['n']} | "
            f"{s['mean_final_overlap_norm']:.6g} | {s['mean_distance']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The direct coordinate baseline is stronger on this smoke set, reaching 20/20 verified overlap repairs. The learned regressor repair remains useful as a negative comparator: active-learning classification does not automatically produce a repair-quality gradient landscape.",
        "",
        "Figure: `figures/optimizer_success_comparison.png`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m chapter6_overlap.optimize_overlap --run-id overlap_repair_al_uncertainty_1750_v1 --train-csv runs/active_learning/al_scale_v1_3k/acquired_uncertainty.csv --train-limit 1750 --epochs 160 --n-starts 20 --mlp-steps 80 --direct-rounds 10",
        "```",
    ])
    (REPORTS_DIR / "optimizer_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zigzag_report() -> None:
    rows = _read_csv(ZIGZAG_RUN / "summary.csv")
    counts = {
        "dual_success": sum(r["success"] == "True" for r in rows),
        "overlap_ok": sum(r["final_overlap_ok"] == "True" for r in rows),
        "assemblable": sum(r["final_assemblable"] == "True" for r in rows),
    }

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(["Dual\nsuccess", "Overlap\nOK", "Assemblable"], list(counts.values()), color="#72b7b2")
    ax.set_ylim(0, len(rows))
    ax.set_ylabel(f"Count out of {len(rows)}")
    ax.set_title("Zig-zag final verification")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "zigzag_final_outcomes.png", dpi=180)
    plt.close(fig)

    final_reasons = Counter(r["final_reason"] for r in rows)
    lines = [
        "# Combined Overlap-First / Assemblability-Second Pipeline",
        "",
        "## Protocol",
        "",
        "- Starts: first 20 holdout rows that were both overlapping under the calibrated verifier and kinematically blocked under the exact assemblability predicate.",
        "- Stage 1: direct coordinate overlap repair.",
        "- Stage 2: coordinate assemblability-margin repair with overlap guardrails.",
        "- Final verification: overlap norm <= `5e-5`, exact kinematic assemblability true, and relaxed Chapter 6 hard constraints satisfied.",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Final dual success | {counts['dual_success']}/{len(rows)} |",
        f"| Final overlap OK | {counts['overlap_ok']}/{len(rows)} |",
        f"| Final assemblable | {counts['assemblable']}/{len(rows)} |",
        "",
        "## Final Assemblability Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in final_reasons.items():
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a smoke-scale but fully dual-verified end-to-end run. It demonstrates that the direct overlap repair baseline can be composed with a guarded exact-assemblability repair stage on the selected starts. It does not yet measure cycling frequency on a large benchmark.",
        "",
        "Figure: `figures/zigzag_final_outcomes.png`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m chapter6_overlap.zigzag_pipeline --run-id zigzag_smoke_v2_calibrated --n-starts 20 --overlap-rounds 10 --assemblability-rounds 12",
        "```",
    ])
    (REPORTS_DIR / "combined_pipeline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _enhance_pool_manifests() -> None:
    hard = ["A3", "B1", "B2_task25", "B3", "B4", "B5", "C1_task25", "D1", "D2", "D3", "E1", "E2_overhang", "E3", "C1_inner_gap"]
    relaxed = ["A1", "A2", "D4", "D5", "D6", "D7"]
    for name, folder in POOL_RUNS.items():
        manifest_path = folder / "manifest.json"
        manifest = _load_json(manifest_path)
        rows = _read_csv(folder / "pool.csv")
        manifest.update({
            "generation_command": f"python -m chapter6_overlap.pool --name {name} --n {manifest['n']} --seed {manifest['seed']} --prefix {manifest['prefix']}",
            "schema_columns": list(rows[0]),
            "hard_constraints_enforced": hard,
            "relaxed_constraints_recorded": relaxed,
            "modified_constraints": {
                "B2_task25": "main_hole_offset_from_open_end + main_hole_radius <= 0.99 * leg_length",
                "C1_task25": "outer_span > wall_thickness + EPS_ACCESS",
                "C1_inner_gap": "outer_span - 2 * wall_thickness > 0",
            },
            "stream_counts": dict(Counter(r["stream"] for r in rows)),
            "relaxed_violation_counts": dict(Counter(v for r in rows for v in r["relaxed_violations"].split("|") if v)),
            "label_policy": "No overlap labels are stored in this pool; labels are generated on demand by LabelCache.",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (folder / "README.md").write_text(
            "# Chapter 6 Params-Only Candidate Pool\n\n"
            f"- Rows: `{manifest['n']}`\n"
            f"- Seed: `{manifest['seed']}`\n"
            f"- Prefix: `{manifest['prefix']}`\n"
            "- Contains overlap labels: `false`\n"
            f"- Sampler version: `{manifest['sampler_version']}`\n\n"
            "Labels must be acquired on demand through the label cache. The CSV includes stream names, intended relaxed-rule families, hard-rule diagnostics, and margins for the relaxed/modified constraints.\n\n"
            "See `manifest.json` for the schema, stream counts, hard constraints, relaxed constraints, and reproduction command.\n",
            encoding="utf-8",
        )


def _write_top_level_readmes() -> None:
    (FIGURES_DIR / "README.md").write_text(
        "# Chapter 6 Figures\n\n"
        "- `calibration_error_speed.png` - generated from `runs/calibration/analytic_calibration_v1_n12/summary.json`.\n"
        "- `active_learning_bac_curve.png` - generated from `runs/active_learning/al_scale_v1_3k/active_learning_log.csv`.\n"
        "- `optimizer_success_comparison.png` - generated from `runs/optimization_mlp/overlap_repair_al_uncertainty_1750_v1/summary.csv`.\n"
        "- `zigzag_final_outcomes.png` - generated from `runs/zigzag/zigzag_smoke_v2_calibrated/summary.csv`.\n",
        encoding="utf-8",
    )
    (TABLES_DIR / "README.md").write_text(
        "# Chapter 6 Tables\n\n"
        "This folder contains thesis/report-facing CSV exports copied or derived from the current Chapter 6 evidence runs.\n\n"
        "- `calibration_rows.csv`\n"
        "- `active_learning_log.csv`\n"
        "- `active_learning_acquired_distribution.csv`\n"
        "- `optimizer_comparison_summary.csv`\n"
        "- `zigzag_summary.csv`\n",
        encoding="utf-8",
    )


def main() -> None:
    _ensure_dirs()
    _copy_tables()
    _calibration_report()
    _active_learning_report()
    _optimizer_report()
    _zigzag_report()
    _enhance_pool_manifests()
    _write_top_level_readmes()
    print(REPORTS_DIR)


if __name__ == "__main__":
    main()
