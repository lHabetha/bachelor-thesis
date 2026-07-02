#!/usr/bin/env python3
"""Thesis-grade aggregator for the Chapter 7 optimization grid.

Read-only over `runs/*/viewer_data.json` + `statistics.json`. Emits, under
`optimizer/analysis/<grid_id>/`, tables whose columns match the Chapter 7
thesis tables so a LaTeX `tabular` is a direct transcription:

    thesis_ready_summary.csv     # tidy: one row per (model, optimizer, tau_decide, p*)
    tables/P*_<optimizer>.csv     # per-family tau/p* sweep tables (K.2 P1..P7)
    tables/T1_recovery.csv        # optimizer x model Verified-OK matrix @ tau_decide=1.0
    tables/T2_false_ok.csv        # optimizer x model False-OK matrix
    tables/T3_proximity.csv       # optimizer x model Mean L2 / L1 / L0
    tables/T4_feature_grad.csv    # paired a<->c and b<->d deltas
    tables/T5_all_frontier.csv    # per-model All: union recovery, best-L2/L0, winner histogram
    tables/T6_quicksim.csv        # aero performance preservation: mean/median Delta L/D, CD0, CD over
                                   #   successful repairs, from each run's sim_eval.json (full-VLM quick sim, §J)
    tables/CF_cross_family.csv    # one headline row per optimizer at its best operating point
    summary.md                    # human digest with a 1-paragraph reading per table
    README.md                     # # SOURCE provenance: run dirs, benchmark, checkpoints

Every CSV starts with a `# SOURCE:` comment line, mirroring the `% SOURCE:` lines
atop `thesis/sections/50_surrogate_based_optimization.tex`.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


from chapter7_aeroforge.optimizer.optimizers import CANONICAL_SUITE  # noqa: E402
from chapter7_aeroforge.optimizer.workbench import VARIANTS  # noqa: E402

from chapter7_aeroforge.release_paths import RUNS_DIR, TABLES_DIR

DEFAULT_RUNS_DIR = RUNS_DIR
DEFAULT_ANALYSIS_DIR = TABLES_DIR
OPTIMIZER_ORDER = CANONICAL_SUITE + ["all"]
GROUP_TO_VARIANT = {v["group_id"]: letter for letter, v in VARIANTS.items()}
VARIANT_TO_MODEL = {letter: v["model_id"] for letter, v in VARIANTS.items()}
HEADLINE_TAU = 1.0
HEADLINE_P_STAR = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def discover_runs(runs_dir: Path) -> list[dict]:
    """One record per run dir, with the fields the tables need."""
    runs: list[dict] = []
    for vd_path in sorted(runs_dir.glob("*/viewer_data.json")):
        try:
            vd = json.loads(vd_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stats = vd.get("statistics", {})
        group_id = vd.get("group_id", "")
        sim_agg = None
        sim_path = vd_path.parent / "sim_eval.json"
        if sim_path.exists():
            try:
                sim_agg = (json.loads(sim_path.read_text(encoding="utf-8")) or {}).get("aggregate")
            except (json.JSONDecodeError, OSError):
                sim_agg = None
        runs.append(
            {
                "run_id": vd.get("run_id", vd_path.parent.name),
                "run_dir": vd_path.parent,
                "group_id": group_id,
                "variant": GROUP_TO_VARIANT.get(group_id),
                "model_id": vd.get("model_id"),
                "optimizer_id": vd.get("optimizer_id"),
                "optimizer_label": vd.get("optimizer_label", vd.get("optimizer_id")),
                "benchmark_id": vd.get("benchmark_id"),
                "verification": stats.get("verification"),
                "n_starts": stats.get("n_starts"),
                "stats": stats,
                "sim": sim_agg,
            }
        )
    return runs


def headline_op(stats: dict) -> dict:
    """Verified/False/L2/L1/L0 at the headline operating point (tau_decide=1.0)."""
    sweep = stats.get("sweep") or []
    chosen = None
    for row in sweep:
        if abs(float(row.get("tau_decide_mm3", -1)) - HEADLINE_TAU) < 1e-9:
            ps = row.get("p_star")
            if ps is None or abs(float(ps) - HEADLINE_P_STAR) < 1e-9:
                chosen = row
                break
    if chosen is not None:
        return {
            "verified_ok": chosen.get("verified_clean_count"),
            "false_ok": chosen.get("false_success_count"),
            "coverage": chosen.get("verified_count"),
            "proximity_n": chosen.get("proximity_n", chosen.get("verified_clean_count")),
            "mean_l2": chosen.get("mean_l2"),
            "mean_l1": chosen.get("mean_l1"),
            "mean_l0": chosen.get("mean_l0"),
        }
    # No sweep block (e.g. the All meta-run): use the headline over-successful
    # proximity (mean_l2_recovered) so Mean L2 stays the thesis "over successful"
    # convention, with a fallback to the over-all movement metric for old runs.
    return {
        "verified_ok": stats.get("verified_clean_count"),
        "false_ok": stats.get("false_success_count"),
        "coverage": stats.get("verified_count"),
        "proximity_n": stats.get("n_recovered_for_proximity", stats.get("verified_clean_count")),
        "mean_l2": stats.get("mean_l2_recovered", stats.get("mean_normalized_distance")),
        "mean_l1": stats.get("mean_l1_recovered", stats.get("mean_l1")),
        "mean_l0": stats.get("mean_l0_recovered", stats.get("mean_active_coordinates")),
    }


def by_key(runs: list[dict]) -> dict[tuple[str, str], dict]:
    """Index runs by (variant, optimizer_id); skips legacy/unmapped groups."""
    idx: dict[tuple[str, str], dict] = {}
    for r in runs:
        if r["variant"] is None:
            continue
        idx[(r["variant"], r["optimizer_id"])] = r
    return idx


def _write_csv(path: Path, header: list[str], rows: list[list], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# SOURCE: {source}\n")
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# --- the master tidy table (one row per operating point) --------------------


def master_rows(runs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in runs:
        if r["variant"] is None:
            continue
        stats = r["stats"]
        sweep = stats.get("sweep") or []
        sim = r.get("sim") or {}
        base = {
            "variant": r["variant"],
            "model_id": r["model_id"],
            "optimizer": r["optimizer_id"],
            "method": r["optimizer_label"],
            "benchmark_id": r["benchmark_id"],
            "n_starts": stats.get("n_starts"),
            "run_id": r["run_id"],
            # full-VLM quick-sim deltas over successful repairs (J); same per run
            "sim_n_success": sim.get("n_success"),
            "sim_baseline_mean_LD": sim.get("baseline_mean_L_over_D"),
            "sim_final_mean_LD": sim.get("final_mean_L_over_D"),
            "sim_mean_dLD": sim.get("mean_delta_L_over_D"),
            "sim_median_dLD": sim.get("median_delta_L_over_D"),
            "sim_mean_dCD0": sim.get("mean_delta_CD0"),
            "sim_mean_dCD": sim.get("mean_delta_CD"),
            "sim_final_vlm_ok_rate": sim.get("final_vlm_ok_rate"),
        }
        if sweep:
            for op in sweep:
                rows.append(
                    {
                        **base,
                        "gate": op.get("gate"),
                        "tau_decide_mm3": op.get("tau_decide_mm3"),
                        "p_star": op.get("p_star"),
                        "verified_ok": op.get("verified_clean_count"),
                        "false_ok": op.get("false_success_count"),
                        "coverage_n": op.get("verified_count"),
                        "proximity_n": op.get("proximity_n"),
                        "mean_l2": op.get("mean_l2"),
                        "mean_l1": op.get("mean_l1"),
                        "mean_l0": op.get("mean_l0"),
                        "mean_final_overlap_mm3": op.get("mean_final_overlap_mm3"),
                    }
                )
        else:
            h = headline_op(stats)
            rows.append(
                {
                    **base,
                    "gate": stats.get("binary_p_star") is not None and "binary_and" or "reg",
                    "tau_decide_mm3": stats.get("tau_decide_mm3", HEADLINE_TAU),
                    "p_star": stats.get("binary_p_star"),
                    "verified_ok": h["verified_ok"],
                    "false_ok": h["false_ok"],
                    "coverage_n": h["coverage"],
                    "proximity_n": h.get("proximity_n"),
                    "mean_l2": h["mean_l2"],
                    "mean_l1": h["mean_l1"],
                    "mean_l0": h["mean_l0"],
                    "mean_final_overlap_mm3": stats.get("mean_final_overlap_mm3"),
                }
            )
    return rows


def write_master(rows: list[dict], out: Path, source: str) -> None:
    header = [
        "variant", "model_id", "optimizer", "method", "gate", "tau_decide_mm3", "p_star",
        "verified_ok", "false_ok", "coverage_n", "proximity_n", "mean_l2", "mean_l1", "mean_l0",
        "mean_final_overlap_mm3",
        "sim_n_success", "sim_baseline_mean_LD", "sim_final_mean_LD", "sim_mean_dLD",
        "sim_median_dLD", "sim_mean_dCD0", "sim_mean_dCD", "sim_final_vlm_ok_rate",
        "n_starts", "benchmark_id", "run_id",
    ]
    out_rows = [[r.get(k) for k in header] for r in rows]
    _write_csv(out, header, out_rows, source)


# --- P-tables: per-optimizer tau/p* sweep across model variants -------------


def write_p_tables(runs: list[dict], tables_dir: Path, variants: list[str], source: str) -> list[str]:
    idx = by_key(runs)
    names: list[str] = []
    for i, opt in enumerate(OPTIMIZER_ORDER, 1):
        header = ["variant", "model_id", "gate", "tau_decide_mm3", "p_star",
                  "verified_ok", "false_ok", "coverage_n", "proximity_n", "mean_l2", "mean_l1", "mean_l0"]
        rows: list[list] = []
        for v in variants:
            r = idx.get((v, opt))
            if not r:
                continue
            sweep = r["stats"].get("sweep") or []
            if not sweep:
                h = headline_op(r["stats"])
                rows.append([v, r["model_id"], "-", HEADLINE_TAU, None,
                             h["verified_ok"], h["false_ok"], h["coverage"], h.get("proximity_n"),
                             h["mean_l2"], h["mean_l1"], h["mean_l0"]])
                continue
            for op in sweep:
                rows.append([
                    v, r["model_id"], op.get("gate"), op.get("tau_decide_mm3"), op.get("p_star"),
                    op.get("verified_clean_count"), op.get("false_success_count"), op.get("verified_count"),
                    op.get("proximity_n"), op.get("mean_l2"), op.get("mean_l1"), op.get("mean_l0"),
                ])
        if rows:
            name = f"P{i}_{opt}.csv"
            _write_csv(tables_dir / name, header, rows, source)
            names.append(name)
    return names


# --- T-matrices + CF --------------------------------------------------------


def write_matrices(runs: list[dict], tables_dir: Path, variants: list[str], source: str) -> None:
    idx = by_key(runs)

    def cell(v, opt, field):
        r = idx.get((v, opt))
        if not r:
            return None
        if opt == "all" and field in ("verified_ok",):
            return r["stats"].get("all_verified_clean_count", headline_op(r["stats"])["verified_ok"])
        return headline_op(r["stats"]).get(field)

    # T1 recovery
    header = ["optimizer", *variants]
    rows = [[opt, *[cell(v, opt, "verified_ok") for v in variants]] for opt in OPTIMIZER_ORDER]
    _write_csv(tables_dir / "T1_recovery.csv", header, rows, source)

    # T2 false-OK
    rows = [[opt, *[cell(v, opt, "false_ok") for v in variants]] for opt in OPTIMIZER_ORDER]
    _write_csv(tables_dir / "T2_false_ok.csv", header, rows, source)

    # T3 proximity (L2, L1, L0 per variant)
    header3 = ["optimizer"]
    for v in variants:
        header3 += [f"{v}_L2", f"{v}_L1", f"{v}_L0"]
    rows = []
    for opt in OPTIMIZER_ORDER:
        row = [opt]
        for v in variants:
            row += [cell(v, opt, "mean_l2"), cell(v, opt, "mean_l1"), cell(v, opt, "mean_l0")]
        rows.append(row)
    _write_csv(tables_dir / "T3_proximity.csv", header3, rows, source)

    # T4 feature-grad effect: a<->c, b<->d
    pairs = [("a", "c"), ("b", "d")]
    header4 = ["optimizer", "pair", "d_verified_ok", "d_mean_l2", "d_mean_l0",
               "from_verified_ok", "to_verified_ok"]
    rows = []
    for opt in OPTIMIZER_ORDER:
        for (a, c) in pairs:
            ha = headline_op(idx[(a, opt)]["stats"]) if (a, opt) in idx else None
            hc = headline_op(idx[(c, opt)]["stats"]) if (c, opt) in idx else None
            if not ha or not hc:
                continue

            def d(field):
                x, y = hc.get(field), ha.get(field)
                return (x - y) if (x is not None and y is not None) else None

            rows.append([
                opt, f"{a}->{c}", d("verified_ok"), d("mean_l2"), d("mean_l0"),
                ha.get("verified_ok"), hc.get("verified_ok"),
            ])
    _write_csv(tables_dir / "T4_feature_grad.csv", header4, rows, source)

    # T5 All frontier
    header5 = ["variant", "model_id", "all_verified_clean_count", "best_L2_average",
               "best_L0_average", "top_winner", "winner_histogram"]
    rows = []
    for v in variants:
        r = idx.get((v, "all"))
        if not r:
            continue
        st = r["stats"]
        hist = st.get("winner_histogram", {}) or {}
        top = max(hist, key=hist.get) if hist else None
        rows.append([
            v, r["model_id"], st.get("all_verified_clean_count"),
            st.get("best_L2_average"), st.get("best_L0_average"), top,
            ";".join(f"{k}={hist[k]}" for k in hist),
        ])
    _write_csv(tables_dir / "T5_all_frontier.csv", header5, rows, source)

    # CF cross-family: per optimizer, its best operating point (max verified, tie -> min L2)
    headerCF = ["optimizer", "best_variant", "model_id", "gate", "tau_decide_mm3", "p_star",
                "verified_ok", "false_ok", "mean_l2", "mean_l1", "mean_l0"]
    rows = []
    for opt in OPTIMIZER_ORDER:
        best = None  # (verified_ok, -l2, payload)
        for v in variants:
            r = idx.get((v, opt))
            if not r:
                continue
            sweep = r["stats"].get("sweep") or []
            ops = sweep or [None]
            for op in ops:
                if op is None:
                    h = headline_op(r["stats"])
                    vo, fo = h["verified_ok"], h["false_ok"]
                    l2, l1, l0 = h["mean_l2"], h["mean_l1"], h["mean_l0"]
                    gate, td, ps = "-", HEADLINE_TAU, None
                else:
                    vo, fo = op.get("verified_clean_count"), op.get("false_success_count")
                    l2, l1, l0 = op.get("mean_l2"), op.get("mean_l1"), op.get("mean_l0")
                    gate, td, ps = op.get("gate"), op.get("tau_decide_mm3"), op.get("p_star")
                if vo is None:
                    continue
                key = (vo, -(l2 if l2 is not None else 1e9))
                if best is None or key > best[0]:
                    best = (key, [opt, v, r["model_id"], gate, td, ps, vo, fo, l2, l1, l0])
        if best:
            rows.append(best[1])
    _write_csv(tables_dir / "CF_cross_family.csv", headerCF, rows, source)


def write_t6_quicksim(runs: list[dict], tables_dir: Path, variants: list[str], source: str) -> bool:
    """T6 — aero performance preservation from the deterministic full-VLM quick-sim
    pass (§J). One row per (variant, optimizer) over its SUCCESSFUL (verified-clean)
    repairs: baseline vs final L/D, mean/median Delta L/D, Delta CD0, Delta CD (drag).
    ``mean_delta_weight`` stays in the schema but is blank — weight needs the CAD
    full report (compute_uav_mass_cog), not the VLM quick sim. Returns False (and
    writes a stub) when no run has a sim_eval.json yet.
    """
    idx = by_key(runs)
    header = [
        "variant", "model_id", "optimizer", "n_success",
        "baseline_mean_L_over_D", "final_mean_L_over_D",
        "mean_delta_L_over_D", "median_delta_L_over_D", "mean_rel_delta_L_over_D",
        "mean_delta_CD0", "median_delta_CD0", "mean_delta_CD",
        "final_vlm_ok_rate", "mean_delta_weight", "note",
    ]
    rows: list[list] = []
    any_sim = False
    for v in variants:
        for opt in OPTIMIZER_ORDER:
            r = idx.get((v, opt))
            if not r or not r.get("sim"):
                continue
            any_sim = True
            sim = r["sim"]
            rows.append([
                v, r["model_id"], opt, sim.get("n_success"),
                sim.get("baseline_mean_L_over_D"), sim.get("final_mean_L_over_D"),
                sim.get("mean_delta_L_over_D"), sim.get("median_delta_L_over_D"), sim.get("mean_rel_delta_L_over_D"),
                sim.get("mean_delta_CD0"), sim.get("median_delta_CD0"), sim.get("mean_delta_CD"),
                sim.get("final_vlm_ok_rate"), None, sim.get("weight_note", ""),
            ])
    if not any_sim:
        rows = [["", "", "", "", "", "", "", "", "", "", "", "", "", "",
                 "no sim_eval.json found — run quicksim_eval (run_sweep does this post-grid)"]]
    _write_csv(tables_dir / "T6_quicksim.csv", header, rows, source)
    return any_sim


# --- markdown digest + provenance -------------------------------------------


def _md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(out)


def write_summary_md(runs: list[dict], variants: list[str], out: Path) -> None:
    idx = by_key(runs)
    lines = [f"# Chapter 7 optimization grid — digest\n", f"Generated {_now()}\n"]

    lines.append("## T1 — Verified-OK recovery (optimizer x model, tau_decide=1.0)\n")
    header = ["optimizer", *variants]
    rows = [[opt, *[(idx.get((v, opt)) and (
        idx[(v, opt)]["stats"].get("all_verified_clean_count") if opt == "all"
        else headline_op(idx[(v, opt)]["stats"])["verified_ok"]
    )) for v in variants]] for opt in OPTIMIZER_ORDER]
    lines.append(_md_table(header, rows))
    lines.append("\n_Reading: higher is better; each cell is the count (of 100) of CAD-verified clean repairs at the headline operating point._\n")

    lines.append("## T2 — False-OK (surrogate false positives)\n")
    rows = [[opt, *[(idx.get((v, opt)) and headline_op(idx[(v, opt)]["stats"])["false_ok"]) for v in variants]] for opt in OPTIMIZER_ORDER]
    lines.append(_md_table(header, rows))
    lines.append("\n_Reading: lower is better; variant f (binary gate) should shrink these vs variant a._\n")

    lines.append("## T5 — 'All' frontier (best of 6 per start)\n")
    header5 = ["variant", "union_recovery", "best_L2_avg", "best_L0_avg", "top_winner"]
    rows = []
    for v in variants:
        r = idx.get((v, "all"))
        if not r:
            continue
        st = r["stats"]
        hist = st.get("winner_histogram", {}) or {}
        top = max(hist, key=hist.get) if hist else None
        rows.append([v, st.get("all_verified_clean_count"), st.get("best_L2_average"), st.get("best_L0_average"), top])
    lines.append(_md_table(header5, rows))
    lines.append("\n_Reading: union_recovery = #starts some optimizer repaired; the winner shows which optimizer earns its keep._\n")

    sim_runs = [r for r in runs if r["variant"] is not None and r.get("sim")]
    if sim_runs:
        lines.append("## T6 — Aero performance preservation (full-VLM quick sim, over successful repairs)\n")
        header6 = ["optimizer", *[f"{v}_dL/D" for v in variants]]
        rows = []
        for opt in OPTIMIZER_ORDER:
            row = [opt]
            for v in variants:
                r = idx.get((v, opt))
                row.append((r.get("sim") or {}).get("mean_delta_L_over_D") if r else None)
            rows.append(row)
        lines.append(_md_table(header6, rows))
        lines.append("\n_Reading: mean Delta(L/D) = final - start over CAD-verified-clean repairs (both designs VLM-ok). "
                     "Near 0 = repair preserved aerodynamics; strongly negative = overlap was fixed by hurting L/D. "
                     "See `tables/T6_quicksim.csv` for CD0/CD/drag and medians; weight is not in the VLM quick path._\n")

    lines.append("\nSee `tables/*.csv` for the full P1-P7 tau/p* sweeps, T3 proximity, T4 feature-grad deltas, T6 aero, and CF.\n")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(runs: list[dict], out: Path, runs_dir: Path) -> None:
    benchmarks = sorted({r["benchmark_id"] for r in runs if r["benchmark_id"]})
    models = sorted({r["model_id"] for r in runs if r["model_id"]})
    run_dirs = sorted(r["run_id"] for r in runs if r["variant"] is not None)
    lines = [
        "# SOURCE provenance\n",
        f"Generated: {_now()}",
        f"Runs dir: {runs_dir}",
        f"Benchmark(s): {', '.join(benchmarks)}",
        f"Checkpoints: {', '.join(models)}",
        f"# subgroups aggregated: {len(run_dirs)}",
        "",
        "## Run dirs",
        *[f"- {rid}" for rid in run_dirs],
        "",
        "Each table CSV repeats this provenance on its `# SOURCE:` header line so a",
        "thesis table is always traceable to the runs that produced it.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate Chapter 7 optimization runs into thesis-ready tables")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--grid-id", type=str, default="grid_100k_v2")
    ap.add_argument("--quicksim", action="store_true",
                    help="(no-op) T6 is auto-populated from each run's sim_eval.json; run quicksim_eval to produce them")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runs = discover_runs(args.runs_dir)
    mapped = [r for r in runs if r["variant"] is not None]
    if not mapped:
        print(f"[summarize] no variant-mapped runs found under {args.runs_dir}")
        return 1
    variants = [v for v in "abcdef" if any(r["variant"] == v for r in mapped)]

    out_dir = args.analysis_dir / args.grid_id
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (
        f"runs_dir={args.runs_dir.name}; benchmark="
        f"{sorted({r['benchmark_id'] for r in mapped if r['benchmark_id']})}; "
        f"n_subgroups={len(mapped)}; generated={_now()}"
    )

    write_master(master_rows(runs), out_dir / "thesis_ready_summary.csv", source)
    p_names = write_p_tables(runs, tables_dir, variants, source)
    write_matrices(runs, tables_dir, variants, source)
    has_sim = write_t6_quicksim(runs, tables_dir, variants, source)
    write_summary_md(runs, variants, out_dir / "summary.md")
    write_readme(runs, out_dir / "README.md", args.runs_dir)

    n_sim = sum(1 for r in mapped if r.get("sim"))
    print(f"[summarize] {len(mapped)} variant-mapped subgroups over variants {variants}")
    print(f"[summarize] wrote -> {out_dir}")
    print(f"[summarize]   thesis_ready_summary.csv, summary.md, README.md")
    print(f"[summarize]   tables/: {', '.join(p_names)} + T1..T6 + CF")
    print(f"[summarize]   quick-sim performance: {n_sim}/{len(mapped)} runs have sim_eval.json "
          f"({'T6 populated' if has_sim else 'T6 stub — run the sim pass'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
