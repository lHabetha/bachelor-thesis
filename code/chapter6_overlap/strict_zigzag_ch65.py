"""Chapter 6.5 strict alternating overlap<->assemblability pipeline, v2 (Chapter 6 #5g).

This is the canonical §6.5 rerun. It replaces the v1 method choices
(``strict_hybrid_zigzag``) with the honestly benchmarked stages:

- **Overlap stage**  -- the Chapter 6.4 single-head ``hybrid_lite`` / ``finish_line``
  repair (``strict_repair_ch64.run_overlap_stage_zigzag``), run with ``tau_vol=0`` so
  the surrogate never triggers an early stop; success is *analytic*
  ``total_overlap_norm <= 1e-6`` only.
- **Assemblability stage** -- the Chapter 5 ``trust_region_hybrid_v1`` search
  (gradient + all coordinate axes + random unit directions per trust radius,
  ``tau=0.75`` on the learned ``P(assemblable)``) with the exact formula oracle as the
  authoritative final gate. The search walks *all* trust radii / directions; ``tau`` is
  used only to rank candidates and flag crossings, never as a lone early-exit.

Both stages run in zig-zag mode: each tries to satisfy its *analytic* objective to
budget exhaustion, then hands control back to the controller. Failures are expected
and recorded honestly via a failure taxonomy:

- ``overlap_exhausted``        -- overlap stage spent its budget while still > 1e-6.
- ``assemblability_exhausted`` -- overlap clean, not reintroduced, but the formula
                                  oracle still says blocked after a full trust-region
                                  search.
- ``max_cycles``               -- the cycle budget ran out without success.

A move in the assemblability stage that pushes overlap back above the strict
threshold is *not* a terminal failure: the controller cycles back to the overlap
stage.

Run two campaigns on ``strict_overlap_blocked_v2`` (50 blocked + overlapping starts):

    python -m task25_overlap.strict_zigzag_ch65 --variant both

Outputs ``runs/zigzag/strict_zigzag_ch65_{hybrid_lite,finish_line}_v3/`` plus the
comparison table ``tables/strict_zigzag_ch65_summary.csv`` and report
``reports/strict_zigzag_ch65.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._public_helpers import FEATURES, _features, _read_csv
from .label_cache import STRICT_THRESHOLD_NORM, params_from_row
from ._public_helpers import _to_params
from .paths import DATA_DIR, REPORTS_DIR, RUNS_DIR, TABLES_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
    ASM_MODEL_DIR,
    ASM_NORMALIZATION_JSON,
    ensure_chapter5_importable,
)
from .sampler import DummyParams, validate_relaxed_params
from ._public_helpers import _bounds_from_pool, _strict_overlap
from .strict_repair_ch64 import (
    V3_MODEL_DIR,
    _make_surrogate,
    _pct_reduction,
    _x_of,
    run_overlap_stage_zigzag,
)
from ._public_helpers import (
    _assemblability_margin,
    _params_from_dict,
    evaluate_exact_assemblability,
)

# Chapter 5 trust-region assemblability optimizer. The feature order is identical
# across Chapter 5 and Chapter 6 (13 raw DummyParams values), so the bridge needs
# no reordering.
ensure_chapter5_importable()

from chapter5_optimization.optimizers.trust_region_hybrid_v1.optimizer import (  # noqa: E402
    BASE_SEED,
    RADII,
    RANDOM_PER_RADIUS,
)
from chapter5_optimization.shared.interface import (  # noqa: E402
    BenchmarkStart,
    NormalizationConfig,
    OptimizationContext,
)
from chapter5_optimization.shared.model_utils import (  # noqa: E402
    load_model_artifact,
)
from chapter5_optimization.shared.optimizer_utils import (  # noqa: E402
    normalized_gradient_direction,
    raw_step_from_norm_direction,
)

# --------------------------------------------------------------------------------------
# Constants / paths.
# --------------------------------------------------------------------------------------

ASM_NORM_JSON = ASM_NORMALIZATION_JSON

BLOCKED_V2_STARTS = CH65_STARTS_CSV
POOL_CSV = POOL_100K_CSV

VARIANTS = ("hybrid_lite", "finish_line")
DEFAULT_TAU = 0.75


# --------------------------------------------------------------------------------------
# Assemblability stage: Chapter 5 trust-region hybrid, zig-zag semantics.
# --------------------------------------------------------------------------------------


def build_asm_context(tau: float = DEFAULT_TAU) -> OptimizationContext:
    """Build the Chapter 5 optimization context for the assemblability stage.

    Reuses the Chapter 5 classifier and the ``blocked_200_v1`` normalization (the same
    global per-parameter scale the reference run ``trust_region_tau0.75_labelblind``
    used). The ``BenchmarkStart`` itself is rebuilt at the *current* geometry on every
    asm stage, so this context is start-agnostic.
    """
    model = load_model_artifact(ASM_MODEL_DIR)
    nd = json.loads(ASM_NORM_JSON.read_text())
    normalization = NormalizationConfig(
        medians=np.array(nd["medians"], dtype=np.float64),
        stds=np.array(nd["stds"], dtype=np.float64),
        parameter_order=nd["parameter_order"],
    )
    return OptimizationContext(
        starts=[],
        model=model,
        model_id=ASM_MODEL_DIR.name,
        normalization=normalization,
        tau=float(tau),
    )


def _asm_frame(p: DummyParams, overlap_norm: float, phase: str) -> dict:
    return {
        "params": {k: float(v) for k, v in asdict(p).items()},
        "verified_overlap_norm": float(overlap_norm),
        "phase": phase,
    }


def run_assemblability_stage_zigzag(
    p0: DummyParams,
    *,
    ctx: OptimizationContext,
    lo: np.ndarray,
    hi: np.ndarray,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
) -> dict:
    """One assemblability-repair stage (Chapter 5 trust-region hybrid), zig-zag mode.

    Walks every trust radius in ``RADII`` and, per radius, the full direction set
    (normalized MLP gradient + all +/- coordinate axes + ``RANDOM_PER_RADIUS`` random
    unit directions) -- the same mix as ``trust_region_tau0.75_labelblind``. It never
    truncates after the first ``tau`` crossing. Candidates are gated on *relaxed*
    validity (consistent with the relaxed overlap-aware design space); the analytic
    voxel overlap is recomputed only on the selected final candidate.

    Selection: the nearest (smallest benchmark-normalized distance) formula-assemblable
    candidate; if none is assemblable, the best fallback (largest formula margin, then
    highest ``P``) is kept as the stage output. Overlap reintroduced by the chosen move
    is flagged for the controller to re-cycle and is *not* a terminal failure.
    """
    x0 = _x_of(p0).astype(np.float64)
    start_overlap = _strict_overlap(p0)
    start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])

    rng = np.random.default_rng(BASE_SEED + int(seed))
    grad = normalized_gradient_direction(ctx, x0)
    axes: list[np.ndarray] = []
    for i in range(len(x0)):
        for sign in (-1.0, 1.0):
            d = np.zeros(len(x0), dtype=np.float64)
            d[i] = sign
            axes.append(d)

    candidates: list[dict] = []
    n_eval = 0
    surrogate_crossed_tau = False
    for radius in RADII:
        dirs: list[np.ndarray] = []
        if grad is not None:
            dirs.append(grad.direction_norm)
        dirs.extend(axes)
        for _ in range(RANDOM_PER_RADIUS):
            d = rng.normal(size=len(x0))
            d /= max(float(np.linalg.norm(d)), 1e-12)
            dirs.append(d.astype(np.float64))
        for d in dirs:
            x = raw_step_from_norm_direction(x0, d, radius, ctx)
            x = np.clip(x, lo, hi)
            cand_p = _to_params(x, p0)
            n_eval += 1
            if not validate_relaxed_params(cand_p).ok:
                continue
            # Use the rounded params (as the geometry will actually be built) for a
            # consistent probability / distance read.
            x_eff = _x_of(cand_p).astype(np.float64)
            prob = float(ctx.model.predict_proba(x_eff.reshape(1, -1))[0])
            if prob >= tau:
                surrogate_crossed_tau = True
            exact = evaluate_exact_assemblability(cand_p, require_validity=False)
            candidates.append({
                "params": cand_p,
                "kinematic": bool(exact.kinematic_assemblable),
                "margin": float(_assemblability_margin(cand_p)),
                "prob": prob,
                "dist": float(ctx.normalization.distance(x0, x_eff)),
            })

    assemblable = [c for c in candidates if c["kinematic"]]
    if assemblable:
        selected = min(assemblable, key=lambda c: (c["dist"], -c["prob"]))
    elif candidates:
        selected = max(candidates, key=lambda c: (c["margin"], c["prob"]))
    else:
        selected = None

    final_p = selected["params"] if selected is not None else p0
    final_overlap = _strict_overlap(final_p)
    final_exact = evaluate_exact_assemblability(final_p, require_validity=False)
    final_assemblable = bool(final_exact.kinematic_assemblable)
    final_relaxed = bool(validate_relaxed_params(final_p).ok)
    reintroduced = bool(final_overlap > STRICT_THRESHOLD_NORM and start_overlap <= STRICT_THRESHOLD_NORM)
    stage_success = bool(final_assemblable and final_relaxed and final_overlap <= STRICT_THRESHOLD_NORM)

    if reintroduced:
        failure_reason = None  # not terminal: controller cycles back to the overlap stage
    elif not stage_success:
        failure_reason = "assemblability_exhausted"
    else:
        failure_reason = None

    frames = [
        _asm_frame(p0, start_overlap, "asm_start"),
        _asm_frame(final_p, final_overlap, "asm_end"),
    ]
    return {
        "stage": "assemblability",
        "method": "trust_region_hybrid_v1",
        "final_params": asdict(final_p),
        "trajectory": frames,
        "start_overlap_norm": float(start_overlap),
        "final_overlap_norm": float(final_overlap),
        "final_p_assemblable": final_assemblable,
        "final_margin": float(selected["margin"]) if selected else float(_assemblability_margin(p0)),
        "final_probability": float(selected["prob"]) if selected else start_prob,
        "start_probability": start_prob,
        "final_relaxed_ok": final_relaxed,
        "reintroduced_overlap": reintroduced,
        "surrogate_crossed_tau": bool(surrogate_crossed_tau),
        "n_candidates": int(len(candidates)),
        "verifier_calls": int(n_eval),
        "stage_success": stage_success,
        "failure_reason": failure_reason,
    }


# --------------------------------------------------------------------------------------
# Controller state machine.
# --------------------------------------------------------------------------------------


def _overlap_args(args: argparse.Namespace) -> argparse.Namespace:
    """Budget namespace for the §6.4 overlap-stage method functions (tau_vol=0)."""
    return argparse.Namespace(
        method_label=None,
        mlp_steps=args.mlp_steps,
        lr=args.lr,
        tau_vol=0.0,
        tau_bin=None,
        lite_rounds=args.lite_rounds,
        finish_coarse=args.finish_coarse,
        finish_refine=args.finish_refine,
        finish_rounds=getattr(args, "finish_rounds", 4),
    )


def run_pipeline_one(
    p0: DummyParams,
    *,
    overlap_method: str,
    surrogate,
    ctx: OptimizationContext,
    lo: np.ndarray,
    hi: np.ndarray,
    overlap_args: argparse.Namespace,
    tau: float,
    max_cycles: int,
    seed: int,
) -> dict:
    """Alternating overlap<->assemblability controller for one start."""
    p = p0
    segments: list[dict] = []
    cycles: list[dict] = []
    seg_idx = 0
    success = False
    failure_reason: str | None = None
    overlap_stage_used = 0
    asm_stage_used = 0
    reintroduced_count = 0
    still_blocked_after_first_overlap: bool | None = None
    first_overlap_pct: float | None = None
    verifier_calls = 0

    for cycle in range(1, max_cycles + 1):
        overlap = _strict_overlap(p)
        exact = evaluate_exact_assemblability(p, require_validity=False)
        asm_ok = bool(exact.kinematic_assemblable)
        relaxed_ok = bool(validate_relaxed_params(p).ok)
        entry = {
            "cycle": cycle,
            "start_overlap_norm": overlap,
            "start_assemblable": asm_ok,
            "start_relaxed_ok": relaxed_ok,
        }

        if overlap <= STRICT_THRESHOLD_NORM and asm_ok and relaxed_ok:
            entry["action"] = "terminal_success"
            cycles.append(entry)
            success = True
            break

        if overlap > STRICT_THRESHOLD_NORM:
            stage = run_overlap_stage_zigzag(p, surrogate, lo, hi, overlap_args, method=overlap_method)
            stage["segment_idx"] = seg_idx
            stage["cycle"] = cycle
            seg_idx += 1
            segments.append(stage)
            overlap_stage_used += 1
            verifier_calls += int(stage["verifier_calls"])
            p = _params_from_dict(stage["final_params"])
            entry["action"] = "overlap"
            entry["overlap_method"] = overlap_method
            entry["overlap_stage_success"] = stage["stage_success"]
            entry["overlap_stage_final_norm"] = stage["final_overlap_norm"]
            entry["overlap_pct_reduction"] = stage["pct_overlap_reduction"]
            if first_overlap_pct is None:
                first_overlap_pct = stage["pct_overlap_reduction"]
                still_blocked_after_first_overlap = not bool(
                    evaluate_exact_assemblability(p, require_validity=False).kinematic_assemblable
                )
            cycles.append(entry)
            if not stage["stage_success"]:
                failure_reason = "overlap_exhausted"
                break
            continue

        # Overlap is strictly clean but the design is kinematically blocked.
        before = overlap
        stage = run_assemblability_stage_zigzag(p, ctx=ctx, lo=lo, hi=hi, tau=tau, seed=seed)
        stage["segment_idx"] = seg_idx
        stage["cycle"] = cycle
        seg_idx += 1
        segments.append(stage)
        asm_stage_used += 1
        verifier_calls += int(stage["verifier_calls"])
        p = _params_from_dict(stage["final_params"])
        after = _strict_overlap(p)
        reintroduced = bool(after > STRICT_THRESHOLD_NORM and before <= STRICT_THRESHOLD_NORM)
        entry["action"] = "assemblability"
        entry["asm_stage_success"] = stage["stage_success"]
        entry["asm_final_overlap_norm"] = after
        entry["asm_final_assemblable"] = stage["final_p_assemblable"]
        entry["asm_surrogate_crossed_tau"] = stage["surrogate_crossed_tau"]
        entry["reintroduced_overlap"] = reintroduced
        cycles.append(entry)
        if reintroduced:
            reintroduced_count += 1
            continue
        if not stage["final_p_assemblable"]:
            failure_reason = "assemblability_exhausted"
            break
        continue

    # Final authoritative recheck (also catches a design solved on the last cycle).
    final_overlap = _strict_overlap(p)
    final_exact = evaluate_exact_assemblability(p, require_validity=False)
    final_assemblable = bool(final_exact.kinematic_assemblable)
    final_relaxed = bool(validate_relaxed_params(p).ok)
    success = bool(final_overlap <= STRICT_THRESHOLD_NORM and final_assemblable and final_relaxed)
    if success:
        failure_reason = None
    elif failure_reason is None:
        failure_reason = "max_cycles"

    start_overlap = _strict_overlap(p0)
    start_exact = evaluate_exact_assemblability(p0, require_validity=False)
    return {
        "start_overlap_norm": float(start_overlap),
        "start_assemblable": bool(start_exact.kinematic_assemblable),
        "final_overlap_norm": float(final_overlap),
        "final_overlap_ok": bool(final_overlap <= STRICT_THRESHOLD_NORM),
        "final_assemblable": final_assemblable,
        "final_relaxed_ok": final_relaxed,
        "final_reason": final_exact.label_reason,
        "success": success,
        "failure_reason": failure_reason,
        "cycles_used": len(cycles),
        "active_stages": overlap_stage_used + asm_stage_used,
        "overlap_stage_used": overlap_stage_used,
        "asm_stage_used": asm_stage_used,
        "reintroduced_overlap_count": reintroduced_count,
        "still_blocked_after_first_overlap": bool(still_blocked_after_first_overlap)
        if still_blocked_after_first_overlap is not None
        else False,
        "first_overlap_pct_reduction": float(first_overlap_pct) if first_overlap_pct is not None else 0.0,
        "overall_pct_overlap_reduction": _pct_reduction(start_overlap, final_overlap),
        "verifier_calls": int(verifier_calls),
        "cycles": cycles,
        "segments": segments,
        "final_params": asdict(p),
    }


# --------------------------------------------------------------------------------------
# Orchestration / IO.
# --------------------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _variant_run_id(variant: str) -> str:
    return f"strict_zigzag_ch65_{variant}_v3"


def _summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    succ = [r for r in results if r["success"]]
    fa = [r for r in results if r["failure_reason"] == "overlap_exhausted"]
    fb = [r for r in results if r["failure_reason"] == "assemblability_exhausted"]
    fmax = [r for r in results if r["failure_reason"] == "max_cycles"]
    return {
        "n_starts": n,
        "pipeline_success": len(succ),
        "fail_overlap_exhausted": len(fa),
        "fail_assemblability_exhausted": len(fb),
        "fail_max_cycles": len(fmax),
        "still_blocked_after_overlap_only": sum(bool(r["still_blocked_after_first_overlap"]) for r in results),
        "runs_using_asm_stage": sum(int(r["asm_stage_used"]) > 0 for r in results),
        "asm_stage_invocations": int(sum(int(r["asm_stage_used"]) for r in results)),
        "overlap_reintroduced_runs": sum(int(r["reintroduced_overlap_count"]) > 0 for r in results),
        "final_overlap_ok": sum(bool(r["final_overlap_ok"]) for r in results),
        "final_assemblable": sum(bool(r["final_assemblable"]) for r in results),
        "mean_first_overlap_pct_reduction": float(np.mean([r["first_overlap_pct_reduction"] for r in results])),
        "mean_overall_pct_overlap_reduction": float(np.mean([r["overall_pct_overlap_reduction"] for r in results])),
        "mean_cycles": float(np.mean([r["cycles_used"] for r in results])),
        "mean_active_stages": float(np.mean([r["active_stages"] for r in results])),
        "mean_verifier_calls": float(np.mean([r["verifier_calls"] for r in results])),
    }


def run_variant(variant: str, args: argparse.Namespace) -> tuple[Path, list[dict]]:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; choose from {VARIANTS}")
    run_id = args.run_id or _variant_run_id(variant)
    run_dir = RUNS_DIR / "zigzag" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    starts = _read_csv(args.starts_csv)[args.start_offset : args.start_offset + args.n_starts]
    lo, hi = _bounds_from_pool(_features(_read_csv(args.pool_csv)))
    surrogate = _make_surrogate("v3", V3_MODEL_DIR)
    ctx = build_asm_context(args.tau)
    overlap_args = _overlap_args(args)

    t0 = time.perf_counter()
    results: list[dict] = []
    cycle_rows: list[dict] = []
    for idx, row in enumerate(starts):
        p0 = params_from_row(row)
        out = run_pipeline_one(
            p0,
            overlap_method=variant,
            surrogate=surrogate,
            ctx=ctx,
            lo=lo,
            hi=hi,
            overlap_args=overlap_args,
            tau=args.tau,
            max_cycles=args.max_cycles,
            seed=args.start_offset + idx,
        )
        result = {
            "sample_id": row["sample_id"],
            "strict_category": row.get("strict_category", ""),
            "start_dominant_pair": row.get("dominant_pair", ""),
            "start_magnitude_bin": row.get("magnitude_bin", ""),
            "overlap_method": variant,
            **out,
        }
        results.append(result)
        for cyc in out["cycles"]:
            cycle_rows.append({"sample_id": row["sample_id"], "overlap_method": variant, **cyc})
        flag = "OK" if result["success"] else (result["failure_reason"] or "fail")
        print(
            f"  [{idx + 1:2d}/{len(starts)}] {row['sample_id']:>14} {flag:>24} "
            f"cyc={result['cycles_used']} asm={result['asm_stage_used']} "
            f"reintro={result['reintroduced_overlap_count']} ov%={result['overall_pct_overlap_reduction']:.0f}"
        )
    wall = time.perf_counter() - t0

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in ("cycles", "segments", "final_params")} for r in results]
    _write_csv(run_dir / "summary.csv", flat)
    _write_csv(run_dir / "cycle_log.csv", cycle_rows)

    summary = _summarize(results)
    summary.update({
        "run_id": run_id,
        "variant": variant,
        "overlap_model_dir": str(V3_MODEL_DIR),
        "asm_model_dir": str(ASM_MODEL_DIR),
        "asm_tau": args.tau,
        "max_cycles": args.max_cycles,
        "strict_threshold_norm": STRICT_THRESHOLD_NORM,
        "starts_csv": str(args.starts_csv),
        "wall_time_s": float(wall),
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_run_report(run_dir, variant, summary)
    print(
        f"[{variant}] success {summary['pipeline_success']}/{summary['n_starts']} "
        f"(a={summary['fail_overlap_exhausted']} b={summary['fail_assemblability_exhausted']} "
        f"max={summary['fail_max_cycles']}) "
        f"asm-used {summary['runs_using_asm_stage']} reintro {summary['overlap_reintroduced_runs']} "
        f"mean_cyc {summary['mean_cycles']:.2f} wall {wall:.0f}s"
    )
    print(run_dir)
    return run_dir, results


def _write_run_report(run_dir: Path, variant: str, summary: dict) -> None:
    n = summary["n_starts"]
    lines = [
        f"# Strict Zig-Zag Ch. 6.5 v2 -- {variant}",
        "",
        f"Strict analytic threshold: `{STRICT_THRESHOLD_NORM}`. Assemblability tau: "
        f"`{summary['asm_tau']}`. Max cycles: `{summary['max_cycles']}`.",
        "",
        "| Metric | Count / Mean |",
        "|---|---:|",
        f"| Pipeline success | {summary['pipeline_success']}/{n} |",
        f"| Fail (a) overlap_exhausted | {summary['fail_overlap_exhausted']}/{n} |",
        f"| Fail (b) assemblability_exhausted | {summary['fail_assemblability_exhausted']}/{n} |",
        f"| Fail max_cycles | {summary['fail_max_cycles']}/{n} |",
        f"| Still blocked after overlap-only stage | {summary['still_blocked_after_overlap_only']}/{n} |",
        f"| Runs using assemblability stage | {summary['runs_using_asm_stage']}/{n} |",
        f"| Overlap reintroduced (>=1 cycle) | {summary['overlap_reintroduced_runs']}/{n} |",
        f"| Final overlap OK | {summary['final_overlap_ok']}/{n} |",
        f"| Final assemblable | {summary['final_assemblable']}/{n} |",
        f"| Mean % overlap reduction (stage 1) | {summary['mean_first_overlap_pct_reduction']:.1f}% |",
        f"| Mean % overlap reduction (overall) | {summary['mean_overall_pct_overlap_reduction']:.1f}% |",
        f"| Mean controller cycles | {summary['mean_cycles']:.2f} |",
        f"| Mean active repair stages | {summary['mean_active_stages']:.2f} |",
        f"| Mean verifier / surrogate calls | {summary['mean_verifier_calls']:.0f} |",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Cross-variant aggregation.
# --------------------------------------------------------------------------------------


def aggregate(variants: tuple[str, ...] = VARIANTS) -> Path:
    """Aggregate both variants into the §6.5 comparison table + report."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for variant in variants:
        run_dir = RUNS_DIR / "zigzag" / _variant_run_id(variant)
        results_path = run_dir / "results.json"
        if not results_path.exists():
            print(f"  (skip {variant}: {results_path} missing)")
            continue
        results = json.loads(results_path.read_text())
        summary = _summarize(results)
        summaries[variant] = summary
        rows.append({"variant": variant, **summary})

    if not rows:
        raise SystemExit("no variant results found to aggregate")

    table_path = TABLES_DIR / "strict_zigzag_ch65_summary.csv"
    _write_csv(table_path, rows)
    _write_comparison_report(summaries)
    print(table_path)
    return table_path


def _write_comparison_report(summaries: dict[str, dict]) -> None:
    def cell(variant: str, key: str, fmt: str = "{}") -> str:
        s = summaries.get(variant)
        return fmt.format(s[key]) if s and key in s else "--"

    n = next((s["n_starts"] for s in summaries.values()), 50)
    metrics = [
        ("Pipeline success", "pipeline_success", "{}"),
        ("Fail (a) overlap_exhausted", "fail_overlap_exhausted", "{}"),
        ("Fail (b) assemblability_exhausted", "fail_assemblability_exhausted", "{}"),
        ("Fail max_cycles", "fail_max_cycles", "{}"),
        ("Still blocked after overlap-only stage", "still_blocked_after_overlap_only", "{}"),
        ("Runs using assemblability stage", "runs_using_asm_stage", "{}"),
        ("Overlap reintroduced (>=1 cycle)", "overlap_reintroduced_runs", "{}"),
        ("Final overlap OK", "final_overlap_ok", "{}"),
        ("Final assemblable", "final_assemblable", "{}"),
        ("Mean % overlap reduction (stage 1)", "mean_first_overlap_pct_reduction", "{:.1f}"),
        ("Mean % overlap reduction (overall)", "mean_overall_pct_overlap_reduction", "{:.1f}"),
        ("Mean controller cycles", "mean_cycles", "{:.2f}"),
        ("Mean active repair stages", "mean_active_stages", "{:.2f}"),
        ("Mean verifier / surrogate calls", "mean_verifier_calls", "{:.0f}"),
    ]
    lines = [
        "# Strict Overlap--Assemblability Pipeline (Ch. 6.5 v2, #5g)",
        "",
        "Benchmark: `strict_overlap_blocked_v2` (50 starts, all strictly overlapping "
        "and kinematically blocked at the start). Overlap stage: Chapter 6.4 single-head "
        "(`hybrid_lite` / corrected path-only `finish_line`, `tau_vol=0`). Assemblability stage: Chapter 5 "
        "`trust_region_hybrid_v1` (`tau=0.75` on the learned `P(assemblable)`), with the "
        "exact formula oracle as the authoritative final gate. Both stages exhaust their "
        "budget before handing back to the controller; no surrogate-only early exit.",
        "",
        f"| Metric (out of {n}) | hybrid-lite | finish-line |",
        "|---|---:|---:|",
    ]
    for label, key, fmt in metrics:
        lines.append(f"| {label} | {cell('hybrid_lite', key, fmt)} | {cell('finish_line', key, fmt)} |")
    lines += [
        "",
        "## Failure taxonomy",
        "",
        "- **(a) `overlap_exhausted`** -- the overlap stage spent its full budget while "
        "the analytic overlap was still above `1e-6` (MLP + coordinate polish for "
        "hybrid-lite, MLP + path extrapolation for finish-line).",
        "- **(b) `assemblability_exhausted`** -- overlap was strictly clean and not "
        "reintroduced, but the exact formula oracle still reported the design blocked "
        "after a full trust-region search over all radii and directions.",
        "- **`max_cycles`** -- the cycle budget ran out (e.g. repeated overlap "
        "reintroduction) without reaching a clean, assemblable, relaxed-valid design.",
        "",
        "A move in the assemblability stage that pushes overlap back above the strict "
        "threshold is logged as a reintroduction and triggers another overlap cycle; it "
        "is not counted as a terminal failure.",
        "",
        "## Artifacts",
        "",
        "- `runs/zigzag/strict_zigzag_ch65_hybrid_lite_v3/`",
        "- `runs/zigzag/strict_zigzag_ch65_finish_line_v3/`",
        "- `tables/strict_zigzag_ch65_summary.csv`",
    ]
    (REPORTS_DIR / "strict_zigzag_ch65.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="both", choices=(*VARIANTS, "both"))
    parser.add_argument("--aggregate", action="store_true", help="only re-aggregate existing runs")
    parser.add_argument("--run-id", default=None, help="override run id (e.g. for smoke tests)")
    parser.add_argument("--starts-csv", type=Path, default=BLOCKED_V2_STARTS)
    parser.add_argument("--pool-csv", type=Path, default=POOL_CSV)
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--mlp-steps", type=int, default=140)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lite-rounds", type=int, default=6)
    parser.add_argument("--finish-coarse", type=int, default=10)
    parser.add_argument("--finish-refine", type=int, default=5)
    parser.add_argument("--finish-rounds", type=int, default=8, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return

    variants = VARIANTS if args.variant == "both" else (args.variant,)
    for variant in variants:
        run_variant(variant, args)
    # Only aggregate the canonical full runs (not overridden / partial smoke runs).
    if args.variant == "both" and args.run_id is None and args.start_offset == 0 and args.n_starts >= 50:
        aggregate()


if __name__ == "__main__":
    main()
