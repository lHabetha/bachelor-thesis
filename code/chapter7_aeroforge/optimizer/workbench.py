#!/usr/bin/env python3
"""Chapter 7 AeroForge overlap-repair optimizer workbench.

Standard pipeline (mirrors Chapter 5's optimizer/viewer split):

  1. A design is the 43-key ADV. The optimizer moves only its numeric drivers in
     normalized [0,1] space; categoricals are frozen.
  2. The optimizer is pure MLP: propose candidate ADVs with arithmetic rules +
     gradients, score every candidate with one MLP forward pass, run to budget.
     NO geometry is ever built during the search.
  3. One global selection rule picks the final design at an *operating point*
     (clean gate + tau_decide + p*): among candidates the gate calls clean take
     the one closest (L2) to the start; otherwise the lowest predicted overlap.
     Gating is selection-only — generation never sees a threshold (§I.1).
  4. CadQuery is called once per distinct selected final (deduped across the
     operating-threshold sweep) to get ground truth. (Skipped with --no-verify.)
  5. Statistics are aggregated over all starts. Artifacts: viewer_data.json,
     statistics.json, manifest.json. NEVER a render -- meshes are produced only
     on demand from the viewer (render_worker.py).

Run under the aeroforge env so the final verification can build geometry:
    conda run -n bachelor-thesis python3 -m chapter7_aeroforge.optimizer.workbench \
        --variant c --optimizer receding_multistep_gradient --workers 15 --seed 12345
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from chapter7_aeroforge.optimizer.model_io import Surrogate, u_from_adv  # noqa: E402
from chapter7_aeroforge.optimizer.optimizers import (  # noqa: E402
    CANONICAL_SUITE,
    OPTIMIZERS,
)
from chapter7_aeroforge.optimizer.perf_surrogate import (  # noqa: E402
    CORE4_NAMES,
    DEFAULT_PERF_CHECKPOINT,
    get_perf_surrogate,
)

from chapter7_aeroforge.release_paths import BENCHMARK_JSON, CHECKPOINTS_DIR, RUNS_DIR

DEFAULT_BENCHMARK = BENCHMARK_JSON
DEFAULT_MODELS_DIR = CHECKPOINTS_DIR
DEFAULT_RUNS_DIR = RUNS_DIR
DEFAULT_TAU_MM3 = 1.0
DEFAULT_WORKERS = 14
DEFAULT_SEED = 12345
DEFAULT_VERIFY_TIMEOUT_S = 180.0  # worst-overlap finals build in ~70s unloaded; headroom under 14-way load
DEFAULT_CHANGE_TOL = 1e-9

# Operating-threshold sweep grids (§I.2). reg-gate models sweep tau_decide; the
# binary_and variant additionally sweeps p* (the binary-head cutoff).
TAU_DECIDE_SWEEP = [1.0, 0.5, 0.1, 0.01]
P_STAR_SWEEP = [0.5, 0.4, 0.3, 0.2]

PART_COLORS = {
    "fuselage": "#8a8f98",
    "wings": "#61afef",
    "tail": "#e5c07b",
    "overlap": "#e06c75",
}

MODEL_TITLES = {
    # canonical 100k cloud-trained checkpoints (production)
    "eng_multitask_gate_strong_100k": "engineered multitask gate-strong (100k)",
    "eng_log_huber_100k": "engineered log-Huber (100k)",
    "eng_gate_aug_strong_100k": "engineered gate-augmented strong (100k)",
    "raw_log_huber_100k": "raw log-Huber baseline (100k)",
    # legacy 20k checkpoints (kept for back-compat)
    "eng_log_huber_20k": "engineered log-Huber (20k)",
    "eng_gate_aug_20k": "engineered gate-augmented (20k)",
}

# Variant a–f wiring. Each variant is one viewer Group.
# group_id is the run-dir prefix; the human-readable group_label is what the viewer
# shows. p_star only applies to the binary_and gate (variant f).
VARIANTS: dict[str, dict] = {
    "a": {
        "model_id": "eng_multitask_gate_strong_100k",
        "feature_grad_mode": "detached",
        "clean_gate": "reg",
        "p_star": 0.5,
        "group_id": "model_a_eng_multitask_gate_strong_featgrad_detached",
        "group_label": "Model A — multitask, feat-grad ignored (detached)",
    },
    "b": {
        "model_id": "eng_log_huber_100k",
        "feature_grad_mode": "detached",
        "clean_gate": "reg",
        "p_star": 0.5,
        "group_id": "model_b_eng_log_huber_featgrad_detached",
        "group_label": "Model B — log-Huber, feat-grad ignored (detached)",
    },
    "c": {
        "model_id": "eng_multitask_gate_strong_100k",
        "feature_grad_mode": "full",
        "clean_gate": "reg",
        "p_star": 0.5,
        "group_id": "model_c_eng_multitask_gate_strong_featgrad_full",
        "group_label": "Model C — multitask, feat-grad backprop (full)",
    },
    "d": {
        "model_id": "eng_log_huber_100k",
        "feature_grad_mode": "full",
        "clean_gate": "reg",
        "p_star": 0.5,
        "group_id": "model_d_eng_log_huber_featgrad_full",
        "group_label": "Model D — log-Huber, feat-grad backprop (full)",
    },
    "e": {
        "model_id": "raw_log_huber_100k",
        "feature_grad_mode": "raw",
        "clean_gate": "reg",
        "p_star": 0.5,
        "group_id": "model_e_raw_log_huber_featgrad_raw",
        "group_label": "Model E — raw baseline (no engineered features)",
    },
    "f": {
        "model_id": "eng_multitask_gate_strong_100k",
        "feature_grad_mode": "detached",
        "clean_gate": "binary_and",
        "p_star": 0.5,
        "group_id": "model_f_eng_multitask_gate_strong_binarygate_detached",
        "group_label": "Model F — multitask, binary-head gate, feat-grad ignored",
    },
}

_SURROGATE_CACHE: dict[str, Surrogate] = {}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_") or "group"


def _adv_hash(adv: dict) -> str:
    items = sorted((k, round(float(v), 6)) for k, v in adv.items() if isinstance(v, (int, float)))
    return hashlib.sha1(repr(items).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class JobSpec:
    optimizer_id: str
    checkpoint: str
    run_dir: str
    start: dict
    cfg: dict
    tau_mm3: float
    verify: bool
    operating_points: tuple = ()  # ordered (gate, tau_decide, p_star); [0] = headline
    sweep: bool = False
    verify_timeout_s: float = DEFAULT_VERIFY_TIMEOUT_S


def _pin_worker() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        from chapter7_aeroforge.overlap_search.core import limit_occt_threads  # noqa: PLC0415

        limit_occt_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _mp():
    import multiprocessing as mp

    return mp.get_context("spawn")


def _get_surrogate(checkpoint: str, tau_mm3: float) -> Surrogate:
    sur = _SURROGATE_CACHE.get(checkpoint)
    if sur is None:
        sur = Surrogate.load(checkpoint, tau_mm3=tau_mm3)
        _SURROGATE_CACHE[checkpoint] = sur
    return sur


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _frame_is_clean(frame: dict, gate: str, tau_decide: float, p_star: float | None) -> bool:
    """Operating-point clean predicate over a stored frame (mirrors Surrogate.is_clean)."""
    if frame["pred_overlap_mm3"] > tau_decide:
        return False
    if gate == "binary_and":
        p = frame.get("p_overlap")
        if p is not None:
            return p < float(p_star if p_star is not None else 0.5)
    return True


def _select_at_op(
    frames: list[dict],
    gate: str,
    tau_decide: float,
    p_star: float | None,
    *,
    aero_selection: str = "standard",
    aero_beta: float | None = None,
) -> tuple[int, str, bool]:
    """Global selection rule at one operating point. Returns (idx, reason, mlp_clean)."""
    clean = [f for f in frames if _frame_is_clean(f, gate, tau_decide, p_star)]
    if clean:
        if aero_selection == "budget_lowest_drift":
            beta = float("inf") if aero_beta is None else float(aero_beta)
            within = [f for f in clean if _safe_float(f.get("aero_D_aero"), float("inf")) <= beta]
            if within:
                best = min(
                    within,
                    key=lambda f: (
                        _safe_float(f.get("aero_R_aero"), float("inf")),
                        f["normalized_distance"],
                        f["pred_overlap_mm3"],
                    ),
                )
                return int(best["frame_idx"]), "aero_budget_lowest_drift_clean", True
        best = min(clean, key=lambda f: (f["normalized_distance"], f["pred_overlap_mm3"]))
        return int(best["frame_idx"]), "closest_predicted_clean", True
    best = min(frames, key=lambda f: (f["pred_overlap_mm3"], f["normalized_distance"]))
    return int(best["frame_idx"]), "lowest_predicted_overlap", False


def _operating_points(
    gate: str, tau: float, p_star: float, sweep: bool
) -> list[tuple[str, float, float | None]]:
    """Ordered operating points; index 0 is always the headline (tau_decide=tau)."""
    head: tuple[str, float, float | None] = (
        gate,
        float(tau),
        (float(p_star) if gate == "binary_and" else None),
    )
    if not sweep:
        return [head]
    ops: list[tuple[str, float, float | None]] = [head]
    if gate == "binary_and":
        for t in TAU_DECIDE_SWEEP:  # tau-axis at the headline p*
            op = ("binary_and", float(t), float(p_star))
            if op not in ops:
                ops.append(op)
        for p in P_STAR_SWEEP:  # p*-axis at the headline tau
            op = ("binary_and", float(tau), float(p))
            if op not in ops:
                ops.append(op)
    else:
        for t in TAU_DECIDE_SWEEP:
            op = ("reg", float(t), None)
            if op not in ops:
                ops.append(op)
    return ops


def _verify_frame(
    idx: int, cand_advs: list[dict], start: dict, cache: dict[str, dict], timeout_s: float
) -> dict:
    """Verify the final at frame `idx` once, cached by GEOMETRY HASH so identical
    finals (selected at several operating points, or duplicate candidates) build
    only once. A frame whose geometry equals the start reuses the benchmark truth.
    """
    adv = cand_advs[idx]
    h = _adv_hash(adv)
    if h in cache:
        return cache[h]
    if idx == 0 or h == _adv_hash(cand_advs[0]):  # geometry == start: reuse benchmark truth
        res = {
            "build_ok": True,
            "verified_overlap_mm3": float(start["overlap_mm3_cut_each"]),
            "verified_raw_overlap_mm3": _safe_float(start.get("overlap_mm3_raw")),
            "error": None,
            "wall_s": None,
            "timed_out": False,
        }
    else:
        from chapter7_aeroforge.optimizer.verify import verify_overlap  # noqa: PLC0415

        res = verify_overlap(adv, timeout_s=timeout_s)
    cache[h] = res
    return res


def _run_job(job: JobSpec) -> dict:
    _pin_worker()
    sur = _get_surrogate(job.checkpoint, job.tau_mm3)
    start = job.start
    adv0 = start["adv"]
    tau = job.tau_mm3

    ops = list(job.operating_points) or _operating_points("reg", tau, 0.5, False)
    perf_sur = None
    if job.cfg.get("perf_checkpoint") or str(job.optimizer_id).startswith("aero_"):
        perf_sur = get_perf_surrogate(job.cfg.get("perf_checkpoint", DEFAULT_PERF_CHECKPOINT))
    # Thread the full operating-point grid into the optimizer so gate-dependent
    # generation (only trust-region shrinkage, §D.8/§I.1) is re-run per operating
    # point; gate-free optimizers ignore it.
    opt_cfg = {**job.cfg, "op_grid": ops}
    if perf_sur is not None:
        opt_cfg["perf_sur"] = perf_sur
    proposal = OPTIMIZERS[job.optimizer_id].fn(adv0, sur, opt_cfg)
    opt_keys = proposal["opt_keys"] or []
    start_top = proposal.get("top_gradients", [])
    u0 = u_from_adv(adv0, opt_keys) if opt_keys else np.zeros(0)

    # Build lean frames (no per-frame ADV/geometry) but keep candidate ADVs in
    # memory so selection + the sweep can address/verify any frame's final.
    frames: list[dict] = []
    cand_advs: list[dict] = []
    frame_active: list[list] = []  # per-frame active deltas; only the selected final's is stored
    for idx, cand in enumerate(proposal["candidates"]):
        adv = cand["adv"]
        cand_advs.append(adv)
        z, pred, p_overlap = sur.predict_with_prob(adv)
        u = u_from_adv(adv, opt_keys) if opt_keys else np.zeros(0)
        delta = (u - u0) if opt_keys else np.zeros(0)
        active = [
            {"key": k, "delta_normalized": float(delta[i]), "start_u": float(u0[i]), "frame_u": float(u[i])}
            for i, k in enumerate(opt_keys)
            if abs(float(delta[i])) > DEFAULT_CHANGE_TOL
        ]
        frame_active.append(active)
        aero_frame = {}
        if perf_sur is not None and opt_keys:
            aero_summary = perf_sur.summary_for_u(adv0, opt_keys, u, u_ref=u0)
            aero_frame = {
                "aero_D_aero": float(aero_summary["D_aero"]),
                "aero_R_aero": float(aero_summary["R_aero"]),
            }
        # Lean frame: metrics only (no per-frame ADV/active-driver list). The viewer
        # never iterates frames; it renders start/final geometry and reads the
        # start-level active_driver_deltas (the selected final's) set below.
        frames.append(
            {
                "frame_idx": idx,
                "stage": cand.get("stage", "step"),
                "note": cand.get("note", ""),
                "step_size": float(cand.get("step_size", 0.0)),
                "pred_z": float(z),
                "pred_overlap_mm3": float(pred),
                "p_overlap": (float(p_overlap) if p_overlap is not None else None),
                "normalized_distance": float(np.linalg.norm(delta)) if opt_keys else 0.0,
                "normalized_l1": float(np.sum(np.abs(delta))) if opt_keys else 0.0,
                "active_coordinate_count": len(active),
                **aero_frame,
            }
        )

    verify_cache: dict[str, dict] = {}

    def _op_row(gate: str, tau_decide: float, p_star: float | None) -> dict:
        idx, reason, mlp_clean = _select_at_op(
            frames,
            gate,
            tau_decide,
            p_star,
            aero_selection=str(job.cfg.get("aero_selection", "standard")),
            aero_beta=job.cfg.get("aero_beta"),
        )
        f = frames[idx]
        verified_overlap = verified_raw = build_ok = verify_error = verify_wall = None
        timed_out = False
        verified_clean = false_success = None
        if job.verify:
            v = _verify_frame(idx, cand_advs, start, verify_cache, job.verify_timeout_s)
            build_ok = v["build_ok"]
            verified_overlap = v["verified_overlap_mm3"]
            verified_raw = v["verified_raw_overlap_mm3"]
            verify_error = v["error"]
            verify_wall = v["wall_s"]
            timed_out = v.get("timed_out", False)
            if verified_overlap is not None:
                verified_clean = verified_overlap <= tau
                false_success = bool(mlp_clean and not verified_clean)
        return {
            "gate": gate,
            "tau_decide_mm3": float(tau_decide),
            "p_star": (float(p_star) if p_star is not None else None),
            "selected_frame_idx": idx,
            "selection_reason": reason,
            "mlp_clean": bool(mlp_clean),
            "adv_hash": _adv_hash(cand_advs[idx]),
            "final_pred_overlap_mm3": f["pred_overlap_mm3"],
            "p_overlap": f.get("p_overlap"),
            "final_verified_overlap_mm3": verified_overlap,
            "final_verified_raw_overlap_mm3": verified_raw,
            "verified_clean": verified_clean,
            "false_success": false_success,
            "build_ok": build_ok,
            "verify_error": verify_error,
            "verify_wall_s": verify_wall,
            "timed_out": timed_out,
            "normalized_distance": f["normalized_distance"],
            "normalized_l1": f["normalized_l1"],
            "active_coordinate_count": f["active_coordinate_count"],
            "aero_D_aero": f.get("aero_D_aero"),
            "aero_R_aero": f.get("aero_R_aero"),
        }

    sweep_rows = [_op_row(g, t, p) for (g, t, p) in ops]
    head = sweep_rows[0]  # ops[0] is the headline operating point
    sel_idx = head["selected_frame_idx"]
    for fr in frames:
        fr["is_selected"] = fr["frame_idx"] == sel_idx
    sel = frames[sel_idx]

    mlp_clean = head["mlp_clean"]
    verified_overlap = head["final_verified_overlap_mm3"]
    verified_clean = head["verified_clean"]
    false_success = head["false_success"]
    build_ok = head["build_ok"]
    missed = (
        bool((not mlp_clean) and verified_clean) if verified_overlap is not None else None
    )

    if not job.verify:
        status = "mlp_clean" if mlp_clean else "mlp_unresolved"
    elif build_ok is False:
        status = "build_fail"
    elif verified_clean:
        status = "verified_clean"
    elif mlp_clean:
        status = "false_success"
    else:
        status = "unresolved"

    row = {
        "rank": int(start["rank"]),
        "start_id": start["start_id"],
        "sample_idx": int(start["sample_idx"]),
        "family": start.get("family"),
        "status": status,
        "selection_reason": head["selection_reason"],
        "selected_frame_idx": sel_idx,
        "selected_step_size": sel["step_size"],
        "changed": bool(sel["normalized_distance"] > DEFAULT_CHANGE_TOL),
        "start_overlap_mm3": float(start["overlap_mm3_cut_each"]),
        "final_pred_overlap_mm3": sel["pred_overlap_mm3"],
        "final_verified_overlap_mm3": verified_overlap,
        "final_verified_raw_overlap_mm3": head["final_verified_raw_overlap_mm3"],
        "mlp_clean": bool(mlp_clean),
        "verified_clean": verified_clean,
        "false_success": false_success,
        "missed": missed,
        "build_ok": build_ok,
        "verify_error": head["verify_error"],
        "verify_wall_s": head["verify_wall_s"],
        "normalized_distance": sel["normalized_distance"],
        "normalized_l1": sel["normalized_l1"],
        "active_coordinate_count": sel["active_coordinate_count"],
        "active_driver_deltas": frame_active[sel_idx],
        "top_gradients": start_top,
        "all_gradients": proposal.get("all_gradients", {}),
        "gradient_direction": proposal.get("gradient_direction", {}),
        "n_candidates": len(frames),
        "adv_start": adv0,
        "adv_final": cand_advs[sel_idx],
        "frames": frames,
    }
    if perf_sur is not None and opt_keys:
        row["aero_preserve"] = {
            **perf_sur.summary_for_u(adv0, opt_keys, u_from_adv(cand_advs[sel_idx], opt_keys), u_ref=u0),
            "operating_param": job.cfg.get("aero_param"),
            "operating_strength": job.cfg.get("aero_strength"),
            "aero_selection": job.cfg.get("aero_selection", "standard"),
            "aero_beta": job.cfg.get("aero_beta"),
            "projection_fallback_count": proposal.get("projection_fallback_count", 0),
            "aero_budget_fallback": bool(
                job.cfg.get("aero_selection") == "budget_lowest_drift"
                and head["selection_reason"] != "aero_budget_lowest_drift_clean"
                and bool(mlp_clean)
            ),
        }
    if job.sweep and len(ops) > 1:
        row["sweep"] = sweep_rows
    return row


def _agg(xs, fn):
    return float(fn(xs)) if xs else None


def _percentile(xs, q: float):
    return float(np.percentile(xs, q)) if xs else None


def _aero_statistics(starts: list[dict], *, verify: bool) -> dict | None:
    rows = [s for s in starts if isinstance(s.get("aero_preserve"), dict)]
    if not rows:
        return None
    raw_successes = [s for s in rows if s.get("verified_clean")] if verify else [s for s in rows if s.get("mlp_clean")]
    if not raw_successes:
        raw_successes = rows
    successes = [s for s in raw_successes if s["aero_preserve"].get("start_within_training_bounds", True)]
    n_excluded = len(raw_successes) - len(successes)
    r_vals = [_safe_float(s["aero_preserve"].get("R_aero")) for s in successes]
    d_vals = [_safe_float(s["aero_preserve"].get("D_aero")) for s in successes]
    per_metric_std: dict[str, list[float]] = {name: [] for name in CORE4_NAMES}
    per_metric_rel: dict[str, list[float]] = {}
    fallback_budget = 0
    fallback_projection = 0
    for s in successes:
        ap = s["aero_preserve"]
        fallback_budget += int(bool(ap.get("aero_budget_fallback")))
        fallback_projection += int(ap.get("projection_fallback_count") or 0)
        for name, value in (ap.get("standardized_delta") or {}).items():
            if name in per_metric_std:
                per_metric_std[name].append(abs(_safe_float(value)))
        for name, value in (ap.get("reportable_relative_delta") or {}).items():
            per_metric_rel.setdefault(name, []).append(abs(_safe_float(value)))
    return {
        "operating_param": rows[0]["aero_preserve"].get("operating_param"),
        "operating_strength": rows[0]["aero_preserve"].get("operating_strength"),
        "n_with_aero": len(rows),
        "n_success_for_aero": len(successes),
        "n_excluded_degenerate_baseline": n_excluded,
        "median_R_aero": _agg(r_vals, np.median),
        "iqr_R_aero": (
            (_percentile(r_vals, 75) - _percentile(r_vals, 25))
            if r_vals and _percentile(r_vals, 75) is not None
            else None
        ),
        "p90_R_aero": _percentile(r_vals, 90),
        "median_D_aero": _agg(d_vals, np.median),
        "within_standardized_radius_0p5": (
            len([v for v in r_vals if v <= 0.5]) / len(r_vals) if r_vals else None
        ),
        "within_standardized_radius_1p0": (
            len([v for v in r_vals if v <= 1.0]) / len(r_vals) if r_vals else None
        ),
        "per_metric_median_abs_standardized": {
            name: _agg(vals, np.median) for name, vals in per_metric_std.items()
        },
        "per_metric_median_abs_rel_reportable": {
            name: _agg(vals, np.median) for name, vals in per_metric_rel.items()
        },
        "aero_budget_fallback_count": fallback_budget,
        "projection_fallback_count": fallback_projection,
    }


def _statistics(
    starts: list[dict],
    *,
    verify: bool,
    tau: float,
    tau_decide: float,
    p_star: float | None,
    seed: int,
    operating_points: list,
) -> dict:
    n = len(starts)
    changed = [s for s in starts if s["changed"]]
    distances = [s["normalized_distance"] for s in starts]
    distances_changed = [s["normalized_distance"] for s in changed]
    l1 = [s.get("normalized_l1", 0.0) for s in starts]
    active = [s["active_coordinate_count"] for s in starts]
    start_ov = [s["start_overlap_mm3"] for s in starts]
    mlp_clean = [s for s in starts if s["mlp_clean"]]

    verified_done = [s for s in starts if s["final_verified_overlap_mm3"] is not None]
    verified_clean = [s for s in verified_done if s["verified_clean"]]
    false_success = [s for s in verified_done if s["false_success"]]
    missed = [s for s in verified_done if s["missed"]]
    build_fail = [s for s in starts if s["status"] == "build_fail"]
    final_ov = [s["final_verified_overlap_mm3"] for s in verified_done]
    log_reduction = [
        math.log10(s["start_overlap_mm3"]) - math.log10(max(s["final_verified_overlap_mm3"], 1e-9))
        for s in verified_done
        if s["start_overlap_mm3"] > 0
    ]

    # Proximity over SUCCESSFUL (verified-clean) repairs only — the thesis §5.2
    # convention (mean L2/L1/L0 "among the repairs they find"; a low-coverage row
    # looks artificially close, so n is always reported). Falls back to all
    # selected finals only when verification was skipped (--no-verify smoke).
    prox = verified_clean if (verify and verified_clean) else starts
    mean_l2_recovered = _agg([s["normalized_distance"] for s in prox], np.mean)
    mean_l1_recovered = _agg([s.get("normalized_l1", 0.0) for s in prox], np.mean)
    mean_l0_recovered = _agg([s["active_coordinate_count"] for s in prox], np.mean)

    # Per-operating-point aggregate (feeds the §K.2 tau-sweep tables). Indexed by
    # position in `operating_points`, which every start's sweep list shares.
    sweep_agg: list[dict] = []
    if operating_points:
        for op_idx, (gate, td, ps) in enumerate(operating_points):
            rows = [s["sweep"][op_idx] for s in starts if s.get("sweep") and op_idx < len(s["sweep"])]
            vdone = [r for r in rows if r["final_verified_overlap_mm3"] is not None]
            vclean = [r for r in vdone if r["verified_clean"]]
            # Mean L2/L1/L0 over the SUCCESSFUL finals at this operating point
            # (thesis convention); all rows only when verification is off.
            prox_rows = vclean if (verify and vclean) else rows
            sweep_agg.append(
                {
                    "gate": gate,
                    "tau_decide_mm3": float(td),
                    "p_star": (float(ps) if ps is not None else None),
                    "verified_count": len(vdone),
                    "verified_clean_count": len(vclean),
                    "false_success_count": len([r for r in vdone if r["false_success"]]),
                    "mlp_clean_count": len([r for r in rows if r["mlp_clean"]]),
                    "proximity_n": len(prox_rows),
                    "mean_l2": _agg([r["normalized_distance"] for r in prox_rows], np.mean),
                    "mean_l1": _agg([r["normalized_l1"] for r in prox_rows], np.mean),
                    "mean_l0": _agg([r["active_coordinate_count"] for r in prox_rows], np.mean),
                    "mean_final_overlap_mm3": _agg(
                        [r["final_verified_overlap_mm3"] for r in vdone], np.mean
                    ),
                }
            )

    stats = {
        "verification": "cad" if verify else "none",
        "tau_mm3": tau,
        "tau_decide_mm3": float(tau_decide),
        "binary_p_star": (float(p_star) if p_star is not None else None),
        "seed": int(seed),
        "n_starts": n,
        # --- how much did designs move ---
        "n_changed": len(changed),
        "n_unchanged": n - len(changed),
        "changed_rate": len(changed) / max(n, 1),
        "mean_normalized_distance": _agg(distances, np.mean),
        "median_normalized_distance": _agg(distances, np.median),
        "mean_normalized_distance_changed": _agg(distances_changed, np.mean),
        "mean_l1": _agg(l1, np.mean),
        "median_l1": _agg(l1, np.median),
        "mean_active_coordinates": _agg(active, np.mean),
        "median_active_coordinates": _agg(active, np.median),
        # proximity over verified-clean repairs (thesis §5.2 convention; headline)
        "mean_l2_recovered": mean_l2_recovered,
        "mean_l1_recovered": mean_l1_recovered,
        "mean_l0_recovered": mean_l0_recovered,
        "n_recovered_for_proximity": len(prox),
        # --- MLP-believed vs ground-truth recovery ---
        "mlp_clean_count": len(mlp_clean),
        "mlp_clean_rate": len(mlp_clean) / max(n, 1),
        "verified_count": len(verified_done),
        "verified_clean_count": len(verified_clean),
        "verified_clean_rate": (len(verified_clean) / len(verified_done)) if verified_done else None,
        "false_success_count": len(false_success),
        "false_success_rate": (len(false_success) / len(verified_done)) if verified_done else None,
        "missed_clean_count": len(missed),
        "build_fail_count": len(build_fail),
        # --- overlap magnitudes ---
        "mean_start_overlap_mm3": _agg(start_ov, np.mean),
        "median_start_overlap_mm3": _agg(start_ov, np.median),
        "mean_final_overlap_mm3": _agg(final_ov, np.mean),
        "median_final_overlap_mm3": _agg(final_ov, np.median),
        "mean_log10_overlap_reduction": _agg(log_reduction, np.mean),
        "sweep": sweep_agg,
    }
    aero_stats = _aero_statistics(starts, verify=verify)
    if aero_stats is not None:
        stats["aero_preserve"] = aero_stats
    return stats


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _is_complete(run_dir: Path, expected_n: int) -> bool:
    manifest = run_dir / "manifest.json"
    if not manifest.exists() or not (run_dir / "viewer_data.json").exists():
        return False
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(m.get("n_starts", -1)) == expected_n


def _run_group(
    *,
    optimizer_id: str,
    model_id: str,
    checkpoint: Path,
    benchmark: dict,
    runs_dir: Path,
    workers: int,
    tau: float,
    verify: bool,
    cfg: dict,
    group_id: str,
    group_label: str,
    gate: str,
    p_star: float,
    seed: int,
    sweep: bool,
    verify_timeout_s: float,
    resume: bool,
    run_suffix: str | None = None,
) -> dict:
    spec = OPTIMIZERS[optimizer_id]
    opt_slug = f"{optimizer_id}__{run_suffix}" if run_suffix else optimizer_id
    subgroup_id = f"{opt_slug}__{model_id}"
    run_id = f"{group_id}__{subgroup_id}"
    model_title = MODEL_TITLES.get(model_id, model_id)
    suffix_label = f" ({run_suffix})" if run_suffix else ""
    subgroup_label = f"{spec.label}{suffix_label} — {model_title}"
    title = subgroup_label
    run_dir = runs_dir / run_id
    expected_n = len(benchmark["starts"])

    if resume and _is_complete(run_dir, expected_n):
        stats = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
        print(f"[{run_id}] complete ({expected_n} starts) -> skip (resume)")
        return stats

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    operating_points = _operating_points(gate, tau, p_star, sweep)
    cfg = {**cfg, "clean_gate": gate, "tau_decide": tau, "p_star": p_star, "seed": seed}

    jobs = [
        JobSpec(
            optimizer_id=optimizer_id,
            checkpoint=str(checkpoint.resolve()),
            run_dir=str(run_dir.resolve()),
            start=start,
            cfg=cfg,
            tau_mm3=tau,
            verify=verify,
            operating_points=tuple(operating_points),
            sweep=sweep,
            verify_timeout_s=verify_timeout_s,
        )
        for start in benchmark["starts"]
    ]
    print(
        f"\n[{run_id}] {len(jobs)} starts @ {workers} workers "
        f"(verify={'cad' if verify else 'none'}, sweep={'on' if sweep else 'off'}, "
        f"gate={gate}, {len(operating_points)} op pts)"
    )
    t0 = time.perf_counter()
    starts: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_pin_worker, mp_context=_mp()) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                row = fut.result()
                starts.append(row)
                print(
                    f"  rank={row['rank']:03d} {row['status']:14s} "
                    f"pred={_safe_float(row['final_pred_overlap_mm3']):9.2f} "
                    f"true={_safe_float(row['final_verified_overlap_mm3']):9.2f} "
                    f"dist={row['normalized_distance']:.4f} active={row['active_coordinate_count']}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {job.start.get('start_id')}: {exc}")

    starts.sort(key=lambda s: int(s["rank"]))
    wall = time.perf_counter() - t0
    stats = _statistics(
        starts,
        verify=verify,
        tau=tau,
        tau_decide=tau,
        p_star=(p_star if gate == "binary_and" else None),
        seed=seed,
        operating_points=operating_points,
    )
    stats["wall_s_total"] = wall
    stats["wall_s_per_start"] = wall / max(len(starts), 1)

    selection_rule = "closest predicted-clean (L2); else lowest predicted overlap"
    viewer_data = {
        "run_id": run_id,
        "title": title,
        "group_id": group_id,
        "group_label": group_label,
        "subgroup_id": subgroup_id,
        "subgroup_label": subgroup_label,
        "model_id": model_id,
        "optimizer_id": optimizer_id,
        "optimizer_label": spec.label,
        "benchmark_id": benchmark.get("benchmark_id"),
        "tau_mm3": tau,
        "tau_decide_mm3": tau,
        "binary_p_star": (p_star if gate == "binary_and" else None),
        "seed": seed,
        "part_colors": PART_COLORS,
        "optimizer_config": {
            **cfg,
            "selection_rule": selection_rule,
            "tau_decide_mm3": tau,
            "binary_p_star": (p_star if gate == "binary_and" else None),
            "seed": seed,
        },
        "statistics": stats,
        "starts": starts,
    }
    _write_json(run_dir / "viewer_data.json", viewer_data)
    _write_json(run_dir / "statistics.json", stats)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "title": title,
            "group_id": group_id,
            "group_label": group_label,
            "subgroup_id": subgroup_id,
            "subgroup_label": subgroup_label,
            "model_id": model_id,
            "optimizer_id": optimizer_id,
            "benchmark_id": benchmark.get("benchmark_id"),
            "n_starts": len(starts),
            "statistics": stats,
        },
    )
    print(f"[{run_id}] saved -> {run_dir}  ({wall:.1f}s, {stats['wall_s_per_start']:.2f}s/start)")
    if verify:
        print(
            f"  verified clean {stats['verified_clean_count']}/{stats['n_starts']} | "
            f"MLP clean {stats['mlp_clean_count']} | false-OK {stats['false_success_count']} | "
            f"changed {stats['n_changed']} | mean L2 {_safe_float(stats['mean_normalized_distance']):.4f}"
        )
    else:
        print(
            f"  MLP clean {stats['mlp_clean_count']}/{stats['n_starts']} | "
            f"changed {stats['n_changed']} | mean L2 {_safe_float(stats['mean_normalized_distance']):.4f}"
        )
    return stats


def _run_all_meta(
    *,
    model_id: str,
    runs_dir: Path,
    group_id: str,
    group_label: str,
    benchmark: dict,
    tau: float,
    seed: int,
    resume: bool,
) -> dict:
    """'All' meta-optimizer: per start pick the best CAD-verified final across the
    six single-optimizer sibling subgroups of this group (§F.3). Reuses already
    computed verifications — no new CadQuery builds.
    """
    optimizer_id = "all"
    subgroup_id = f"{optimizer_id}__{model_id}"
    run_id = f"{group_id}__{subgroup_id}"
    model_title = MODEL_TITLES.get(model_id, model_id)
    subgroup_label = f"All (best of 6, CAD-adjudicated) — {model_title}"
    run_dir = runs_dir / run_id
    expected_n = len(benchmark["starts"])

    if resume and _is_complete(run_dir, expected_n):
        stats = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
        print(f"[{run_id}] complete -> skip (resume)")
        return stats

    sibling_starts: dict[str, dict[int, dict]] = {}
    for opt_id in CANONICAL_SUITE:
        vd = runs_dir / f"{group_id}__{opt_id}__{model_id}" / "viewer_data.json"
        if not vd.exists():
            raise SystemExit(
                f"[all] missing sibling subgroup {opt_id} for {group_id} "
                f"(expected {vd}); run the six optimizers before 'all'."
            )
        data = json.loads(vd.read_text(encoding="utf-8"))
        sibling_starts[opt_id] = {int(s["rank"]): s for s in data["starts"]}

    ranks = sorted({int(s["rank"]) for s in benchmark["starts"]})
    chosen: list[dict] = []
    best_l2_vals: list[float] = []
    best_l0_vals: list[float] = []
    winner_hist: dict[str, int] = {opt: 0 for opt in CANONICAL_SUITE}

    for rank in ranks:
        cand = [(opt_id, sibling_starts[opt_id][rank]) for opt_id in CANONICAL_SUITE if rank in sibling_starts[opt_id]]
        if not cand:
            continue
        clean = [(o, s) for (o, s) in cand if s.get("verified_clean")]
        if clean:
            # default pick = best L0 (then L2); also record best-L2 for stats
            o_l0, s_l0 = min(clean, key=lambda t: (t[1]["active_coordinate_count"], t[1]["normalized_distance"]))
            o_l2, s_l2 = min(clean, key=lambda t: (t[1]["normalized_distance"], t[1]["active_coordinate_count"]))
            best_l0_vals.append(float(s_l0["active_coordinate_count"]))
            best_l2_vals.append(float(s_l2["normalized_distance"]))
            winner = (o_l0, s_l0, "all_best_l0")
        else:
            verified = [(o, s) for (o, s) in cand if s.get("final_verified_overlap_mm3") is not None]
            if verified:
                o_b, s_b = min(verified, key=lambda t: t[1]["final_verified_overlap_mm3"])
                winner = (o_b, s_b, "all_lowest_overlap")
            else:
                o_b, s_b = min(cand, key=lambda t: t[1]["final_pred_overlap_mm3"])
                winner = (o_b, s_b, "all_lowest_overlap")
        win_opt, win_row, reason = winner
        winner_hist[win_opt] = winner_hist.get(win_opt, 0) + 1
        merged = dict(win_row)
        merged["selection_reason"] = reason
        merged["source_optimizer"] = win_opt
        merged.pop("sweep", None)  # the All run reports a single adjudicated final
        chosen.append(merged)

    chosen.sort(key=lambda s: int(s["rank"]))
    stats = _statistics(
        chosen, verify=True, tau=tau, tau_decide=tau, p_star=None, seed=seed, operating_points=[]
    )
    stats["all_verified_clean_count"] = len([s for s in chosen if s.get("verified_clean")])
    stats["best_L2_average"] = _agg(best_l2_vals, np.mean)
    stats["best_L0_average"] = _agg(best_l0_vals, np.mean)
    stats["winner_histogram"] = winner_hist
    stats["wall_s_total"] = 0.0
    stats["wall_s_per_start"] = 0.0

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    viewer_data = {
        "run_id": run_id,
        "title": subgroup_label,
        "group_id": group_id,
        "group_label": group_label,
        "subgroup_id": subgroup_id,
        "subgroup_label": subgroup_label,
        "model_id": model_id,
        "optimizer_id": optimizer_id,
        "optimizer_label": "All (best of 6, CAD-adjudicated)",
        "benchmark_id": benchmark.get("benchmark_id"),
        "tau_mm3": tau,
        "tau_decide_mm3": tau,
        "binary_p_star": None,
        "seed": seed,
        "part_colors": PART_COLORS,
        "optimizer_config": {
            "meta": "all",
            "members": CANONICAL_SUITE,
            "selection_rule": "per start: best-L0 among CAD-verified-clean finals; else lowest verified overlap",
            "seed": seed,
        },
        "statistics": stats,
        "starts": chosen,
    }
    _write_json(run_dir / "viewer_data.json", viewer_data)
    _write_json(run_dir / "statistics.json", stats)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "title": subgroup_label,
            "group_id": group_id,
            "group_label": group_label,
            "subgroup_id": subgroup_id,
            "subgroup_label": subgroup_label,
            "model_id": model_id,
            "optimizer_id": optimizer_id,
            "benchmark_id": benchmark.get("benchmark_id"),
            "n_starts": len(chosen),
            "statistics": stats,
        },
    )
    print(
        f"[{run_id}] saved -> {run_dir}  (All meta) | union recovery "
        f"{stats['all_verified_clean_count']}/{len(chosen)} | "
        f"best-L0 avg {_safe_float(stats['best_L0_average']):.2f} | "
        f"best-L2 avg {_safe_float(stats['best_L2_average']):.4f}"
    )
    return stats


def _filter_starts(benchmark: dict, *, rank_stride, ranks, limit_starts) -> dict:
    out = dict(benchmark)
    starts = out["starts"]
    if rank_stride is not None:
        starts = [s for s in starts if int(s.get("rank", 0)) % rank_stride == 0]
    if ranks is not None:
        wanted = {int(x.strip()) for x in ranks.split(",") if x.strip()}
        starts = [s for s in starts if int(s.get("rank", 0)) in wanted]
    if limit_starts is not None:
        starts = starts[:limit_starts]
    out["starts"] = starts
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chapter 7 AeroForge overlap-repair optimizer workbench")
    ap.add_argument(
        "--optimizer",
        type=str,
        default="one_shot_gradient_line_search",
        choices=sorted(OPTIMIZERS) + ["all"],
        help="Optimizer registry key, or 'all' for the per-variant best-of-6 meta-run.",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=sorted(VARIANTS),
        help="Model variant a–f (§B.1): sets checkpoint + feature_grad_mode + gate + group.",
    )
    ap.add_argument(
        "--group",
        type=str,
        default=None,
        help="Override the top-level group label (defaults to the variant's label).",
    )
    ap.add_argument("--group-id", type=str, default=None, help="Override the top-level group id/run-dir prefix.")
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Legacy: comma-separated model_ids (each <models-dir>/<id>.pt). Ignored when --variant is set.",
    )
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tau-mm3", type=float, default=DEFAULT_TAU_MM3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-verify", action="store_true", help="Skip CadQuery; MLP-only stats (fast)")
    sweep = ap.add_mutually_exclusive_group()
    sweep.add_argument("--sweep", dest="sweep", action="store_true", help="Operating-threshold sweep (default on)")
    sweep.add_argument("--no-sweep", dest="sweep", action="store_false", help="Single headline operating point only")
    ap.set_defaults(sweep=True)
    ap.add_argument("--resume", action="store_true", help="Skip subgroups whose artifacts are already complete")
    ap.add_argument("--verify-timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT_S)
    ap.add_argument("--perf-checkpoint", type=Path, default=DEFAULT_PERF_CHECKPOINT)
    ap.add_argument("--run-suffix", type=str, default=None, help="Optional optimizer-strength slug in subgroup/run_id.")
    ap.add_argument("--aero-param", type=str, default=None, help="Aero-preserve strength parameter name (lambda/alpha/beta).")
    ap.add_argument("--aero-strength", type=float, default=None, help="Aero-preserve strength value for reporting.")
    ap.add_argument("--aero-lambda", type=float, default=None)
    ap.add_argument("--aero-alpha", type=float, default=None)
    ap.add_argument("--aero-beta", type=float, default=None)
    ap.add_argument(
        "--aero-selection",
        choices=["standard", "budget_lowest_drift"],
        default=None,
        help="Selection mode for aero-preserving candidates.",
    )
    ap.add_argument("--limit-starts", type=int, default=None)
    ap.add_argument("--rank-stride", type=int, default=None, help="Only ranks divisible by N (e.g. 10)")
    ap.add_argument("--ranks", type=str, default=None, help="Exact ranks, e.g. 10,50,100")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark = _filter_starts(
        benchmark, rank_stride=args.rank_stride, ranks=args.ranks, limit_starts=args.limit_starts
    )

    if args.variant is None and args.optimizer == "all":
        raise SystemExit("--optimizer all requires --variant (the All meta-run is per model variant)")

    # Resolve the (model, group, gate, feature_grad_mode) tuple(s) to run.
    if args.variant is not None:
        v = VARIANTS[args.variant]
        targets = [
            {
                "model_id": v["model_id"],
                "feature_grad_mode": v["feature_grad_mode"],
                "clean_gate": v["clean_gate"],
                "p_star": v["p_star"],
                "group_id": args.group_id or v["group_id"],
                "group_label": args.group or v["group_label"],
            }
        ]
    else:
        model_ids = [m.strip() for m in (args.models or "eng_log_huber_20k").split(",") if m.strip()]
        group_label = args.group or "Ungrouped"
        targets = [
            {
                "model_id": mid,
                "feature_grad_mode": "full",
                "clean_gate": "reg",
                "p_star": 0.5,
                "group_id": args.group_id or _slug(group_label),
                "group_label": group_label,
            }
            for mid in model_ids
        ]

    summary = []
    for t in targets:
        print(f"\nGroup: {t['group_label']!r} (id={t['group_id']})")
        if args.optimizer == "all":
            stats = _run_all_meta(
                model_id=t["model_id"],
                runs_dir=args.runs_dir,
                group_id=t["group_id"],
                group_label=t["group_label"],
                benchmark=benchmark,
                tau=args.tau_mm3,
                seed=args.seed,
                resume=args.resume,
            )
            summary.append((f"{t['group_id']}__all__{t['model_id']}", stats))
            continue

        checkpoint = args.models_dir / f"{t['model_id']}.pt"
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        cfg = dict(OPTIMIZERS[args.optimizer].default_cfg)
        cfg["tau_mm3"] = args.tau_mm3
        cfg["feature_grad_mode"] = t["feature_grad_mode"]
        if args.optimizer.startswith("aero_"):
            if args.variant != "a":
                raise SystemExit("aero-preserving optimizers must run with --variant a (model-a parity guard)")
            if (
                t["model_id"] != "eng_multitask_gate_strong_100k"
                or t["feature_grad_mode"] != "detached"
                or t["clean_gate"] != "reg"
            ):
                raise SystemExit(f"model-a parity guard failed: {t}")
            cfg["perf_checkpoint"] = str(args.perf_checkpoint.resolve())
            cfg["aero_param"] = args.aero_param
            cfg["aero_strength"] = args.aero_strength
            if args.aero_lambda is not None:
                cfg["aero_lambda"] = float(args.aero_lambda)
            if args.aero_alpha is not None:
                cfg["aero_alpha"] = float(args.aero_alpha)
            if args.aero_beta is not None:
                beta = float(args.aero_beta)
                cfg["aero_beta"] = 1.0e99 if not math.isfinite(beta) else beta
            if args.aero_selection is not None:
                cfg["aero_selection"] = args.aero_selection
        stats = _run_group(
            optimizer_id=args.optimizer,
            model_id=t["model_id"],
            checkpoint=checkpoint,
            benchmark=benchmark,
            runs_dir=args.runs_dir,
            workers=args.workers,
            tau=args.tau_mm3,
            verify=not args.no_verify,
            cfg=cfg,
            group_id=t["group_id"],
            group_label=t["group_label"],
            gate=t["clean_gate"],
            p_star=t["p_star"],
            seed=args.seed,
            sweep=args.sweep,
            verify_timeout_s=args.verify_timeout,
            resume=args.resume,
            run_suffix=args.run_suffix,
        )
        summary.append((f"{t['group_id']}__{args.optimizer}__{t['model_id']}", stats))

    print("\n=== summary ===")
    for run_id, stats in summary:
        if stats.get("verification") == "cad":
            print(
                f"  {run_id}: verified {stats['verified_clean_count']}/{stats['n_starts']} "
                f"(false-OK {stats['false_success_count']}), changed {stats['n_changed']}, "
                f"mean L2 {_safe_float(stats['mean_normalized_distance']):.4f}"
            )
        else:
            print(
                f"  {run_id}: MLP-clean {stats['mlp_clean_count']}/{stats['n_starts']}, "
                f"changed {stats['n_changed']}, mean L2 {_safe_float(stats['mean_normalized_distance']):.4f}"
            )


if __name__ == "__main__":
    main()
