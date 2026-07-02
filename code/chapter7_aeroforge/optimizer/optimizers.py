#!/usr/bin/env python3
"""Optimizer registry for the Chapter 7 overlap-repair workbench.

An optimizer is a *pure-MLP* function: given a starting ADV and a `Surrogate`, it
proposes a trajectory of candidate ADVs using only MLP inference and arithmetic
rules on the normalized design vector. It never builds geometry and never calls
CadQuery. The workbench scores every candidate with the MLP, applies one global
selection rule, and runs a single CadQuery verification on the chosen final
design (see `workbench.py`).

Each optimizer returns a dict:
    {
      "opt_keys": list[str],                 # normalized drivers it moved
      "top_gradients": list[dict],           # start-design top gradients (display)
      "gradient_direction": dict[str, float],# start-design descent direction
      "candidates": list[dict],              # candidate[0] is always the start
    }
Each candidate is {"adv", "step_size", "stage", "note", optional per-frame
"top_gradients"/"gradient_direction"}.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

from dataclasses import dataclass, field
from typing import Callable

import sys
from pathlib import Path

import numpy as np

from chapter7_aeroforge.optimizer.model_io import (  # noqa: E402
    Surrogate,
    adv_from_u,
    opt_keys_for_adv,
    u_from_adv,
)
from chapter7_aeroforge.optimizer.perf_surrogate import get_perf_surrogate  # noqa: E402


def _direction_dict(opt_keys: list[str], direction: np.ndarray) -> dict[str, float]:
    return {k: float(direction[i]) for i, k in enumerate(opt_keys) if abs(direction[i]) > 0}


def one_shot_top5(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Single gradient evaluation; sweep step sizes along the fixed top-k descent.

    Diagnostic baseline: take the k most active normalized gradients at the start,
    fix that descent direction, and probe `steps` (log-spaced) step magnitudes.
    Pure MLP; the workbench picks the closest predicted-clean probe.
    """
    k = int(cfg.get("top_k", 5))
    steps = list(cfg.get("steps", np.logspace(-3, 0, 10).tolist()))
    top, direction, opt_keys = sur.gradient_top_k(adv0, k)
    grad_full = sur.gradient(adv0, opt_keys)
    all_gradients = {key: float(grad_full[i]) for i, key in enumerate(opt_keys)}
    u0 = u_from_adv(adv0, opt_keys)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    for step in steps:
        u = np.clip(u0 + float(step) * direction, 0.0, 1.0)
        candidates.append(
            {
                "adv": adv_from_u(adv0, opt_keys, u),
                "step_size": float(step),
                "stage": "grid_step",
                "note": f"top-{k} one-shot step={step:.4g}",
            }
        )
    return {
        "opt_keys": opt_keys,
        "top_gradients": top,
        "all_gradients": all_gradients,
        "gradient_direction": _direction_dict(opt_keys, direction),
        "candidates": candidates,
    }


def topk_descent(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Iterative top-k gradient descent in normalized space, MLP-only.

    At each iterate: recompute the gradient, restrict to the k most active
    coordinates, line-search a small log-spaced step grid, and accept the probe
    that minimizes predicted overlap. Stop when predicted overlap <= tau, when no
    probe improves, or after `max_iters`. This is the canonical "evaluate with the
    MLP, nudge the vector, re-evaluate until a criterion is met" repair loop.
    """
    k = int(cfg.get("top_k", 5))
    max_iters = int(cfg.get("max_iters", 12))
    probe_steps = list(cfg.get("probe_steps", np.logspace(-2.5, -0.3, 6).tolist()))
    tau = float(cfg.get("tau_mm3", sur.tau_mm3))

    opt_keys0 = None
    start_top: list[dict] = []
    start_dir: dict[str, float] = {}
    start_all: dict[str, float] = {}

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    adv = adv0
    _, pred = sur.predict(adv)

    for it in range(max_iters):
        if pred <= tau:
            break
        top, direction, opt_keys = sur.gradient_top_k(adv, k)
        if opt_keys0 is None:
            opt_keys0, start_top, start_dir = opt_keys, top, _direction_dict(opt_keys, direction)
            grad_full = sur.gradient(adv0, opt_keys)
            start_all = {key: float(grad_full[i]) for i, key in enumerate(opt_keys)}
        u_cur = u_from_adv(adv, opt_keys)

        best = None  # (pred, step, adv)
        for step in probe_steps:
            u = np.clip(u_cur + float(step) * direction, 0.0, 1.0)
            adv_probe = adv_from_u(adv, opt_keys, u)
            _, p = sur.predict(adv_probe)
            if best is None or p < best[0]:
                best = (p, float(step), adv_probe)

        if best is None or best[0] >= pred - 1e-9:  # no improvement
            break
        pred, step, adv = best
        candidates.append(
            {
                "adv": adv,
                "step_size": step,
                "stage": "descent_iter",
                "note": f"iter {it + 1} top-{k} descent step={step:.4g}",
                "top_gradients": top,
                "gradient_direction": _direction_dict(opt_keys, direction),
            }
        )

    return {
        "opt_keys": opt_keys0 or [],
        "top_gradients": start_top,
        "all_gradients": start_all,
        "gradient_direction": start_dir,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Canonical v3.1 optimizer suite
#
# Design contract:
#   * "Run to budget": optimizers generate their full candidate set independent
#     of the clean gate and operating thresholds. They never early-stop because
#     predicted overlap dropped below tau. The workbench applies the gate as a
#     *selection* rule afterwards, which makes the operating-threshold sweep a
#     cheap post-hoc re-selection over the same candidates.
#   * `feature_grad_mode` ("full" | "detached" | "raw") is read from cfg and
#     threaded into every gradient call so variants a/c (and b/d) differ only by
#     how the engineered-feature pathway contributes to the gradient.
#   * The single exception is trust-region *shrinkage*, where the clean gate
#     enters generation (rollback must stay clean); see §D.8.
# ---------------------------------------------------------------------------


def _start_gradient_block(adv0: dict, sur: Surrogate, mode: str, k) -> tuple:
    """Compute start-design gradient artifacts for display + a descent direction."""
    top, direction, opt_keys = sur.gradient_top_k(adv0, k, feature_grad_mode=mode)
    grad_full = sur.gradient(adv0, opt_keys, feature_grad_mode=mode)
    all_gradients = {key: float(grad_full[i]) for i, key in enumerate(opt_keys)}
    return top, direction, opt_keys, all_gradients, grad_full


def _perf_from_cfg(cfg: dict):
    perf = cfg.get("perf_sur")
    if perf is not None:
        return perf
    return get_perf_surrogate(cfg.get("perf_checkpoint"))


def _overlap_z(sur: Surrogate, adv: dict) -> float:
    z, _pred = sur.predict(adv)
    return float(z)


def _topk_mask(grad: np.ndarray, k) -> np.ndarray:
    mask = np.zeros_like(grad, dtype=np.float64)
    if k == "all":
        mask[:] = 1.0
        return mask
    kk = max(1, min(int(k), len(grad)))
    idx = np.argsort(-np.abs(grad))[:kk]
    mask[idx] = 1.0
    return mask


def aero_penalized_receding(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Receding gradient descent with standardized performance-drift penalty.

    Internal score uses overlap z-space + distance + lambda_aero * D_aero. The
    final workbench selection still applies the normal CAD-verified repair
    contract; aero metadata is stored for reporting/viewer use.
    """
    mode = cfg.get("feature_grad_mode", "detached")
    k = cfg.get("top_k", 5)
    max_iters = int(cfg.get("max_iters", 12))
    lam_dist = float(cfg.get("lambda_dist", cfg.get("penalty_lambda", 0.05)))
    lam_aero = float(cfg.get("aero_lambda", 1.0))
    probe_steps = list(cfg.get("probe_steps", np.logspace(-2.5, -0.3, 6).tolist()))
    perf = _perf_from_cfg(cfg)

    opt_keys = opt_keys_for_adv(adv0)
    u_start = u_from_adv(adv0, opt_keys)
    top0, dir0, _, start_all, grad0 = _start_gradient_block(adv0, sur, mode, k)
    start_dir = _direction_dict(opt_keys, dir0)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    adv = adv0
    u_cur = u_start.copy()
    score = _overlap_z(sur, adv)

    for it in range(max_iters):
        grad = sur.gradient(adv, opt_keys, feature_grad_mode=mode)
        direction = -grad * _topk_mask(grad, k)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            break
        direction /= norm

        best = None  # (score, z, drift, step, adv, u)
        for step in probe_steps:
            u = np.clip(u_cur + float(step) * direction, 0.0, 1.0)
            adv_probe = adv_from_u(adv0, opt_keys, u)
            z = _overlap_z(sur, adv_probe)
            drift = perf.drift_from_u(adv0, opt_keys, u, u_ref=u_start)
            sc = z + lam_dist * float(np.linalg.norm(u - u_start)) + lam_aero * drift
            if best is None or sc < best[0]:
                best = (sc, z, drift, float(step), adv_probe, u)

        if best is None or best[0] >= score - 1e-9:
            break
        score, _z, drift, step, adv, u_cur = best
        candidates.append(
            {
                "adv": adv,
                "step_size": step,
                "stage": "aero_penalized_iter",
                "note": f"iter {it + 1} lambda_aero={lam_aero:g} D_aero={drift:.4g} step={step:.4g}",
                "top_gradients": top0 if it == 0 else [],
                "gradient_direction": _direction_dict(opt_keys, direction),
            }
        )

    return {
        "opt_keys": opt_keys,
        "top_gradients": top0,
        "all_gradients": {key: float(grad0[i]) for i, key in enumerate(opt_keys)},
        "gradient_direction": start_dir,
        "candidates": candidates,
    }


def aero_tangent_receding(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Receding gradient descent projected away from standardized aero drift."""
    mode = cfg.get("feature_grad_mode", "detached")
    k = cfg.get("top_k", 5)
    max_iters = int(cfg.get("max_iters", 12))
    alpha = float(cfg.get("aero_alpha", 0.9))
    eps = float(cfg.get("projection_eps", 1e-4))
    probe_steps = list(cfg.get("probe_steps", np.logspace(-2.5, -0.3, 6).tolist()))
    perf = _perf_from_cfg(cfg)

    opt_keys = opt_keys_for_adv(adv0)
    top0, _dir0, _, start_all, grad0 = _start_gradient_block(adv0, sur, mode, k)
    u_cur = u_from_adv(adv0, opt_keys)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    adv = adv0
    pred_score = _overlap_z(sur, adv)
    fallback_count = 0

    for it in range(max_iters):
        grad_cur = sur.gradient(adv, opt_keys, feature_grad_mode=mode)
        g = grad_cur * _topk_mask(grad_cur, k)
        j = perf.jacobian_core4(adv0, opt_keys, u=u_cur)
        try:
            aero_component = j.T @ np.linalg.solve(j @ j.T + eps * np.eye(j.shape[0]), j @ g)
        except np.linalg.LinAlgError:
            aero_component = np.zeros_like(g)
        direction = -(g - alpha * aero_component)
        norm = float(np.linalg.norm(direction))
        projection_degenerate = norm < 1e-12
        if projection_degenerate:
            direction = -g
            norm = float(np.linalg.norm(direction))
            fallback_count += 1
        if norm < 1e-12:
            break
        direction /= norm

        best = None  # (z, step, adv, u)
        for step in probe_steps:
            u = np.clip(u_cur + float(step) * direction, 0.0, 1.0)
            adv_probe = adv_from_u(adv0, opt_keys, u)
            z = _overlap_z(sur, adv_probe)
            if best is None or z < best[0]:
                best = (z, float(step), adv_probe, u)
        if best is None or best[0] >= pred_score - 1e-9:
            break
        pred_score, step, adv, u_cur = best
        note = f"iter {it + 1} alpha={alpha:g} step={step:.4g}"
        if projection_degenerate:
            note += " projection_fallback"
        candidates.append(
            {
                "adv": adv,
                "step_size": step,
                "stage": "aero_tangent_iter",
                "note": note,
                "gradient_direction": _direction_dict(opt_keys, direction),
            }
        )

    return {
        "opt_keys": opt_keys,
        "top_gradients": top0,
        "all_gradients": start_all,
        "gradient_direction": _direction_dict(opt_keys, -grad0 / max(float(np.linalg.norm(grad0)), 1e-12)),
        "projection_fallback_count": fallback_count,
        "candidates": candidates,
    }


def aero_budget_trust_region(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Trust-region cloud with aero-budget-aware selection in the workbench.

    Candidate generation intentionally reuses the existing shrinkage implementation.
    The workbench filters clean candidates by ``D_aero <= aero_beta`` and selects
    the lowest predicted drift among survivors.
    """
    tr_cfg = {**cfg, "shrinkage": True}
    out = trust_region_hybrid(adv0, sur, tr_cfg)
    beta = cfg.get("aero_beta", float("inf"))
    out["aero_budget"] = float(beta)
    return out


def one_shot_gradient_line_search(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Single gradient evaluation; line-search step magnitudes along fixed descent.

    Generalizes `one_shot_top5` with a configurable `top_k` (int or ``"all"``)
    and `feature_grad_mode`. Pure MLP; the workbench selects the closest
    predicted-clean probe at each operating point.
    """
    mode = cfg.get("feature_grad_mode", "full")
    k = cfg.get("top_k", 5)
    steps = list(cfg.get("steps", np.logspace(-3, 0, 10).tolist()))
    top, direction, opt_keys, all_gradients, _ = _start_gradient_block(adv0, sur, mode, k)
    u0 = u_from_adv(adv0, opt_keys)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    for step in steps:
        u = np.clip(u0 + float(step) * direction, 0.0, 1.0)
        candidates.append(
            {
                "adv": adv_from_u(adv0, opt_keys, u),
                "step_size": float(step),
                "stage": "grid_step",
                "note": f"line-search step={step:.4g}",
            }
        )
    return {
        "opt_keys": opt_keys,
        "top_gradients": top,
        "all_gradients": all_gradients,
        "gradient_direction": _direction_dict(opt_keys, direction),
        "candidates": candidates,
    }


def receding_multistep_gradient(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Iterative top-k gradient descent, MLP-only, run to budget.

    At each iterate recompute the gradient, restrict to the k most active
    coordinates, line-search a small log-spaced step grid, and accept the probe
    that minimizes predicted overlap. Stop ONLY when no probe improves the
    prediction (stall) or after `max_iters` — never because pred dropped below
    tau (run-to-budget; the gate is applied later as a selection rule).
    """
    mode = cfg.get("feature_grad_mode", "full")
    k = cfg.get("top_k", 5)
    max_iters = int(cfg.get("max_iters", 12))
    probe_steps = list(cfg.get("probe_steps", np.logspace(-2.5, -0.3, 6).tolist()))

    opt_keys0: list[str] | None = None
    start_top: list[dict] = []
    start_dir: dict[str, float] = {}
    start_all: dict[str, float] = {}

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    adv = adv0
    _, pred = sur.predict(adv)

    for it in range(max_iters):
        top, direction, opt_keys = sur.gradient_top_k(adv, k, feature_grad_mode=mode)
        if opt_keys0 is None:
            opt_keys0, start_top, start_dir = opt_keys, top, _direction_dict(opt_keys, direction)
            _t, _d, _ok, start_all, _g = _start_gradient_block(adv0, sur, mode, k)
        u_cur = u_from_adv(adv, opt_keys)

        best = None  # (pred, step, adv)
        for step in probe_steps:
            u = np.clip(u_cur + float(step) * direction, 0.0, 1.0)
            adv_probe = adv_from_u(adv, opt_keys, u)
            _, p = sur.predict(adv_probe)
            if best is None or p < best[0]:
                best = (p, float(step), adv_probe)

        if best is None or best[0] >= pred - 1e-9:  # stall: no probe improves
            break
        pred, step, adv = best
        candidates.append(
            {
                "adv": adv,
                "step_size": step,
                "stage": "descent_iter",
                "note": f"iter {it + 1} top-{k} descent step={step:.4g}",
                "top_gradients": top,
                "gradient_direction": _direction_dict(opt_keys, direction),
            }
        )

    return {
        "opt_keys": opt_keys0 or [],
        "top_gradients": start_top,
        "all_gradients": start_all,
        "gradient_direction": start_dir,
        "candidates": candidates,
    }


def receding_multistep_penalty(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Receding-horizon descent that minimizes pred + lambda * ||u - u_start||.

    Same loop as `receding_multistep_gradient`, but each probe is scored by a
    distance-penalized objective so the search trades raw overlap reduction
    against staying close to the original design. Run to budget (stop on score
    stall or `max_iters`); selection / gating happen later in the workbench.
    """
    mode = cfg.get("feature_grad_mode", "full")
    k = cfg.get("top_k", 5)
    max_iters = int(cfg.get("max_iters", 12))
    lam = float(cfg.get("penalty_lambda", 0.05))
    probe_steps = list(cfg.get("probe_steps", np.logspace(-2.5, -0.3, 6).tolist()))

    opt_keys = opt_keys_for_adv(adv0)
    u_start0 = u_from_adv(adv0, opt_keys)

    top0, dir0, _, start_all, _ = _start_gradient_block(adv0, sur, mode, k)
    start_top, start_dir = top0, _direction_dict(opt_keys, dir0)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    adv = adv0
    _, pred = sur.predict(adv)
    score = pred  # distance term is 0 at the start

    for it in range(max_iters):
        top, direction, _ok = sur.gradient_top_k(adv, k, feature_grad_mode=mode)
        u_cur = u_from_adv(adv, opt_keys)

        best = None  # (score, pred, step, adv)
        for step in probe_steps:
            u = np.clip(u_cur + float(step) * direction, 0.0, 1.0)
            adv_probe = adv_from_u(adv, opt_keys, u)
            _, p = sur.predict(adv_probe)
            sc = p + lam * float(np.linalg.norm(u - u_start0))
            if best is None or sc < best[0]:
                best = (sc, p, float(step), adv_probe)

        if best is None or best[0] >= score - 1e-9:  # stall on the penalized score
            break
        score, pred, step, adv = best
        candidates.append(
            {
                "adv": adv,
                "step_size": step,
                "stage": "descent_iter",
                "note": f"iter {it + 1} penalized (lambda={lam:g}) step={step:.4g}",
                "top_gradients": top,
                "gradient_direction": _direction_dict(opt_keys, direction),
            }
        )

    return {
        "opt_keys": opt_keys,
        "top_gradients": start_top,
        "all_gradients": start_all,
        "gradient_direction": start_dir,
        "candidates": candidates,
    }


def full_coordinate_grid_refine(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Gradient-free coordinate search: probe each driver +/- on a coarse grid,
    then refine around the most promising single-driver moves.

    Generation never uses gradients (the start-design gradient is still computed
    for display only). Produces a fan of single-coordinate candidates; the
    workbench selects the closest predicted-clean one at each operating point.
    """
    mode = cfg.get("feature_grad_mode", "full")
    coarse = list(cfg.get("coarse_grid", [0.02, 0.05, 0.10, 0.20, 0.35]))
    refine_factors = list(cfg.get("refine_factors", [0.5, 0.75, 1.25, 1.5]))
    n_refine = int(cfg.get("n_refine", 3))

    opt_keys = opt_keys_for_adv(adv0)
    u0 = u_from_adv(adv0, opt_keys)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    probes: list[tuple] = []  # (pred, idx, key, sign, step)
    for i, key in enumerate(opt_keys):
        for sign in (1.0, -1.0):
            for s in coarse:
                u = u0.copy()
                u[i] = float(np.clip(u0[i] + sign * s, 0.0, 1.0))
                adv_probe = adv_from_u(adv0, opt_keys, u)
                _, p = sur.predict(adv_probe)
                sgn = "+" if sign > 0 else "-"
                candidates.append(
                    {
                        "adv": adv_probe,
                        "step_size": float(sign * s),
                        "stage": "coord_coarse",
                        "note": f"{key} {sgn}{s:g}",
                    }
                )
                probes.append((p, i, key, sign, s))

    for (_p, i, key, sign, s) in sorted(probes, key=lambda r: r[0])[:n_refine]:
        for f in refine_factors:
            snew = s * f
            u = u0.copy()
            u[i] = float(np.clip(u0[i] + sign * snew, 0.0, 1.0))
            sgn = "+" if sign > 0 else "-"
            candidates.append(
                {
                    "adv": adv_from_u(adv0, opt_keys, u),
                    "step_size": float(sign * snew),
                    "stage": "coord_refine",
                    "note": f"refine {key} {sgn}{snew:.4g}",
                }
            )

    top, direction, _ok, all_gradients, _g = _start_gradient_block(adv0, sur, mode, 5)
    return {
        "opt_keys": opt_keys,
        "top_gradients": top,
        "all_gradients": all_gradients,
        "gradient_direction": _direction_dict(opt_keys, direction),
        "candidates": candidates,
    }


def trust_region_hybrid(adv0: dict, sur: Surrogate, cfg: dict) -> dict:
    """Trust-region cloud search over the top-x active drivers across a radius
    ladder, optionally followed by a clean-preserving shrinkage cleanup.

    For each radius r in a growing ladder, sample a fixed pool of unit directions
    supported on the top-x most active drivers (signed coordinate axes + the
    gradient-descent direction + seeded random directions) and probe u0 + r*v.
    Run to budget over all radii. If `shrinkage` is on, the clean gate enters
    generation: starting from the closest predicted-clean candidate, each moved
    driver is rolled back toward the start as far as it can while the design
    stays clean at the variant's operating point (§D.6, §D.8).
    """
    mode = cfg.get("feature_grad_mode", "full")
    top_x = int(cfg.get("top_x", 10))
    dirs_per_radius = int(cfg.get("dirs_per_radius", 80))
    radii = list(cfg.get("radii", [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]))
    seed = int(cfg.get("seed", 12345))
    grad_bias_eta = float(cfg.get("grad_bias_eta", 0.5))
    shrinkage = bool(cfg.get("shrinkage", False))
    rng = np.random.default_rng(seed)

    top, direction, opt_keys, all_gradients, grad_full = _start_gradient_block(adv0, sur, mode, top_x)
    u0 = u_from_adv(adv0, opt_keys)
    n = len(opt_keys)
    topx_idx = np.argsort(-np.abs(grad_full))[: min(top_x, n)]

    # Direction pool on the top-x subspace (§D.5): (1) signed coordinate axes,
    # (3) gradient-biased noisy dirs normalize(-g_topx + eta*noise) incl. the pure
    # descent dir, then (2) seeded random unit vectors fill the remainder.
    g_topx = -grad_full[topx_idx]
    g_norm = float(np.linalg.norm(g_topx))
    g_unit = g_topx / g_norm if g_norm > 1e-12 else None

    pool: list[np.ndarray] = []
    for i in topx_idx:  # (1) signed coordinate axes on the active subspace
        e = np.zeros(n)
        e[i] = 1.0
        pool.append(e.copy())
        e[i] = -1.0
        pool.append(e.copy())
    if g_unit is not None:  # pure gradient-descent direction (eta=0 special case)
        v = np.zeros(n)
        v[topx_idx] = g_unit
        pool.append(v)
        # (3) gradient-biased noisy directions: ~1/3 of the remaining budget
        n_grad_biased = max(0, (dirs_per_radius - len(pool)) // 3)
        for _ in range(n_grad_biased):
            noise = rng.standard_normal(len(topx_idx))
            nn = np.linalg.norm(noise)
            if nn < 1e-12:
                continue
            mix = g_unit + grad_bias_eta * (noise / nn)
            mn = np.linalg.norm(mix)
            if mn < 1e-12:
                continue
            v = np.zeros(n)
            v[topx_idx] = mix / mn
            pool.append(v)
    while len(pool) < dirs_per_radius:  # (2) seeded random unit directions on the subspace
        r = rng.standard_normal(len(topx_idx))
        nr = np.linalg.norm(r)
        if nr < 1e-12:
            continue
        v = np.zeros(n)
        v[topx_idx] = r / nr
        pool.append(v)

    candidates = [{"adv": adv0, "step_size": 0.0, "stage": "start", "note": "original design"}]
    for r in radii:
        for v in pool:
            u = np.clip(u0 + r * v, 0.0, 1.0)
            candidates.append(
                {
                    "adv": adv_from_u(adv0, opt_keys, u),
                    "step_size": float(r),
                    "stage": "trust_radius",
                    "note": f"r={r:g}",
                }
            )

    if shrinkage:
        # Clean-preserving cleanup (§D.6), re-run PER OPERATING POINT (§I.1): the
        # cloud above is gate-free; here, for each (gate, tau_decide, p*) in the
        # sweep, start from the closest candidate clean at THAT operating point and
        # pull each moved driver back toward the start, accepting the LARGEST
        # rollback (rho=1.0 resets the driver to its start value) that stays clean.
        # Repeat over `passes` so earlier resets unlock later ones. Dedup keeps the
        # frame count bounded. This is the one place the gate enters generation
        # (§D.8) and what keeps the tau_decide/p* sweep faithful for shrinkage.
        rho_grid = list(cfg.get("rollback_rhos", [1.0, 0.5, 0.25]))
        passes = int(cfg.get("shrinkage_passes", 3))
        default_op = (
            cfg.get("clean_gate", "reg"),
            float(cfg.get("tau_decide", cfg.get("tau_mm3", sur.tau_mm3))),
            float(cfg.get("p_star", 0.5)),
        )
        op_grid = [tuple(op) for op in (cfg.get("op_grid") or [default_op])]
        pre_shrink = list(candidates)  # gate-free cloud + start
        seen: set = set()
        for (gate, tau_decide, p_star) in op_grid:
            ps = float(p_star) if p_star is not None else 0.5
            td = float(tau_decide)
            clean = [
                (float(np.linalg.norm(u_from_adv(c["adv"], opt_keys) - u0)), c["adv"])
                for c in pre_shrink[1:]
                if sur.is_clean(c["adv"], gate=gate, tau_decide=td, p_star=ps)
            ]
            if not clean:
                continue
            _d, base_adv = min(clean, key=lambda t: t[0])
            u_cur = u_from_adv(base_adv, opt_keys)
            for pass_i in range(passes):
                changed = False
                order = sorted(range(n), key=lambda i: abs(u_cur[i] - u0[i]), reverse=True)
                for i in order:
                    if abs(u_cur[i] - u0[i]) < 1e-9:
                        continue
                    for rho in rho_grid:  # largest rollback first (rho=1.0 -> reset to start)
                        u_try = u_cur.copy()
                        u_try[i] = u_cur[i] + rho * (u0[i] - u_cur[i])
                        # reference adv0 so a full reset copies the original value verbatim (L0 drops)
                        adv_try = adv_from_u(adv0, opt_keys, u_try)
                        if sur.is_clean(adv_try, gate=gate, tau_decide=td, p_star=ps):
                            if not np.allclose(u_try, u_cur):
                                u_cur = u_try
                                changed = True
                                key = tuple(np.round(u_try, 9).tolist())
                                if key not in seen:
                                    seen.add(key)
                                    candidates.append(
                                        {
                                            "adv": adv_try,
                                            "step_size": float(np.linalg.norm(u_try - u0)),
                                            "stage": "shrinkage",
                                            "note": (
                                                f"op(g={gate},t={td:g},p={ps:g}) "
                                                f"pass{pass_i + 1} rollback {opt_keys[i]} rho={rho:g}"
                                            ),
                                        }
                                    )
                            break
                if not changed:
                    break

    return {
        "opt_keys": opt_keys,
        "top_gradients": top,
        "all_gradients": all_gradients,
        "gradient_direction": _direction_dict(opt_keys, direction),
        "candidates": candidates,
    }


@dataclass(frozen=True)
class OptimizerSpec:
    optimizer_id: str
    label: str
    fn: Callable[[dict, Surrogate, dict], dict]
    default_cfg: dict = field(default_factory=dict)
    description: str = ""


OPTIMIZERS: dict[str, OptimizerSpec] = {
    "one_shot_top5": OptimizerSpec(
        optimizer_id="one_shot_top5",
        label="One-shot top-5 gradient sweep",
        fn=one_shot_top5,
        default_cfg={"top_k": 5, "steps": np.logspace(-3, 0, 10).tolist()},
        description="Single start gradient; 10 log-spaced steps along fixed top-5 descent.",
    ),
    "topk_descent": OptimizerSpec(
        optimizer_id="topk_descent",
        label="Iterative top-5 gradient descent (legacy, early-stop)",
        fn=topk_descent,
        default_cfg={
            "top_k": 5,
            "max_iters": 12,
            "probe_steps": np.logspace(-2.5, -0.3, 6).tolist(),
        },
        description="MLP-only line-search descent on the top-5 active drivers until pred<=tau.",
    ),
    # --- canonical v3.1 suite (run-to-budget, feature_grad_mode-aware) ---------
    "one_shot_gradient_line_search": OptimizerSpec(
        optimizer_id="one_shot_gradient_line_search",
        label="One-shot gradient line search",
        fn=one_shot_gradient_line_search,
        default_cfg={"top_k": 5, "steps": np.logspace(-3, 0, 10).tolist()},
        description="Single start gradient; 10 log-spaced steps along the fixed top-k descent (D.1/L.5).",
    ),
    "receding_multistep_gradient": OptimizerSpec(
        optimizer_id="receding_multistep_gradient",
        label="Receding-horizon multistep gradient",
        fn=receding_multistep_gradient,
        default_cfg={
            "top_k": 5,
            "max_iters": 12,
            "probe_steps": np.logspace(-2.5, -0.3, 6).tolist(),
        },
        description="Iterative top-k line-search descent, run to budget (stall/max_iters only).",
    ),
    "receding_multistep_penalty": OptimizerSpec(
        optimizer_id="receding_multistep_penalty",
        label="Receding-horizon multistep (distance-penalized)",
        fn=receding_multistep_penalty,
        default_cfg={
            "top_k": 5,
            "max_iters": 12,
            "penalty_lambda": 0.05,
            "probe_steps": np.logspace(-2.5, -0.3, 6).tolist(),
        },
        description="Descent that minimizes pred + lambda*||u-u_start||, run to budget.",
    ),
    "full_coordinate_grid_refine": OptimizerSpec(
        optimizer_id="full_coordinate_grid_refine",
        label="Full coordinate grid + refine",
        fn=full_coordinate_grid_refine,
        default_cfg={
            "coarse_grid": [0.02, 0.05, 0.10, 0.20, 0.35],
            "refine_factors": [0.5, 0.75, 1.25, 1.5],
            "n_refine": 3,
        },
        description="Gradient-free per-driver +/- coarse grid then refine around the best moves.",
    ),
    "trust_region_hybrid": OptimizerSpec(
        optimizer_id="trust_region_hybrid",
        label="Trust-region top-x cloud (no shrinkage)",
        fn=trust_region_hybrid,
        default_cfg={
            "top_x": 10,
            "dirs_per_radius": 80,
            "radii": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
            "shrinkage": False,
        },
        description="Seeded direction cloud on the top-x active drivers over a radius ladder.",
    ),
    "trust_region_hybrid_shrinkage": OptimizerSpec(
        optimizer_id="trust_region_hybrid_shrinkage",
        label="Trust-region top-x cloud + clean-preserving shrinkage",
        fn=trust_region_hybrid,
        default_cfg={
            "top_x": 10,
            "dirs_per_radius": 80,
            "radii": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
            "shrinkage": True,
            "rollback_rhos": [1.0, 0.5, 0.25],
            "shrinkage_passes": 3,
        },
        description="Trust-region cloud followed by gate-preserving rollback toward the start (D.6).",
    ),
    # --- aero-preserving suite -----------------------------------------------
    "aero_penalized_receding": OptimizerSpec(
        optimizer_id="aero_penalized_receding",
        label="Aero-penalized receding gradient",
        fn=aero_penalized_receding,
        default_cfg={
            "top_k": 5,
            "max_iters": 12,
            "lambda_dist": 0.05,
            "aero_lambda": 1.0,
            "probe_steps": np.logspace(-2.5, -0.3, 6).tolist(),
            "aero_selection": "standard",
        },
        description="Receding overlap descent scored in z-space with a standardized core-4 aero drift penalty.",
    ),
    "aero_tangent_receding": OptimizerSpec(
        optimizer_id="aero_tangent_receding",
        label="Aero-tangent projected receding gradient",
        fn=aero_tangent_receding,
        default_cfg={
            "top_k": 5,
            "max_iters": 12,
            "aero_alpha": 0.9,
            "projection_eps": 1e-4,
            "probe_steps": np.logspace(-2.5, -0.3, 6).tolist(),
            "aero_selection": "standard",
        },
        description="Projects overlap descent away from the standardized performance-MLP core-4 Jacobian.",
    ),
    "aero_budget_trust_region": OptimizerSpec(
        optimizer_id="aero_budget_trust_region",
        label="Aero-budgeted trust region + shrinkage",
        fn=aero_budget_trust_region,
        default_cfg={
            "top_x": 10,
            "dirs_per_radius": 80,
            "radii": [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
            "rollback_rhos": [1.0, 0.5, 0.25],
            "shrinkage_passes": 3,
            "aero_beta": 1.0,
            "aero_selection": "budget_lowest_drift",
        },
        description="Trust-region cloud/shrinkage, selected by standardized aero drift budget among clean candidates.",
    ),
}


# Canonical six-optimizer suite executed per model variant (in order). The
# "all" meta-optimizer (best-of across these six, real-CAD adjudicated) is
# handled by the workbench, not as an entry here.
CANONICAL_SUITE: list[str] = [
    "one_shot_gradient_line_search",
    "receding_multistep_gradient",
    "receding_multistep_penalty",
    "full_coordinate_grid_refine",
    "trust_region_hybrid",
    "trust_region_hybrid_shrinkage",
]
