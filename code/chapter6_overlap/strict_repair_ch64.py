"""Chapter 6.4 overlap-only strict repair benchmark (Chapter 6 #5f).

This module re-runs the Chapter 6.4 repair experiments on the frozen 50-start
benchmark (``strict_overlap_repair_ch64_v1``) with an honest, surrogate-centred
method matrix. It deliberately keeps the legacy :mod:`strict_repair` /
:mod:`strict_hybrid_repair` defaults (used elsewhere) untouched.

Success is strict analytic non-overlap: ``total_overlap_norm <= 1e-6``. The goal
is overlap *removal* only; assemblability is not required here.

Method matrix (one process per ``method x model x tau`` cell; aggregate later):

- ``mlp``           -- learned gradient repair, analytic-verified line search.
- ``hybrid_lite``   -- MLP bulk + a short, bounded coordinate polish (few rounds).
- ``hybrid_lite_reduced_calls``
                     -- MLP bulk + a reduced coordinate polish that only probes
                        the top-gradient signed axes.
- ``finish_line``   -- MLP-only repair, then a one-dimensional line search that
                       keeps walking along the same start-to-MLP displacement.
- ``full_hybrid``   -- MLP proposal + a deep coordinate cleanup (upper-bound only).
- ``random_direct`` -- negative control: random search directions + direct analytic
                       verify, no surrogate (verifier-budget capped).
- ``axis_direct``   -- negative control: coordinate-axis probes (~26 analytic evals
                       per round) + direct verify, no surrogate.

Surrogate backends:

- ``v3``        single-head regressor; volume-head gradient; ``tau_vol`` stop on the
                *predicted* overlap norm (``tau_vol = 0`` => analytic verify drives
                the stop, the surrogate only supplies the descent direction).
- ``multitask`` shared-trunk model; volume head for gradients, binary head for a
                ``P(overlap) < tau_bin`` stop (the ``grad_vol_stop_bin`` mode).

Primary cost metric is ``verifier_calls`` (analytic overlap evaluations); wall time
is only faithful when a cell is run uncontended (see Chapter 6).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ._public_helpers import FEATURES, _features, _read_csv
from .label_cache import DEFAULT_THRESHOLD_NORM, STRICT_THRESHOLD_NORM, params_from_row
from .models import predict_overlap_norm
from .multitask_overlap_model import (
    load_multitask_overlap_model,
    predict_overlap_norm_multitask,
    predict_p_overlap,
)
from ._public_helpers import _distance, _to_params
from .paths import DATA_DIR, MODELS_DIR, RUNS_DIR
from .release_paths import (
    ACQUIRED_RANDOM_15K,
    CH64_STARTS_CSV,
    CH65_STARTS_CSV,
    HOLDOUT_5K_CSV,
    HOLDOUT_LABELED_5K,
    POOL_100K_CSV,
    REPAIR_CH64_DIR,
)
from .regression_metrics import PAIR_COLUMNS
from ._public_helpers import _load_model
from .sampler import DummyParams, validate_relaxed_params
from ._public_helpers import (
    _active_coordinates,
    _boring_score,
    _bounds_from_pool,
    _strict_label,
    _strict_overlap,
)

V3_MODEL_DIR = MODELS_DIR / "overlap_regressor_regression_v3_selected"
MULTITASK_MODEL_DIR = MODELS_DIR / "overlap_regressor_multitask_v1_selected"

CH64_STARTS_CSV = CH64_STARTS_CSV
POOL_CSV = POOL_100K_CSV


# --------------------------------------------------------------------------------------
# Surrogate wrapper: one interface for the v3 single-head and multitask backends.
# --------------------------------------------------------------------------------------


class _Surrogate:
    """Unified predict/gradient interface over both overlap surrogate backends.

    Predictions are inverted with ``tau0 = 5e-5`` (the scale the heads were trained
    on), so ``predict_overlap_norm`` returns overlap in true normalized units and a
    ``tau_vol`` comparison is meaningful.
    """

    def __init__(self, kind: str, model, std) -> None:
        self.kind = kind  # "v3" | "multitask" | "none"
        self.model = model
        self.std = std
        if model is not None:
            self._mean = torch.tensor(std.mean_, dtype=torch.float32)
            self._scale = torch.tensor(std.scale_, dtype=torch.float32)

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def predict_overlap_norm(self, x2d: np.ndarray) -> np.ndarray:
        if self.kind == "multitask":
            return predict_overlap_norm_multitask(self.model, self.std, x2d, threshold_norm=DEFAULT_THRESHOLD_NORM)
        return predict_overlap_norm(self.model, self.std, x2d, threshold_norm=DEFAULT_THRESHOLD_NORM)

    def predict_p_overlap(self, x2d: np.ndarray):
        if self.kind == "multitask":
            return predict_p_overlap(self.model, self.std, x2d)
        return None

    def grad_volume(self, x1d: np.ndarray) -> np.ndarray:
        """Gradient of the predicted (log-)overlap w.r.t. parameters (volume head)."""
        x = torch.tensor(x1d.astype(np.float32), dtype=torch.float32, requires_grad=True)
        z = ((x - self._mean) / self._scale).unsqueeze(0)
        out = self.model(z)
        if self.kind == "multitask":
            out = out[0]  # volume head only
        out.backward()
        return x.grad.detach().numpy()


def _make_surrogate(model_kind: str, model_dir: Path) -> _Surrogate:
    if model_kind == "none":
        return _Surrogate("none", None, None)
    if model_kind == "multitask":
        model, std, _ = load_multitask_overlap_model(model_dir)
        return _Surrogate("multitask", model, std)
    model, std, _ = _load_model(model_dir)
    return _Surrogate("v3", model, std)


# --------------------------------------------------------------------------------------
# Trajectory frames and per-start metrics.
# --------------------------------------------------------------------------------------


def _x_of(p: DummyParams) -> np.ndarray:
    return np.array([float(getattr(p, k)) for k in FEATURES], dtype=np.float32)


def _frame(step: int, x: np.ndarray, p0: DummyParams, *, phase: str, verified: float, pred) -> dict:
    p = _to_params(x, p0)
    return {
        "step": int(step),
        "phase": phase,
        "params": {k: float(v) for k, v in asdict(p).items()},
        "pred_overlap_norm": (None if pred is None else float(pred)),
        "verified_overlap_norm": float(verified),
        "overlap_ok": bool(float(verified) <= STRICT_THRESHOLD_NORM),
    }


def _pct_reduction(start_norm: float, final_norm: float) -> float:
    """100 * (start - final) / start, capped at 100 (final = 0); honest if it worsens."""
    if start_norm <= STRICT_THRESHOLD_NORM:
        return 100.0 if final_norm <= STRICT_THRESHOLD_NORM else 0.0
    return min(100.0, 100.0 * (start_norm - final_norm) / start_norm)


def _build_result(
    method: str,
    p0: DummyParams,
    x_final: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    frames: list[dict],
    verifier_calls: int,
    wall_time_s: float,
    *,
    extra: dict | None = None,
) -> dict:
    x0 = _x_of(p0)
    final_p = _to_params(x_final, p0)
    final_label = _strict_label(final_p)
    start_label = _strict_label(p0)
    start_norm = float(start_label["total_overlap_norm"])
    final_norm = float(final_label["total_overlap_norm"])
    result = {
        "method": method,
        "success": bool(final_norm <= STRICT_THRESHOLD_NORM and validate_relaxed_params(final_p).ok),
        "steps": int(len(frames) - 1),
        "start_overlap_norm": start_norm,
        "final_overlap_norm": final_norm,
        "pct_overlap_reduction": _pct_reduction(start_norm, final_norm),
        "dominant_pair_start": start_label["dominant_pair"],
        "dominant_pair_final": final_label["dominant_pair"],
        "distance": _distance(x_final, x0, lo, hi),
        "active_coordinates": _active_coordinates(x_final, x0),
        "boring_radius_change_score": _boring_score(x_final, x0, lo, hi),
        "verifier_calls": int(verifier_calls),
        "wall_time_s": float(wall_time_s),
        "trajectory": frames,
        "final_params": asdict(final_p),
        **{f"final_{k}": final_label[f"pair_norm_{k}"] for k in PAIR_COLUMNS},
        **{f"start_{k}": start_label[f"pair_norm_{k}"] for k in PAIR_COLUMNS},
    }
    if extra:
        result.update(extra)
    return result


# --------------------------------------------------------------------------------------
# Phases: MLP gradient descent and bounded coordinate (direct) descent.
# --------------------------------------------------------------------------------------


def _mlp_phase(
    p0: DummyParams,
    surrogate: _Surrogate,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_steps: int,
    lr: float,
    tau_vol: float,
    tau_bin: float | None,
    near_zero_handoff: float | None,
) -> tuple[np.ndarray, list[dict], int, str]:
    """MLP gradient repair with an analytic-verified backtracking line search.

    Stops on (a) true analytic success, (b) an optional near-zero handoff kept for
    legacy experiments, (c) a surrogate-believes-done early stop
    (``pred <= tau_vol`` or ``P(overlap) < tau_bin``), (d) a vanishing gradient, or
    (e) no accepted move / step budget. ``tau_vol = 0`` disables the predicted-volume
    stop so the analytic verifier alone decides when the run is finished.
    """
    x = _x_of(p0)
    frames: list[dict] = []
    verifier_calls = 0
    best_overlap = float("inf")
    best_pred = float("inf")
    stop_reason = "max_steps"
    for step in range(max_steps + 1):
        p = _to_params(x, p0)
        verified = _strict_overlap(p)
        verifier_calls += 1
        pred = float(surrogate.predict_overlap_norm(x.reshape(1, -1))[0])
        p_over = surrogate.predict_p_overlap(x.reshape(1, -1))
        frames.append(_frame(step, x, p0, phase="mlp", verified=verified, pred=pred))
        best_overlap = min(best_overlap, verified)
        best_pred = min(best_pred, pred)
        if validate_relaxed_params(p).ok and verified <= STRICT_THRESHOLD_NORM:
            stop_reason = "analytic_success"
            break
        if near_zero_handoff is not None and verified <= near_zero_handoff:
            stop_reason = "near_zero_handoff"
            break
        if tau_vol > 0.0 and pred <= tau_vol:
            stop_reason = "tau_vol_stop"
            break
        if tau_bin is not None and p_over is not None and float(p_over[0]) < tau_bin:
            stop_reason = "tau_bin_stop"
            break
        grad = surrogate.grad_volume(x)
        norm = float(np.linalg.norm(grad))
        if norm < 1e-12:
            stop_reason = "vanishing_gradient"
            break
        direction = -grad / norm
        accepted = False
        for step_size in (lr, lr * 0.5, lr * 0.25, lr * 0.1, lr * 0.05, lr * 0.02, lr * 0.01):
            cand = np.clip(x + step_size * direction, lo, hi)
            cand_p = _to_params(cand, p0)
            if not validate_relaxed_params(cand_p).ok:
                continue
            cand_overlap = _strict_overlap(cand_p)
            verifier_calls += 1
            cand_pred = float(surrogate.predict_overlap_norm(cand.reshape(1, -1))[0])
            if cand_overlap <= best_overlap or cand_pred < best_pred:
                x = cand
                best_overlap = min(best_overlap, cand_overlap)
                best_pred = min(best_pred, cand_pred)
                accepted = True
                break
        if not accepted:
            stop_reason = "no_accepted_move"
            break
    return x, frames, verifier_calls, stop_reason


def _direct_phase(
    x_start: np.ndarray,
    p0: DummyParams,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_rounds: int,
    step_fracs: tuple[float, ...],
    start_step: int,
    max_verifier_calls: int | None = None,
    dist_reg: float = 2e-5,
) -> tuple[np.ndarray, list[dict], int]:
    """One-coordinate-per-round descent on the analytic overlap label.

    Each round probes every parameter at ``+/-`` each step fraction, keeps the single
    best overlap-reducing change, and stops at the strict threshold, on stalling, or
    at an optional verifier-call budget. A tiny distance regularizer breaks ties
    toward the smaller edit.
    """
    x = x_start.copy()
    x0 = _x_of(p0)
    frames: list[dict] = []
    verifier_calls = 0
    step = start_step
    for _ in range(max_rounds):
        best_x = x.copy()
        best_val = _strict_overlap(_to_params(x, p0))
        verifier_calls += 1
        best_score = best_val + dist_reg * _distance(x, x0, lo, hi)
        for j in range(len(FEATURES)):
            span = hi[j] - lo[j]
            for frac in step_fracs:
                for sign in (-1.0, 1.0):
                    cand = x.copy()
                    cand[j] = np.clip(cand[j] + sign * frac * span, lo[j], hi[j])
                    cand_p = _to_params(cand, p0)
                    if not validate_relaxed_params(cand_p).ok:
                        continue
                    val = _strict_overlap(cand_p)
                    verifier_calls += 1
                    score = val + dist_reg * _distance(cand, x0, lo, hi)
                    if score < best_score and val <= best_val:
                        best_score = score
                        best_val = val
                        best_x = cand
                    if max_verifier_calls is not None and verifier_calls >= max_verifier_calls:
                        break
                if max_verifier_calls is not None and verifier_calls >= max_verifier_calls:
                    break
            if max_verifier_calls is not None and verifier_calls >= max_verifier_calls:
                break
        if np.allclose(best_x, x):
            break
        x = best_x
        step += 1
        frames.append(_frame(step, x, p0, phase="direct", verified=best_val, pred=None))
        if best_val <= STRICT_THRESHOLD_NORM:
            break
        if max_verifier_calls is not None and verifier_calls >= max_verifier_calls:
            break
    return x, frames, verifier_calls


def _reduced_gradient_direct_phase(
    x_start: np.ndarray,
    p0: DummyParams,
    lo: np.ndarray,
    hi: np.ndarray,
    surrogate: _Surrogate,
    *,
    max_rounds: int,
    top_k: int,
    step_fracs: tuple[float, ...],
    start_step: int,
    dist_reg: float = 2e-5,
) -> tuple[np.ndarray, list[dict], int, dict]:
    """Coordinate polish restricted to the top-gradient signed axes.

    Each round recomputes the surrogate gradient, keeps the ``top_k`` coordinates
    by absolute gradient magnitude, and probes only the signed descent direction
    for each selected coordinate.
    """
    x = x_start.copy()
    x0 = _x_of(p0)
    frames: list[dict] = []
    verifier_calls = 0
    gradient_evals = 0
    signed_candidates = 0
    step = start_step
    selected_by_round: list[list[int]] = []

    for _ in range(max_rounds):
        grad = surrogate.grad_volume(x)
        gradient_evals += 1
        ranked = [int(i) for i in np.argsort(-np.abs(grad)) if abs(float(grad[i])) > 1e-12]
        selected = ranked[: max(0, int(top_k))]
        selected_by_round.append(selected)
        if not selected:
            break

        best_x = x.copy()
        best_val = _strict_overlap(_to_params(x, p0))
        verifier_calls += 1
        best_score = best_val + dist_reg * _distance(x, x0, lo, hi)
        for j in selected:
            signed_direction = -1.0 if float(grad[j]) > 0.0 else 1.0
            span = hi[j] - lo[j]
            for frac in step_fracs:
                cand = x.copy()
                cand[j] = np.clip(cand[j] + signed_direction * frac * span, lo[j], hi[j])
                cand_p = _to_params(cand, p0)
                if not validate_relaxed_params(cand_p).ok:
                    continue
                val = _strict_overlap(cand_p)
                verifier_calls += 1
                signed_candidates += 1
                score = val + dist_reg * _distance(cand, x0, lo, hi)
                if score < best_score and val <= best_val:
                    best_score = score
                    best_val = val
                    best_x = cand
        if np.allclose(best_x, x):
            break
        x = best_x
        step += 1
        frames.append(_frame(step, x, p0, phase="reduced_direct", verified=best_val, pred=None))
        if best_val <= STRICT_THRESHOLD_NORM:
            break

    extra = {
        "reduced_polish_rounds": len(frames),
        "reduced_gradient_evals": int(gradient_evals),
        "reduced_top_k": int(top_k),
        "reduced_signed_candidates": int(signed_candidates),
        "reduced_selected_axes": json.dumps(selected_by_round),
    }
    return x, frames, verifier_calls, extra


def _path_line_search(
    x0: np.ndarray,
    x_mlp: np.ndarray,
    p0: DummyParams,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    coarse_n: int,
    refine_n: int,
    start_step: int,
) -> tuple[np.ndarray, list[dict], int, dict]:
    """Search only along the MLP displacement, extrapolating past the MLP endpoint."""
    direction = x_mlp - x0
    if float(np.linalg.norm(direction)) < 1e-12:
        return x_mlp, [], 0, {
            "finish_line_path_reason": "zero_mlp_displacement",
            "finish_line_alpha": 1.0,
            "finish_line_candidates": 0,
        }

    base_grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], dtype=float)
    coarse_n = max(1, min(int(coarse_n), len(base_grid)))
    coarse = base_grid[:coarse_n]
    seen: set[float] = set()
    candidates: list[dict] = []
    verifier_calls = 0

    def probe(alpha: float, source: str) -> None:
        nonlocal verifier_calls
        key = round(float(alpha), 10)
        if key in seen:
            return
        seen.add(key)
        x = np.clip(x0 + float(alpha) * direction, lo, hi)
        cand_p = _to_params(x, p0)
        if not validate_relaxed_params(cand_p).ok:
            return
        overlap = _strict_overlap(cand_p)
        verifier_calls += 1
        candidates.append({
            "alpha": float(alpha),
            "x": x,
            "overlap": float(overlap),
            "success": bool(overlap <= STRICT_THRESHOLD_NORM),
            "source": source,
        })

    for alpha in coarse:
        probe(float(alpha), "coarse")

    if candidates and refine_n > 0:
        coarse_candidates = [c for c in candidates if c["source"] == "coarse"]
        best = min(coarse_candidates, key=lambda c: (not c["success"], c["overlap"], c["alpha"]))
        best_alpha = float(best["alpha"])
        idx = int(np.where(coarse == best_alpha)[0][0]) if best_alpha in coarse else 0
        lo_alpha = float(coarse[max(0, idx - 1)])
        hi_alpha = float(coarse[min(len(coarse) - 1, idx + 1)])
        if lo_alpha == hi_alpha:
            lo_alpha = max(1.0, best_alpha - 0.25)
            hi_alpha = best_alpha + 0.25
        for alpha in np.linspace(lo_alpha, hi_alpha, max(1, int(refine_n))):
            probe(float(alpha), "refine")

    if not candidates:
        return x_mlp, [], verifier_calls, {
            "finish_line_path_reason": "no_valid_path_candidates",
            "finish_line_alpha": 1.0,
            "finish_line_candidates": 0,
        }

    successful = [c for c in candidates if c["success"]]
    if successful:
        selected = min(successful, key=lambda c: c["alpha"])
        reason = "strict_success"
    else:
        selected = min(candidates, key=lambda c: (c["overlap"], c["alpha"]))
        reason = "lowest_overlap"

    frame = _frame(
        start_step + 1,
        selected["x"],
        p0,
        phase="finish_line_path",
        verified=selected["overlap"],
        pred=None,
    )
    return selected["x"], [frame], verifier_calls, {
        "finish_line_path_reason": reason,
        "finish_line_alpha": float(selected["alpha"]),
        "finish_line_candidates": int(len(candidates)),
    }


# --------------------------------------------------------------------------------------
# Methods.
# --------------------------------------------------------------------------------------


def _method_mlp(p0, surrogate, lo, hi, args) -> dict:
    t0 = time.perf_counter()
    x, frames, vcalls, stop_reason = _mlp_phase(
        p0, surrogate, lo, hi,
        max_steps=args.mlp_steps, lr=args.lr, tau_vol=args.tau_vol, tau_bin=args.tau_bin,
        near_zero_handoff=None,
    )
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, vcalls, time.perf_counter() - t0,
        extra={"mlp_stop_reason": stop_reason},
    )


def _method_hybrid_lite(p0, surrogate, lo, hi, args) -> dict:
    t0 = time.perf_counter()
    x, frames, vcalls, stop_reason = _mlp_phase(
        p0, surrogate, lo, hi,
        max_steps=args.mlp_steps, lr=args.lr, tau_vol=args.tau_vol, tau_bin=args.tau_bin,
        near_zero_handoff=None,
    )
    overlap_after_mlp = frames[-1]["verified_overlap_norm"]
    x, dframes, dcalls = _direct_phase(
        x, p0, lo, hi,
        max_rounds=args.lite_rounds, step_fracs=(0.005, 0.01, 0.02, 0.05, 0.10),
        start_step=frames[-1]["step"],
    )
    frames.extend(dframes)
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, vcalls + dcalls, time.perf_counter() - t0,
        extra={"mlp_stop_reason": stop_reason, "overlap_after_mlp": float(overlap_after_mlp), "polish_rounds": len(dframes)},
    )


def _method_hybrid_lite_reduced_calls(p0, surrogate, lo, hi, args) -> dict:
    t0 = time.perf_counter()
    x, frames, vcalls, stop_reason = _mlp_phase(
        p0, surrogate, lo, hi,
        max_steps=args.mlp_steps, lr=args.lr, tau_vol=args.tau_vol, tau_bin=args.tau_bin,
        near_zero_handoff=None,
    )
    overlap_after_mlp = frames[-1]["verified_overlap_norm"]
    x, dframes, dcalls, reduced_extra = _reduced_gradient_direct_phase(
        x, p0, lo, hi, surrogate,
        max_rounds=args.reduced_rounds,
        top_k=args.reduced_top_k,
        step_fracs=(0.005, 0.01, 0.02, 0.05, 0.10),
        start_step=frames[-1]["step"],
    )
    frames.extend(dframes)
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, vcalls + dcalls, time.perf_counter() - t0,
        extra={
            "mlp_stop_reason": stop_reason,
            "overlap_after_mlp": float(overlap_after_mlp),
            **reduced_extra,
        },
    )


def _method_finish_line(p0, surrogate, lo, hi, args) -> dict:
    t0 = time.perf_counter()
    x0 = _x_of(p0)
    x, frames, vcalls, stop_reason = _mlp_phase(
        p0, surrogate, lo, hi,
        max_steps=args.mlp_steps, lr=args.lr, tau_vol=args.tau_vol, tau_bin=args.tau_bin,
        near_zero_handoff=None,
    )
    overlap_before_finish = frames[-1]["verified_overlap_norm"]
    success_before_finish = bool(overlap_before_finish <= STRICT_THRESHOLD_NORM)
    x, dframes, dcalls, path_extra = _path_line_search(
        x0, x, p0, lo, hi,
        coarse_n=getattr(args, "finish_coarse", 10),
        refine_n=getattr(args, "finish_refine", 5),
        start_step=frames[-1]["step"],
    )
    frames.extend(dframes)
    final_overlap = frames[-1]["verified_overlap_norm"]
    success_after = bool(final_overlap <= STRICT_THRESHOLD_NORM)
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, vcalls + dcalls, time.perf_counter() - t0,
        extra={
            "mlp_stop_reason": stop_reason,
            "overlap_before_finish_line": float(overlap_before_finish),
            "finish_line_steps": len(dframes),
            "finish_line_converted": bool((not success_before_finish) and success_after),
            **path_extra,
        },
    )


def _method_full_hybrid(p0, surrogate, lo, hi, args) -> dict:
    t0 = time.perf_counter()
    x, frames, vcalls, stop_reason = _mlp_phase(
        p0, surrogate, lo, hi,
        max_steps=args.mlp_steps, lr=args.lr, tau_vol=args.tau_vol, tau_bin=args.tau_bin,
        near_zero_handoff=None,
    )
    x, dframes, dcalls = _direct_phase(
        x, p0, lo, hi,
        max_rounds=args.full_rounds, step_fracs=(0.005, 0.01, 0.02, 0.05, 0.10),
        start_step=frames[-1]["step"],
    )
    frames.extend(dframes)
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, vcalls + dcalls, time.perf_counter() - t0,
        extra={"mlp_stop_reason": stop_reason, "cleanup_rounds": len(dframes)},
    )


def _method_axis_direct(p0, surrogate, lo, hi, args) -> dict:
    """Negative control: coordinate-axis probes (~26 analytic evals/round), no MLP."""
    t0 = time.perf_counter()
    x0 = _x_of(p0)
    start_overlap = _strict_overlap(p0)
    frames = [_frame(0, x0, p0, phase="axis", verified=start_overlap, pred=None)]
    x, dframes, vcalls = _direct_phase(
        x0, p0, lo, hi,
        max_rounds=args.axis_rounds, step_fracs=(args.axis_frac,),
        start_step=0, max_verifier_calls=args.control_max_verifier,
    )
    frames.extend(dframes)
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, 1 + vcalls, time.perf_counter() - t0,
        extra={"control_rounds": len(dframes)},
    )


def _method_random_direct(p0, surrogate, lo, hi, args) -> dict:
    """Negative control: random search directions + direct analytic verify, no MLP."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    span = hi - lo
    x = _x_of(p0)
    frames: list[dict] = []
    verifier_calls = 0
    step_menu = (0.005, 0.01, 0.02, 0.05, 0.10)
    for step in range(args.random_steps + 1):
        cur = _strict_overlap(_to_params(x, p0))
        verifier_calls += 1
        frames.append(_frame(step, x, p0, phase="random", verified=cur, pred=None))
        if validate_relaxed_params(_to_params(x, p0)).ok and cur <= STRICT_THRESHOLD_NORM:
            break
        if verifier_calls >= args.control_max_verifier:
            break
        best_x = x.copy()
        best_val = cur
        for _ in range(args.random_k):
            d = rng.normal(size=len(FEATURES)).astype(np.float32)
            n = float(np.linalg.norm(d))
            if n < 1e-12:
                continue
            d /= n
            for frac in step_menu:
                cand = np.clip(x + frac * span * d, lo, hi)
                cand_p = _to_params(cand, p0)
                if not validate_relaxed_params(cand_p).ok:
                    continue
                val = _strict_overlap(cand_p)
                verifier_calls += 1
                if val < best_val:
                    best_val = val
                    best_x = cand
                if verifier_calls >= args.control_max_verifier:
                    break
            if verifier_calls >= args.control_max_verifier:
                break
        if np.allclose(best_x, x):
            break
        x = best_x
    return _build_result(
        args.method_label, p0, x, lo, hi, frames, verifier_calls, time.perf_counter() - t0,
        extra={"random_k": args.random_k},
    )


_METHODS = {
    "mlp": _method_mlp,
    "hybrid_lite": _method_hybrid_lite,
    "hybrid_lite_reduced_calls": _method_hybrid_lite_reduced_calls,
    "finish_line": _method_finish_line,
    "full_hybrid": _method_full_hybrid,
    "axis_direct": _method_axis_direct,
    "random_direct": _method_random_direct,
}

_REQUIRES_MODEL = {"mlp", "hybrid_lite", "hybrid_lite_reduced_calls", "finish_line", "full_hybrid"}


# --------------------------------------------------------------------------------------
# Zig-zag mode (Chapter 6 #5g): overlap stage for the Ch. 6.5 alternating pipeline.
# --------------------------------------------------------------------------------------

_ZIGZAG_OVERLAP_METHODS = {"hybrid_lite": _method_hybrid_lite, "finish_line": _method_finish_line}


def run_overlap_stage_zigzag(
    p0: DummyParams,
    surrogate: _Surrogate,
    lo: np.ndarray,
    hi: np.ndarray,
    args: argparse.Namespace,
    *,
    method: str,
) -> dict:
    """Run one overlap-repair stage for the Ch. 6.5 zig-zag controller.

    This wraps the §6.4 ``hybrid_lite`` / ``finish_line`` repair (run with
    ``tau_vol=0`` so the *surrogate* never triggers an early stop) and re-frames the
    result as a controller stage with explicit zig-zag semantics:

    - **Success** is *analytic* ``total_overlap_norm <= STRICT_THRESHOLD_NORM`` only;
      a tiny *predicted* overlap never counts.
    - The MLP phase + direct polish/tail run to exhaustion (the direct phase stops
      only when no probed coordinate improves the analytic overlap, or the round /
      verifier budget is spent). A stage that exhausts its budget while still above
      the strict threshold returns ``failure_reason="overlap_exhausted"``.
    - The full per-step trajectory (params at every accepted step) is preserved for
      the multi-segment viewer.

    The standalone §6.4 ``run()`` / CLI is unaffected: this is an additive entry
    point that reuses the existing method functions.
    """
    if method not in _ZIGZAG_OVERLAP_METHODS:
        raise ValueError(
            f"unknown overlap method {method!r}; choose from {sorted(_ZIGZAG_OVERLAP_METHODS)}"
        )
    args.method_label = method
    result = _ZIGZAG_OVERLAP_METHODS[method](p0, surrogate, lo, hi, args)
    stage_success = bool(result["final_overlap_norm"] <= STRICT_THRESHOLD_NORM)
    return {
        "stage": "overlap",
        "method": method,
        "final_params": result["final_params"],
        "trajectory": result["trajectory"],
        "start_overlap_norm": result["start_overlap_norm"],
        "final_overlap_norm": result["final_overlap_norm"],
        "pct_overlap_reduction": result["pct_overlap_reduction"],
        "verifier_calls": int(result["verifier_calls"]),
        "active_coordinates": int(result.get("active_coordinates", 0)),
        "mlp_stop_reason": result.get("mlp_stop_reason"),
        "stage_success": stage_success,
        "failure_reason": None if stage_success else "overlap_exhausted",
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


def _default_method_label(args: argparse.Namespace) -> str:
    parts = [args.method, args.model_kind]
    if args.method in _REQUIRES_MODEL:
        if args.tau_bin is not None:
            parts.append(f"taubin{args.tau_bin}")
        else:
            parts.append(f"tauvol{args.tau_vol:g}")
    return "__".join(parts)


def run(args: argparse.Namespace) -> Path:
    if args.method not in _METHODS:
        raise SystemExit(f"unknown method {args.method!r}; choose from {sorted(_METHODS)}")
    if args.method in _REQUIRES_MODEL and args.model_kind == "none":
        raise SystemExit(f"method {args.method!r} needs a surrogate; set --model-kind v3 or multitask")

    args.method_label = args.method_label or _default_method_label(args)
    run_id = args.run_id or args.method_label
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    start_offset = getattr(args, "start_offset", 0)
    starts = _read_csv(args.starts_csv)[start_offset : start_offset + args.n_starts]
    lo, hi = _bounds_from_pool(_features(_read_csv(args.pool_csv)))
    surrogate = _make_surrogate(args.model_kind, args.model_dir)
    method_fn = _METHODS[args.method]

    results = []
    for row in starts:
        p0 = params_from_row(row)
        base = {
            "sample_id": row["sample_id"],
            "strict_category": row.get("strict_category", ""),
            "start_dominant_pair": row.get("dominant_pair", ""),
            "start_magnitude_bin": row.get("magnitude_bin", ""),
            "model_kind": args.model_kind,
            "tau_vol": args.tau_vol,
            "tau_bin": (None if args.tau_bin is None else float(args.tau_bin)),
        }
        results.append({**base, **method_fn(p0, surrogate, lo, hi, args)})

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    flat = [{k: v for k, v in r.items() if k not in ("trajectory", "final_params")} for r in results]
    _write_csv(run_dir / "summary.csv", flat)
    summary = {
        "run_id": run_id,
        "method": args.method,
        "method_label": args.method_label,
        "model_kind": args.model_kind,
        "model_dir": (None if args.model_kind == "none" else str(args.model_dir)),
        "tau_vol": args.tau_vol,
        "tau_bin": (None if args.tau_bin is None else float(args.tau_bin)),
        "strict_threshold_norm": STRICT_THRESHOLD_NORM,
        "starts_csv": str(args.starts_csv),
        "n_starts": len(starts),
        "strict_success": int(sum(bool(r["success"]) for r in results)),
        "mean_pct_overlap_reduction": float(np.mean([r["pct_overlap_reduction"] for r in results])),
        "median_pct_overlap_reduction": float(np.median([r["pct_overlap_reduction"] for r in results])),
        "mean_verifier_calls": float(np.mean([r["verifier_calls"] for r in results])),
        "mean_active_coordinates": float(np.mean([r["active_coordinates"] for r in results])),
        "mean_wall_time_s": float(np.mean([r["wall_time_s"] for r in results])),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{args.method_label}] success {summary['strict_success']}/{len(starts)} "
        f"mean_reduction {summary['mean_pct_overlap_reduction']:.1f}% "
        f"mean_verifier {summary['mean_verifier_calls']:.0f} "
        f"mean_wall {summary['mean_wall_time_s']:.3f}s"
    )
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(_METHODS))
    parser.add_argument("--model-kind", default="v3", choices=("v3", "multitask", "none"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--starts-csv", type=Path, default=CH64_STARTS_CSV)
    parser.add_argument("--pool-csv", type=Path, default=POOL_CSV)
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--tau-vol", type=float, default=0.0)
    parser.add_argument("--tau-bin", type=float, default=None)
    parser.add_argument("--mlp-steps", type=int, default=140)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--lite-rounds", type=int, default=3)
    parser.add_argument("--reduced-rounds", type=int, default=3)
    parser.add_argument("--reduced-top-k", type=int, default=5)
    parser.add_argument("--finish-coarse", type=int, default=10)
    parser.add_argument("--finish-refine", type=int, default=5)
    parser.add_argument("--finish-rounds", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--full-rounds", type=int, default=10)
    parser.add_argument("--axis-frac", type=float, default=0.02)
    parser.add_argument("--axis-rounds", type=int, default=6)
    parser.add_argument("--random-k", type=int, default=16)
    parser.add_argument("--random-steps", type=int, default=80)
    parser.add_argument("--control-max-verifier", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=REPAIR_CH64_DIR)
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = MULTITASK_MODEL_DIR if args.model_kind == "multitask" else V3_MODEL_DIR
    run(args)


if __name__ == "__main__":
    main()
