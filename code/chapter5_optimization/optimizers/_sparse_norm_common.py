"""Shared sparse-norm optimizer variants for Chapter 5."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.interface import (
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
    StepRecord,
)
from shared.model_utils import compute_gradient
from shared.optimizer_utils import (
    constrained_direction_norm,
    evaluate_candidate,
    finalize_result,
    log_magnitudes,
    no_gradient_result,
    normalized_gradient_direction,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_sparse_success,
)


TRUST_RADII = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
TRUST_RANDOM_PER_RADIUS = 32
TRUST_BASE_SEED = 27201

AXIS_MAGNITUDES = log_magnitudes(2, -4, multipliers=(1.0, 0.5, 0.2), descending=False)

PROX_STEP_SIZE = 0.03
PROX_N_STEPS = 220
PROX_DISTANCE_PENALTY = 0.10
PROX_SOFT_THRESHOLD_FRAC = 0.35
PROX_TOP_K_RELAXATION = (1, 2, 3)


@dataclass
class SparseTrustRegion:
    optimizer_id: str
    sparse_mode: str

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        rng = np.random.default_rng(TRUST_BASE_SEED + int(start.start_id.split("_")[-1]))
        steps = [evaluate_candidate(step_idx=0, x0=x0, x=x0, ctx=ctx, gradient_direction=None, step_magnitude=0.0)]

        grad = normalized_gradient_direction(ctx, x0)
        axes = []
        for i in range(len(x0)):
            for sign in (-1.0, 1.0):
                d = np.zeros(len(x0), dtype=np.float64)
                d[i] = sign
                axes.append(d)

        for radius in TRUST_RADII:
            dirs: list[np.ndarray] = []
            if grad is not None:
                dirs.append(grad.direction_norm)
            dirs.extend(axes)
            for _ in range(TRUST_RANDOM_PER_RADIUS):
                d = rng.normal(size=len(x0))
                d = constrained_direction_norm(ctx, d)
                if float(np.linalg.norm(d)) > 1e-12:
                    dirs.append(d.astype(np.float64))
            for d in dirs:
                x = raw_step_from_norm_direction(x0, d, radius, ctx)
                steps.append(evaluate_candidate(
                    step_idx=len(steps),
                    x0=x0,
                    x=x,
                    ctx=ctx,
                    gradient_direction=d,
                    step_magnitude=radius,
                ))

        selected = select_sparse_success(steps, x0=x0, ctx=ctx, mode=self.sparse_mode)
        if selected is None:
            selected, status = select_best_fallback(steps)
            stop_reason = status if status != "no_surrogate_crossing" else "trust_region_exhausted"
        else:
            status = "surrogate_success"
            stop_reason = f"tau_crossed_sparse_{self.sparse_mode}"
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=stop_reason,
            start_probability=start_prob,
        )


@dataclass
class SparseCoordinateAxis:
    optimizer_id: str
    sparse_mode: str

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        steps = [evaluate_candidate(step_idx=0, x0=x0, x=x0, ctx=ctx, gradient_direction=None, step_magnitude=0.0)]

        for axis in range(len(x0)):
            for sign in (-1.0, 1.0):
                direction = np.zeros(len(x0), dtype=np.float64)
                direction[axis] = sign
                for mag in AXIS_MAGNITUDES:
                    x = raw_step_from_norm_direction(x0, direction, mag, ctx)
                    steps.append(evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x,
                        ctx=ctx,
                        gradient_direction=direction,
                        step_magnitude=mag,
                    ))

        selected = select_sparse_success(steps, x0=x0, ctx=ctx, mode=self.sparse_mode)
        if selected is None:
            selected, status = select_best_fallback(steps)
            stop_reason = status
        else:
            status = "surrogate_success"
            stop_reason = f"tau_crossed_sparse_{self.sparse_mode}"
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=stop_reason,
            start_probability=start_prob,
        )


@dataclass
class SparsePenalizedProximity:
    optimizer_id: str
    sparse_mode: str

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        grad_raw, start_prob = compute_gradient(ctx.model, x0)
        if float(np.linalg.norm(grad_raw)) < 1e-12:
            return no_gradient_result(start=start, ctx=ctx, optimizer_id=self.optimizer_id, probability=start_prob)

        steps: list[StepRecord] = []
        for k in self._k_schedule():
            x = x0.copy()
            for _ in range(PROX_N_STEPS):
                grad_raw, _prob = compute_gradient(ctx.model, x)
                grad_norm = grad_raw * ctx.normalization.stds
                displacement_norm = (x - x0) / ctx.normalization.stds
                direction = grad_norm - PROX_DISTANCE_PENALTY * displacement_norm
                direction = self._sparsify_direction(direction, k=k, ctx=ctx)
                if float(np.linalg.norm(direction)) < 1e-12:
                    break
                x_next = raw_step_from_norm_direction(x, direction, PROX_STEP_SIZE, ctx)
                rec = evaluate_candidate(
                    step_idx=len(steps),
                    x0=x0,
                    x=x_next,
                    ctx=ctx,
                    gradient_direction=direction,
                    step_magnitude=PROX_STEP_SIZE,
                )
                steps.append(rec)
                if rec.valid:
                    x = x_next

        selected = select_sparse_success(steps, x0=x0, ctx=ctx, mode=self.sparse_mode)
        if selected is None:
            selected, status = select_best_fallback(steps)
            stop_reason = status if status != "no_surrogate_crossing" else "sparse_penalty_step_budget_exhausted"
        else:
            status = "surrogate_success"
            stop_reason = f"tau_crossed_sparse_{self.sparse_mode}"
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=stop_reason,
            start_probability=start_prob,
        )

    def _k_schedule(self) -> tuple[int, ...]:
        if self.sparse_mode == "l0_l2":
            return PROX_TOP_K_RELAXATION
        if self.sparse_mode == "l1_l2_l0":
            return (2, 3)
        return (len(PROX_TOP_K_RELAXATION) + 10,)

    def _sparsify_direction(self, direction: np.ndarray, *, k: int, ctx: OptimizationContext) -> np.ndarray:
        direction = constrained_direction_norm(ctx, direction)
        if float(np.linalg.norm(direction)) < 1e-12:
            return direction
        if self.sparse_mode == "l1":
            abs_d = np.abs(direction)
            threshold = PROX_SOFT_THRESHOLD_FRAC * float(np.max(abs_d))
            direction = np.sign(direction) * np.maximum(abs_d - threshold, 0.0)
            return constrained_direction_norm(ctx, direction)
        if self.sparse_mode in {"l0_l2", "l1_l2_l0"}:
            abs_d = np.abs(direction)
            keep = np.argsort(abs_d)[-min(k, len(direction)):]
            sparse = np.zeros_like(direction)
            sparse[keep] = direction[keep]
            return constrained_direction_norm(ctx, sparse)
        return direction
