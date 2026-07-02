"""Expanding trust-region hybrid optimizer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.interface import BenchmarkStart, OptimizationContext, OptimizationResult
from shared.optimizer_utils import (
    evaluate_candidate,
    finalize_result,
    normalized_gradient_direction,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
)

OPTIMIZER_ID = "trust_region_hybrid_v1"
RADII = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
RANDOM_PER_RADIUS = 32
BASE_SEED = 26201


@dataclass
class TrustRegionHybrid:
    optimizer_id: str = OPTIMIZER_ID

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        rng = np.random.default_rng(BASE_SEED + int(start.start_id.split("_")[-1]))
        steps = [evaluate_candidate(step_idx=0, x0=x0, x=x0, ctx=ctx, gradient_direction=None, step_magnitude=0.0)]

        grad = normalized_gradient_direction(ctx, x0)
        axes = []
        for i in range(len(x0)):
            for sign in (-1.0, 1.0):
                d = np.zeros(len(x0), dtype=np.float64)
                d[i] = sign
                axes.append(d)

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
                steps.append(evaluate_candidate(
                    step_idx=len(steps), x0=x0, x=x, ctx=ctx,
                    gradient_direction=d, step_magnitude=radius,
                ))
            selected = select_nearest_success(steps, ctx.tau)
            if selected is not None:
                return finalize_result(
                    start=start, ctx=ctx, optimizer_id=self.optimizer_id, steps=steps,
                    selected=selected, status="surrogate_success", stop_reason="tau_crossed",
                    start_probability=start_prob,
                )

        selected, status = select_best_fallback(steps)
        return finalize_result(
            start=start, ctx=ctx, optimizer_id=self.optimizer_id, steps=steps,
            selected=selected, status=status,
            stop_reason=status if status != "no_surrogate_crossing" else "trust_region_exhausted",
            start_probability=start_prob,
        )
