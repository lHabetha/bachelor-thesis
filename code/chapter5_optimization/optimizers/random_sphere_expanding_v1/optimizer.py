"""Gradient-free expanding random sphere baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.interface import (
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
)
from shared.optimizer_utils import (
    evaluate_candidate,
    finalize_result,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
)

OPTIMIZER_ID = "random_sphere_expanding_v1"
RADII = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
SAMPLES_PER_RADIUS = 64
BASE_SEED = 26001


@dataclass
class RandomSphereExpanding:
    optimizer_id: str = OPTIMIZER_ID
    samples_per_radius: int = SAMPLES_PER_RADIUS
    radii: tuple[float, ...] = RADII
    base_seed: int = BASE_SEED

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        seed = self.base_seed + int(start.start_id.split("_")[-1])
        rng = np.random.default_rng(seed)
        steps = []

        start_step = evaluate_candidate(
            step_idx=0,
            x0=x0,
            x=x0,
            ctx=ctx,
            gradient_direction=None,
            step_magnitude=0.0,
        )
        steps.append(start_step)

        for radius in self.radii:
            for _ in range(self.samples_per_radius):
                direction = rng.normal(size=len(x0))
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                x = raw_step_from_norm_direction(x0, direction, radius, ctx)
                steps.append(evaluate_candidate(
                    step_idx=len(steps),
                    x0=x0,
                    x=x,
                    ctx=ctx,
                    gradient_direction=direction,
                    step_magnitude=radius,
                ))
            selected = select_nearest_success(steps, ctx.tau)
            if selected is not None:
                return finalize_result(
                    start=start,
                    ctx=ctx,
                    optimizer_id=self.optimizer_id,
                    steps=steps,
                    selected=selected,
                    status="surrogate_success",
                    stop_reason="tau_crossed",
                    start_probability=start_prob,
                )

        selected, status = select_best_fallback(steps)
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=status if status != "no_surrogate_crossing" else "radius_budget_exhausted",
            start_probability=start_prob,
        )
