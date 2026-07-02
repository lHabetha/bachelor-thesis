"""Receding-gradient multi-scale beam search optimizer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.interface import (
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
    StepRecord,
)
from shared.optimizer_utils import (
    evaluate_candidate,
    finalize_result,
    no_gradient_result,
    normalized_gradient_direction,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
)

OPTIMIZER_ID = "receding_gradient_multiscale_v1"
STEP_SIZES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MAX_DEPTH = 20
BEAM_WIDTH = 12


@dataclass
class RecedingGradientMultiscale:
    optimizer_id: str = OPTIMIZER_ID

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        first_direction = normalized_gradient_direction(ctx, x0)
        if first_direction is None:
            prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
            return no_gradient_result(start=start, ctx=ctx, optimizer_id=self.optimizer_id, probability=prob)

        start_step = evaluate_candidate(
            step_idx=0,
            x0=x0,
            x=x0,
            ctx=ctx,
            gradient_direction=first_direction.direction_norm,
            step_magnitude=0.0,
        )
        steps: list[StepRecord] = [start_step]
        frontier = [(x0.copy(), start_step.probability, 0.0)]
        seen = {tuple(np.round((x0 - x0) / ctx.normalization.stds, 6))}

        for _depth in range(MAX_DEPTH):
            new_frontier: list[tuple[np.ndarray, float, float]] = []
            for x_parent, _prob_parent, _dist_parent in frontier:
                direction = normalized_gradient_direction(ctx, x_parent)
                if direction is None:
                    continue
                for step_size in STEP_SIZES:
                    x_candidate = raw_step_from_norm_direction(
                        x_parent, direction.direction_norm, step_size, ctx
                    )
                    key = tuple(np.round((x_candidate - x0) / ctx.normalization.stds, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    rec = evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x_candidate,
                        ctx=ctx,
                        gradient_direction=direction.direction_norm,
                        step_magnitude=step_size,
                    )
                    steps.append(rec)
                    if rec.valid:
                        new_frontier.append((rec.params.copy(), rec.probability, rec.normalized_distance))

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
                    start_probability=first_direction.probability,
                )
            if not new_frontier:
                break

            def rank(item: tuple[np.ndarray, float, float]) -> tuple[float, float]:
                _x, prob, dist = item
                # Prefer high progress with moderate distance.
                return (-(prob - start_step.probability) / max(dist, 1e-6), dist)

            new_frontier.sort(key=rank)
            frontier = new_frontier[:BEAM_WIDTH]

        selected, status = select_best_fallback(steps)
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=status if status != "no_surrogate_crossing" else "frontier_exhausted",
            start_probability=first_direction.probability,
        )
