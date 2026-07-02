"""Probability-guided coordinate search for non-gradient surrogates."""
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
    log_magnitudes,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
)

OPTIMIZER_ID = "probability_guided_coordinate_v1"
MAGNITUDES = log_magnitudes(2, -4, multipliers=(1.0, 0.5, 0.2, 0.1), descending=False)
PROGRESS_PENALTY = 0.08


@dataclass
class ProbabilityGuidedCoordinate:
    optimizer_id: str = OPTIMIZER_ID

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        steps = [
            evaluate_candidate(
                step_idx=0,
                x0=x0,
                x=x0,
                ctx=ctx,
                gradient_direction=None,
                step_magnitude=0.0,
            )
        ]

        for axis in range(len(x0)):
            for sign in (-1.0, 1.0):
                direction = np.zeros(len(x0), dtype=np.float64)
                direction[axis] = sign
                best_prob_seen = start_prob
                for mag in MAGNITUDES:
                    x = raw_step_from_norm_direction(x0, direction, mag, ctx)
                    step = evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x,
                        ctx=ctx,
                        gradient_direction=direction,
                        step_magnitude=mag,
                    )
                    steps.append(step)
                    # If probability stopped improving along this axis at large
                    # enough distance, avoid spending more probes in that branch.
                    if step.valid and step.probability > best_prob_seen:
                        best_prob_seen = step.probability
                    elif mag > 1.0 and best_prob_seen <= start_prob + 0.01:
                        break

        selected = select_nearest_success(steps, ctx.tau)
        if selected is not None:
            status = "surrogate_success"
            stop_reason = "tau_crossed"
        else:
            valid = [s for s in steps if s.valid]
            if valid:
                selected = max(
                    valid,
                    key=lambda s: (s.probability - start_prob) - PROGRESS_PENALTY * s.normalized_distance,
                )
                status = "no_surrogate_crossing"
                stop_reason = "best_probability_gain_minus_distance"
            else:
                selected, status = select_best_fallback(steps)
                stop_reason = status
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
