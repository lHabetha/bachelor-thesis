"""Axis-aligned coordinate bracket optimizer."""
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

OPTIMIZER_ID = "coordinate_axis_bracket_v1"
MAGNITUDES = log_magnitudes(2, -4, multipliers=(1.0, 0.5, 0.2), descending=False)


@dataclass
class CoordinateAxisBracket:
    optimizer_id: str = OPTIMIZER_ID

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
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

        for axis in range(len(x0)):
            for sign in (-1.0, 1.0):
                direction = np.zeros(len(x0), dtype=np.float64)
                direction[axis] = sign
                for mag in MAGNITUDES:
                    x = raw_step_from_norm_direction(x0, direction, mag, ctx)
                    steps.append(evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x,
                        ctx=ctx,
                        gradient_direction=direction,
                        step_magnitude=mag,
                    ))

        selected = select_nearest_success(steps, ctx.tau)
        if selected is None:
            selected, status = select_best_fallback(steps)
            stop_reason = status
        else:
            status = "surrogate_success"
            stop_reason = "tau_crossed"
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
