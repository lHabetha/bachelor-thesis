"""Shared penalized/sparse receding multiscale optimizers."""
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
    select_sparse_success,
    sparse_metrics,
)

STEP_SIZES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MAX_DEPTH = 14
BEAM_WIDTH = 12
MAX_EVALUATIONS = 900
L2_PENALTY = 0.35
L0_PENALTY = 0.08
STRONG_L0_PENALTY = 0.35


@dataclass
class RecedingMultiscalePenalized:
    optimizer_id: str
    mode: str

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
                    if len(steps) >= MAX_EVALUATIONS:
                        break
                    x_candidate = raw_step_from_norm_direction(x_parent, direction.direction_norm, step_size, ctx)
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
                if len(steps) >= MAX_EVALUATIONS:
                    break

            selected = self._select_success(steps=steps, x0=x0, ctx=ctx)
            if selected is not None:
                return finalize_result(
                    start=start,
                    ctx=ctx,
                    optimizer_id=self.optimizer_id,
                    steps=steps,
                    selected=selected,
                    status="surrogate_success",
                    stop_reason=f"tau_crossed_{self.mode}",
                    start_probability=first_direction.probability,
                )
            if not new_frontier or len(steps) >= MAX_EVALUATIONS:
                break

            new_frontier.sort(key=lambda item: self._frontier_key(item, x0=x0, ctx=ctx, start_prob=start_step.probability))
            frontier = new_frontier[:BEAM_WIDTH]

        selected, status = select_best_fallback(steps)
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=status if status != "no_surrogate_crossing" else f"{self.mode}_frontier_exhausted",
            start_probability=first_direction.probability,
        )

    def _select_success(self, *, steps: list[StepRecord], x0: np.ndarray, ctx: OptimizationContext) -> StepRecord | None:
        if self.mode == "sparse_l0_l2":
            return select_sparse_success(steps, x0=x0, ctx=ctx, mode="l0_l2")
        return select_nearest_success(steps, ctx.tau)

    def _frontier_key(
        self,
        item: tuple[np.ndarray, float, float],
        *,
        x0: np.ndarray,
        ctx: OptimizationContext,
        start_prob: float,
    ) -> tuple:
        x, prob, dist = item
        progress = prob - start_prob
        if self.mode == "penalized_l2":
            utility = progress - L2_PENALTY * dist
            return (-utility, dist)
        metrics = sparse_metrics(x0, x, ctx)
        l0_penalty = STRONG_L0_PENALTY if self.mode == "sparse_l0_l2_strong" else L0_PENALTY
        utility = progress - L2_PENALTY * metrics.l2_distance - l0_penalty * metrics.l0_active
        return (metrics.l0_active, -utility, metrics.l2_distance)
