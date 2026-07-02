"""Multi-fixed-step gradient optimizers with proximity penalties."""
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
    no_gradient_result,
    normalized_gradient_direction,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
    select_sparse_success,
)


STEP_SIZES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MAX_DEPTH = 20
BEAM_WIDTH = 12
BRACKET_ITERS = 10


@dataclass(frozen=True)
class MultiFixedPenaltyConfig:
    """Configuration for fixed-step-menu proximity variants."""

    penalty_lambda: float = 0.10
    use_penalized_direction: bool = True
    use_penalized_frontier: bool = True
    sparse_mode: str | None = None


@dataclass
class MultiFixedPenalty:
    optimizer_id: str
    config: MultiFixedPenaltyConfig = MultiFixedPenaltyConfig()

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        first_direction = self._direction(x=x0, x0=x0, ctx=ctx)
        if first_direction is None:
            prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
            return no_gradient_result(start=start, ctx=ctx, optimizer_id=self.optimizer_id, probability=prob)

        start_step = evaluate_candidate(
            step_idx=0,
            x0=x0,
            x=x0,
            ctx=ctx,
            gradient_direction=first_direction,
            step_magnitude=0.0,
        )
        steps: list[StepRecord] = [start_step]
        frontier = [(x0.copy(), start_step.probability, 0.0)]
        seen = {tuple(np.round((x0 - x0) / ctx.normalization.stds, 6))}

        for _depth in range(MAX_DEPTH):
            new_frontier: list[tuple[np.ndarray, float, float]] = []
            for x_parent, _prob_parent, _dist_parent in frontier:
                direction = self._direction(x=x_parent, x0=x0, ctx=ctx)
                if direction is None:
                    continue
                for step_size in STEP_SIZES:
                    x_candidate = raw_step_from_norm_direction(x_parent, direction, step_size, ctx)
                    key = tuple(np.round((x_candidate - x0) / ctx.normalization.stds, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    rec = evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x_candidate,
                        ctx=ctx,
                        gradient_direction=direction,
                        step_magnitude=step_size,
                    )
                    steps.append(rec)
                    if rec.valid and rec.probability >= ctx.tau:
                        bracketed = self._bracket_crossing(
                            x0=x0,
                            lo=x_parent,
                            hi=rec.params,
                            direction=direction,
                            step_size=step_size,
                            steps=steps,
                            ctx=ctx,
                        )
                        if bracketed is not None:
                            rec = bracketed
                    if rec.valid:
                        new_frontier.append((rec.params.copy(), rec.probability, rec.normalized_distance))

            selected = self._select_success(steps=steps, x0=x0, ctx=ctx)
            if selected is not None:
                return finalize_result(
                    start=start,
                    ctx=ctx,
                    optimizer_id=self.optimizer_id,
                    steps=steps,
                    selected=selected,
                    status="surrogate_success",
                    stop_reason=self._success_reason(),
                    start_probability=start_step.probability,
                )
            if not new_frontier:
                break

            new_frontier.sort(
                key=lambda item: self._frontier_key(
                    item,
                    x0=x0,
                    start_probability=start_step.probability,
                )
            )
            frontier = new_frontier[:BEAM_WIDTH]

        selected, status = select_best_fallback(steps)
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status=status,
            stop_reason=status if status != "no_surrogate_crossing" else "fixed_step_penalty_frontier_exhausted",
            start_probability=start_step.probability,
        )

    def _direction(self, *, x: np.ndarray, x0: np.ndarray, ctx: OptimizationContext) -> np.ndarray | None:
        if not self.config.use_penalized_direction:
            direction = normalized_gradient_direction(ctx, x)
            return None if direction is None else direction.direction_norm

        grad_raw, _prob = compute_gradient(ctx.model, x)
        grad_norm = grad_raw * ctx.normalization.stds
        displacement_norm = (x - x0) / ctx.normalization.stds
        penalized = grad_norm - self.config.penalty_lambda * displacement_norm
        direction = constrained_direction_norm(ctx, penalized)
        if float(np.linalg.norm(direction)) < 1e-12:
            return None
        return direction

    def _select_success(
        self,
        *,
        steps: list[StepRecord],
        x0: np.ndarray,
        ctx: OptimizationContext,
    ) -> StepRecord | None:
        if self.config.sparse_mode:
            return select_sparse_success(steps, x0=x0, ctx=ctx, mode=self.config.sparse_mode)
        return select_nearest_success(steps, ctx.tau)

    def _frontier_key(
        self,
        item: tuple[np.ndarray, float, float],
        *,
        x0: np.ndarray,
        start_probability: float,
    ) -> tuple[float, float]:
        x, prob, dist = item
        if self.config.use_penalized_frontier:
            # dist is already the L2 norm in normalized space.
            objective = prob - 0.5 * self.config.penalty_lambda * dist * dist
            start_objective = start_probability
            progress = objective - start_objective
        else:
            progress = prob - start_probability
        return (-progress / max(dist, 1e-6), dist)

    def _bracket_crossing(
        self,
        *,
        x0: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        direction: np.ndarray,
        step_size: float,
        steps: list[StepRecord],
        ctx: OptimizationContext,
    ) -> StepRecord | None:
        selected: StepRecord | None = None
        lo_x = lo.copy()
        hi_x = hi.copy()
        for _ in range(BRACKET_ITERS):
            mid = 0.5 * (lo_x + hi_x)
            rec = evaluate_candidate(
                step_idx=len(steps),
                x0=x0,
                x=mid,
                ctx=ctx,
                gradient_direction=direction,
                step_magnitude=step_size,
                invalid_reasons=["fixed_step_penalty_bracket"],
            )
            steps.append(rec)
            if rec.valid and rec.probability >= ctx.tau:
                selected = rec
                hi_x = rec.params.copy()
            else:
                lo_x = mid
        return selected

    def _success_reason(self) -> str:
        parts = ["tau_crossed_fixed_step_penalty"]
        if self.config.use_penalized_direction:
            parts.append("direction")
        if self.config.use_penalized_frontier:
            parts.append("frontier")
        if self.config.sparse_mode:
            parts.append(self.config.sparse_mode)
        return "_".join(parts)
