"""Shared adaptive multi-step gradient optimizers for Chapter 5.

The variants in this module take one accepted step at a time. Step size is
adapted from local surrogate progress, validity failures, and optional momentum;
the exact formula label is used only by ``finalize_result`` after candidate
selection.
"""
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
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
    select_sparse_success,
)


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for an adaptive Chapter 5 optimizer variant."""

    step_rule: str = "momentum_gap"
    penalty_lambda: float = 0.0
    sparse_mode: str | None = None
    max_iters: int = 180
    max_evaluations: int = 900
    initial_step_min: float = 0.02
    initial_step_max: float = 0.75
    min_step: float = 1e-4
    max_step: float = 1.25
    shrink: float = 0.5
    expand: float = 1.35
    max_backtracks: int = 8
    momentum: float = 0.72
    armijo_c: float = 0.03
    bracket_iters: int = 12
    sparse_refine_passes: int = 4


TUNING_CONFIGS: dict[str, AdaptiveConfig] = {
    "gap_backtrack": AdaptiveConfig(step_rule="gap_backtrack", momentum=0.0),
    "bold_backtrack": AdaptiveConfig(step_rule="bold_backtrack", momentum=0.0),
    "momentum_gap": AdaptiveConfig(step_rule="momentum_gap", momentum=0.72),
    "armijo": AdaptiveConfig(step_rule="armijo", momentum=0.0),
    "trust_clip": AdaptiveConfig(step_rule="trust_clip", momentum=0.55, max_step=0.60),
}


@dataclass
class AdaptiveMultistepOptimizer:
    """Adaptive multi-step optimizer with optional proximity penalty."""

    optimizer_id: str
    config: AdaptiveConfig = AdaptiveConfig()

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        grad_raw, start_prob = compute_gradient(ctx.model, x0)
        if float(np.linalg.norm(grad_raw)) < 1e-12:
            return no_gradient_result(
                start=start,
                ctx=ctx,
                optimizer_id=self.optimizer_id,
                probability=start_prob,
            )

        x = x0.copy()
        current_prob = float(start_prob)
        current_objective = self._objective(x=x, x0=x0, prob=current_prob, ctx=ctx)
        step_size = self._initial_step(current_prob, ctx.tau)
        velocity: np.ndarray | None = None
        steps: list[StepRecord] = [
            evaluate_candidate(
                step_idx=0,
                x0=x0,
                x=x0,
                ctx=ctx,
                gradient_direction=None,
                step_magnitude=0.0,
            )
        ]

        for _iteration in range(self.config.max_iters):
            if len(steps) >= self.config.max_evaluations:
                break

            direction = self._direction(x=x, x0=x0, ctx=ctx)
            if direction is None:
                break
            if self.config.momentum > 0.0:
                if velocity is None:
                    velocity = direction
                else:
                    velocity = constrained_direction_norm(
                        ctx,
                        self.config.momentum * velocity + (1.0 - self.config.momentum) * direction,
                    )
                direction = velocity

            accepted: StepRecord | None = None
            accepted_step = step_size
            local_step = self._proposed_step(step_size, current_prob, ctx.tau)
            for backtrack_idx in range(self.config.max_backtracks + 1):
                if len(steps) >= self.config.max_evaluations:
                    break
                candidate = raw_step_from_norm_direction(x, direction, local_step, ctx)
                rec = evaluate_candidate(
                    step_idx=len(steps),
                    x0=x0,
                    x=candidate,
                    ctx=ctx,
                    gradient_direction=direction,
                    step_magnitude=local_step,
                    invalid_reasons=(
                        [f"adaptive_backtrack_{backtrack_idx}"]
                        if backtrack_idx
                        else None
                    ),
                )
                steps.append(rec)
                if self._accept(
                    rec=rec,
                    x=x,
                    x0=x0,
                    current_prob=current_prob,
                    current_objective=current_objective,
                    step_size=local_step,
                    ctx=ctx,
                ):
                    accepted = rec
                    accepted_step = local_step
                    break
                local_step *= self.config.shrink
                if local_step < self.config.min_step:
                    break

            if accepted is None:
                step_size = max(step_size * self.config.shrink, self.config.min_step)
                if step_size <= self.config.min_step:
                    break
                velocity = None
                continue

            previous_x = x.copy()
            x = accepted.params.copy()
            progress = accepted.probability - current_prob
            current_prob = accepted.probability
            current_objective = self._objective(x=x, x0=x0, prob=current_prob, ctx=ctx)

            if accepted.valid and accepted.probability >= ctx.tau:
                bracketed = self._bracket_crossing(
                    x0=x0,
                    lo=previous_x,
                    hi=accepted.params,
                    direction=direction,
                    step_size=accepted_step,
                    steps=steps,
                    ctx=ctx,
                )
                if bracketed is not None:
                    x = bracketed.params.copy()
                self._prepare_success_selection(
                    x0=x0,
                    crossing=x,
                    direction=direction,
                    steps=steps,
                    ctx=ctx,
                )
                selected = self._select_success(steps=steps, x0=x0, ctx=ctx)
                return finalize_result(
                    start=start,
                    ctx=ctx,
                    optimizer_id=self.optimizer_id,
                    steps=steps,
                    selected=selected,
                    status="surrogate_success",
                    stop_reason=self._success_reason(),
                    start_probability=start_prob,
                )

            step_size = self._next_step(
                accepted_step=accepted_step,
                progress=progress,
                current_prob=current_prob,
                tau=ctx.tau,
            )

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
            stop_reason=(
                status
                if status != "no_surrogate_crossing"
                else f"adaptive_{self.config.step_rule}_budget_exhausted"
            ),
            start_probability=start_prob,
        )

    def _direction(
        self,
        *,
        x: np.ndarray,
        x0: np.ndarray,
        ctx: OptimizationContext,
    ) -> np.ndarray | None:
        grad_raw, _prob = compute_gradient(ctx.model, x)
        grad_norm = grad_raw * ctx.normalization.stds
        if self.config.penalty_lambda:
            displacement = (x - x0) / ctx.normalization.stds
            grad_norm = grad_norm - self.config.penalty_lambda * displacement
        direction = constrained_direction_norm(ctx, grad_norm)
        if float(np.linalg.norm(direction)) < 1e-12:
            return None
        return direction

    def _objective(
        self,
        *,
        x: np.ndarray,
        x0: np.ndarray,
        prob: float,
        ctx: OptimizationContext,
    ) -> float:
        if not self.config.penalty_lambda:
            return float(prob)
        return float(prob - self.config.penalty_lambda * ctx.normalization.distance(x0, x))

    def _prepare_success_selection(
        self,
        *,
        x0: np.ndarray,
        crossing: np.ndarray,
        direction: np.ndarray,
        steps: list[StepRecord],
        ctx: OptimizationContext,
    ) -> None:
        if self.config.sparse_mode:
            self._add_sparse_refinement_candidates(
                x0=x0,
                crossing=crossing,
                direction=direction,
                steps=steps,
                ctx=ctx,
            )

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

    def _initial_step(self, prob: float, tau: float) -> float:
        gap = max(float(tau) - float(prob), 0.0)
        if self.config.step_rule == "trust_clip":
            value = 0.05 + 0.8 * gap
        else:
            value = 0.03 + 1.2 * gap
        return float(np.clip(value, self.config.initial_step_min, self.config.initial_step_max))

    def _proposed_step(self, step_size: float, prob: float, tau: float) -> float:
        gap = max(float(tau) - float(prob), 0.0)
        if self.config.step_rule in {"gap_backtrack", "momentum_gap"}:
            gap_step = 0.015 + 1.0 * gap
            return float(np.clip(max(step_size, gap_step), self.config.min_step, self.config.max_step))
        if self.config.step_rule == "trust_clip":
            gap_step = 0.02 + 0.65 * gap
            return float(np.clip(min(max(step_size, gap_step), 0.60), self.config.min_step, self.config.max_step))
        return float(np.clip(step_size, self.config.min_step, self.config.max_step))

    def _accept(
        self,
        *,
        rec: StepRecord,
        x: np.ndarray,
        x0: np.ndarray,
        current_prob: float,
        current_objective: float,
        step_size: float,
        ctx: OptimizationContext,
    ) -> bool:
        if not rec.valid:
            return False
        if rec.probability >= ctx.tau:
            return True
        next_objective = self._objective(x=rec.params, x0=x0, prob=rec.probability, ctx=ctx)
        if self.config.step_rule == "armijo":
            return rec.probability >= current_prob + self.config.armijo_c * max(step_size, 1e-9)
        if self.config.penalty_lambda:
            return next_objective > current_objective + 1e-6 or rec.probability > current_prob + 1e-4
        return rec.probability > current_prob + 1e-5

    def _next_step(
        self,
        *,
        accepted_step: float,
        progress: float,
        current_prob: float,
        tau: float,
    ) -> float:
        gap = max(float(tau) - float(current_prob), 0.0)
        if self.config.step_rule == "bold_backtrack":
            factor = self.config.expand if progress > 0.03 else 0.85
        elif self.config.step_rule == "armijo":
            factor = 1.12 if progress > 0.01 else 0.75
        elif self.config.step_rule == "trust_clip":
            factor = 1.20 if progress > 0.02 else 0.70
        else:
            factor = self.config.expand if progress > 0.02 or gap > 0.35 else 0.90
        gap_cap = self.config.initial_step_max if gap > 0.35 else max(0.08, 0.35 * gap + 0.03)
        next_step = min(accepted_step * factor, gap_cap, self.config.max_step)
        return float(max(next_step, self.config.min_step))

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
        for _ in range(self.config.bracket_iters):
            if len(steps) >= self.config.max_evaluations:
                break
            mid = 0.5 * (lo_x + hi_x)
            rec = evaluate_candidate(
                step_idx=len(steps),
                x0=x0,
                x=mid,
                ctx=ctx,
                gradient_direction=direction,
                step_magnitude=step_size,
                invalid_reasons=["adaptive_bracket"],
            )
            steps.append(rec)
            if rec.valid and rec.probability >= ctx.tau:
                selected = rec
                hi_x = rec.params.copy()
            else:
                lo_x = mid
        return selected

    def _add_sparse_refinement_candidates(
        self,
        *,
        x0: np.ndarray,
        crossing: np.ndarray,
        direction: np.ndarray,
        steps: list[StepRecord],
        ctx: OptimizationContext,
    ) -> None:
        candidate = crossing.copy()
        for _pass_idx in range(self.config.sparse_refine_passes):
            delta_norm = np.abs((candidate - x0) / ctx.normalization.stds)
            order = list(np.argsort(delta_norm))
            improved = False
            for idx in order:
                if delta_norm[idx] <= 1e-4 or idx in ctx.locked_indices:
                    continue
                trial = candidate.copy()
                trial[idx] = x0[idx]
                rec = evaluate_candidate(
                    step_idx=len(steps),
                    x0=x0,
                    x=trial,
                    ctx=ctx,
                    gradient_direction=direction,
                    step_magnitude=ctx.normalization.distance(x0, trial),
                    invalid_reasons=["adaptive_sparse_refine"],
                )
                steps.append(rec)
                if rec.valid and rec.probability >= ctx.tau:
                    candidate = rec.params.copy()
                    improved = True
            if not improved:
                break

    def _success_reason(self) -> str:
        if self.config.sparse_mode:
            return f"tau_crossed_adaptive_{self.config.step_rule}_sparse_{self.config.sparse_mode}"
        if self.config.penalty_lambda:
            return f"tau_crossed_adaptive_{self.config.step_rule}_lambda_{self.config.penalty_lambda:.2f}"
        return f"tau_crossed_adaptive_{self.config.step_rule}"
