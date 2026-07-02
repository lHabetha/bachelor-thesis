"""Demo optimizers using a 40/60 mixed L2/L0 proximity score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)
from optimizers.random_sphere_coordinate_shrink_v1.optimizer import (
    RandomSphereCoordinateShrink,
)
from optimizers.trust_region_hybrid_v1.optimizer import (
    BASE_SEED as TRUST_BASE_SEED,
    RADII as TRUST_RADII,
    RANDOM_PER_RADIUS as TRUST_RANDOM_PER_RADIUS,
)
from shared.interface import (
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
    StepRecord,
)
from shared.optimizer_utils import (
    evaluate_candidate,
    finalize_result,
    mixed_l2_l0_score,
    normalized_gradient_direction,
    raw_step_from_norm_direction,
    select_best_fallback,
    select_mixed_l2_l0_success,
)

L2_WEIGHT = 0.4
L0_WEIGHT = 0.6


@dataclass
class MixedAdaptivePenalty(AdaptiveMultistepOptimizer):
    """Adaptive penalty demo with mixed L2/L0 scoring."""

    optimizer_id: str = "adaptive_multistep_penalty_mixed_l2_l0_40_60_demo_v1"
    config: AdaptiveConfig = AdaptiveConfig(
        step_rule="bold_backtrack",
        penalty_lambda=0.10,
        sparse_mode=None,
        momentum=0.0,
    )

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
        return float(
            prob
            - self.config.penalty_lambda
            * mixed_l2_l0_score(
                x0,
                x,
                ctx,
                l2_weight=L2_WEIGHT,
                l0_weight=L0_WEIGHT,
            )
        )

    def _select_success(
        self,
        *,
        steps: list[StepRecord],
        x0: np.ndarray,
        ctx: OptimizationContext,
    ) -> StepRecord | None:
        return select_mixed_l2_l0_success(
            steps,
            x0=x0,
            ctx=ctx,
            l2_weight=L2_WEIGHT,
            l0_weight=L0_WEIGHT,
        )

    def _success_reason(self) -> str:
        return "tau_crossed_adaptive_bold_backtrack_mixed_l2_l0_40_60"


@dataclass
class MixedTrustRegion:
    """Trust-region demo with unchanged candidates and mixed L2/L0 selection."""

    optimizer_id: str = "trust_region_mixed_l2_l0_40_60_demo_v1"

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        rng = np.random.default_rng(TRUST_BASE_SEED + int(start.start_id.split("_")[-1]))
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
                d /= max(float(np.linalg.norm(d)), 1e-12)
                dirs.append(d.astype(np.float64))
            for d in dirs:
                x = raw_step_from_norm_direction(x0, d, radius, ctx)
                steps.append(
                    evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x,
                        ctx=ctx,
                        gradient_direction=d,
                        step_magnitude=radius,
                    )
                )
            selected = select_mixed_l2_l0_success(
                steps,
                x0=x0,
                ctx=ctx,
                l2_weight=L2_WEIGHT,
                l0_weight=L0_WEIGHT,
            )
            if selected is not None:
                return finalize_result(
                    start=start,
                    ctx=ctx,
                    optimizer_id=self.optimizer_id,
                    steps=steps,
                    selected=selected,
                    status="surrogate_success",
                    stop_reason="tau_crossed_mixed_l2_l0_40_60",
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
            stop_reason=status if status != "no_surrogate_crossing" else "trust_region_exhausted",
            start_probability=start_prob,
        )


@dataclass
class MixedSphereShrink128(RandomSphereCoordinateShrink):
    """128-direction sphere-shrink demo with mixed L2/L0 scoring."""

    optimizer_id: str = "sphere_shrink_128_mixed_l2_l0_40_60_demo_v1"
    samples_per_radius: int = 128
    base_seed: int = 26001

    def _sphere_search(
        self,
        *,
        x0: np.ndarray,
        ctx: OptimizationContext,
        rng: np.random.Generator,
        steps: list[StepRecord],
    ) -> StepRecord | None:
        for radius in self.radii:
            for _ in range(self.samples_per_radius):
                direction = rng.normal(size=len(x0))
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                x = raw_step_from_norm_direction(x0, direction, radius, ctx)
                steps.append(
                    evaluate_candidate(
                        step_idx=len(steps),
                        x0=x0,
                        x=x,
                        ctx=ctx,
                        gradient_direction=direction,
                        step_magnitude=radius,
                    )
                )
            selected = select_mixed_l2_l0_success(
                steps,
                x0=x0,
                ctx=ctx,
                l2_weight=L2_WEIGHT,
                l0_weight=L0_WEIGHT,
            )
            if selected is not None:
                return selected
        return None

    def _coordinate_shrink(
        self,
        *,
        x0: np.ndarray,
        ctx: OptimizationContext,
        steps: list[StepRecord],
        initial: StepRecord,
    ) -> StepRecord:
        current = initial.params.copy()
        selected = initial
        selected_score = mixed_l2_l0_score(
            x0,
            selected.params,
            ctx,
            l2_weight=L2_WEIGHT,
            l0_weight=L0_WEIGHT,
        )

        for _sweep in range(self.shrink_sweeps):
            changed = False
            for coord_idx in range(len(x0)):
                candidate = self._shrink_coordinate(
                    x0=x0,
                    current=current,
                    ctx=ctx,
                    coord_idx=coord_idx,
                    steps=steps,
                )
                if candidate is None:
                    continue
                candidate_score = mixed_l2_l0_score(
                    x0,
                    candidate.params,
                    ctx,
                    l2_weight=L2_WEIGHT,
                    l0_weight=L0_WEIGHT,
                )
                if candidate_score < selected_score - 1e-12:
                    selected = candidate
                    selected_score = candidate_score
                if not np.allclose(candidate.params[coord_idx], current[coord_idx], rtol=0.0, atol=1e-12):
                    current = candidate.params.copy()
                    changed = True
            if not changed:
                break
        return selected
