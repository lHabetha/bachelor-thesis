"""Random sphere search followed by coordinate-wise shrink toward the start."""
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
    raw_step_from_norm_direction,
    select_best_fallback,
    select_nearest_success,
)

OPTIMIZER_ID = "random_sphere_coordinate_shrink_v1"
RADII = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
SAMPLES_PER_RADIUS = 64
SHRINK_SWEEPS = 10
BINARY_STEPS = 18
BASE_SEED = 26301


@dataclass
class RandomSphereCoordinateShrink:
    optimizer_id: str = OPTIMIZER_ID
    samples_per_radius: int = SAMPLES_PER_RADIUS
    radii: tuple[float, ...] = RADII
    shrink_sweeps: int = SHRINK_SWEEPS
    binary_steps: int = BINARY_STEPS
    base_seed: int = BASE_SEED

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        start_prob = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])
        seed = self.base_seed + int(start.start_id.split("_")[-1])
        rng = np.random.default_rng(seed)
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

        initial_success = self._sphere_search(x0=x0, ctx=ctx, rng=rng, steps=steps)
        if initial_success is None:
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

        selected = self._coordinate_shrink(
            x0=x0,
            ctx=ctx,
            steps=steps,
            initial=initial_success,
        )
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=steps,
            selected=selected,
            status="surrogate_success",
            stop_reason="tau_crossed_after_coordinate_shrink",
            start_probability=start_prob,
        )

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
            selected = select_nearest_success(steps, ctx.tau)
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
                if candidate.normalized_distance < selected.normalized_distance - 1e-12:
                    selected = candidate
                if not np.allclose(candidate.params[coord_idx], current[coord_idx], rtol=0.0, atol=1e-12):
                    current = candidate.params.copy()
                    changed = True
            if not changed:
                break
        return selected

    def _shrink_coordinate(
        self,
        *,
        x0: np.ndarray,
        current: np.ndarray,
        ctx: OptimizationContext,
        coord_idx: int,
        steps: list[StepRecord],
    ) -> StepRecord | None:
        if abs(float(current[coord_idx] - x0[coord_idx])) < 1e-12:
            return None

        direction = np.zeros(len(x0), dtype=np.float64)
        direction[coord_idx] = np.sign(x0[coord_idx] - current[coord_idx])
        if direction[coord_idx] == 0.0:
            return None

        full = current.copy()
        full[coord_idx] = x0[coord_idx]
        full_step = self._record_shrink_probe(x0=x0, x=full, ctx=ctx, direction=direction, steps=steps)
        if self._is_acceptable(full_step, ctx):
            return full_step

        lo = float(x0[coord_idx])
        hi = float(current[coord_idx])
        best: StepRecord | None = None
        for _ in range(self.binary_steps):
            mid = 0.5 * (lo + hi)
            probe = current.copy()
            probe[coord_idx] = mid
            step = self._record_shrink_probe(x0=x0, x=probe, ctx=ctx, direction=direction, steps=steps)
            if self._is_acceptable(step, ctx):
                best = step
                hi = mid
            else:
                lo = mid
        return best

    def _record_shrink_probe(
        self,
        *,
        x0: np.ndarray,
        x: np.ndarray,
        ctx: OptimizationContext,
        direction: np.ndarray,
        steps: list[StepRecord],
    ) -> StepRecord:
        step = evaluate_candidate(
            step_idx=len(steps),
            x0=x0,
            x=x,
            ctx=ctx,
            gradient_direction=direction,
            step_magnitude=ctx.normalization.distance(x0, x),
        )
        steps.append(step)
        return step

    @staticmethod
    def _is_acceptable(step: StepRecord, ctx: OptimizationContext) -> bool:
        return bool(step.valid and step.probability >= ctx.tau)
