"""Random sphere with coordinate shrink and sparse L0-L2 final selection."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers.random_sphere_coordinate_shrink_v1.optimizer import (
    RandomSphereCoordinateShrink,
)
from shared.interface import (
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
)
from shared.optimizer_utils import (
    finalize_result,
    select_best_fallback,
    select_sparse_success,
)

OPTIMIZER_ID = "random_sphere_coordinate_shrink_128_sparse_l0_l2_v1"


@dataclass
class RandomSphereCoordinateShrink128SparseL0L2(RandomSphereCoordinateShrink):
    optimizer_id: str = OPTIMIZER_ID
    samples_per_radius: int = 128
    base_seed: int = 26001
    sparse_mode: str = "l0_l2"

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        result = super().optimize(start, ctx)
        selected = select_sparse_success(
            result.steps,
            x0=start.params,
            ctx=ctx,
            mode=self.sparse_mode,
        )
        if selected is None:
            selected, status = select_best_fallback(result.steps)
            stop_reason = status if status != "no_surrogate_crossing" else "radius_budget_exhausted"
        else:
            status = "surrogate_success"
            stop_reason = f"tau_crossed_sparse_{self.sparse_mode}"
        return finalize_result(
            start=start,
            ctx=ctx,
            optimizer_id=self.optimizer_id,
            steps=result.steps,
            selected=selected,
            status=status,
            stop_reason=stop_reason,
            start_probability=result.start_probability,
        )
