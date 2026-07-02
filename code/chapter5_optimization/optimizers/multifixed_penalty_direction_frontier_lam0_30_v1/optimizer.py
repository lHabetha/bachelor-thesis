"""Fixed-step-menu proximity optimizer variant multifixed_penalty_direction_frontier_lam0_30_v1."""
from __future__ import annotations

from optimizers._multifixed_penalty_common import (
    MultiFixedPenalty,
    MultiFixedPenaltyConfig,
)

OPTIMIZER_ID = "multifixed_penalty_direction_frontier_lam0_30_v1"


class MultiFixedPenaltyDirectionFrontierLam030(MultiFixedPenalty):
    def __init__(self) -> None:
        super().__init__(
            optimizer_id=OPTIMIZER_ID,
            config=MultiFixedPenaltyConfig(
                penalty_lambda=0.30,
                use_penalized_direction=True,
                use_penalized_frontier=True,
                sparse_mode=None,
            ),
        )
