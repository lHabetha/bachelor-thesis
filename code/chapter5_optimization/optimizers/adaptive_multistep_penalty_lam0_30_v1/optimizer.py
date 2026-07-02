"""Adaptive multi-step penalized optimizer, lambda 0.30."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)

OPTIMIZER_ID = "adaptive_multistep_penalty_lam0_30_v1"


@dataclass
class AdaptiveMultistepPenaltyLam030(AdaptiveMultistepOptimizer):
    optimizer_id: str = OPTIMIZER_ID
    config: AdaptiveConfig = AdaptiveConfig(step_rule="bold_backtrack", penalty_lambda=0.30, momentum=0.0)
