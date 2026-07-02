"""Adaptive multi-step penalized optimizer, lambda 0.20."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)

OPTIMIZER_ID = "adaptive_multistep_penalty_lam0_20_v1"


@dataclass
class AdaptiveMultistepPenaltyLam020(AdaptiveMultistepOptimizer):
    optimizer_id: str = OPTIMIZER_ID
    config: AdaptiveConfig = AdaptiveConfig(step_rule="bold_backtrack", penalty_lambda=0.20, momentum=0.0)
