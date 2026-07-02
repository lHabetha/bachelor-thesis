"""Adaptive multi-step penalized optimizer, lambda 0.05."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)

OPTIMIZER_ID = "adaptive_multistep_penalty_lam0_05_v1"


@dataclass
class AdaptiveMultistepPenaltyLam005(AdaptiveMultistepOptimizer):
    optimizer_id: str = OPTIMIZER_ID
    config: AdaptiveConfig = AdaptiveConfig(step_rule="bold_backtrack", penalty_lambda=0.05, momentum=0.0)
