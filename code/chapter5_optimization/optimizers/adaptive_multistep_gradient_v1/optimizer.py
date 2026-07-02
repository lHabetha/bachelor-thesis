"""Adaptive multi-step gradient optimizer."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)

OPTIMIZER_ID = "adaptive_multistep_gradient_v1"


@dataclass
class AdaptiveMultistepGradient(AdaptiveMultistepOptimizer):
    optimizer_id: str = OPTIMIZER_ID
    config: AdaptiveConfig = AdaptiveConfig(step_rule="bold_backtrack", momentum=0.0)
