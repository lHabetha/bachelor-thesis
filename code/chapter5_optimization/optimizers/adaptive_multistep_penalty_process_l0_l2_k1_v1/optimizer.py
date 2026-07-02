"""Adaptive penalty with process-level L0/L2 sparsity, top-1 direction."""
from __future__ import annotations

from dataclasses import dataclass

from optimizers._adaptive_l0_l2_process_common import (
    AdaptivePenaltyProcessL0L2,
    PROCESS_L0_L2_CONFIG,
)
from optimizers._adaptive_multistep_common import AdaptiveConfig

OPTIMIZER_ID = "adaptive_multistep_penalty_process_l0_l2_k1_v1"


@dataclass
class AdaptivePenaltyProcessL0L2K1(AdaptivePenaltyProcessL0L2):
    optimizer_id: str = OPTIMIZER_ID
    config: AdaptiveConfig = PROCESS_L0_L2_CONFIG
    keep_top_k: int = 1
