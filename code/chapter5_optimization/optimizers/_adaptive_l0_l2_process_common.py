"""Adaptive penalty variants with process-level L0/L2 sparsity pressure."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizers._adaptive_multistep_common import (
    AdaptiveConfig,
    AdaptiveMultistepOptimizer,
)
from shared.interface import OptimizationContext
from shared.model_utils import compute_gradient
from shared.optimizer_utils import (
    constrained_direction_norm,
    sparse_metrics,
)


@dataclass
class AdaptivePenaltyProcessL0L2(AdaptiveMultistepOptimizer):
    """Use the adaptive penalty path, but sparsify each update direction."""

    keep_top_k: int = 3
    l0_lambda: float = 0.03

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

        keep_count = max(1, min(int(self.keep_top_k), len(direction)))
        abs_direction = np.abs(direction)
        keep = np.argsort(abs_direction)[-keep_count:]
        sparse = np.zeros_like(direction)
        sparse[keep] = direction[keep]
        sparse = constrained_direction_norm(ctx, sparse)
        if float(np.linalg.norm(sparse)) < 1e-12:
            return None
        return sparse

    def _objective(
        self,
        *,
        x: np.ndarray,
        x0: np.ndarray,
        prob: float,
        ctx: OptimizationContext,
    ) -> float:
        metrics = sparse_metrics(x0, x, ctx)
        l0_fraction = metrics.l0_active / max(1, len(x0))
        return float(
            prob
            - self.config.penalty_lambda * metrics.l2_distance
            - self.l0_lambda * l0_fraction
        )

    def _success_reason(self) -> str:
        return (
            "tau_crossed_adaptive_"
            f"{self.config.step_rule}_process_l0_l2_k{self.keep_top_k}_"
            f"lambda_{self.config.penalty_lambda:.2f}"
        )


PROCESS_L0_L2_CONFIG = AdaptiveConfig(
    step_rule="bold_backtrack",
    penalty_lambda=0.10,
    sparse_mode=None,
    momentum=0.0,
)
