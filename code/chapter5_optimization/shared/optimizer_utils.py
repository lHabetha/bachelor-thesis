"""Reusable optimizer helpers for Chapter 5.

The workbench compares many algorithms, but the bookkeeping must stay identical:
distances are measured in the benchmark-normalized 13-D space, validity is
checked before trusting a surrogate probability, and the exact formula oracle is
only used after candidate evaluation for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .interface import BenchmarkStart, OptimizationContext, OptimizationResult, StepRecord
from .model_utils import compute_gradient
from .oracle import is_valid, oracle_check


@dataclass(frozen=True)
class DirectionInfo:
    """A unit direction in benchmark-normalized space and its raw-space form."""

    direction_norm: np.ndarray
    direction_raw: np.ndarray
    probability: float
    gradient_raw: np.ndarray | None = None


SPARSE_ACTIVE_EPS = 1e-4


@dataclass(frozen=True)
class SparseMetrics:
    """Sparse normalized displacement metrics for one candidate."""

    l0_active: int
    l1_distance: float
    l2_distance: float


def sparse_metrics(
    x0: np.ndarray,
    x: np.ndarray,
    ctx: OptimizationContext,
    *,
    active_eps: float = SPARSE_ACTIVE_EPS,
) -> SparseMetrics:
    """Compute L0/L1/L2 metrics in benchmark-normalized parameter space."""
    delta = (x.astype(np.float64) - x0.astype(np.float64)) / ctx.normalization.stds
    abs_delta = np.abs(delta)
    return SparseMetrics(
        l0_active=int(np.count_nonzero(abs_delta > active_eps)),
        l1_distance=float(np.sum(abs_delta)),
        l2_distance=float(np.linalg.norm(delta)),
    )


def sparse_success_key(step: StepRecord, x0: np.ndarray, ctx: OptimizationContext, mode: str) -> tuple:
    """Selection key for sparse norm modes."""
    metrics = sparse_metrics(x0, step.params, ctx)
    if mode == "l1_l2_l0":
        return (metrics.l0_active, metrics.l1_distance, metrics.l2_distance, step.step_idx)
    if mode == "l0_l2":
        return (metrics.l0_active, metrics.l2_distance, step.step_idx)
    if mode == "l1":
        return (metrics.l1_distance, metrics.l2_distance, step.step_idx)
    raise ValueError(f"Unknown sparse mode: {mode}")


def select_sparse_success(
    steps: list[StepRecord],
    *,
    x0: np.ndarray,
    ctx: OptimizationContext,
    mode: str,
) -> StepRecord | None:
    """Select a valid tau-crossing by sparse norm criteria instead of pure L2."""
    successes = [s for s in steps if s.valid and s.probability >= ctx.tau]
    if not successes:
        return None
    return min(successes, key=lambda step: sparse_success_key(step, x0, ctx, mode))


def mixed_l2_l0_score(
    x0: np.ndarray,
    x: np.ndarray,
    ctx: OptimizationContext,
    *,
    l2_weight: float = 0.4,
    l0_weight: float = 0.6,
) -> float:
    """Weighted proximity score with L0 normalized by parameter count."""
    metrics = sparse_metrics(x0, x, ctx)
    l0_fraction = metrics.l0_active / max(1, len(x0))
    return float(l2_weight * metrics.l2_distance + l0_weight * l0_fraction)


def select_mixed_l2_l0_success(
    steps: list[StepRecord],
    *,
    x0: np.ndarray,
    ctx: OptimizationContext,
    l2_weight: float = 0.4,
    l0_weight: float = 0.6,
) -> StepRecord | None:
    """Select a valid tau-crossing by a weighted L2/L0 proximity score."""
    successes = [s for s in steps if s.valid and s.probability >= ctx.tau]
    if not successes:
        return None
    return min(
        successes,
        key=lambda step: (
            mixed_l2_l0_score(
                x0,
                step.params,
                ctx,
                l2_weight=l2_weight,
                l0_weight=l0_weight,
            ),
            step.step_idx,
        ),
    )


def has_parameter_locks(ctx: OptimizationContext) -> bool:
    """Return True when the current per-start context freezes coordinates."""
    return bool(ctx.locked_indices)


def project_locked_params(ctx: OptimizationContext, x0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return a candidate with locked coordinates restored to the start values."""
    projected = x.astype(np.float64).copy()
    if ctx.locked_indices:
        projected[list(ctx.locked_indices)] = x0[list(ctx.locked_indices)]
    return projected


def constrained_direction_norm(ctx: OptimizationContext, direction_norm: np.ndarray) -> np.ndarray:
    """Zero locked normalized-space direction components and renormalize."""
    direction = direction_norm.astype(np.float64).copy()
    if ctx.locked_indices:
        direction[list(ctx.locked_indices)] = 0.0
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < 1e-12:
        return np.zeros_like(direction)
    return direction / length


def normalized_gradient_direction(ctx: OptimizationContext, x: np.ndarray) -> DirectionInfo | None:
    """Return the MLP gradient direction as a unit vector in normalized space."""
    grad_raw, prob = compute_gradient(ctx.model, x)
    grad_norm_space = grad_raw * ctx.normalization.stds
    if ctx.locked_indices:
        grad_norm_space[list(ctx.locked_indices)] = 0.0
    length = float(np.linalg.norm(grad_norm_space))
    if not np.isfinite(length) or length < 1e-12:
        return None
    direction_norm = grad_norm_space / length
    return DirectionInfo(
        direction_norm=direction_norm.astype(np.float64),
        direction_raw=(direction_norm * ctx.normalization.stds).astype(np.float64),
        probability=float(prob),
        gradient_raw=grad_raw.astype(np.float64),
    )


def raw_step_from_norm_direction(
    x: np.ndarray,
    direction_norm: np.ndarray,
    magnitude: float,
    ctx: OptimizationContext,
) -> np.ndarray:
    """Step from ``x`` by ``magnitude`` along a normalized-space direction."""
    direction = constrained_direction_norm(ctx, direction_norm)
    return x + float(magnitude) * direction * ctx.normalization.stds


def log_magnitudes(
    max_order: int,
    min_order: int,
    multipliers: Iterable[float] = (1.0, 0.5),
    *,
    descending: bool = True,
) -> list[float]:
    """Build deterministic log-spaced normalized step magnitudes."""
    mags = [
        float(mult) * (10.0 ** int(order))
        for order in range(int(max_order), int(min_order) - 1, -1)
        for mult in multipliers
    ]
    mags = sorted(set(mags), reverse=descending)
    return mags


def fine_between(lo: float, hi: float, n: int) -> list[float]:
    """Return ``n`` interior points between two magnitudes."""
    if n <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    return [float(v) for v in np.linspace(lo, hi, n + 2)[1:-1]]


def evaluate_candidate(
    *,
    step_idx: int,
    x0: np.ndarray,
    x: np.ndarray,
    ctx: OptimizationContext,
    gradient_direction: np.ndarray | None,
    step_magnitude: float | None,
    include_oracle: bool = False,
    invalid_reasons: list[str] | None = None,
) -> StepRecord:
    """Evaluate one candidate consistently for all optimizers."""
    x = project_locked_params(ctx, x0, x)
    if gradient_direction is not None:
        gradient_direction = constrained_direction_norm(ctx, gradient_direction)
        if float(np.linalg.norm(gradient_direction)) < 1e-12:
            gradient_direction = None
    valid = is_valid(x)
    if valid:
        probability = float(ctx.model.predict_proba(x.reshape(1, -1))[0])
    else:
        probability = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])

    record = StepRecord(
        step_idx=int(step_idx),
        params=x.astype(np.float64).copy(),
        probability=probability,
        gradient_direction=(
            gradient_direction.astype(np.float64).copy()
            if gradient_direction is not None
            else None
        ),
        step_magnitude=float(step_magnitude) if step_magnitude is not None else None,
        valid=bool(valid),
        invalid_reasons=list(invalid_reasons or ([] if valid else ["invalid_design_space"])),
        normalized_distance=ctx.normalization.distance(x0, x),
    )
    if include_oracle:
        attach_oracle(record)
    return record


def attach_oracle(step: StepRecord) -> None:
    """Attach exact formula oracle fields to a step in-place."""
    result = oracle_check(step.params)
    step.oracle_label = result.get("label", 0)
    step.oracle_reason = result.get("formula_reason", "unknown")


def attach_oracle_to_steps(steps: list[StepRecord]) -> None:
    for step in steps:
        attach_oracle(step)


def select_nearest_success(steps: list[StepRecord], tau: float) -> StepRecord | None:
    """Smallest-distance valid step with surrogate probability >= tau."""
    successes = [s for s in steps if s.valid and s.probability >= tau]
    if not successes:
        return None
    return min(successes, key=lambda s: (s.normalized_distance, s.step_idx))


def select_best_fallback(steps: list[StepRecord]) -> tuple[StepRecord | None, str]:
    """Fallback for algorithms that did not cross tau."""
    valid_steps = [s for s in steps if s.valid]
    if valid_steps:
        return max(valid_steps, key=lambda s: (s.probability, s.normalized_distance)), "no_surrogate_crossing"
    return None, "no_valid_step"


def finalize_result(
    *,
    start: BenchmarkStart,
    ctx: OptimizationContext,
    optimizer_id: str,
    steps: list[StepRecord],
    selected: StepRecord | None,
    status: str,
    stop_reason: str,
    start_probability: float | None = None,
) -> OptimizationResult:
    """Mark selection, attach oracle data, and return an OptimizationResult."""
    x0 = start.params.copy()
    if start_probability is None:
        start_probability = float(ctx.model.predict_proba(x0.reshape(1, -1))[0])

    for step in steps:
        step.is_selected = False
        step.is_terminated = False
    if selected is None:
        selected = evaluate_candidate(
            step_idx=len(steps),
            x0=x0,
            x=x0,
            ctx=ctx,
            gradient_direction=None,
            step_magnitude=0.0,
        )
        steps.append(selected)

    if selected is not None:
        selected.is_selected = True
        selected.is_terminated = True
        selected.stop_reason = stop_reason
        final_params = selected.params.copy()
        final_probability = float(selected.probability)
        valid_final = bool(selected.valid)
        final_distance = float(selected.normalized_distance)
    attach_oracle_to_steps(steps)
    oracle = oracle_check(final_params)

    return OptimizationResult(
        start_id=start.start_id,
        optimizer_id=optimizer_id,
        model_id=ctx.model_id,
        status=status,
        start_params=x0,
        final_params=final_params,
        start_probability=float(start_probability),
        final_probability=final_probability,
        threshold=float(ctx.tau),
        valid_final=valid_final,
        oracle_label=oracle.get("label", 0),
        oracle_reason=oracle.get("formula_reason", "unknown"),
        normalized_distance=final_distance,
        n_steps=len(steps),
        n_evaluations=len(steps),
        stop_reason=stop_reason,
        steps=steps,
    )


def no_gradient_result(
    *,
    start: BenchmarkStart,
    ctx: OptimizationContext,
    optimizer_id: str,
    probability: float,
) -> OptimizationResult:
    """Return a schema-valid no-gradient result with the start selected."""
    step = evaluate_candidate(
        step_idx=0,
        x0=start.params,
        x=start.params,
        ctx=ctx,
        gradient_direction=None,
        step_magnitude=0.0,
        include_oracle=True,
    )
    return finalize_result(
        start=start,
        ctx=ctx,
        optimizer_id=optimizer_id,
        steps=[step],
        selected=step,
        status="no_gradient",
        stop_reason="zero_gradient",
        start_probability=probability,
    )
