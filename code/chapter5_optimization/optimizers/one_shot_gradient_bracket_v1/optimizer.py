"""One-shot gradient bracket line-search optimizer.

Algorithm:
1. Compute one gradient at the blocked starting point.
2. Normalize the gradient direction.
3. Test 16 step magnitudes: 8 orders of magnitude (10^3 down to 10^-4),
   with multipliers 1.0 and 0.5 at each order → 16 steps total.
4. Find the bracket: adjacent pair where smaller step fails tau, larger succeeds.
5. Evaluate 20 equally-spaced steps between the bracket bounds.
6. Select the smallest valid step whose surrogate probability >= tau.
7. If no bracket found (none of 16 reach tau), use the largest valid step.

The optimizer NEVER uses the oracle to decide the selected step.
Oracle is evaluated afterward for reporting only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys


from shared.interface import (
    BaseOptimizer,
    BenchmarkStart,
    OptimizationContext,
    OptimizationResult,
    StepRecord,
)
from shared.model_utils import (
    compute_gradient,
    FEATURE_NAMES,
)
from shared.oracle import (
    is_valid,
    oracle_check,
)
from shared.optimizer_utils import (
    constrained_direction_norm,
    project_locked_params,
)

OPTIMIZER_ID = "one_shot_gradient_bracket_v1"

ORDERS_OF_MAGNITUDE = [3, 2, 1, 0, -1, -2, -3, -4]
MULTIPLIERS = [1.0, 0.5]
N_BRACKET_STEPS = 20


@dataclass
class OneShotGradientBracket:
    optimizer_id: str = OPTIMIZER_ID

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        x0 = start.params.copy()
        tau = ctx.tau
        model = ctx.model
        stds = ctx.normalization.stds

        grad_raw, prob_start = compute_gradient(model, x0)
        grad_norm = np.linalg.norm(grad_raw)
        if grad_norm < 1e-12:
            return self._no_gradient_result(start, ctx, prob_start)

        # Transform gradient to normalized space: dP/dx_norm_i = dP/dx_i * std_i
        # (chain rule: x_i = x_norm_i * std_i + median_i, so dx_i/dx_norm_i = std_i)
        grad_in_norm_space = grad_raw * stds
        norm_space_len = np.linalg.norm(grad_in_norm_space)
        if norm_space_len < 1e-12:
            return self._no_gradient_result(start, ctx, prob_start)
        direction_norm = constrained_direction_norm(ctx, grad_in_norm_space / norm_space_len)
        if float(np.linalg.norm(direction_norm)) < 1e-12:
            return self._no_gradient_result(start, ctx, prob_start)

        # Step in normalized space → raw space: Δx = mag * direction_norm * stds
        # This ensures: normalized_distance = mag * ||direction_norm|| = mag
        direction_raw = direction_norm * stds

        step_magnitudes = []
        for order in ORDERS_OF_MAGNITUDE:
            for mult in MULTIPLIERS:
                step_magnitudes.append(mult * (10.0 ** order))
        step_magnitudes.sort(reverse=True)

        coarse_steps: list[StepRecord] = []

        for idx, mag in enumerate(step_magnitudes):
            x_candidate = project_locked_params(ctx, x0, x0 + mag * direction_raw)
            valid = is_valid(x_candidate)
            if valid:
                prob = float(model.predict_proba(x_candidate.reshape(1, -1))[0])
            else:
                prob = prob_start

            dist = ctx.normalization.distance(x0, x_candidate)
            step_rec = StepRecord(
                step_idx=idx,
                params=x_candidate.copy(),
                probability=prob,
                gradient_direction=direction_norm.copy(),
                step_magnitude=mag,
                valid=valid,
                normalized_distance=dist,
            )
            coarse_steps.append(step_rec)

        # Find the tightest bracket: smallest adjacent pair where the larger
        # magnitude has P >= tau (valid) and the smaller magnitude has P < tau.
        # Steps are sorted descending, so we scan from small to large index
        # (large to small magnitude) looking for the LAST crossing.
        bracket_lo = None
        bracket_hi = None
        for idx in range(len(coarse_steps) - 1, 0, -1):
            curr = coarse_steps[idx]  # smaller magnitude
            prev = coarse_steps[idx - 1]  # larger magnitude
            if prev.valid and prev.probability >= tau and curr.probability < tau:
                bracket_lo = step_magnitudes[idx]
                bracket_hi = step_magnitudes[idx - 1]
                break
        # Edge case: smallest step already crosses tau
        if bracket_lo is None and coarse_steps[-1].valid and coarse_steps[-1].probability >= tau:
            bracket_hi = step_magnitudes[-1]
            bracket_lo = step_magnitudes[-1] / 2.0

        all_steps = list(coarse_steps)
        n_evaluations = len(step_magnitudes)

        if bracket_lo is not None and bracket_hi is not None:
            fine_mags = np.linspace(bracket_lo, bracket_hi, N_BRACKET_STEPS + 2)[1:-1]
            fine_steps: list[StepRecord] = []
            for fidx, fmag in enumerate(fine_mags):
                x_candidate = project_locked_params(ctx, x0, x0 + fmag * direction_raw)
                valid = is_valid(x_candidate)
                if valid:
                    prob = float(model.predict_proba(x_candidate.reshape(1, -1))[0])
                else:
                    prob = prob_start

                dist = ctx.normalization.distance(x0, x_candidate)
                fine_steps.append(StepRecord(
                    step_idx=len(all_steps) + fidx,
                    params=x_candidate.copy(),
                    probability=prob,
                    gradient_direction=direction_norm.copy(),
                    step_magnitude=float(fmag),
                    valid=valid,
                    normalized_distance=dist,
                ))

            all_steps.extend(fine_steps)
            n_evaluations += len(fine_mags)

            successful = [s for s in all_steps if s.valid and s.probability >= tau]
            if successful:
                selected = min(successful, key=lambda s: s.step_magnitude)
            else:
                selected = None
        else:
            successful_coarse = [s for s in coarse_steps if s.valid and s.probability >= tau]
            if successful_coarse:
                selected = min(successful_coarse, key=lambda s: s.step_magnitude)
            else:
                selected = None

        if selected is None:
            valid_steps = [s for s in all_steps if s.valid]
            if valid_steps:
                selected = max(valid_steps, key=lambda s: s.step_magnitude)
                status = "no_surrogate_crossing"
                stop_reason = "no_surrogate_crossing"
            else:
                selected = StepRecord(
                    step_idx=len(all_steps),
                    params=x0.copy(),
                    probability=prob_start,
                    gradient_direction=None,
                    step_magnitude=0.0,
                    valid=True,
                    normalized_distance=0.0,
                )
                all_steps.append(selected)
                n_evaluations += 1
                status = "no_valid_step"
                stop_reason = "no_valid_step"
        else:
            status = "surrogate_success"
            stop_reason = "tau_crossed"

        if selected:
            selected.is_selected = True
            selected.is_terminated = True
            selected.stop_reason = stop_reason
            final_params = selected.params
            final_prob = selected.probability
            valid_final = selected.valid
            final_dist = selected.normalized_distance
        else:
            final_params = x0
            final_prob = prob_start
            valid_final = True
            final_dist = 0.0

        oracle_result = oracle_check(final_params)

        for s in all_steps:
            s_oracle = oracle_check(s.params)
            s.oracle_label = s_oracle.get("label", 0)
            s.oracle_reason = s_oracle.get("formula_reason", "unknown")

        return OptimizationResult(
            start_id=start.start_id,
            optimizer_id=self.optimizer_id,
            model_id=ctx.model_id,
            status=status,
            start_params=x0,
            final_params=final_params,
            start_probability=prob_start,
            final_probability=final_prob,
            threshold=tau,
            valid_final=valid_final,
            oracle_label=oracle_result.get("label", 0),
            oracle_reason=oracle_result.get("formula_reason", "unknown"),
            normalized_distance=final_dist,
            n_steps=len(all_steps),
            n_evaluations=n_evaluations,
            stop_reason=stop_reason,
            steps=all_steps,
        )

    def _no_gradient_result(self, start: BenchmarkStart, ctx: OptimizationContext, prob: float) -> OptimizationResult:
        x0 = start.params.copy()
        oracle_result = oracle_check(x0)
        step = StepRecord(
            step_idx=0,
            params=x0.copy(),
            probability=prob,
            gradient_direction=None,
            step_magnitude=0.0,
            valid=is_valid(x0),
            oracle_label=oracle_result.get("label", 0),
            oracle_reason=oracle_result.get("formula_reason", "unknown"),
            normalized_distance=0.0,
            is_selected=True,
            is_terminated=True,
            stop_reason="zero_gradient",
        )
        return OptimizationResult(
            start_id=start.start_id,
            optimizer_id=self.optimizer_id,
            model_id=ctx.model_id,
            status="no_gradient",
            start_params=x0,
            final_params=x0,
            start_probability=prob,
            final_probability=prob,
            threshold=ctx.tau,
            valid_final=step.valid,
            oracle_label=oracle_result.get("label", 0),
            oracle_reason=oracle_result.get("formula_reason", "unknown"),
            normalized_distance=0.0,
            n_steps=1,
            n_evaluations=1,
            stop_reason="zero_gradient",
            steps=[step],
        )
