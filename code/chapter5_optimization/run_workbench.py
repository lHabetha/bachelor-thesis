"""Chapter 5 workbench runner: execute an optimizer over the active benchmark.

Handles loading benchmark/model, running the optimizer, writing schema-v1
trajectories, 50-frame viewer exports, statistics JSON/MD, and run manifests.

Usage:
    python -m run_workbench \
        --optimizer one_shot_gradient_bracket_v1 --tau 0.60
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.paths import (
    BENCHMARK_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
)
from shared.model_utils import (
    load_model_artifact,
    FEATURE_NAMES,
)
from shared.interface import (
    BenchmarkStart,
    NormalizationConfig,
    OptimizationContext,
    OptimizationResult,
    StepRecord,
)
from shared.oracle import oracle_check, is_valid
from shared.parameter_locks import (
    LockSpec,
    resolve_lock_spec,
)

N_VIEWER_FRAMES = 50
_PARALLEL_OPTIMIZER = None
_PARALLEL_CTX = None


def _init_parallel_worker(
    optimizer_id: str,
    model_dir_str: str,
    model_id: str,
    normalization: NormalizationConfig,
    tau: float,
) -> None:
    global _PARALLEL_OPTIMIZER, _PARALLEL_CTX
    model = load_model_artifact(Path(model_dir_str))
    _PARALLEL_OPTIMIZER = load_optimizer(optimizer_id)
    _PARALLEL_CTX = OptimizationContext(
        starts=[],
        model=model,
        model_id=model_id,
        normalization=normalization,
        tau=tau,
    )


def _run_parallel_start(args: tuple[int, BenchmarkStart]) -> tuple[int, OptimizationResult]:
    idx, start = args
    if _PARALLEL_OPTIMIZER is None or _PARALLEL_CTX is None:
        raise RuntimeError("Parallel worker not initialized")
    result = _PARALLEL_OPTIMIZER.optimize(start, _PARALLEL_CTX)
    return idx, result


def load_benchmark() -> tuple[list[BenchmarkStart], NormalizationConfig]:
    """Load the active blocked benchmark set and normalization."""
    from shared.paths import BENCHMARK_ID, BENCHMARK_JSONL, BENCHMARK_NORMALIZATION

    jsonl_path = BENCHMARK_JSONL
    norm_path = BENCHMARK_NORMALIZATION

    assert jsonl_path.exists(), f"Benchmark not found: {jsonl_path}"
    assert norm_path.exists(), f"Normalization not found: {norm_path}"

    with open(norm_path) as f:
        norm_data = json.load(f)
    normalization = NormalizationConfig(
        medians=np.array(norm_data["medians"]),
        stds=np.array(norm_data["stds"]),
        parameter_order=norm_data["parameter_order"],
    )

    starts = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            params = np.array([rec[n] for n in FEATURE_NAMES], dtype=np.float64)
            starts.append(BenchmarkStart(
                start_id=rec["start_id"],
                param_id=rec["param_id"],
                params=params,
                subgroup=rec["subgroup"],
                formula_reason=rec["formula_reason"],
                margins=rec.get("margins", {}),
                blocked_explanation=rec.get("blocked_explanation", ""),
            ))
    return starts, normalization


def load_optimizer(optimizer_id: str):
    """Dynamically load an optimizer by ID."""
    if optimizer_id == "one_shot_gradient_bracket_v1":
        from optimizers.one_shot_gradient_bracket_v1.optimizer import (
            OneShotGradientBracket,
        )
        return OneShotGradientBracket()
    if optimizer_id == "one_shot_gradient_shrink_v1":
        from optimizers.one_shot_gradient_shrink_v1.optimizer import (
            OneShotGradientShrink,
        )
        return OneShotGradientShrink()
    if optimizer_id == "one_shot_gradient_bracket_extreme_v1":
        from optimizers.one_shot_gradient_bracket_extreme_v1.optimizer import (
            OneShotGradientBracketExtreme,
        )
        return OneShotGradientBracketExtreme()
    if optimizer_id == "receding_gradient_multiscale_v1":
        from optimizers.receding_gradient_multiscale_v1.optimizer import (
            RecedingGradientMultiscale,
        )
        return RecedingGradientMultiscale()
    if optimizer_id == "random_sphere_expanding_v1":
        from optimizers.random_sphere_expanding_v1.optimizer import (
            RandomSphereExpanding,
        )
        return RandomSphereExpanding()
    random_sphere_expanding_samples = {
        "random_sphere_expanding_32_v1": 32,
        "random_sphere_expanding_128_v1": 128,
        "random_sphere_expanding_256_v1": 256,
    }
    if optimizer_id in random_sphere_expanding_samples:
        from optimizers.random_sphere_expanding_v1.optimizer import (
            RandomSphereExpanding,
        )
        return RandomSphereExpanding(
            optimizer_id=optimizer_id,
            samples_per_radius=random_sphere_expanding_samples[optimizer_id],
        )
    if optimizer_id == "random_sphere_coordinate_shrink_v1":
        from optimizers.random_sphere_coordinate_shrink_v1.optimizer import (
            RandomSphereCoordinateShrink,
        )
        return RandomSphereCoordinateShrink()
    if optimizer_id == "random_sphere_coordinate_shrink_128_sparse_l0_l2_v1":
        from optimizers.random_sphere_coordinate_shrink_128_sparse_l0_l2_v1.optimizer import (
            RandomSphereCoordinateShrink128SparseL0L2,
        )
        return RandomSphereCoordinateShrink128SparseL0L2()
    if optimizer_id == "sphere_shrink_128_mixed_l2_l0_40_60_demo_v1":
        from optimizers.sphere_shrink_128_mixed_l2_l0_40_60_demo_v1.optimizer import (
            SphereShrink128MixedL2L0Demo,
        )
        return SphereShrink128MixedL2L0Demo()
    random_sphere_shrink_samples = {
        "random_sphere_coordinate_shrink_32_v1": 32,
        "random_sphere_coordinate_shrink_64_matched_seed_v1": 64,
        "random_sphere_coordinate_shrink_128_v1": 128,
        "random_sphere_coordinate_shrink_256_v1": 256,
    }
    if optimizer_id in random_sphere_shrink_samples:
        from optimizers.random_sphere_coordinate_shrink_v1.optimizer import (
            RandomSphereCoordinateShrink,
        )
        return RandomSphereCoordinateShrink(
            optimizer_id=optimizer_id,
            samples_per_radius=random_sphere_shrink_samples[optimizer_id],
            base_seed=26001,
        )
    if optimizer_id == "coordinate_axis_bracket_v1":
        from optimizers.coordinate_axis_bracket_v1.optimizer import (
            CoordinateAxisBracket,
        )
        return CoordinateAxisBracket()
    if optimizer_id == "gradient_jitter_bracket_v1":
        from optimizers.gradient_jitter_bracket_v1.optimizer import (
            GradientJitterBracket,
        )
        return GradientJitterBracket()
    if optimizer_id == "trust_region_hybrid_v1":
        from optimizers.trust_region_hybrid_v1.optimizer import (
            TrustRegionHybrid,
        )
        return TrustRegionHybrid()
    if optimizer_id == "trust_region_mixed_l2_l0_40_60_demo_v1":
        from optimizers.trust_region_mixed_l2_l0_40_60_demo_v1.optimizer import (
            TrustRegionMixedL2L0Demo,
        )
        return TrustRegionMixedL2L0Demo()
    if optimizer_id == "trust_region_mixed_l2_l0_40_60_v1":
        from optimizers.trust_region_mixed_l2_l0_40_60_v1.optimizer import (
            TrustRegionMixedL2L0,
        )
        return TrustRegionMixedL2L0()
    if optimizer_id == "trust_region_no_gradient_v1":
        from optimizers.trust_region_no_gradient_v1.optimizer import (
            TrustRegionNoGradient,
        )
        return TrustRegionNoGradient()
    if optimizer_id == "momentum_receding_gradient_v1":
        from optimizers.momentum_receding_gradient_v1.optimizer import (
            MomentumRecedingGradient,
        )
        return MomentumRecedingGradient()
    if optimizer_id == "penalized_proximity_descent_v1":
        from optimizers.penalized_proximity_descent_v1.optimizer import (
            PenalizedProximityDescent,
        )
        return PenalizedProximityDescent()
    if optimizer_id == "penalized_proximity_descent_w0_20_v1":
        from optimizers.penalized_proximity_descent_w0_20_v1.optimizer import (
            PenalizedProximityDescentW020,
        )
        return PenalizedProximityDescentW020()
    if optimizer_id == "penalized_proximity_descent_w0_30_v1":
        from optimizers.penalized_proximity_descent_w0_30_v1.optimizer import (
            PenalizedProximityDescentW030,
        )
        return PenalizedProximityDescentW030()
    if optimizer_id == "penalized_proximity_descent_w0_50_v1":
        from optimizers.penalized_proximity_descent_w0_50_v1.optimizer import (
            PenalizedProximityDescentW050,
        )
        return PenalizedProximityDescentW050()
    if optimizer_id == "multifixed_penalty_direction_lam0_10_v1":
        from optimizers.multifixed_penalty_direction_lam0_10_v1.optimizer import (
            MultiFixedPenaltyDirectionLam010,
        )
        return MultiFixedPenaltyDirectionLam010()
    if optimizer_id == "multifixed_penalty_frontier_lam0_10_v1":
        from optimizers.multifixed_penalty_frontier_lam0_10_v1.optimizer import (
            MultiFixedPenaltyFrontierLam010,
        )
        return MultiFixedPenaltyFrontierLam010()
    if optimizer_id == "multifixed_penalty_direction_frontier_lam0_10_v1":
        from optimizers.multifixed_penalty_direction_frontier_lam0_10_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierLam010,
        )
        return MultiFixedPenaltyDirectionFrontierLam010()
    if optimizer_id == "multifixed_penalty_direction_frontier_lam0_20_v1":
        from optimizers.multifixed_penalty_direction_frontier_lam0_20_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierLam020,
        )
        return MultiFixedPenaltyDirectionFrontierLam020()
    if optimizer_id == "multifixed_penalty_direction_frontier_lam0_30_v1":
        from optimizers.multifixed_penalty_direction_frontier_lam0_30_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierLam030,
        )
        return MultiFixedPenaltyDirectionFrontierLam030()
    if optimizer_id == "multifixed_penalty_direction_frontier_lam0_50_v1":
        from optimizers.multifixed_penalty_direction_frontier_lam0_50_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierLam050,
        )
        return MultiFixedPenaltyDirectionFrontierLam050()
    if optimizer_id == "multifixed_penalty_direction_frontier_sparse_l1_l2_lam0_10_v1":
        from optimizers.multifixed_penalty_direction_frontier_sparse_l1_l2_lam0_10_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierSparseL1L2Lam010,
        )
        return MultiFixedPenaltyDirectionFrontierSparseL1L2Lam010()
    if optimizer_id == "multifixed_penalty_direction_frontier_sparse_l0_l2_lam0_10_v1":
        from optimizers.multifixed_penalty_direction_frontier_sparse_l0_l2_lam0_10_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierSparseL0L2Lam010,
        )
        return MultiFixedPenaltyDirectionFrontierSparseL0L2Lam010()
    if optimizer_id == "multifixed_penalty_direction_frontier_sparse_l0_l1_l2_lam0_10_v1":
        from optimizers.multifixed_penalty_direction_frontier_sparse_l0_l1_l2_lam0_10_v1.optimizer import (
            MultiFixedPenaltyDirectionFrontierSparseL0L1L2Lam010,
        )
        return MultiFixedPenaltyDirectionFrontierSparseL0L1L2Lam010()
    if optimizer_id == "cem_local_search_v1":
        from optimizers.cem_local_search_v1.optimizer import (
            CemLocalSearch,
        )
        return CemLocalSearch()
    if optimizer_id == "probability_guided_coordinate_v1":
        from optimizers.probability_guided_coordinate_v1.optimizer import (
            ProbabilityGuidedCoordinate,
        )
        return ProbabilityGuidedCoordinate()
    if optimizer_id == "adaptive_sphere_refine_v1":
        from optimizers.adaptive_sphere_refine_v1.optimizer import (
            AdaptiveSphereRefine,
        )
        return AdaptiveSphereRefine()
    if optimizer_id == "trust_region_sparse_l1_l2_l0_v1":
        from optimizers.trust_region_sparse_l1_l2_l0_v1.optimizer import TrustRegionSparseL1L2L0
        return TrustRegionSparseL1L2L0()
    if optimizer_id == "trust_region_sparse_l0_l2_v1":
        from optimizers.trust_region_sparse_l0_l2_v1.optimizer import TrustRegionSparseL0L2
        return TrustRegionSparseL0L2()
    if optimizer_id == "trust_region_sparse_l1_v1":
        from optimizers.trust_region_sparse_l1_v1.optimizer import TrustRegionSparseL1
        return TrustRegionSparseL1()
    if optimizer_id == "coordinate_axis_sparse_l1_l2_l0_v1":
        from optimizers.coordinate_axis_sparse_l1_l2_l0_v1.optimizer import CoordinateAxisSparseL1L2L0
        return CoordinateAxisSparseL1L2L0()
    if optimizer_id == "coordinate_axis_sparse_l0_l2_v1":
        from optimizers.coordinate_axis_sparse_l0_l2_v1.optimizer import CoordinateAxisSparseL0L2
        return CoordinateAxisSparseL0L2()
    if optimizer_id == "coordinate_axis_sparse_l1_v1":
        from optimizers.coordinate_axis_sparse_l1_v1.optimizer import CoordinateAxisSparseL1
        return CoordinateAxisSparseL1()
    if optimizer_id == "penalized_proximity_sparse_l1_l2_l0_v1":
        from optimizers.penalized_proximity_sparse_l1_l2_l0_v1.optimizer import PenalizedProximitySparseL1L2L0
        return PenalizedProximitySparseL1L2L0()
    if optimizer_id == "penalized_proximity_sparse_l0_l2_v1":
        from optimizers.penalized_proximity_sparse_l0_l2_v1.optimizer import PenalizedProximitySparseL0L2
        return PenalizedProximitySparseL0L2()
    if optimizer_id == "penalized_proximity_sparse_l1_v1":
        from optimizers.penalized_proximity_sparse_l1_v1.optimizer import PenalizedProximitySparseL1
        return PenalizedProximitySparseL1()
    if optimizer_id == "receding_multiscale_penalized_l2_v1":
        from optimizers.receding_multiscale_penalized_l2_v1.optimizer import RecedingMultiscalePenalizedL2
        return RecedingMultiscalePenalizedL2()
    if optimizer_id == "receding_multiscale_sparse_l0_l2_v1":
        from optimizers.receding_multiscale_sparse_l0_l2_v1.optimizer import RecedingMultiscaleSparseL0L2
        return RecedingMultiscaleSparseL0L2()
    if optimizer_id == "receding_multiscale_sparse_l0_l2_strong_v1":
        from optimizers.receding_multiscale_sparse_l0_l2_strong_v1.optimizer import RecedingMultiscaleSparseL0L2Strong
        return RecedingMultiscaleSparseL0L2Strong()
    if optimizer_id == "adaptive_multistep_gradient_v1":
        from optimizers.adaptive_multistep_gradient_v1.optimizer import AdaptiveMultistepGradient
        return AdaptiveMultistepGradient()
    if optimizer_id == "adaptive_multistep_penalty_lam0_05_v1":
        from optimizers.adaptive_multistep_penalty_lam0_05_v1.optimizer import AdaptiveMultistepPenaltyLam005
        return AdaptiveMultistepPenaltyLam005()
    if optimizer_id == "adaptive_multistep_penalty_lam0_10_v1":
        from optimizers.adaptive_multistep_penalty_lam0_10_v1.optimizer import AdaptiveMultistepPenaltyLam010
        return AdaptiveMultistepPenaltyLam010()
    if optimizer_id == "adaptive_multistep_penalty_mixed_l2_l0_40_60_demo_v1":
        from optimizers.adaptive_multistep_penalty_mixed_l2_l0_40_60_demo_v1.optimizer import AdaptivePenaltyMixedL2L0Demo
        return AdaptivePenaltyMixedL2L0Demo()
    if optimizer_id == "adaptive_multistep_penalty_lam0_20_v1":
        from optimizers.adaptive_multistep_penalty_lam0_20_v1.optimizer import AdaptiveMultistepPenaltyLam020
        return AdaptiveMultistepPenaltyLam020()
    if optimizer_id == "adaptive_multistep_penalty_lam0_30_v1":
        from optimizers.adaptive_multistep_penalty_lam0_30_v1.optimizer import AdaptiveMultistepPenaltyLam030
        return AdaptiveMultistepPenaltyLam030()
    if optimizer_id == "adaptive_multistep_penalty_lam0_50_v1":
        from optimizers.adaptive_multistep_penalty_lam0_50_v1.optimizer import AdaptiveMultistepPenaltyLam050
        return AdaptiveMultistepPenaltyLam050()
    if optimizer_id == "adaptive_multistep_penalty_sparse_l1_l2_v1":
        from optimizers.adaptive_multistep_penalty_sparse_l1_l2_v1.optimizer import AdaptiveMultistepPenaltySparseL1L2
        return AdaptiveMultistepPenaltySparseL1L2()
    if optimizer_id == "adaptive_multistep_penalty_sparse_l0_l2_v1":
        from optimizers.adaptive_multistep_penalty_sparse_l0_l2_v1.optimizer import AdaptiveMultistepPenaltySparseL0L2
        return AdaptiveMultistepPenaltySparseL0L2()
    if optimizer_id == "adaptive_multistep_penalty_sparse_l0_l1_l2_v1":
        from optimizers.adaptive_multistep_penalty_sparse_l0_l1_l2_v1.optimizer import AdaptiveMultistepPenaltySparseL0L1L2
        return AdaptiveMultistepPenaltySparseL0L1L2()
    if optimizer_id == "adaptive_multistep_penalty_process_l0_l2_k1_v1":
        from optimizers.adaptive_multistep_penalty_process_l0_l2_k1_v1.optimizer import AdaptivePenaltyProcessL0L2K1
        return AdaptivePenaltyProcessL0L2K1()
    if optimizer_id == "adaptive_multistep_penalty_process_l0_l2_k2_v1":
        from optimizers.adaptive_multistep_penalty_process_l0_l2_k2_v1.optimizer import AdaptivePenaltyProcessL0L2K2
        return AdaptivePenaltyProcessL0L2K2()
    if optimizer_id == "adaptive_multistep_penalty_process_l0_l2_k3_v1":
        from optimizers.adaptive_multistep_penalty_process_l0_l2_k3_v1.optimizer import AdaptivePenaltyProcessL0L2K3
        return AdaptivePenaltyProcessL0L2K3()
    raise ValueError(f"Unknown optimizer: {optimizer_id}")


def _result_to_trajectory_json(result: OptimizationResult, lock_spec: LockSpec | None = None) -> dict:
    """Convert an OptimizationResult to schema-v1 JSON."""
    steps_json = []
    for s in result.steps:
        steps_json.append({
            "step_idx": s.step_idx,
            "params": {n: float(s.params[i]) for i, n in enumerate(FEATURE_NAMES)},
            "probability": s.probability,
            "gradient_direction": s.gradient_direction.tolist() if s.gradient_direction is not None else None,
            "step_magnitude": s.step_magnitude,
            "valid": s.valid,
            "invalid_reasons": s.invalid_reasons,
            "oracle_label": s.oracle_label,
            "oracle_reason": s.oracle_reason,
            "normalized_distance": s.normalized_distance,
            "is_selected": s.is_selected,
            "is_terminated": s.is_terminated,
            "stop_reason": s.stop_reason,
        })

    trajectory = {
        "schema_version": "v1",
        "start_id": result.start_id,
        "optimizer_id": result.optimizer_id,
        "model_id": result.model_id,
        "status": result.status,
        "start_params": {n: float(result.start_params[i]) for i, n in enumerate(FEATURE_NAMES)},
        "final_params": {n: float(result.final_params[i]) for i, n in enumerate(FEATURE_NAMES)},
        "start_probability": result.start_probability,
        "final_probability": result.final_probability,
        "threshold": result.threshold,
        "valid_final": result.valid_final,
        "oracle_label": result.oracle_label,
        "oracle_reason": result.oracle_reason,
        "normalized_distance": result.normalized_distance,
        "n_steps": result.n_steps,
        "n_evaluations": result.n_evaluations,
        "stop_reason": result.stop_reason,
        "steps": steps_json,
    }
    if lock_spec is not None:
        trajectory["constraint"] = lock_spec.to_json()
    return trajectory


def _build_viewer_frames(result: OptimizationResult, ctx: OptimizationContext) -> list[dict]:
    """Build 50 viewer frames from true start to selected final design.

    The raw optimizer steps remain in ``trajectories.json``. The browser animation
    should show the design morph from the blocked start to the optimizer's chosen
    final candidate, not the coarse line-search probes ordered from huge invalid
    steps down to tiny valid ones.
    """
    if result.optimizer_id == "one_shot_gradient_shrink_v1":
        shrink_frames = _build_shrink_path_viewer_frames(result, ctx)
        if shrink_frames is not None:
            return shrink_frames

    selected = next((step for step in result.steps if step.is_selected), None)
    gradient_direction = (
        selected.gradient_direction.tolist()
        if selected and selected.gradient_direction is not None
        else None
    )

    frames = []
    for frame_idx in range(N_VIEWER_FRAMES):
        alpha = frame_idx / (N_VIEWER_FRAMES - 1)
        params = (1.0 - alpha) * result.start_params + alpha * result.final_params
        valid = is_valid(params)
        probability = float(ctx.model.predict_proba(params.reshape(1, -1))[0]) if valid else result.start_probability
        oracle = oracle_check(params)
        distance = ctx.normalization.distance(result.start_params, params)

        frames.append({
            "frame_idx": frame_idx,
            "params": {n: float(params[i]) for i, n in enumerate(FEATURE_NAMES)},
            "probability": probability,
            "gradient_direction": gradient_direction,
            "step_magnitude": distance,
            "normalized_distance": distance,
            "valid": valid,
            "oracle_label": oracle.get("label", 0),
            "oracle_reason": oracle.get("formula_reason", "unknown"),
            "terminated": frame_idx == N_VIEWER_FRAMES - 1,
            "termination_frame": N_VIEWER_FRAMES - 1 if frame_idx == N_VIEWER_FRAMES - 1 else None,
            "stop_reason": result.stop_reason if frame_idx == N_VIEWER_FRAMES - 1 else None,
        })

    return frames


def _build_shrink_path_viewer_frames(result: OptimizationResult, ctx: OptimizationContext) -> list[dict] | None:
    """Animate the accepted one-shot-plus-shrink path instead of a final morph."""
    selected = next((step for step in result.steps if step.is_selected), None)
    path_steps = [
        step for step in result.steps
        if step.stop_reason in {"one_shot_pre_shrink", "accepted_coordinate_shrink"}
    ]
    if selected is not None and not any(np.allclose(step.params, selected.params) for step in path_steps):
        path_steps.append(selected)
    if len(path_steps) < 2:
        return None

    keyframes: list[tuple[np.ndarray, np.ndarray | None, str | None]] = [
        (result.start_params.copy(), None, "start")
    ]
    for step in path_steps:
        if np.allclose(keyframes[-1][0], step.params):
            continue
        keyframes.append((
            step.params.copy(),
            step.gradient_direction.copy() if step.gradient_direction is not None else None,
            step.stop_reason,
        ))
    if len(keyframes) < 2:
        return None

    frames = []
    n_segments = len(keyframes) - 1
    for frame_idx in range(N_VIEWER_FRAMES):
        path_pos = (frame_idx / (N_VIEWER_FRAMES - 1)) * n_segments
        segment_idx = min(int(np.floor(path_pos)), n_segments - 1)
        alpha = path_pos - segment_idx
        params_a, _dir_a, _reason_a = keyframes[segment_idx]
        params_b, direction_b, reason_b = keyframes[segment_idx + 1]
        params = (1.0 - alpha) * params_a + alpha * params_b
        valid = is_valid(params)
        probability = float(ctx.model.predict_proba(params.reshape(1, -1))[0]) if valid else result.start_probability
        oracle = oracle_check(params)
        distance = ctx.normalization.distance(result.start_params, params)
        is_final = frame_idx == N_VIEWER_FRAMES - 1

        frames.append({
            "frame_idx": frame_idx,
            "params": {n: float(params[i]) for i, n in enumerate(FEATURE_NAMES)},
            "probability": probability,
            "gradient_direction": direction_b.tolist() if direction_b is not None else None,
            "step_magnitude": distance,
            "normalized_distance": distance,
            "valid": valid,
            "oracle_label": oracle.get("label", 0),
            "oracle_reason": oracle.get("formula_reason", "unknown"),
            "terminated": is_final,
            "termination_frame": N_VIEWER_FRAMES - 1 if is_final else None,
            "stop_reason": result.stop_reason if is_final else reason_b,
        })

    return frames


def _select_frame_indices(steps: list[StepRecord], selected_idx: int | None) -> list[int]:
    """Pick up to N_VIEWER_FRAMES indices from the steps list, evenly spaced."""
    n = len(steps)
    if n <= N_VIEWER_FRAMES:
        return list(range(n))

    if selected_idx is not None and selected_idx < n:
        end = selected_idx + 1
    else:
        end = n

    indices = np.linspace(0, end - 1, N_VIEWER_FRAMES, dtype=int).tolist()
    return sorted(set(indices))[:N_VIEWER_FRAMES]


def _trajectory_final_params(record: dict) -> np.ndarray:
    return np.array([float(record["final_params"][name]) for name in FEATURE_NAMES], dtype=np.float64)


def _load_baseline_records(run_id: str | None) -> dict[str, dict]:
    if not run_id:
        return {}
    path = RUNS_DIR / run_id / "trajectories.json"
    assert path.exists(), f"Baseline run not found: {path}"
    return {rec["start_id"]: rec for rec in json.loads(path.read_text())}


def compute_statistics(
    results: list[OptimizationResult],
    starts: list[BenchmarkStart],
    *,
    normalization: NormalizationConfig | None = None,
    constraints_by_start: dict[str, LockSpec] | None = None,
    baseline_by_start: dict[str, dict] | None = None,
) -> dict:
    """Compute aggregate and subgroup statistics."""
    start_map = {s.start_id: s for s in starts}

    total = len(results)
    surrogate_success = [r for r in results if r.status == "surrogate_success"]
    oracle_confirmed = [r for r in results if r.oracle_label == 1]
    false_success = [r for r in surrogate_success if r.oracle_label != 1]
    no_crossing = [r for r in results if r.status == "no_surrogate_crossing"]
    no_valid = [r for r in results if r.status == "no_valid_step"]

    dists_all = [r.normalized_distance for r in results]
    dists_oracle_ok = [r.normalized_distance for r in oracle_confirmed]
    dists_false = [r.normalized_distance for r in false_success]
    dists_stuck = [r.normalized_distance for r in results if r.status != "surrogate_success"]

    stats = {
        "total_starts": total,
        "surrogate_success_count": len(surrogate_success),
        "oracle_confirmed_count": len(oracle_confirmed),
        "false_success_count": len(false_success),
        "false_success_rate": len(false_success) / max(len(surrogate_success), 1),
        "no_crossing_count": len(no_crossing),
        "validity_failure_count": len(no_valid),
        "distances": {
            "all_mean": float(np.mean(dists_all)) if dists_all else 0.0,
            "all_median": float(np.median(dists_all)) if dists_all else 0.0,
            "oracle_confirmed_mean": float(np.mean(dists_oracle_ok)) if dists_oracle_ok else 0.0,
            "oracle_confirmed_median": float(np.median(dists_oracle_ok)) if dists_oracle_ok else 0.0,
            "false_success_mean": float(np.mean(dists_false)) if dists_false else 0.0,
            "stuck_mean": float(np.mean(dists_stuck)) if dists_stuck else 0.0,
        },
    }

    if normalization is not None:
        active_counts = []
        l1_distances = []
        moved_counts = {name: 0 for name in FEATURE_NAMES}
        for r in results:
            delta = (r.final_params - r.start_params) / normalization.stds
            abs_delta = np.abs(delta)
            active = abs_delta > 1e-4
            active_counts.append(int(np.count_nonzero(active)))
            l1_distances.append(float(np.sum(abs_delta)))
            for idx, is_active in enumerate(active):
                if bool(is_active):
                    moved_counts[FEATURE_NAMES[idx]] += 1
        stats["sparse_metrics"] = {
            "active_epsilon": 1e-4,
            "active_coordinate_count_mean": float(np.mean(active_counts)) if active_counts else 0.0,
            "active_coordinate_count_median": float(np.median(active_counts)) if active_counts else 0.0,
            "l1_distance_mean": float(np.mean(l1_distances)) if l1_distances else 0.0,
            "l1_distance_median": float(np.median(l1_distances)) if l1_distances else 0.0,
            "moved_parameter_counts": moved_counts,
        }

    subgroups: dict[str, list[OptimizationResult]] = {}
    for r in results:
        sg = start_map[r.start_id].subgroup
        subgroups.setdefault(sg, []).append(r)

    stats["subgroups"] = {}
    for sg, sg_results in sorted(subgroups.items()):
        sg_surr = [r for r in sg_results if r.status == "surrogate_success"]
        sg_oracle = [r for r in sg_results if r.oracle_label == 1]
        sg_dists = [r.normalized_distance for r in sg_results]
        stats["subgroups"][sg] = {
            "count": len(sg_results),
            "surrogate_success": len(sg_surr),
            "oracle_confirmed": len(sg_oracle),
            "mean_distance": float(np.mean(sg_dists)) if sg_dists else 0.0,
        }

    if constraints_by_start:
        locked_counts = [len(spec.locked_indices) for spec in constraints_by_start.values()]
        gradient_masses = [
            spec.locked_gradient_mass
            for spec in constraints_by_start.values()
            if spec.locked_gradient_mass is not None
        ]
        delta_masses = [
            spec.locked_delta_mass
            for spec in constraints_by_start.values()
            if spec.locked_delta_mass is not None
        ]
        parameter_counts = {name: 0 for name in FEATURE_NAMES}
        for spec in constraints_by_start.values():
            for name in spec.locked_names:
                parameter_counts[name] += 1
        constraint_id = next(iter(constraints_by_start.values())).constraint_id
        stats["parameter_locks"] = {
            "constraint_id": constraint_id,
            "scenario_kind": next(iter(constraints_by_start.values())).scenario_kind,
            "starts_with_locks": len(constraints_by_start),
            "locked_count_mean": float(np.mean(locked_counts)) if locked_counts else 0.0,
            "locked_gradient_mass_mean": float(np.mean(gradient_masses)) if gradient_masses else None,
            "locked_delta_mass_mean": float(np.mean(delta_masses)) if delta_masses else None,
            "parameter_lock_counts": parameter_counts,
        }

        if baseline_by_start:
            baseline_solved = 0
            constrained_recovered = 0
            baseline_oracle = 0
            constrained_oracle = 0
            paired_distance_deltas = []
            same_or_better_success_distance = 0
            for result in results:
                baseline = baseline_by_start.get(result.start_id)
                if baseline is None:
                    continue
                baseline_ok = int(baseline.get("oracle_label", 0)) == 1
                constrained_ok = result.oracle_label == 1
                baseline_oracle += int(baseline_ok)
                constrained_oracle += int(constrained_ok)
                baseline_solved += int(baseline_ok)
                constrained_recovered += int(baseline_ok and constrained_ok)
                baseline_dist = float(baseline.get("normalized_distance", 0.0))
                delta = float(result.normalized_distance - baseline_dist)
                paired_distance_deltas.append(delta)
                same_or_better_success_distance += int(
                    baseline_ok and constrained_ok and result.normalized_distance <= baseline_dist + 1e-12
                )
            stats["parameter_locks"]["paired_baseline"] = {
                "baseline_run_id": next(iter(constraints_by_start.values())).baseline_run_id,
                "baseline_oracle_confirmed": baseline_oracle,
                "constrained_oracle_confirmed": constrained_oracle,
                "oracle_success_drop": baseline_oracle - constrained_oracle,
                "recoverability_among_baseline_successes": constrained_recovered / max(baseline_solved, 1),
                "mean_distance_delta_vs_baseline": (
                    float(np.mean(paired_distance_deltas)) if paired_distance_deltas else 0.0
                ),
                "median_distance_delta_vs_baseline": (
                    float(np.median(paired_distance_deltas)) if paired_distance_deltas else 0.0
                ),
                "same_or_better_distance_success_count": same_or_better_success_distance,
            }

    return stats


def render_statistics_md(stats: dict, optimizer_id: str, model_id: str) -> str:
    """Render statistics as a Markdown page for the viewer."""
    lines = [
        f"# Optimization Statistics: {optimizer_id}",
        "",
        f"Model artifact: `{model_id}`",
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total starts | {stats['total_starts']} |",
        f"| Surrogate success | {stats['surrogate_success_count']} |",
        f"| Oracle confirmed | {stats['oracle_confirmed_count']} |",
        f"| False successes | {stats['false_success_count']} |",
        f"| False success rate | {stats['false_success_rate']:.1%} |",
        f"| No crossing (stuck) | {stats['no_crossing_count']} |",
        f"| Validity failures | {stats.get('validity_failure_count', 0)} |",
        "",
        "## Normalized Distances",
        "",
        "| Group | Mean | Median |",
        "|-------|------|--------|",
        f"| All {stats['total_starts']} starts | {stats['distances']['all_mean']:.4f} | {stats['distances']['all_median']:.4f} |",
        f"| Oracle confirmed | {stats['distances']['oracle_confirmed_mean']:.4f} | {stats['distances']['oracle_confirmed_median']:.4f} |",
        f"| False success | {stats['distances']['false_success_mean']:.4f} | — |",
        f"| Stuck / no crossing | {stats['distances']['stuck_mean']:.4f} | — |",
        "",
        "## By Blocked Subgroup",
        "",
        "| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |",
        "|----------|-------|---------------|-----------|-----------|",
    ]
    for sg, sg_data in sorted(stats.get("subgroups", {}).items()):
        lines.append(
            f"| {sg} | {sg_data['count']} | {sg_data['surrogate_success']} | "
            f"{sg_data['oracle_confirmed']} | {sg_data['mean_distance']:.4f} |"
        )
    sparse = stats.get("sparse_metrics")
    if sparse:
        lines.extend([
            "",
            "## Sparse Edit Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Active-coordinate epsilon | {sparse['active_epsilon']:.1e} |",
            f"| Mean active coordinates | {sparse['active_coordinate_count_mean']:.4f} |",
            f"| Median active coordinates | {sparse['active_coordinate_count_median']:.4f} |",
            f"| Mean L1 distance | {sparse['l1_distance_mean']:.4f} |",
            f"| Median L1 distance | {sparse['l1_distance_median']:.4f} |",
            "",
            "### Moved Parameter Frequency",
            "",
            "| Parameter | Active Finals |",
            "|-----------|---------------|",
        ])
        for name, count in sorted(
            sparse.get("moved_parameter_counts", {}).items(),
            key=lambda item: (-int(item[1]), item[0]),
        ):
            if count:
                lines.append(f"| `{name}` | {count} |")
    lock_stats = stats.get("parameter_locks")
    if lock_stats:
        lines.extend([
            "",
            "## Parameter Locks",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Constraint ID | `{lock_stats['constraint_id']}` |",
            f"| Scenario kind | `{lock_stats['scenario_kind']}` |",
            f"| Starts with locks | {lock_stats['starts_with_locks']} |",
            f"| Mean locked parameter count | {lock_stats['locked_count_mean']:.2f} |",
            f"| Mean locked gradient mass | {_fmt_optional(lock_stats.get('locked_gradient_mass_mean'))} |",
            f"| Mean locked delta mass | {_fmt_optional(lock_stats.get('locked_delta_mass_mean'))} |",
        ])
        paired = lock_stats.get("paired_baseline")
        if paired:
            lines.extend([
                f"| Paired baseline | `{paired.get('baseline_run_id')}` |",
                f"| Baseline oracle OK | {paired['baseline_oracle_confirmed']} |",
                f"| Constrained oracle OK | {paired['constrained_oracle_confirmed']} |",
                f"| Oracle success drop | {paired['oracle_success_drop']} |",
                f"| Recoverability | {paired['recoverability_among_baseline_successes']:.1%} |",
                f"| Mean distance delta | {paired['mean_distance_delta_vs_baseline']:.4f} |",
                f"| Median distance delta | {paired['median_distance_delta_vs_baseline']:.4f} |",
                f"| Same-or-better distance successes | {paired['same_or_better_distance_success_count']} |",
            ])
        lines.extend([
            "",
            "### Locked Parameter Frequency",
            "",
            "| Parameter | Locked Starts |",
            "|-----------|---------------|",
        ])
        for name, count in sorted(lock_stats.get("parameter_lock_counts", {}).items()):
            if count:
                lines.append(f"| `{name}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser(description="Chapter 5 workbench runner")
    parser.add_argument("--optimizer", required=True, help="Optimizer ID")
    parser.add_argument("--tau", type=float, default=0.60, help="Success threshold")
    parser.add_argument("--model-dir", type=str, default=None, help="Model artifact dir")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (auto-generated if not set)")
    parser.add_argument("--constraint-id", type=str, default=None, help="Parameter-lock scenario ID")
    parser.add_argument(
        "--locked-params",
        type=str,
        default="",
        help="Comma-separated manual locked parameter names; implies constraint-id=manual if set",
    )
    parser.add_argument("--baseline-run-id", type=str, default=None, help="Unconstrained baseline run for paired lock stats")
    parser.add_argument("--workers", type=int, default=1, help="Parallel start-level workers")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Override output runs directory (default: results/chapter5_optimization/runs/)",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help="Limit benchmark starts for smoke tests (default: all 200)",
    )
    args = parser.parse_args()

    starts, normalization = load_benchmark()
    if args.max_starts is not None and args.max_starts > 0:
        starts = starts[: args.max_starts]
    optimizer = load_optimizer(args.optimizer)

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else MODELS_DIR / "row1_uncertainty_disagreement_B1000_T2500_best"
    )
    model = load_model_artifact(model_dir)
    model_id = model_dir.name
    card_path = model_dir / "model_card.json"
    if card_path.exists():
        try:
            model_id = json.loads(card_path.read_text()).get("model_id", model_id)
        except json.JSONDecodeError:
            pass
    model_dir_resolved = model_dir.resolve()
    if model_dir_resolved.is_relative_to(PROJECT_ROOT.resolve()):
        model_dir_public = str(model_dir_resolved.relative_to(PROJECT_ROOT.resolve()))
    else:
        model_dir_public = model_dir_resolved.name

    manual_locked_names = tuple(name.strip() for name in args.locked_params.split(",") if name.strip())
    constraint_id = args.constraint_id
    if manual_locked_names and not constraint_id:
        constraint_id = "manual"
    baseline_by_start = _load_baseline_records(args.baseline_run_id)

    ctx = OptimizationContext(
        starts=starts,
        model=model,
        model_id=model_id,
        normalization=normalization,
        tau=args.tau,
        constraint_id=constraint_id,
        baseline_run_id=args.baseline_run_id,
    )

    constraint_suffix = f"_lock_{constraint_id}" if constraint_id else ""
    run_id = args.run_id or f"{args.optimizer}_tau{args.tau:.2f}{constraint_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_root = args.runs_dir.resolve() if args.runs_dir else RUNS_DIR
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_starts = len(starts)
    benchmark_id = BENCHMARK_DIR.name
    print(f"Running optimizer '{args.optimizer}' on {n_starts} starts (tau={args.tau})")
    print(f"Output: {run_dir}")

    t0 = time.time()
    results: list[OptimizationResult] = []
    trajectories = []
    viewer_data = []
    constraints_by_start: dict[str, LockSpec] = {}

    parallel_ok = (
        args.workers > 1
        and constraint_id is None
        and not manual_locked_names
        and not baseline_by_start
    )

    if parallel_ok:
        print(f"Using {args.workers} parallel workers")
        ordered_results: list[OptimizationResult | None] = [None] * len(starts)
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_ctx,
            initializer=_init_parallel_worker,
            initargs=(args.optimizer, str(model_dir), model_id, normalization, args.tau),
        ) as executor:
            futures = {
                executor.submit(_run_parallel_start, (i, start)): (i, start)
                for i, start in enumerate(starts)
            }
            completed = 0
            for future in as_completed(futures):
                i, start = futures[future]
                result = future.result()
                result_idx, opt_result = result
                ordered_results[result_idx] = opt_result
                completed += 1
                status_char = "✓" if opt_result.oracle_label == 1 else ("~" if opt_result.status == "surrogate_success" else "·")
                print(
                    f"  [{completed:3d}/{n_starts}] {start.start_id} {status_char} "
                    f"P={opt_result.final_probability:.3f} d={opt_result.normalized_distance:.4f} "
                    f"[{opt_result.stop_reason}]"
                )
        results = [r for r in ordered_results if r is not None]
    else:
        if args.workers > 1:
            print("Parallel workers requested, but constraints/baseline locks require sequential execution.")
        for i, start in enumerate(starts):
            baseline_record = baseline_by_start.get(start.start_id)
            lock_spec = resolve_lock_spec(
                constraint_id=constraint_id,
                start=start,
                model=model,
                normalization=normalization,
                manual_locked_names=manual_locked_names,
                baseline_record=baseline_record,
                baseline_run_id=args.baseline_run_id,
            )
            if lock_spec is not None:
                constraints_by_start[start.start_id] = lock_spec
                run_ctx = replace(
                    ctx,
                    locked_indices=lock_spec.locked_indices,
                    locked_names=lock_spec.locked_names,
                )
            else:
                run_ctx = ctx

            result = optimizer.optimize(start, run_ctx)
            results.append(result)

            status_char = "✓" if result.oracle_label == 1 else ("~" if result.status == "surrogate_success" else "·")
            print(f"  [{i+1:3d}/{n_starts}] {start.start_id} {status_char} P={result.final_probability:.3f} d={result.normalized_distance:.4f} [{result.stop_reason}]")

    lock_specs_by_start = constraints_by_start
    for result in results:
        start = next(s for s in starts if s.start_id == result.start_id)
        lock_spec = lock_specs_by_start.get(result.start_id)
        run_ctx = ctx
        if lock_spec is not None:
            run_ctx = replace(
                ctx,
                locked_indices=lock_spec.locked_indices,
                locked_names=lock_spec.locked_names,
            )

        traj_json = _result_to_trajectory_json(result, lock_spec)
        trajectories.append(traj_json)

        frames = _build_viewer_frames(result, run_ctx)
        viewer_record = {
            "start_id": start.start_id,
            "subgroup": start.subgroup,
            "status": result.status,
            "oracle_label": result.oracle_label,
            "oracle_reason": result.oracle_reason,
            "normalized_distance": result.normalized_distance,
            "start_probability": result.start_probability,
            "final_probability": result.final_probability,
            "threshold": result.threshold,
            "stop_reason": result.stop_reason,
            "blocked_explanation": start.blocked_explanation,
            "frames": frames,
        }
        if lock_spec is not None:
            viewer_record["constraint"] = lock_spec.to_json()
        viewer_data.append(viewer_record)

    wall_time = time.time() - t0

    (run_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=1))
    (run_dir / "viewer_data.json").write_text(json.dumps(viewer_data, indent=1))

    stats = compute_statistics(
        results,
        starts,
        normalization=normalization,
        constraints_by_start=constraints_by_start or None,
        baseline_by_start=baseline_by_start or None,
    )
    (run_dir / "statistics.json").write_text(json.dumps(stats, indent=2))

    stats_md = render_statistics_md(stats, args.optimizer, model_id)
    (run_dir / "statistics.md").write_text(stats_md)

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "optimizer_id": args.optimizer,
        "model_id": model_id,
        "model_dir": model_dir_public,
        "benchmark_set": benchmark_id,
        "tau": args.tau,
        "n_starts": len(starts),
        "constraint_id": constraint_id,
        "locked_params": list(manual_locked_names),
        "baseline_run_id": args.baseline_run_id,
        "wall_time_s": wall_time,
        "timestamp": datetime.now().isoformat(),
        "results_summary": {
            "surrogate_success": stats["surrogate_success_count"],
            "oracle_confirmed": stats["oracle_confirmed_count"],
            "false_success": stats["false_success_count"],
            "no_crossing": stats["no_crossing_count"],
            "mean_distance": stats["distances"]["all_mean"],
        },
    }
    if constraints_by_start:
        manifest["constraint"] = {
            "constraint_id": constraint_id,
            "scenario_kind": next(iter(constraints_by_start.values())).scenario_kind,
            "baseline_run_id": args.baseline_run_id,
            "manual_locked_params": list(manual_locked_names),
            "locked_names_vary_by_start": len({spec.locked_names for spec in constraints_by_start.values()}) > 1,
        }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    algo_md_src = Path(__file__).resolve().parent / "optimizers" / args.optimizer / "algorithm.md"
    if algo_md_src.exists():
        algorithm_md = algo_md_src.read_text()
        if args.model_dir:
            algorithm_md += (
                "\n\n## Run-Specific Model Note\n\n"
                f"This run used model artifact `{model_id}` from `{model_dir}` rather than the default "
                "Chapter 4 hard-label MLP. Optimizer mechanics are unchanged, but probabilities and "
                "gradients come from this alternate surrogate.\n"
            )
        (run_dir / "algorithm.md").write_text(algorithm_md)

    print(f"\nDone in {wall_time:.1f}s")
    print(f"  Surrogate success: {stats['surrogate_success_count']}/{n_starts}")
    print(f"  Oracle confirmed:  {stats['oracle_confirmed_count']}/{n_starts}")
    print(f"  False successes:   {stats['false_success_count']}")
    print(f"  Mean distance:     {stats['distances']['all_mean']:.4f}")
    print(f"\nArtifacts in {run_dir}")


if __name__ == "__main__":
    main()
