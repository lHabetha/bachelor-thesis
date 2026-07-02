"""Parameter-lock scenario definitions for Chapter 5 constrained runs.

The lock study intentionally uses deterministic, per-start scenarios so
constrained runs can be compared pairwise against their unconstrained baselines.
No exact oracle labels are used to choose locked coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

import numpy as np

from .interface import BenchmarkStart, NormalizationConfig
from .model_utils import FEATURE_NAMES, FittedModel, compute_gradient


SEMANTIC_LOCKS: dict[str, tuple[str, ...]] = {
    "lock_pin": ("main_pin_length", "main_pin_radius"),
    "lock_pin_user": ("main_pin_length", "main_pin_radius", "main_hole_radius"),
    "lock_bracket": (
        "wall_thickness",
        "outer_span",
        "leg_length",
        "depth",
        "main_hole_offset_from_open_end",
        "main_hole_radius",
        "overhang_span_y",
    ),
    "lock_main_bracket_user": (
        "wall_thickness",
        "outer_span",
        "leg_length",
        "depth",
        "main_hole_offset_from_open_end",
        "overhang_span_y",
    ),
    "lock_splint": (
        "cross_hole_radius",
        "cross_hole_distance_from_free_end",
        "splint_radius",
        "splint_length",
    ),
    "lock_splint_user": (
        "cross_hole_radius",
        "splint_radius",
        "splint_length",
    ),
    "lock_roof_path": ("leg_length", "overhang_span_y", "main_pin_length"),
}

GRADIENT_SCENARIOS = {
    "grad_top_1",
    "grad_top_2",
    "grad_top_3",
    "grad_bottom_1",
    "grad_bottom_2",
    "grad_bottom_3",
    "grad_bottom_10",
    "grad_bottom_11",
    "grad_bottom_12",
    "grad_middle_2",
}

DELTA_SCENARIOS = {"delta_top_1", "delta_top_2", "delta_top_3"}

RANDOM_PATTERN = re.compile(r"^random_(?P<k>[123])_seed(?P<seed>[0-2])$")


@dataclass(frozen=True)
class LockSpec:
    """Resolved locked coordinates for one start in one constrained run."""

    constraint_id: str
    locked_indices: tuple[int, ...]
    locked_names: tuple[str, ...]
    scenario_kind: str
    locked_gradient_mass: float | None = None
    locked_delta_mass: float | None = None
    baseline_run_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "scenario_kind": self.scenario_kind,
            "locked_indices": list(self.locked_indices),
            "locked_names": list(self.locked_names),
            "locked_count": len(self.locked_indices),
            "locked_gradient_mass": self.locked_gradient_mass,
            "locked_delta_mass": self.locked_delta_mass,
            "baseline_run_id": self.baseline_run_id,
        }


def all_scenario_ids() -> list[str]:
    """Return the deterministic scenario IDs used by the lock study."""
    random_ids = [f"random_{k}_seed{seed}" for k in (1, 2, 3) for seed in range(3)]
    return [
        "grad_top_1",
        "grad_top_2",
        "grad_top_3",
        "grad_bottom_1",
        "grad_bottom_2",
        "grad_bottom_3",
        "grad_middle_2",
        "delta_top_1",
        "delta_top_2",
        "delta_top_3",
        *SEMANTIC_LOCKS.keys(),
        *random_ids,
    ]


def scenario_registry() -> list[dict[str, Any]]:
    """Machine-readable scenario registry for study artifacts."""
    rows = []
    for scenario_id in all_scenario_ids():
        rows.append({
            "constraint_id": scenario_id,
            "kind": scenario_kind(scenario_id),
            "description": scenario_description(scenario_id),
            "fixed_locked_names": list(SEMANTIC_LOCKS.get(scenario_id, ())),
            "requires_baseline": scenario_id in DELTA_SCENARIOS,
        })
    return rows


def scenario_kind(constraint_id: str) -> str:
    if constraint_id in GRADIENT_SCENARIOS:
        return "gradient_ranked"
    if constraint_id in DELTA_SCENARIOS:
        return "baseline_delta_ranked"
    if constraint_id in SEMANTIC_LOCKS:
        return "semantic_group"
    if RANDOM_PATTERN.match(constraint_id):
        return "random_control"
    if constraint_id == "manual":
        return "manual"
    raise ValueError(f"Unknown parameter-lock constraint_id: {constraint_id}")


def scenario_description(constraint_id: str) -> str:
    descriptions = {
        "grad_top_1": "Lock the largest absolute normalized MLP-gradient coordinate.",
        "grad_top_2": "Lock the two largest absolute normalized MLP-gradient coordinates.",
        "grad_top_3": "Lock the three largest absolute normalized MLP-gradient coordinates.",
        "grad_bottom_1": "Lock the smallest absolute normalized MLP-gradient coordinate.",
        "grad_bottom_2": "Lock the two smallest absolute normalized MLP-gradient coordinates.",
        "grad_bottom_3": "Lock the three smallest absolute normalized MLP-gradient coordinates.",
        "grad_bottom_10": "Lock the ten smallest absolute normalized MLP-gradient coordinates.",
        "grad_bottom_11": "Lock the eleven smallest absolute normalized MLP-gradient coordinates.",
        "grad_bottom_12": "Lock the twelve smallest absolute normalized MLP-gradient coordinates.",
        "grad_middle_2": "Lock the two middle-ranked absolute normalized MLP-gradient coordinates.",
        "delta_top_1": "Lock the coordinate with largest unconstrained normalized movement.",
        "delta_top_2": "Lock the two coordinates with largest unconstrained normalized movement.",
        "delta_top_3": "Lock the three coordinates with largest unconstrained normalized movement.",
        "lock_pin": "Lock main pin length and radius.",
        "lock_pin_user": "Lock main pin length, main pin radius, and main-hole radius.",
        "lock_bracket": "Lock bracket body and main-hole geometry.",
        "lock_main_bracket_user": "Lock main bracket and roof-path geometry except main-hole radius.",
        "lock_splint": "Lock cross-hole and splint geometry.",
        "lock_splint_user": "Lock splint radius, splint length, and cross-hole radius.",
        "lock_roof_path": "Lock leg length, overhang span, and main pin length.",
    }
    if constraint_id in descriptions:
        return descriptions[constraint_id]
    match = RANDOM_PATTERN.match(constraint_id)
    if match:
        return f"Lock {match.group('k')} random parameter(s), deterministic seed {match.group('seed')}."
    if constraint_id == "manual":
        return "Lock a manually supplied parameter list."
    raise ValueError(f"Unknown parameter-lock constraint_id: {constraint_id}")


def indices_from_names(names: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    name_to_idx = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    try:
        indices = sorted({name_to_idx[name] for name in names})
    except KeyError as exc:
        raise ValueError(f"Unknown parameter name in lock set: {exc.args[0]}") from exc
    return tuple(indices)


def names_from_indices(indices: tuple[int, ...] | list[int]) -> tuple[str, ...]:
    return tuple(FEATURE_NAMES[int(i)] for i in indices)


def locked_gradient_mass(
    *,
    model: FittedModel,
    params: np.ndarray,
    normalization: NormalizationConfig,
    locked_indices: tuple[int, ...],
) -> float:
    grad_raw, _prob = compute_gradient(model, params)
    grad_norm = np.abs(grad_raw * normalization.stds)
    total = float(np.sum(grad_norm))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(grad_norm[list(locked_indices)]) / total)


def locked_delta_mass(
    *,
    start_params: np.ndarray,
    final_params: np.ndarray,
    normalization: NormalizationConfig,
    locked_indices: tuple[int, ...],
) -> float:
    delta = np.abs((final_params - start_params) / normalization.stds)
    total = float(np.sum(delta))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(delta[list(locked_indices)]) / total)


def resolve_lock_spec(
    *,
    constraint_id: str | None,
    start: BenchmarkStart,
    model: FittedModel,
    normalization: NormalizationConfig,
    manual_locked_names: tuple[str, ...] = (),
    baseline_record: dict[str, Any] | None = None,
    baseline_run_id: str | None = None,
) -> LockSpec | None:
    """Resolve a scenario into the concrete lock set for one start."""
    if not constraint_id:
        return None

    params = start.params
    if constraint_id == "manual":
        indices = indices_from_names(manual_locked_names)
    elif constraint_id in SEMANTIC_LOCKS:
        indices = indices_from_names(SEMANTIC_LOCKS[constraint_id])
    elif constraint_id in GRADIENT_SCENARIOS:
        indices = _gradient_ranked_indices(
            constraint_id=constraint_id,
            model=model,
            params=params,
            normalization=normalization,
        )
    elif constraint_id in DELTA_SCENARIOS:
        if baseline_record is None:
            raise ValueError(f"{constraint_id} requires --baseline-run-id")
        indices = _delta_ranked_indices(
            constraint_id=constraint_id,
            start_params=params,
            baseline_record=baseline_record,
            normalization=normalization,
        )
    else:
        match = RANDOM_PATTERN.match(constraint_id)
        if not match:
            raise ValueError(f"Unknown parameter-lock constraint_id: {constraint_id}")
        indices = _random_indices(
            k=int(match.group("k")),
            seed=int(match.group("seed")),
            start_id=start.start_id,
        )

    baseline_final = _baseline_final_params(baseline_record) if baseline_record else None
    return LockSpec(
        constraint_id=constraint_id,
        locked_indices=indices,
        locked_names=names_from_indices(indices),
        scenario_kind=scenario_kind(constraint_id),
        locked_gradient_mass=locked_gradient_mass(
            model=model,
            params=params,
            normalization=normalization,
            locked_indices=indices,
        ),
        locked_delta_mass=(
            locked_delta_mass(
                start_params=params,
                final_params=baseline_final,
                normalization=normalization,
                locked_indices=indices,
            )
            if baseline_final is not None
            else None
        ),
        baseline_run_id=baseline_run_id,
    )


def _gradient_ranked_indices(
    *,
    constraint_id: str,
    model: FittedModel,
    params: np.ndarray,
    normalization: NormalizationConfig,
) -> tuple[int, ...]:
    grad_raw, _prob = compute_gradient(model, params)
    mass = np.abs(grad_raw * normalization.stds)
    order = np.argsort(mass)
    if constraint_id.startswith("grad_top_"):
        k = int(constraint_id.rsplit("_", 1)[1])
        selected = order[-k:]
    elif constraint_id.startswith("grad_bottom_"):
        k = int(constraint_id.rsplit("_", 1)[1])
        selected = order[:k]
    elif constraint_id == "grad_middle_2":
        mid = len(order) // 2
        selected = order[mid - 1: mid + 1]
    else:
        raise ValueError(f"Unsupported gradient scenario: {constraint_id}")
    return tuple(sorted(int(i) for i in selected))


def _delta_ranked_indices(
    *,
    constraint_id: str,
    start_params: np.ndarray,
    baseline_record: dict[str, Any],
    normalization: NormalizationConfig,
) -> tuple[int, ...]:
    final_params = _baseline_final_params(baseline_record)
    delta = np.abs((final_params - start_params) / normalization.stds)
    k = int(constraint_id.rsplit("_", 1)[1])
    selected = np.argsort(delta)[-k:]
    return tuple(sorted(int(i) for i in selected))


def _random_indices(*, k: int, seed: int, start_id: str) -> tuple[int, ...]:
    digest = hashlib.sha256(f"{start_id}:{seed}".encode("utf-8")).digest()
    stable_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(stable_seed)
    return tuple(sorted(int(i) for i in rng.choice(len(FEATURE_NAMES), size=k, replace=False)))


def _baseline_final_params(baseline_record: dict[str, Any]) -> np.ndarray:
    final = baseline_record.get("final_params", {})
    return np.array([float(final[name]) for name in FEATURE_NAMES], dtype=np.float64)
