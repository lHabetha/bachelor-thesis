"""Optimizer plugin interface for the Chapter 5 workbench.

Every optimizer must implement the BaseOptimizer protocol and return
OptimizationResult objects. The workbench handles writing manifests,
trajectories, summaries, and viewer exports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .model_utils import FittedModel, FEATURE_NAMES, N_FEATURES
from .oracle import oracle_check, is_valid


@dataclass
class BenchmarkStart:
    """One canonical blocked starting assembly from the active benchmark."""
    start_id: str
    param_id: str
    params: np.ndarray  # shape (13,)
    subgroup: str
    formula_reason: str
    margins: dict
    blocked_explanation: str


@dataclass
class StepRecord:
    """One evaluation step during optimization."""
    step_idx: int
    params: np.ndarray
    probability: float
    gradient_direction: np.ndarray | None = None
    step_magnitude: float | None = None
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    oracle_label: int | None = None
    oracle_reason: str | None = None
    normalized_distance: float = 0.0
    is_selected: bool = False
    is_terminated: bool = False
    stop_reason: str | None = None


@dataclass
class OptimizationResult:
    """Output of one optimizer run on one blocked start."""
    start_id: str
    optimizer_id: str
    model_id: str
    status: str  # "surrogate_success", "no_surrogate_crossing", "no_valid_step", "no_gradient"
    start_params: np.ndarray
    final_params: np.ndarray
    start_probability: float
    final_probability: float
    threshold: float
    valid_final: bool
    oracle_label: int | None = None
    oracle_reason: str | None = None
    normalized_distance: float = 0.0
    n_steps: int = 0
    n_evaluations: int = 0
    stop_reason: str = ""
    steps: list[StepRecord] = field(default_factory=list)


@dataclass
class NormalizationConfig:
    """Median/std normalization over the active benchmark starts."""
    medians: np.ndarray
    stds: np.ndarray
    parameter_order: list[str]

    def distance(self, x0: np.ndarray, x1: np.ndarray) -> float:
        d = (x1 - x0) / self.stds
        return float(np.linalg.norm(d))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.medians) / self.stds


@dataclass
class OptimizationContext:
    """Shared context passed to every optimizer."""
    starts: list[BenchmarkStart]
    model: FittedModel
    model_id: str
    normalization: NormalizationConfig
    tau: float = 0.60
    constraint_id: str | None = None
    locked_indices: tuple[int, ...] = ()
    locked_names: tuple[str, ...] = ()
    baseline_run_id: str | None = None


class BaseOptimizer(Protocol):
    """Protocol that every optimizer must implement."""
    optimizer_id: str

    def optimize(self, start: BenchmarkStart, ctx: OptimizationContext) -> OptimizationResult:
        ...
