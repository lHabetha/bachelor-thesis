"""Acquisition strategies for Chapter 4.

Chapter 4 uses label-blind strategies only:
- `uncertainty_disagreement`: model uncertainty plus ensemble disagreement.
- `diverse_uncertainty`: uncertainty base + greedy diversity in param space.

The legacy `hybrid` helpers are kept for provenance only. They use
formula-derived fields and must not be used for label-scarcity claims.

Label taxonomy (Chapter 3):
  assemb-roof_clearance, assemb-splint_clearance, assemb-inward_movement,
  blocked, overlap.
"""
from __future__ import annotations

import numpy as np

from .formula_oracle import FEATURE_NAMES, REASON_BLOCKED, REASON_INWARD_MOVEMENT
from .model_mlp64 import Standardizer


def _minmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi <= lo + 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def score_hybrid(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    f_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
) -> np.ndarray:
    """Compute deprecated formula-informed hybrid acquisition score.

    Weights: 30% model uncertainty, 20% ensemble disagreement, 25% formula
    margin proximity, 15% blocked-class targeting, 10% inward-movement targeting.
    This score is not label-blind because it consumes formula-derived fields.
    """
    return (
        0.30 * _minmax(-np.abs(p_mean - 0.5))
        + 0.20 * _minmax(p_std)
        + 0.25 * _minmax(-f_margin)
        + 0.15 * is_blocked.astype(np.float64)
        + 0.10 * is_inward_movement.astype(np.float64)
    )


def score_uncertainty_disagreement(
    p_mean: np.ndarray,
    p_std: np.ndarray,
    *,
    uncertainty_weight: float = 0.60,
    disagreement_weight: float = 0.40,
) -> np.ndarray:
    """Label-blind score using only model uncertainty and ensemble disagreement."""
    return (
        uncertainty_weight * _minmax(-np.abs(p_mean - 0.5))
        + disagreement_weight * _minmax(p_std)
    )


def score_diverse_uncertainty(p_mean: np.ndarray) -> np.ndarray:
    """Base score for diverse_uncertainty: proximity to decision boundary."""
    return -np.abs(p_mean - 0.5)


def greedy_diverse_select(
    X: np.ndarray, score: np.ndarray, k: int, *, preselect: int = 3000
) -> list[int]:
    """Greedy max-min diversity selection from top-scoring candidates.

    Verbatim from Chapter 4 `_greedy_diverse_select`:
    - Take top `preselect` by score
    - Standardize within that subset
    - Iteratively pick argmax(0.70 * score_norm + 0.30 * min_dist_norm)
    """
    n = len(score)
    top_n = min(n, max(preselect, k * 20))
    top = np.argsort(-score)[:top_n]
    x_top = X[top]
    std = Standardizer.fit(x_top)
    xz = std.transform(x_top)
    score_z = _minmax(score[top])

    selected_local: list[int] = [int(np.argmax(score_z))]
    min_dist = np.linalg.norm(xz - xz[selected_local[0]], axis=1)
    used = np.zeros(top_n, dtype=bool)
    used[selected_local[0]] = True

    while len(selected_local) < k:
        div_z = _minmax(min_dist)
        combined = 0.70 * score_z + 0.30 * div_z
        combined[used] = -np.inf
        j = int(np.argmax(combined))
        if not np.isfinite(combined[j]):
            break
        selected_local.append(j)
        used[j] = True
        min_dist = np.minimum(min_dist, np.linalg.norm(xz - xz[j], axis=1))

    return [int(top[j]) for j in selected_local]


def select_hybrid(
    pool_X: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    f_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
    k: int,
) -> list[int]:
    """Select top-k by hybrid score (deterministic)."""
    scores = score_hybrid(p_mean, p_std, f_margin, is_blocked, is_inward_movement)
    return np.argsort(-scores)[:k].tolist()


def select_uncertainty_disagreement(
    pool_X: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    k: int,
) -> list[int]:
    """Select top-k by label-blind uncertainty/disagreement score."""
    del pool_X
    scores = score_uncertainty_disagreement(p_mean, p_std)
    return np.argsort(-scores)[:k].tolist()


def select_diverse_uncertainty(
    pool_X: np.ndarray,
    p_mean: np.ndarray,
    k: int,
) -> list[int]:
    """Select k candidates using diverse_uncertainty strategy."""
    base_score = score_diverse_uncertainty(p_mean)
    return greedy_diverse_select(pool_X, base_score, k)


def select_disjoint_round(
    pool_X: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    f_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
    n_hybrid: int = 125,
    n_diverse: int = 125,
) -> tuple[list[int], list[int]]:
    """Row 2 disjoint selection: hybrid first, then diverse from remainder.

    Returns (hybrid_indices, diverse_indices) into the pool arrays.
    Both index sets are guaranteed disjoint.
    """
    hybrid_idx = select_hybrid(
        pool_X, p_mean, p_std, f_margin, is_blocked, is_inward_movement, n_hybrid
    )
    hybrid_set = set(hybrid_idx)

    remaining_mask = np.ones(len(pool_X), dtype=bool)
    remaining_mask[list(hybrid_set)] = False
    remaining_global_idx = np.where(remaining_mask)[0]

    if len(remaining_global_idx) == 0:
        return hybrid_idx, []

    rem_X = pool_X[remaining_global_idx]
    rem_p_mean = p_mean[remaining_global_idx]

    local_diverse = select_diverse_uncertainty(rem_X, rem_p_mean, n_diverse)
    diverse_idx = [int(remaining_global_idx[i]) for i in local_diverse]

    return hybrid_idx, diverse_idx
