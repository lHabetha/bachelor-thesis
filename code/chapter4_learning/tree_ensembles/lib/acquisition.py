"""Chapter 4 active-learning acquisition helpers.

Chapter 4 label-blind rows use only model probabilities, model uncertainty, and
raw parameter-space diversity. The legacy hybrid helpers are retained for
provenance only because they consume formula-derived oracle fields.
"""
from __future__ import annotations

import numpy as np


def minmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi <= lo + 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def hybrid_score(
    p_mean: np.ndarray,
    p_uncertainty: np.ndarray,
    formula_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
) -> np.ndarray:
    """Deprecated formula-informed score; do not use for label-scarcity claims."""
    return (
        0.30 * minmax(-np.abs(p_mean - 0.5))
        + 0.20 * minmax(p_uncertainty)
        + 0.25 * minmax(-formula_margin)
        + 0.15 * is_blocked.astype(np.float64)
        + 0.10 * is_inward_movement.astype(np.float64)
    )


def uncertainty_disagreement_score(
    p_mean: np.ndarray,
    p_uncertainty: np.ndarray,
    *,
    uncertainty_weight: float = 0.60,
    disagreement_weight: float = 0.40,
) -> np.ndarray:
    """Label-blind score using model uncertainty and model disagreement."""
    return (
        uncertainty_weight * minmax(-np.abs(p_mean - 0.5))
        + disagreement_weight * minmax(p_uncertainty)
    )


def select_hybrid(
    X: np.ndarray,
    p_mean: np.ndarray,
    p_uncertainty: np.ndarray,
    formula_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
    k: int,
) -> list[int]:
    scores = hybrid_score(p_mean, p_uncertainty, formula_margin, is_blocked, is_inward_movement)
    return np.argsort(-scores)[:k].astype(int).tolist()


def select_uncertainty_disagreement(
    X: np.ndarray,
    p_mean: np.ndarray,
    p_uncertainty: np.ndarray,
    k: int,
) -> list[int]:
    del X
    scores = uncertainty_disagreement_score(p_mean, p_uncertainty)
    return np.argsort(-scores)[:k].astype(int).tolist()


def greedy_diverse_select(
    X: np.ndarray,
    score: np.ndarray,
    k: int,
    *,
    preselect: int = 3000,
    score_weight: float = 0.70,
    diversity_weight: float = 0.30,
) -> list[int]:
    if k <= 0:
        return []
    n = len(score)
    top_n = min(n, max(preselect, k * 20))
    top = np.argsort(-score)[:top_n]
    x_top = X[top].astype(np.float64)
    mu = x_top.mean(axis=0)
    sd = x_top.std(axis=0)
    sd[sd < 1e-9] = 1.0
    xz = (x_top - mu) / sd
    score_z = minmax(score[top])
    first = int(np.argmax(score_z))
    selected_local = [first]
    used = np.zeros(top_n, dtype=bool)
    used[first] = True
    min_dist = np.linalg.norm(xz - xz[first], axis=1)
    while len(selected_local) < k:
        combined = score_weight * score_z + diversity_weight * minmax(min_dist)
        combined[used] = -np.inf
        j = int(np.argmax(combined))
        if not np.isfinite(combined[j]):
            break
        selected_local.append(j)
        used[j] = True
        min_dist = np.minimum(min_dist, np.linalg.norm(xz - xz[j], axis=1))
    return [int(top[j]) for j in selected_local]


def select_diverse_uncertainty(X: np.ndarray, p_mean: np.ndarray, k: int) -> list[int]:
    return greedy_diverse_select(X, -np.abs(p_mean - 0.5), k)


def select_disjoint_hybrid_diverse(
    X: np.ndarray,
    p_mean: np.ndarray,
    p_uncertainty: np.ndarray,
    formula_margin: np.ndarray,
    is_blocked: np.ndarray,
    is_inward_movement: np.ndarray,
    *,
    n_hybrid: int,
    n_diverse: int,
) -> tuple[list[int], list[int]]:
    hybrid_idx = select_hybrid(
        X, p_mean, p_uncertainty, formula_margin, is_blocked, is_inward_movement, n_hybrid
    )
    mask = np.ones(len(X), dtype=bool)
    mask[hybrid_idx] = False
    rem_idx = np.where(mask)[0]
    diverse_local = select_diverse_uncertainty(X[rem_idx], p_mean[rem_idx], n_diverse)
    diverse_idx = [int(rem_idx[i]) for i in diverse_local]
    return hybrid_idx, diverse_idx
