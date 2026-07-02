"""Design-space validity and oracle utilities for Chapter 5 optimization."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .paths import ensure_chapter3_importable

ensure_chapter3_importable()

from chapter3_clevis_setup.design_space import DummyParams, validate_params  # noqa: E402
from chapter3_clevis_setup.exact_assemblability import (  # noqa: E402
    evaluate_exact_assemblability,
)

from .model_utils import FEATURE_NAMES, N_FEATURES

__all__ = [
    "DummyParams",
    "validate_params",
    "is_valid",
    "array_to_params",
    "params_to_array",
    "oracle_check",
    "FEATURE_NAMES",
    "N_FEATURES",
]


def is_valid(x: np.ndarray) -> bool:
    try:
        kwargs = {name: float(x[i]) for i, name in enumerate(FEATURE_NAMES)}
        kwargs["exploded_gap"] = 30.0
        p = DummyParams(**kwargs)
        ok, _ = validate_params(p)
        return ok
    except (TypeError, ValueError):
        return False


def array_to_params(x: np.ndarray) -> DummyParams | None:
    try:
        kwargs = {name: float(x[i]) for i, name in enumerate(FEATURE_NAMES)}
        kwargs["exploded_gap"] = 30.0
        p = DummyParams(**kwargs)
        ok, _ = validate_params(p)
        return p if ok else None
    except (TypeError, ValueError):
        return None


def params_to_array(p: DummyParams) -> np.ndarray:
    return np.array([float(getattr(p, n)) for n in FEATURE_NAMES], dtype=np.float64)


def oracle_check(x: np.ndarray) -> dict:
    p = array_to_params(x)
    if p is None:
        return {"validity_ok": False, "label": 0, "formula_reason": "overlap"}

    result = evaluate_exact_assemblability(p)
    terms = asdict(result.terms)
    out = {
        "validity_ok": bool(result.validity_ok),
        "label": int(result.assemblable),
        "label_class": result.label_class,
        "label_subclass": result.label_subclass,
        "formula_reason": result.label_reason,
    }
    out.update(terms)
    return out
