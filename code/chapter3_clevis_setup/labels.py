"""Minimal formula-label helper for the public Chapter 3 release."""

from __future__ import annotations

from typing import Any

from .design_space import DummyParams, params_to_dict
from .exact_assemblability import evaluate_exact_assemblability

FEATURE_NAMES: tuple[str, ...] = (
    "wall_thickness",
    "outer_span",
    "leg_length",
    "depth",
    "main_hole_offset_from_open_end",
    "main_hole_radius",
    "main_pin_length",
    "main_pin_radius",
    "cross_hole_radius",
    "cross_hole_distance_from_free_end",
    "splint_radius",
    "splint_length",
    "overhang_span_y",
)


def params_from_dict(data: dict[str, Any]) -> DummyParams:
    kwargs = {name: float(data[name]) for name in FEATURE_NAMES}
    kwargs["exploded_gap"] = float(data.get("exploded_gap", 30.0))
    return DummyParams(**kwargs)


def label_params(params: DummyParams) -> dict[str, Any]:
    """Return a flat record with validity and analytic label fields."""
    result = evaluate_exact_assemblability(params)
    record = params_to_dict(params)
    record.update(
        {
            "validity_ok": result.validity_ok,
            "validity_reasons": list(result.validity_reasons),
            "label_class": result.label_class,
            "label_subclass": result.label_subclass,
            "label_reason": result.label_reason,
            "assemblable": int(result.assemblable),
            "kinematic_assemblable": int(result.kinematic_assemblable),
            "terms": result.to_dict()["terms"],
        }
    )
    return record


def label_record_to_json(record: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serializable nested values if needed."""
    out = dict(record)
    if "terms" in out:
        out["terms"] = dict(out["terms"])
    return out
