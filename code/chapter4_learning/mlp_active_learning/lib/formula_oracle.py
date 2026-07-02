"""Formula oracle for Chapter 4 — wraps Chapter 3 exact assemblability labeling."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_CH4_ROOT = Path(__file__).resolve().parents[2]
if str(_CH4_ROOT) not in sys.path:
    sys.path.insert(0, str(_CH4_ROOT))

from release_paths import ensure_chapter3_importable  # noqa: E402

ensure_chapter3_importable()

from chapter3_clevis_setup.design_space import DummyParams  # noqa: E402
from chapter3_clevis_setup.exact_assemblability import (  # noqa: E402
    evaluate_exact_assemblability,
)

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
N_FEATURES = len(FEATURE_NAMES)

REASON_ROOF_CLEARANCE = "assemb-roof_clearance"
REASON_SPLINT_CLEARANCE = "assemb-splint_clearance"
REASON_INWARD_MOVEMENT = "assemb-inward_movement"
REASON_BLOCKED = "blocked"
REASON_OVERLAP = "overlap"

REASON_Q1 = REASON_ROOF_CLEARANCE
REASON_Q2 = REASON_SPLINT_CLEARANCE
REASON_Q2B = REASON_INWARD_MOVEMENT
REASON_INVALID = REASON_OVERLAP
REASON_ORDER = (
    REASON_ROOF_CLEARANCE,
    REASON_SPLINT_CLEARANCE,
    REASON_INWARD_MOVEMENT,
    REASON_BLOCKED,
    REASON_OVERLAP,
)


def params_from_dict(d: dict[str, Any]) -> DummyParams:
    kwargs = {name: float(d[name]) for name in FEATURE_NAMES}
    kwargs["exploded_gap"] = float(d.get("exploded_gap", 30.0))
    return DummyParams(**kwargs)


def params_to_array(p: DummyParams) -> np.ndarray:
    return np.array([float(getattr(p, name)) for name in FEATURE_NAMES], dtype=np.float64)


def label_params(p: DummyParams) -> dict[str, Any]:
    result = evaluate_exact_assemblability(p)
    terms = result.terms
    terms_dict = asdict(terms)

    out: dict[str, Any] = {
        "label": int(result.assemblable),
        "label_class": result.label_class,
        "label_subclass": result.label_subclass,
        "formula_reason": result.label_reason,
        "validity_ok": bool(result.validity_ok),
    }
    out.update(terms_dict)
    out["formula_margin_min"] = min(
        abs(float(terms_dict.get("vertical_extraction_margin", 0.0))),
        abs(float(terms_dict.get("lateral_escape_margin", 0.0))),
    )
    return out


def label_record(param_id: str, source: str, p: DummyParams) -> dict[str, Any]:
    rec: dict[str, Any] = {"param_id": param_id, "source": source}
    for name in FEATURE_NAMES:
        rec[name] = float(getattr(p, name))
    rec.update(label_params(p))
    return rec
