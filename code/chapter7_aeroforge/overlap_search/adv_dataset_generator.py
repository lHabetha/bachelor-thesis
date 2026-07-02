"""Wide 43-key ADV generator for the overlap-surrogate training pool.

This is the single entry point for generating large *unlabeled* pools of airframe
design vectors (e.g. 100k JSONs). It is intentionally a **family of distributions**,
not one narrow search distribution, designed against two competing goals:

1. **Very wide coverage** of the 43-D ADV space so that any airframe a downstream
   optimizer might visit lies inside (or near) the training manifold, and an MLP
   trained on a labeled subset generalizes there.
2. **A deliberate but conservative tilt toward non-assemblable (overlapping)
   designs**, covering *many different overlap mechanisms* rather than one, so the
   defect class is well represented near a ~40-50% label split.

Design rules (informed by the Chapter 7 §18 discovery experiment):

- **Coverage first.** Most families use ``defaults="wide"`` (full constructible
  range, uniform on every driver) and only *constrain the specific driver(s)* that
  define the family. This keeps every non-targeted dimension full-width, so the
  pool's marginals stay broad. Only a couple of families use ``defaults="broad"``
  (overlap-leaning truncated normals) to add sampling density right at the
  assemblable/non-assemblable boundary.
- **Never widen past the constructible ranges.** The ``"wide"`` sampler already
  spans the validated build-success ranges (Chapter 7 §17); widening further would
  just inflate build failures and waste label budget.
- **Many overlap types.** Overlap is driven mainly by tail type (conv vs v) and the
  longitudinal spacing triad (``wing_position`` aft, ``tail_position`` forward with
  a sharp threshold ~0.82, ``length`` short), with tailplane size as a secondary
  volume amplifier. Families probe these mechanisms individually and in
  combination, plus envelope-size variety, so the defect class is diverse.
- **Sensible weights, not uniform.** Families carry explicit integer weights;
  coverage/safe families are up-weighted so the bulk of the pool stays wide, while
  a spread of overlap families guarantees defect diversity.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import random
from typing import Any

from .discovery_sampler import sample_from_spec


def _u(lo: float, hi: float) -> dict[str, Any]:
    return {"dist": "uniform", "lo": lo, "hi": hi}


def _c(values: list[Any], weights: list[float] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"dist": "choice", "values": values}
    if weights is not None:
        spec["weights"] = weights
    return spec


# ---------------------------------------------------------------------------
# Family mixture.
#
# Each family: name, weight (relative integer), defaults ("wide"|"broad"),
# drivers (per-key distribution overrides), role + expected_overlap (documentation
# only; the realized rate is measured by verify_adv_dataset.py and may be tuned).
# ---------------------------------------------------------------------------
ADV_DATASET_FAMILIES: list[dict[str, Any]] = [
    # ---- Wide baselines: guarantee full-space coverage + non-overlap diversity --
    {
        "name": "wide_unconditioned",
        "weight": 4,
        "role": "Full-width baseline, both tail types (anchors overall coverage)",
        "expected_overlap": "low",
        "defaults": "wide",
        "drivers": {},
    },
    {
        "name": "conv_tail_wide",
        "weight": 4,
        "role": "Full-width within the conventional-tail regime (where overlap lives)",
        "expected_overlap": "low-mid",
        "defaults": "wide",
        "drivers": {"tail_type": _c(["conv_tail"])},
    },
    {
        "name": "v_tail_wide",
        "weight": 2,
        "role": "Full-width v-tail regime (categorical coverage; almost always safe)",
        "expected_overlap": "very low",
        "defaults": "wide",
        "drivers": {"tail_type": _c(["v_tail"])},
    },
    {
        "name": "wide_categorical_coverage",
        "weight": 2,
        "role": "Force even spread over design templates + NACA codes, conv-tail biased",
        "expected_overlap": "low",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail", "v_tail"], weights=[0.75, 0.25]),
            "design_name": _c(["standard", "simple", "transport", "racing", "blended"]),
            "root_naca_code": _c(["2412", "2410", "0012"]),
            "tip_naca_code": _c(["2412", "2408", "0009"]),
        },
    },
    # ---- Safe region: cover the clearly-assemblable interior explicitly ----------
    {
        "name": "safe_long_forward",
        "weight": 1,
        "role": "Validated safe counter-region: forward wing, aft tail, long fuselage",
        "expected_overlap": "near zero",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.40, 0.50),
            "tail_position": _u(0.92, 0.98),
            "length": _u(1150, 1400),
        },
    },
    {
        "name": "safe_v_tail_varied",
        "weight": 1,
        "role": "Safe spacing within v-tail regime, wide elsewhere",
        "expected_overlap": "near zero",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["v_tail"]),
            "wing_position": _u(0.40, 0.55),
            "length": _u(1050, 1400),
        },
    },
    # ---- Envelope-size variety: cover small vs large airframes jointly -----------
    {
        "name": "compact_aircraft",
        "weight": 2,
        "role": "Small/compact envelope (short, narrow, small span) - tends to overlap",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "length": _u(800, 1050),
            "wingspan": _u(1200, 1600),
            "max_width": _u(100, 150),
            "max_height": _u(75, 110),
        },
    },
    {
        "name": "large_aircraft",
        "weight": 2,
        "role": "Large envelope (long, wide, big span + tailplane) - mostly safe but big surfaces",
        "expected_overlap": "low-mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "length": _u(1150, 1400),
            "wingspan": _u(1700, 2000),
            "max_width": _u(150, 200),
            "max_height": _u(110, 140),
            "hstab_semispan": _u(280, 350),
        },
    },
    # ---- Single-mechanism overlap (wide elsewhere): many distinct defect types --
    {
        "name": "aft_wing_wide",
        "weight": 3,
        "role": "Primary driver: aft wing_position, everything else full-width",
        "expected_overlap": "mid-high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.60, 0.70),
        },
    },
    {
        "name": "forward_tail_wide",
        "weight": 3,
        "role": "Primary driver: forward tail_position (plateau band), wide elsewhere",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "tail_position": _u(0.82, 0.88),
        },
    },
    {
        "name": "extreme_forward_tail_wide",
        "weight": 2,
        "role": "Nonlinear tail_position threshold (<0.82): high overlap, wide elsewhere",
        "expected_overlap": "high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "tail_position": _u(0.78, 0.82),
        },
    },
    {
        "name": "short_fuselage_wide",
        "weight": 3,
        "role": "Primary driver: short length, wide elsewhere",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "length": _u(800, 1000),
        },
    },
    {
        "name": "large_hstab_wide",
        "weight": 2,
        "role": "Secondary volume amplifier: large + low-AR horizontal tail, wide elsewhere",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "hstab_semispan": _u(300, 350),
            "hstab_aspect_ratio": _u(2.5, 3.4),
        },
    },
    {
        "name": "low_ar_chord_wide",
        "weight": 1,
        "role": "Weak secondary: low wing aspect ratio (large chord), wide elsewhere",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "aspect_ratio": _u(5.5, 7.0),
        },
    },
    # ---- Multi-mechanism overlap: different combined defect "shapes" ------------
    {
        "name": "aft_wing_short_fuse",
        "weight": 3,
        "role": "Two-factor spacing: aft wing + short fuselage, wide elsewhere",
        "expected_overlap": "high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.60, 0.68),
            "length": _u(800, 1000),
        },
    },
    {
        "name": "forward_tail_aft_wing",
        "weight": 3,
        "role": "Two-factor spacing: forward tail + aft wing, wide elsewhere",
        "expected_overlap": "high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.58, 0.66),
            "tail_position": _u(0.80, 0.86),
        },
    },
    {
        "name": "moderate_spacing_triad",
        "weight": 3,
        "role": "Dense high-overlap cluster: full spacing triad, overlap-leaning background",
        "expected_overlap": "very high",
        "defaults": "broad",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.58, 0.64),
            "tail_position": _u(0.84, 0.88),
            "length": _u(950, 1100),
        },
    },
    {
        "name": "forward_tail_large_hstab",
        "weight": 2,
        "role": "Tail threshold + tailplane-volume amplifier, wide elsewhere",
        "expected_overlap": "high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "tail_position": _u(0.78, 0.84),
            "hstab_semispan": _u(290, 350),
        },
    },
    # ---- Boundary density: overlap-leaning backgrounds (broad) -------------------
    {
        "name": "conv_tail_broad",
        "weight": 2,
        "role": "Overlap-leaning conventional-tail cluster (dense near decision boundary)",
        "expected_overlap": "mid-high",
        "defaults": "broad",
        "drivers": {"tail_type": _c(["conv_tail"])},
    },
    {
        "name": "overlap_volume_diverse",
        "weight": 2,
        "role": "High-overlap with volume/shape diversity (span, chord, tailplane)",
        "expected_overlap": "high",
        "defaults": "wide",
        "drivers": {
            "tail_type": _c(["conv_tail"]),
            "wing_position": _u(0.58, 0.68),
            "tail_position": _u(0.78, 0.88),
            "hstab_semispan": _u(270, 350),
            "aspect_ratio": _u(5.5, 8.0),
        },
    },
    # ---- Frontier: deliberately beyond the standard constructible ranges --------
    # Pushes EVERY driver past the validated [lo, hi] used by the other families,
    # into "far but still buildable" territory. Small weight so it does not dominate
    # the training signal, but present enough to double as an out-of-distribution
    # stress generator for the test/validation set (how does the model do on
    # airframes unlike anything in the bulk training pool?). Ranges below were
    # build-verified; extremes that broke the geometry build were pulled back.
    {
        "name": "extreme_wide_frontier",
        "weight": 3,
        "role": "OOD frontier: every driver beyond standard ranges, still buildable",
        "expected_overlap": "mid",
        "defaults": "wide",
        "drivers": {
            "wing_position": _u(0.34, 0.76),
            "tail_position": _u(0.75, 0.99),
            "length": _u(700, 1600),
            "wingspan": _u(1000, 2300),
            "aspect_ratio": _u(4.5, 13.0),
            "taper": _u(0.25, 0.70),
            "sweep": _u(5.0, 35.0),
            "dihedral": _u(0.0, 14.0),
            "twist": _u(0.0, 6.0),
            "root_incidence": _u(-5.0, 7.0),
            "wing_height_ratio": _u(0.25, 0.75),
            "max_width": _u(85, 230),
            "max_height": _u(60, 160),
            "wall_thickness": _u(2.0, 11.0),
            "end_cap_percent": _u(0.02, 0.11),
            "hstab_semispan": _u(160, 420),
            "hstab_aspect_ratio": _u(2.0, 6.3),
            "hstab_taper": _u(0.25, 0.70),
            "hstab_sweep": _u(8.0, 40.0),
            "hstab_dihedral": _u(0.0, 12.0),
            "hstab_root_incidence": _u(-8.0, 4.0),
            "vstab_height": _u(90, 280),
            "vstab_aspect_ratio": _u(0.9, 2.8),
            "vstab_taper": _u(0.25, 0.75),
            "vstab_sweep": _u(10.0, 54.0),
        },
    },
]

_WEIGHTS = [float(f.get("weight", 1)) for f in ADV_DATASET_FAMILIES]
_GLOBAL_RNG = random.Random()


def _pick_family(rng: random.Random) -> dict[str, Any]:
    return rng.choices(ADV_DATASET_FAMILIES, weights=_WEIGHTS, k=1)[0]


def sample_adv_dataset(rng: random.Random | None = None) -> dict[str, Any]:
    """Sample one 43-key ADV from the wide, weighted training-pool mixture.

    This is the one-call generator for large unlabeled pools. Pass a seeded
    ``random.Random`` for reproducibility; generation is pure-Python and fast
    (no geometry build), so 100k vectors are effectively instant.
    """
    rng = rng or _GLOBAL_RNG
    return sample_from_spec(rng, _pick_family(rng))


def sample_adv_dataset_with_family(rng: random.Random | None = None) -> tuple[dict[str, Any], str]:
    """Like :func:`sample_adv_dataset` but also returns the family name (for audits)."""
    rng = rng or _GLOBAL_RNG
    family = _pick_family(rng)
    return sample_from_spec(rng, family), family["name"]


def family_weights() -> dict[str, float]:
    """Return each family's normalized selection probability (sums to 1)."""
    total = sum(_WEIGHTS)
    return {f["name"]: w / total for f, w in zip(ADV_DATASET_FAMILIES, _WEIGHTS)}
