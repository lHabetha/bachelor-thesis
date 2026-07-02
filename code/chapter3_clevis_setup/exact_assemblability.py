"""Assumption-bound exact assemblability geometry for DummyParams.

This module intentionally separates:
1) geometric validity (A1-E3 checks in ``design_space.validate_params``), and
2) a deterministic kinematic assemblability predicate under frozen assumptions.

Frozen assumptions for the exact claim in this file:
- Rigid parts.
- No rotations.
- Before splint extraction, the main pin may be pushed in the "wrong"
  direction (``-Y``) until the nail-head touches the +Y outer wall.
- The splint center may additionally shift inside the cross-hole by radial
  slack in the helpful direction.
- The splint is extracted by a pure ``+Z`` translation, possibly followed by
  a lateral move after its -Z end clears the main pin core cylinder.
- Once the splint is removed, the main pin is pulled out in the intended
  ``+Y`` direction by its nail-head.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .design_space import DummyParams, SPLINT_HEAD_RATIO, validate_params


@dataclass(frozen=True)
class ExactAssemblabilityTerms:
    """Analytic geometry terms derived directly from ``DummyParams``."""

    # Raw landmarks
    y_overhang_outer_neg: float
    y_overhang_outer_pos: float
    y_splint_center: float
    y_splint_head_neg_edge: float
    y_splint_head_pos_edge: float
    z_roof_bottom: float
    z_splint_head_top: float
    z_pin_top: float
    z_splint_bottom: float
    head_radius: float

    # Assemb-roof_clearance: Does roof cover the splint in the stale pose?
    roof_clearance: bool
    roof_blocks_splint: bool

    # Assemb-splint_clearance: raise splint until head touches roof, then move sideways
    available_travel: float
    needed_travel: float
    raw_pin_top_to_roof_bottom: float
    main_pin_vertical_slack: float
    effective_pin_top_to_roof_bottom: float
    splint_total_height: float
    splint_clearance_margin: float
    splint_clearance_margin_fraction: float
    splint_clearance: bool
    # Backwards-compatible aliases
    vertical_extraction_margin: float
    vertical_extraction_margin_fraction: float
    can_extract_vertically: bool

    # Assemb-inward_movement: push main pin -Y first, then pull splint +Z
    pin_inward_travel: float
    cross_hole_radial_slack: float
    available_inward_splint_shift: float
    needed_inward_splint_shift: float
    inward_movement_margin: float
    inward_movement: bool
    # Backwards-compatible aliases
    max_lateral_shift: float
    needed_lateral_shift: float
    lateral_escape_margin: float
    lateral_escape: bool


@dataclass(frozen=True)
class ExactAssemblabilityResult:
    """Outcome of the exact assumption-bound predicate."""

    validity_ok: bool
    validity_reasons: list[str]
    kinematic_assemblable: bool
    assemblable: bool
    label_class: str
    label_subclass: str
    label_reason: str
    terms: ExactAssemblabilityTerms

    def to_dict(self) -> dict:
        return {
            "validity_ok": self.validity_ok,
            "validity_reasons": list(self.validity_reasons),
            "kinematic_assemblable": self.kinematic_assemblable,
            "assemblable": self.assemblable,
            "label_class": self.label_class,
            "label_subclass": self.label_subclass,
            "label_reason": self.label_reason,
            "terms": asdict(self.terms),
        }


def _splint_head_len(p: DummyParams) -> float:
    return 0.05 * p.splint_length


def compute_exact_terms(p: DummyParams) -> ExactAssemblabilityTerms:
    """Compute all physically correct Task-16 geometric terms."""
    head_r = SPLINT_HEAD_RATIO * p.splint_radius
    head_len = _splint_head_len(p)

    y_overhang_outer_neg = -0.5 * p.overhang_span_y
    y_overhang_outer_pos = 0.5 * p.overhang_span_y
    z_roof_bottom = 0.5 * p.wall_thickness

    y_splint_center = p.cross_y
    y_splint_head_neg_edge = y_splint_center - head_r
    y_splint_head_pos_edge = y_splint_center + head_r

    z_splint_head_top = p.hole_z + 0.5 * p.splint_length + head_len
    z_pin_top = p.hole_z + p.main_pin_radius
    z_splint_bottom = p.hole_z - 0.5 * p.splint_length

    # assemb-roof_clearance: Does the roof even block the splint's vertical path?
    # If the +Y edge of the splint head is further -Y than the -Y edge of the roof,
    # it clears completely.
    roof_clearance = y_splint_head_pos_edge <= y_overhang_outer_neg
    roof_blocks_splint = not roof_clearance

    # assemb-splint_clearance: vertical clearance race.
    #
    # Equivalent physical statements:
    # 1) Motion view:
    #    available_travel = how far the head top can move up before touching roof
    #    needed_travel = how far the shaft bottom must move up to clear pin top
    # 2) Static-space view:
    #    effective_pin_top_to_roof_bottom must fit the whole splint shaft plus
    #    head height.
    # The second view is usually easier to validate by eye.
    main_pin_vertical_slack = max(0.0, p.main_hole_radius - p.main_pin_radius)
    available_travel = z_roof_bottom - z_splint_head_top + main_pin_vertical_slack
    needed_travel = z_pin_top - z_splint_bottom
    raw_pin_top_to_roof_bottom = z_roof_bottom - z_pin_top
    effective_pin_top_to_roof_bottom = raw_pin_top_to_roof_bottom + main_pin_vertical_slack
    splint_total_height = p.splint_length + head_len
    splint_clearance_margin = effective_pin_top_to_roof_bottom - splint_total_height
    splint_clearance_margin_fraction = (
        splint_clearance_margin / splint_total_height
        if splint_total_height > 0.0
        else 0.0
    )
    splint_clearance = splint_clearance_margin >= 0.0

    # assemb-inward_movement: push the main pin in the wrong direction (-Y)
    # until its nail-head touches the +Y outer wall, plus radial slack of the
    # splint inside the cross-hole. If this moves the splint head past the -Y
    # roof edge, the splint can be pulled +Z and then the pin can be pulled out +Y.
    pin_inward_travel = p.pin_half - (p.outer_span / 2.0)
    cross_hole_radial_slack = max(0.0, p.cross_hole_radius - p.splint_radius)
    available_inward_splint_shift = pin_inward_travel + cross_hole_radial_slack

    # How much shift is needed to get the splint head's +Y edge past the roof's -Y edge?
    needed_inward_splint_shift = max(0.0, y_splint_head_pos_edge - y_overhang_outer_neg)
    inward_movement_margin = available_inward_splint_shift - needed_inward_splint_shift
    inward_movement = inward_movement_margin >= 0.0

    # Backwards-compatible aliases for older scripts/reports.
    vertical_extraction_margin = splint_clearance_margin
    vertical_extraction_margin_fraction = splint_clearance_margin_fraction
    can_extract_vertically = splint_clearance
    max_lateral_shift = available_inward_splint_shift
    needed_lateral_shift = needed_inward_splint_shift
    lateral_escape_margin = inward_movement_margin
    lateral_escape = inward_movement

    return ExactAssemblabilityTerms(
        y_overhang_outer_neg=y_overhang_outer_neg,
        y_overhang_outer_pos=y_overhang_outer_pos,
        y_splint_center=y_splint_center,
        y_splint_head_neg_edge=y_splint_head_neg_edge,
        y_splint_head_pos_edge=y_splint_head_pos_edge,
        z_roof_bottom=z_roof_bottom,
        z_splint_head_top=z_splint_head_top,
        z_pin_top=z_pin_top,
        z_splint_bottom=z_splint_bottom,
        head_radius=head_r,
        roof_clearance=roof_clearance,
        roof_blocks_splint=roof_blocks_splint,
        available_travel=available_travel,
        needed_travel=needed_travel,
        raw_pin_top_to_roof_bottom=raw_pin_top_to_roof_bottom,
        main_pin_vertical_slack=main_pin_vertical_slack,
        effective_pin_top_to_roof_bottom=effective_pin_top_to_roof_bottom,
        splint_total_height=splint_total_height,
        splint_clearance_margin=splint_clearance_margin,
        splint_clearance_margin_fraction=splint_clearance_margin_fraction,
        splint_clearance=splint_clearance,
        vertical_extraction_margin=vertical_extraction_margin,
        vertical_extraction_margin_fraction=vertical_extraction_margin_fraction,
        can_extract_vertically=can_extract_vertically,
        pin_inward_travel=pin_inward_travel,
        cross_hole_radial_slack=cross_hole_radial_slack,
        available_inward_splint_shift=available_inward_splint_shift,
        needed_inward_splint_shift=needed_inward_splint_shift,
        inward_movement_margin=inward_movement_margin,
        inward_movement=inward_movement,
        max_lateral_shift=max_lateral_shift,
        needed_lateral_shift=needed_lateral_shift,
        lateral_escape_margin=lateral_escape_margin,
        lateral_escape=lateral_escape,
    )


def evaluate_exact_assemblability(
    p: DummyParams,
    *,
    require_validity: bool = True,
) -> ExactAssemblabilityResult:
    """Evaluate the physically exact Task-16 predicate."""
    validity_ok, validity_reasons = validate_params(p)
    t = compute_exact_terms(p)

    if not validity_ok and require_validity:
        label_class = "overlap"
        label_subclass = ""
        label_reason = "overlap"
        kinematic_assemblable = False
    elif t.roof_clearance:
        label_class = "assemb"
        label_subclass = "roof_clearance"
        label_reason = "assemb-roof_clearance"
        kinematic_assemblable = True
    elif t.splint_clearance:
        label_class = "assemb"
        label_subclass = "splint_clearance"
        label_reason = "assemb-splint_clearance"
        kinematic_assemblable = True
    elif t.inward_movement:
        label_class = "assemb"
        label_subclass = "inward_movement"
        label_reason = "assemb-inward_movement"
        kinematic_assemblable = True
    else:
        label_class = "blocked"
        label_subclass = ""
        label_reason = "blocked"
        kinematic_assemblable = False

    assemblable = kinematic_assemblable and (validity_ok or not require_validity)

    return ExactAssemblabilityResult(
        validity_ok=validity_ok,
        validity_reasons=validity_reasons,
        kinematic_assemblable=kinematic_assemblable,
        assemblable=assemblable,
        label_class=label_class,
        label_subclass=label_subclass,
        label_reason=label_reason,
        terms=t,
    )

__all__ = [
    "ExactAssemblabilityTerms",
    "ExactAssemblabilityResult",
    "compute_exact_terms",
    "evaluate_exact_assemblability",
]
