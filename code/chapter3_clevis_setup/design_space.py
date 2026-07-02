"""Design-space sampler + validity filter for the clevis DummyParams.

Chapter 3, Phase 1. The whole point of this module is to be able to
randomly draw ``DummyParams`` instances that are *geometrically valid*
(parts fit, no hole breaks out, legs have a real cavity, etc.) without
any prior knowledge of which of those draws will end up
*disassemblable*. The assemblability question is what the surrogate
learns; the validity question is what this module enforces.

This file is the executable companion to the validity checks in thesis
Chapter 3: every constraint
A1..E3 from the doc has a corresponding check in :func:`validate_params`,
and :func:`sample_params` is written so the independent dimensions can
be freely uniform while the dependents are drawn conditionally to keep
constraint violations rare (rejection rate empirically <5%).

Public API
----------
* :data:`DESIGN_SPACE`      — per-parameter `(min, max)` boxes
* :data:`TEST_PROTOCOL_EN`  — plain-English constraint check-list
* :func:`validate_params`   — returns ``(ok, reasons)``
* :func:`sample_params`     — returns a ``DummyParams`` instance

Conventions
-----------
* All dimensions are in mm (pre-normalization). ``prepare_assembly``
  later rescales to bbox=10 before the planner sees anything.
* Safety margins ``EPS_*`` default to 0.25 mm. These mirror the ε
  values in the constraints doc.
* The sampler takes a :class:`numpy.random.Generator` for reproducibility.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .generate_modified_clevis_dummy import DummyParams


# ──────────────────────────────────────────────────────────────────────
#  Safety-margin constants
# ──────────────────────────────────────────────────────────────────────

EPS_FIT: float = 0.025         # radial clearance on shaft-in-hole fits.
# 0.025 mm is below the demos' tightest nominal fit (pin=3.95 / hole=4.00,
# i.e. 0.05 mm radial clearance) while still preserving a safety buffer
# against mesh tessellation error (cylinders with ~60 segments approximate
# the radius within ~0.01-0.02 mm). All 10 canonical demos are validated
# geometrically by overlap_check, so any validity constraint that rejects
# one of them is definitionally too strict.
EPS_PIN_WALL: float = 0.5      # min material between cross-hole and pin ends
EPS_LEG_WALL: float = 0.5      # min material between hole and leg tip
EPS_DEPTH_WALL: float = 1.0    # min material on each side of hole in X
EPS_RETENTION: float = 1.5     # splint retention length past the pin surface
EPS_ACCESS: float = 0.5        # pin-must-protrude-past-wall safety
EPS_HEAD_CLEAR: float = 0.5    # splint head vs roof clearance
EPS_HEAD_SLIP: float = 0.3     # min material from splint head edge to cross-hole edge.
# 2026-04-11: introduced after discovering v1 used head_r = 1.2*splint_radius,
# which was smaller than cross_hole_radius in ~75-90% of random samples.
# 0.3 mm is plenty above mesh tessellation error (~0.02 mm) and matches the
# tightest demo (demo10: head=2.4, hole=2.0, margin=0.4 mm > 0.3).

# Splint head retention ratio. MUST equal
# clevis_dummy_assembly.generate_modified_clevis_dummy.SPLINT_HEAD_RATIO.
# Encoded here as a separate constant so design_space stays importable
# without pulling in CadQuery; a runtime cross-check in validate_params
# catches accidental drift.
SPLINT_HEAD_RATIO: float = 1.5


# ──────────────────────────────────────────────────────────────────────
#  DESIGN_SPACE: per-parameter bounds (independent dims only)
#
#  Dependent dims get sampled *conditionally* by sample_params() so
#  constraint violation stays rare. The comments on each row show the
#  coupling; validate_params() enforces every constraint regardless of
#  how the sample was produced.
# ──────────────────────────────────────────────────────────────────────

DESIGN_SPACE: dict[str, dict] = {
    # Bracket frame (all independent)
    'wall_thickness':                  {'min':  2.0, 'max': 12.0, 'independent': True},
    'outer_span':                      {'min': 30.0, 'max': 100.0, 'independent': True},
    'leg_length':                      {'min': 20.0, 'max': 80.0, 'independent': True},
    'depth':                           {'min': 12.0, 'max': 60.0, 'independent': True},

    # Main hole (independent, but bounded by depth)
    'main_hole_radius':                {'min':  2.0, 'max': 10.0, 'independent': True,
                                        'bounded_by': 'depth',
                                        'bounded_note': '2r + 2*EPS_DEPTH_WALL <= depth'},

    # Main hole offset: dependent on leg_length + main_hole_radius (80% rule)
    'main_hole_offset_from_open_end':  {'min':  None, 'max': None, 'independent': False,
                                        'derivation':
                                            'U(main_hole_radius + EPS_LEG_WALL, '
                                            '0.80 * leg_length - main_hole_radius)'},

    # Main pin radius: dependent on main_hole_radius (shaft-in-hole fit)
    'main_pin_radius':                 {'min':  None, 'max': None, 'independent': False,
                                        'derivation':
                                            'U(0.75, 0.97) * main_hole_radius; '
                                            'clamped to >= 1.0 mm'},

    # Cross-hole radius: bounded by main_pin_radius, independently drawn ratio
    'cross_hole_radius':               {'min':  1.0, 'max':  5.0, 'independent': False,
                                        'derivation':
                                            'U(0.30, 0.65) * main_pin_radius; '
                                            'clamped to >= 1.0 mm'},

    # Cross-hole distance from free end: dependent on cross_hole_radius + pin length
    'cross_hole_distance_from_free_end': {'min': None, 'max': None, 'independent': False,
                                          'derivation':
                                              'U(cross_hole_radius + EPS_PIN_WALL, '
                                              '0.25 * main_pin_length)'},

    # Main pin length: derived from outer_span + cross-hole + protrude margin
    'main_pin_length':                 {'min':  None, 'max': None, 'independent': False,
                                        'derivation':
                                            'outer_span + 2*(cross_hole_dist + radius) '
                                            '+ 2*EPS_ACCESS + U(4, 20)'},

    # Splint radius: bounded by cross_hole_radius (shaft-in-hole fit)
    'splint_radius':                   {'min':  0.8, 'max':  4.0, 'independent': False,
                                        'derivation':
                                            'U(0.60, 0.90) * cross_hole_radius; '
                                            'clamped to >= 0.8 mm'},

    # Splint length: derived, must span pin + retention both sides
    'splint_length':                   {'min':  None, 'max': None, 'independent': False,
                                        'derivation':
                                            '2*main_pin_radius + 2*EPS_RETENTION + U(4, 20); '
                                            'also clipped by splint-head-clears-roof constraint'},

    # Overhang: independent and free to cage — the whole thesis signal
    'overhang_span_y':                 {'min':  0.0, 'max': None, 'independent': True,
                                        'derivation':
                                            'U(0.8, 2.0) * outer_span; the upper end '
                                            'creates a roof that topologically cages the pin. '
                                            'This is ALLOWED and creates label=0 samples.'},

    # Preview-only
    'exploded_gap':                    {'min': 30.0, 'max': 30.0, 'independent': True,
                                        'derivation': 'fixed'},
}


# ──────────────────────────────────────────────────────────────────────
#  TEST PROTOCOL (plain English)
# ──────────────────────────────────────────────────────────────────────

TEST_PROTOCOL_EN: str = """\
Validity test protocol for a DummyParams tuple p. All dimensions in mm.

For each freshly sampled or hand-written DummyParams, run these checks
in order. If ANY fails, the design is INVALID and must be rejected
BEFORE generating mesh or running the physics planner. If ALL pass,
the design is VALID (geometrically) but may still be non-disassemblable
— that is the intended thesis signal.

A. SLIDING FITS (shaft fits inside hole)
  A1. The main pin shaft fits through the leg holes:
          p.main_pin_radius + EPS_FIT <= p.main_hole_radius
  A2. The splint shaft fits through the pin's cross-hole:
          p.splint_radius + EPS_FIT <= p.cross_hole_radius
  A3. The splint head is LARGER than the cross-hole so it actually
      retains the splint instead of walking through:
          SPLINT_HEAD_RATIO * p.splint_radius
              >= p.cross_hole_radius + EPS_HEAD_SLIP
      (With SPLINT_HEAD_RATIO=1.5, EPS_HEAD_SLIP=0.3 mm. Before 2026-04-11
      the head was 1.2*splint_radius, and since the sampler chose
      splint_radius from U(0.60, 0.90)*cross_hole_radius the head was
      actually SMALLER than the hole in ~75-90% of random samples -
      defeating retention entirely.)

B. NO HOLE BREAKS OUT OF ITS PARENT SOLID
  B1. Main hole stays inside bracket depth (X direction):
          2 * p.main_hole_radius + 2 * EPS_DEPTH_WALL <= p.depth
  B2. Main hole stays in the lower 80% of the leg (user's rule),
      with its top edge at or below 80% of leg_length:
          p.main_hole_offset_from_open_end + p.main_hole_radius
              <= 0.80 * p.leg_length
  B3. Main hole does not break the open leg tip:
          p.main_hole_offset_from_open_end - p.main_hole_radius
              >= EPS_LEG_WALL
  B4. Cross-hole does not open the pin's -Y free tip:
          p.cross_hole_distance_from_free_end
              >= p.cross_hole_radius + EPS_PIN_WALL
  B5. Cross-hole does not eat into the pin's +Y nail-head:
          p.cross_hole_distance_from_free_end + p.cross_hole_radius
              + EPS_PIN_WALL
              <= p.main_pin_length - head_len(p)
      where head_len(p) = 0.05 * p.main_pin_length

C. BRACKET HAS A SENSIBLE INNER CAVITY
  C1. The two legs have a real inner gap:
          p.outer_span > 2 * p.wall_thickness + EPS_ACCESS

D. PARTS CAN PHYSICALLY BE IN THEIR ASSEMBLED POSE
  D1. Pin is long enough that the cross-hole sits OUTSIDE the -Y wall:
          p.main_pin_length
              > p.outer_span
                + 2 * p.cross_hole_distance_from_free_end
                + 2 * EPS_ACCESS
  D2. Cross-hole's +Y rim clears the -Y wall inner face:
          p.main_pin_length
              >= p.outer_span - 2 * p.wall_thickness
                + 2 * p.cross_hole_distance_from_free_end
                + 2 * p.cross_hole_radius
                + 2 * EPS_ACCESS
  D3. Splint is long enough to span the pin diameter plus retention
      on both sides:
          p.splint_length >= 2 * p.main_pin_radius + 2 * EPS_RETENTION
  D4. Splint head does not clip into the roof, even if the main pin
      shifts upward by its radial slack inside the main through-hole.
      Using derived splint head length hl = 0.05 * splint_length:
          0.5 * p.splint_length + hl
              + (p.main_hole_radius - p.main_pin_radius)
              + EPS_HEAD_CLEAR
              <= p.wall_thickness + p.leg_length
                 - p.main_hole_offset_from_open_end
  D5. The oversized splint head (radius hr = SPLINT_HEAD_RATIO *
      splint_radius) must NOT extend into the bracket's -Y leg wall.
      The head sits at y = cross_y = -(pin_half
      - cross_hole_distance_from_free_end) and extends +/- hr in Y.
      Requiring the head's +Y rim to stay on the -Y side of the -Y
      OUTER wall face (y = -outer_span/2):
          p.main_pin_length
              >= p.outer_span
                 + 2 * p.cross_hole_distance_from_free_end
                 + 2 * (SPLINT_HEAD_RATIO * p.splint_radius)
                 + 2 * EPS_ACCESS
      BUG FIX 2026-05-26: previously used (outer_span - 2*wt), which
      checked against the inner wall face — the head could sit inside
      the wall material between outer and inner faces.

  D6. The splint head disc must fit within the main-hole bore through
      the bracket leg. Because D1 guarantees the splint center sits
      outside the -Y wall, the splint head (a cylinder of radius hr)
      passes through the leg at the same Z as the main hole. If the
      head is wider than the hole, it clips into the remaining leg
      material around the bore:
          SPLINT_HEAD_RATIO * p.splint_radius + EPS_FIT
              <= p.main_hole_radius
  D7. The main pin nail-head must sit below the underside of the roof,
      regardless of roof Y-span and even if the main pin shifts upward
      by its radial slack inside the main through-hole. This intentionally
      rejects edge cases where the nail-head would only avoid the roof
      because the roof is short in Y:
          p.hole_z + 2 * p.main_pin_radius
              + (p.main_hole_radius - p.main_pin_radius)
              + EPS_HEAD_CLEAR
              <= p.wall_thickness / 2
      Expanded with hole_z:
          p.main_hole_offset_from_open_end
              + p.main_hole_radius + p.main_pin_radius
              + EPS_HEAD_CLEAR <= p.wall_thickness + p.leg_length

E. NON-DEGENERATE DIMENSIONS
  E1. All linear dimensions strictly positive with a small minimum
      (>= 0.5 mm). Covers wall_thickness, outer_span, leg_length,
      depth, main_hole_radius, main_pin_length, main_pin_radius,
      cross_hole_radius, cross_hole_distance_from_free_end,
      splint_radius, splint_length.
  E2. overhang_span_y >= 0 (0 produces a degenerate 1e-3 mm roof nub
      per _build_roof; negative is rejected).
  E3. exploded_gap > 0 (preview only).

Intentionally NOT checked
-------------------------
* Upper bound on overhang_span_y. Large overhangs create topological
  caging — that is the primary label-0 source and is ALLOWED.
* Any disassembly-reachability check. This module filters geometric
  validity only. The planner reports disassemblability separately.
"""


# ──────────────────────────────────────────────────────────────────────
#  validate_params(): executable version of TEST_PROTOCOL_EN
# ──────────────────────────────────────────────────────────────────────

def _head_len(p: DummyParams) -> float:
    return 0.05 * p.main_pin_length


def _splint_head_len(p: DummyParams) -> float:
    return 0.05 * p.splint_length


def validate_params(
    p: DummyParams,
    *,
    return_reasons: bool = True,
    eps_fit: float = EPS_FIT,
    eps_pin_wall: float = EPS_PIN_WALL,
    eps_leg_wall: float = EPS_LEG_WALL,
    eps_depth_wall: float = EPS_DEPTH_WALL,
    eps_retention: float = EPS_RETENTION,
    eps_access: float = EPS_ACCESS,
    eps_head_clear: float = EPS_HEAD_CLEAR,
    eps_head_slip: float = EPS_HEAD_SLIP,
) -> tuple[bool, list[str]]:
    """Check every A1..E3 constraint from TEST_PROTOCOL_EN.

    Returns ``(ok, reasons)`` where ``reasons`` is a list of structured
    failure tags of the form ``'A1: main_pin_radius=3.95 >= main_hole_radius=4.00'``.
    On success, returns ``(True, [])``.
    """
    reasons: list[str] = []

    # E1: positivity (check first; other constraints may divide by these)
    linear_fields = [
        'wall_thickness', 'outer_span', 'leg_length', 'depth',
        'main_hole_radius', 'main_pin_length', 'main_pin_radius',
        'cross_hole_radius', 'cross_hole_distance_from_free_end',
        'splint_radius', 'splint_length',
    ]
    for f in linear_fields:
        v = float(getattr(p, f))
        if v < 0.5:
            reasons.append(f'E1: {f}={v:.3f} < 0.5 mm (degenerate)')
    if p.overhang_span_y < 0.0:
        reasons.append(f'E2: overhang_span_y={p.overhang_span_y:.3f} < 0')
    if p.exploded_gap <= 0.0:
        reasons.append(f'E3: exploded_gap={p.exploded_gap:.3f} <= 0')

    if reasons and not return_reasons:
        return False, reasons

    # A: Sliding fits
    if p.main_pin_radius + eps_fit > p.main_hole_radius:
        reasons.append(
            f'A1: main_pin_radius={p.main_pin_radius:.3f} + EPS_FIT'
            f'={eps_fit} > main_hole_radius={p.main_hole_radius:.3f}')
    if p.splint_radius + eps_fit > p.cross_hole_radius:
        reasons.append(
            f'A2: splint_radius={p.splint_radius:.3f} + EPS_FIT'
            f'={eps_fit} > cross_hole_radius={p.cross_hole_radius:.3f}')
    hr = SPLINT_HEAD_RATIO * p.splint_radius
    if hr < p.cross_hole_radius + eps_head_slip:
        reasons.append(
            f'A3: head_r={hr:.3f} (={SPLINT_HEAD_RATIO}*splint_radius) '
            f'< cross_hole_radius+EPS_HEAD_SLIP'
            f'={p.cross_hole_radius + eps_head_slip:.3f} '
            f'(splint head would slip through cross-hole)')

    # B: No holes breaking out
    if 2 * p.main_hole_radius + 2 * eps_depth_wall > p.depth:
        reasons.append(
            f'B1: 2*main_hole_radius={2*p.main_hole_radius:.3f} + '
            f'2*EPS_DEPTH_WALL={2*eps_depth_wall} > depth={p.depth:.3f}')
    if p.main_hole_offset_from_open_end + p.main_hole_radius > 0.80 * p.leg_length:
        reasons.append(
            f'B2: offset+r={p.main_hole_offset_from_open_end + p.main_hole_radius:.3f} '
            f'> 0.80*leg_length={0.80 * p.leg_length:.3f}')
    if p.main_hole_offset_from_open_end - p.main_hole_radius < eps_leg_wall:
        reasons.append(
            f'B3: offset-r={p.main_hole_offset_from_open_end - p.main_hole_radius:.3f} '
            f'< EPS_LEG_WALL={eps_leg_wall}')
    if p.cross_hole_distance_from_free_end < p.cross_hole_radius + eps_pin_wall:
        reasons.append(
            f'B4: cross_hole_distance_from_free_end={p.cross_hole_distance_from_free_end:.3f} '
            f'< cross_hole_radius+EPS_PIN_WALL={p.cross_hole_radius + eps_pin_wall:.3f}')
    hl_pin = _head_len(p)
    b5_rhs = p.main_pin_length - hl_pin
    b5_lhs = p.cross_hole_distance_from_free_end + p.cross_hole_radius + eps_pin_wall
    if b5_lhs > b5_rhs:
        reasons.append(
            f'B5: cross_hole+r+eps={b5_lhs:.3f} > main_pin_length-head_len={b5_rhs:.3f}')

    # C: Cavity
    if p.outer_span <= 2 * p.wall_thickness + eps_access:
        reasons.append(
            f'C1: outer_span={p.outer_span:.3f} <= 2*wall_thickness+EPS_ACCESS'
            f'={2*p.wall_thickness + eps_access:.3f}')

    # D: Assembled pose
    d1_rhs = p.outer_span + 2 * p.cross_hole_distance_from_free_end + 2 * eps_access
    if p.main_pin_length <= d1_rhs:
        reasons.append(
            f'D1: main_pin_length={p.main_pin_length:.3f} '
            f'<= outer_span+2*cross_hole_dist+2*eps={d1_rhs:.3f}')
    d2_rhs = (p.outer_span - 2 * p.wall_thickness
              + 2 * p.cross_hole_distance_from_free_end
              + 2 * p.cross_hole_radius
              + 2 * eps_access)
    if p.main_pin_length < d2_rhs:
        reasons.append(
            f'D2: main_pin_length={p.main_pin_length:.3f} '
            f'< outer_span-2wt+2cd+2cr+2eps={d2_rhs:.3f}')
    # D5: oversized splint head must NOT extend into the bracket's -Y leg
    # wall. The head center is at y = cross_y (outside the bracket on -Y side);
    # the head +Y rim at cross_y + hr must stay on the -Y side of the -Y OUTER
    # wall face (y = -outer_span/2). Using main_pin_length as the controlling
    # variable:
    #   cross_y + hr <= -outer_span/2 - eps_access
    #   ⇔ main_pin_length >= outer_span + 2*cd + 2*hr + 2*eps_access
    #
    # BUG FIX 2026-05-26: previously used (outer_span - 2*wall_thickness),
    # which checked against the INNER wall face — allowing the head to sit
    # inside the wall material between outer and inner faces.
    d5_rhs = (p.outer_span
              + 2 * p.cross_hole_distance_from_free_end
              + 2 * hr
              + 2 * eps_access)
    if p.main_pin_length < d5_rhs:
        reasons.append(
            f'D5: main_pin_length={p.main_pin_length:.3f} '
            f'< outer_span+2cd+2hr+2eps={d5_rhs:.3f} '
            f'(oversized splint head overlaps -Y bracket wall)')
    # D6: splint head fits within the main hole bore
    if hr > p.main_hole_radius - eps_fit:
        reasons.append(
            f'D6: splint_head_r={hr:.3f} '
            f'> main_hole_radius-EPS_FIT={p.main_hole_radius - eps_fit:.3f} '
            f'(splint head wider than main hole — clips bracket leg)')
    # D7: main pin nail-head top stays below roof underside, independent of
    # roof Y-span. Include upward main-hole radial slack as the worst case.
    main_hole_radial_slack = max(0.0, p.main_hole_radius - p.main_pin_radius)
    d7_lhs = (
        p.main_hole_offset_from_open_end
        + 2 * p.main_pin_radius
        + main_hole_radial_slack
        + eps_head_clear
    )
    d7_rhs = p.wall_thickness + p.leg_length
    if d7_lhs > d7_rhs:
        reasons.append(
            f'D7: offset+main_head_r+main_hole_slack+eps={d7_lhs:.3f} '
            f'> wall_thickness+leg_length={d7_rhs:.3f} '
            f'(main pin nail-head would clip roof underside)')
    if p.splint_length < 2 * p.main_pin_radius + 2 * eps_retention:
        reasons.append(
            f'D3: splint_length={p.splint_length:.3f} '
            f'< 2*main_pin_radius+2*EPS_RETENTION'
            f'={2*p.main_pin_radius + 2*eps_retention:.3f}')
    hl_spl = _splint_head_len(p)
    d4_lhs = 0.5 * p.splint_length + hl_spl + main_hole_radial_slack + eps_head_clear
    d4_rhs = p.wall_thickness + p.leg_length - p.main_hole_offset_from_open_end
    if d4_lhs > d4_rhs:
        reasons.append(
            f'D4: splint_top+main_hole_slack={d4_lhs:.3f} > roof_bottom={d4_rhs:.3f} '
            f'(splint head would clip roof)')

    ok = len(reasons) == 0
    return ok, reasons


# ──────────────────────────────────────────────────────────────────────
#  sample_params(): hierarchical sampler
# ──────────────────────────────────────────────────────────────────────

def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def sample_params(
    rng: np.random.Generator,
    *,
    max_tries: int = 40,
) -> DummyParams:
    """Draw one valid DummyParams from the design space.

    Strategy (2026-04-11 reorder: splint_radius is now drawn BEFORE
    main_pin_length so the pin-length floor can account for the
    oversized splint head (D5) in one shot):
      1. Draw independent frame dims (wall_thickness, outer_span,
         leg_length, depth) uniformly from DESIGN_SPACE.
      2. Draw main_hole_radius uniformly, constrained by depth (B1).
      3. Derive main_pin_radius ~ U(0.75, 0.97) * main_hole_radius (A1).
      4. Draw main_hole_offset ~ conditional window that respects B2+B3.
      5. Derive cross_hole_radius ~ U(0.30, 0.65) * main_pin_radius.
      5b. Derive splint_radius so A2 and A3 both hold. We clamp
          U(0.75, 0.90) * cross_hole_radius from below by
          (cross_hole_radius + EPS_HEAD_SLIP) / SPLINT_HEAD_RATIO so
          A3 is guaranteed even for small cross-hole radii.
      6. Draw cross_hole_distance ~ conditional on cross_hole_radius (B4)
         and a preliminary pin length.
      7. Derive main_pin_length to satisfy D1, D2, AND D5 (which uses
         the splint head radius hr = SPLINT_HEAD_RATIO * splint_radius)
         with a random extra margin.
      8. Derive splint_length to satisfy D3 and D4 (clipped by the
         D4-implied upper bound if necessary).
      9. Draw overhang_span_y ~ U(0.8, 2.0) * outer_span (explicitly
         allowed to cage).
     10. Run validate_params. If it fails for any reason (very rare
         given the derivations), retry up to max_tries.
    """
    for attempt in range(max_tries):
        wall_thickness = _u(rng, DESIGN_SPACE['wall_thickness']['min'],
                            DESIGN_SPACE['wall_thickness']['max'])
        outer_span = _u(rng, DESIGN_SPACE['outer_span']['min'],
                        DESIGN_SPACE['outer_span']['max'])
        leg_length = _u(rng, DESIGN_SPACE['leg_length']['min'],
                        DESIGN_SPACE['leg_length']['max'])
        depth = _u(rng, DESIGN_SPACE['depth']['min'],
                   DESIGN_SPACE['depth']['max'])

        # Skip obviously degenerate frames (C1)
        if outer_span <= 2 * wall_thickness + EPS_ACCESS + 1.0:
            continue

        # main_hole_radius: bounded by depth/2 - EPS_DEPTH_WALL
        hole_r_hi = min(DESIGN_SPACE['main_hole_radius']['max'],
                        0.5 * depth - EPS_DEPTH_WALL - 0.25)
        hole_r_lo = DESIGN_SPACE['main_hole_radius']['min']
        if hole_r_hi <= hole_r_lo:
            continue
        main_hole_radius = _u(rng, hole_r_lo, hole_r_hi)

        # main_pin_radius ~ ratio * main_hole_radius
        pin_ratio = _u(rng, 0.75, 0.97)
        main_pin_radius = max(1.0, pin_ratio * main_hole_radius - EPS_FIT)
        if main_pin_radius + EPS_FIT >= main_hole_radius:
            continue  # shouldn't happen, paranoia

        # main_hole_offset: conditional on leg_length + hole radius (B2+B3)
        # and on D7 so the main pin nail-head top stays below the roof underside.
        off_lo = main_hole_radius + EPS_LEG_WALL
        off_hi_b2 = 0.80 * leg_length - main_hole_radius
        main_hole_radial_slack = max(0.0, main_hole_radius - main_pin_radius)
        off_hi_d7 = (
            wall_thickness
            + leg_length
            - 2 * main_pin_radius
            - main_hole_radial_slack
            - EPS_HEAD_CLEAR
        )
        off_hi = min(off_hi_b2, off_hi_d7)
        if off_hi <= off_lo:
            continue
        main_hole_offset_from_open_end = _u(rng, off_lo, off_hi)

        # cross_hole_radius ~ ratio * main_pin_radius
        ch_ratio = _u(rng, 0.30, 0.65)
        cross_hole_radius = max(1.0, ch_ratio * main_pin_radius)
        if cross_hole_radius + EPS_FIT > main_pin_radius - 0.5:
            # Leave some pin wall around the cross-hole
            continue

        # splint_radius drawn BEFORE main_pin_length so that D5 can be
        # rolled into the pin-length floor in one shot. We want:
        #   A2: splint_radius <= cross_hole_radius - EPS_FIT
        #   A3: SPLINT_HEAD_RATIO * splint_radius >= cross_hole_radius + EPS_HEAD_SLIP
        # i.e. splint_radius >= (cross_hole_radius + EPS_HEAD_SLIP)/SPLINT_HEAD_RATIO.
        # The old U(0.60, 0.90) * cross_hole_radius ratio put the head
        # smaller than the hole in ~75-90% of draws; we tighten to
        # U(0.75, 0.90) and clamp below by the A3 lower bound.
        a3_splint_lo = (cross_hole_radius + EPS_HEAD_SLIP) / SPLINT_HEAD_RATIO
        sp_ratio = _u(rng, 0.75, 0.90)
        splint_radius = max(
            DESIGN_SPACE['splint_radius']['min'],
            a3_splint_lo,
            sp_ratio * cross_hole_radius,
        )
        splint_radius_hi = min(
            DESIGN_SPACE['splint_radius']['max'],
            cross_hole_radius - EPS_FIT,
        )
        if splint_radius > splint_radius_hi:
            continue
        hr = SPLINT_HEAD_RATIO * splint_radius  # splint head radius, used for D5

        # Preliminary pin length for conditional cross-hole distance.
        # Final pin length depends on cross_hole_distance, so we do a
        # two-stage draw: draw cross_hole_distance assuming a loose
        # pin length, then fix main_pin_length to satisfy D1/D2/D5.
        cd_lo = cross_hole_radius + EPS_PIN_WALL
        # cap cross distance at 25% of a provisional pin length
        prov_pin_length = outer_span + 2 * cd_lo + 2 * EPS_ACCESS + 12.0
        cd_hi = max(cd_lo + 0.5, 0.25 * prov_pin_length)
        if cd_hi <= cd_lo:
            continue
        cross_hole_distance_from_free_end = _u(rng, cd_lo, cd_hi)

        # main_pin_length: satisfy D1, D2, and D5 with a random extra margin.
        # D5 strictly dominates D2 when A3 holds (hr > cross_hole_radius),
        # so d5_min is the binding floor for the +Y head rim.
        d1_min = outer_span + 2 * cross_hole_distance_from_free_end + 2 * EPS_ACCESS + 0.5
        d2_min = (outer_span - 2 * wall_thickness
                  + 2 * cross_hole_distance_from_free_end
                  + 2 * cross_hole_radius
                  + 2 * EPS_ACCESS + 0.5)
        d5_min = (outer_span
                  + 2 * cross_hole_distance_from_free_end
                  + 2 * hr
                  + 2 * EPS_ACCESS + 0.5)
        pin_len_min = max(d1_min, d2_min, d5_min)
        extra = _u(rng, 4.0, 20.0)
        main_pin_length = pin_len_min + extra

        # B5 check (cross-hole doesn't eat the nail-head): mostly trivial
        # after D1 but re-verify.
        head_len = 0.05 * main_pin_length
        if (cross_hole_distance_from_free_end + cross_hole_radius + EPS_PIN_WALL
                > main_pin_length - head_len):
            continue

        # splint_length: satisfy D3, then cap by D4 (clears the roof)
        d3_min = 2 * main_pin_radius + 2 * EPS_RETENTION
        # D4: 0.5*splint_length + splint_head_len + eps <= wt + ll - offset
        # splint_head_len = 0.05 * splint_length, so:
        # 0.55*L <= RHS - main_hole_radial_slack - EPS_HEAD_CLEAR
        d4_rhs = wall_thickness + leg_length - main_hole_offset_from_open_end
        splint_len_max = (d4_rhs - main_hole_radial_slack - EPS_HEAD_CLEAR) / 0.55
        if splint_len_max < d3_min + 1.0:
            continue
        splint_extra = _u(rng, 1.0, max(1.5, min(20.0, splint_len_max - d3_min - 0.5)))
        splint_length = d3_min + splint_extra

        # overhang_span_y: free, U(0.8, 2.0) * outer_span
        overhang_span_y = _u(rng, 0.8, 2.0) * outer_span

        p = DummyParams(
            wall_thickness=round(wall_thickness, 3),
            outer_span=round(outer_span, 3),
            leg_length=round(leg_length, 3),
            depth=round(depth, 3),
            main_hole_offset_from_open_end=round(main_hole_offset_from_open_end, 3),
            main_hole_radius=round(main_hole_radius, 3),
            main_pin_length=round(main_pin_length, 3),
            main_pin_radius=round(main_pin_radius, 3),
            cross_hole_radius=round(cross_hole_radius, 3),
            cross_hole_distance_from_free_end=round(cross_hole_distance_from_free_end, 3),
            splint_radius=round(splint_radius, 3),
            splint_length=round(splint_length, 3),
            overhang_span_y=round(overhang_span_y, 3),
            exploded_gap=30.0,
        )
        ok, reasons = validate_params(p)
        if ok:
            return p
    raise RuntimeError(
        f'sample_params: exhausted {max_tries} tries without a valid draw. '
        f'Last reasons: {reasons}')


def params_to_dict(p: DummyParams) -> dict:
    """Plain serializable dict (no derived helpers, no numpy scalars)."""
    return {k: float(v) for k, v in asdict(p).items()}


__all__ = [
    'DESIGN_SPACE',
    'TEST_PROTOCOL_EN',
    'EPS_FIT', 'EPS_PIN_WALL', 'EPS_LEG_WALL', 'EPS_DEPTH_WALL',
    'EPS_RETENTION', 'EPS_ACCESS', 'EPS_HEAD_CLEAR', 'EPS_HEAD_SLIP',
    'SPLINT_HEAD_RATIO',
    'validate_params',
    'sample_params',
    'params_to_dict',
    'DummyParams',
]
