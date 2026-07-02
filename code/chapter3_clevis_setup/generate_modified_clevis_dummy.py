"""Modified clevis dummy assembly generator.

Produces three rigid parts (bracket, main_pin, splint) for path-planning
assemblability experiments.  Run directly to generate all output files.

Coordinate system (verified by trimesh bounding-box check after every run):

    Y-axis = main pin axis.  Nail-head at +Y, free tip at -Y.
    Z-axis = vertical.       Bridge at +Z top, legs grow downward (-Z).
    X-axis = depth            (front-to-back).

    Splint axis is +Z, through the cross-hole near the pin's -Y free end.

    CadQuery gotcha -- Workplane("XZ") normal = -Y:
        offset = +V  -->  world y = -V      extrude advances toward -Y
        offset = -V  -->  world y = +V      extrude advances toward -Y
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cadquery as cq

# ── User toggle ─────────────────────────────────────────────────────────────
EXPORT_SUBPARTS = True  # False = only assembled / exploded previews

# ── Geometry constants ──────────────────────────────────────────────────────
# Retention ratio for the splint's +Z head disc. 2026-04-11: bumped from 1.2
# to 1.5 after discovering head=1.2*splint_radius was smaller than the
# cross-hole in ~75-90% of random samples, defeating retention (the splint
# head walked right through the hole during disassembly). Any consumer that
# needs the same ratio must read this constant rather than hard-code 1.5 or 1.2.
SPLINT_HEAD_RATIO: float = 1.5

# ── Per-part colors for multi-format colored previews ───────────────────────
_PART_COLORS_RGBA: dict[str, tuple[int, int, int, int]] = {
    "bracket":  (90, 160, 200, 255),
    "main_pin": (180, 180, 190, 255),
    "splint":   (230, 80, 120, 255),
}


# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DummyParams:
    """All dimensions in millimeters."""

    # U-frame (upside-down U: bridge on top +Z, legs toward -Z)
    wall_thickness: float = 8.0   # uniform wall / bridge / roof thickness
    outer_span:     float = 60.0  # outside-to-outside distance between legs (Y)
    leg_length:     float = 50.0  # how far legs extend downward from the bridge
    depth:          float = 24.0  # frame extent in X (front-to-back)

    # Through-hole in both legs for the main pin (axis Y)
    main_hole_offset_from_open_end: float = 20.0  # from open leg tips upward to hole center
    main_hole_radius:               float = 4.0   # clearance-hole radius

    # Main pin (axis Y; nail-head at +Y, free tip at -Y)
    main_pin_length: float = 90.0  # shaft length (must exceed outer_span)
    main_pin_radius: float = 3.95  # shaft radius (must be <= main_hole_radius)

    # Cross-hole through pin for the splint (axis Z)
    cross_hole_radius:              float = 2.5  # bore radius
    cross_hole_distance_from_free_end: float = 5.0  # from -Y free tip inward to hole center

    # Splint through cross-hole (axis +Z; head toward bridge/roof)
    splint_radius: float = 2.2   # shaft radius (must be < cross_hole_radius)
    splint_length: float = 30.0  # shaft length along Z

    # Roof slab (fused into bracket; sits on top of bridge)
    overhang_span_y: float = 110.0  # total Y-extent; > outer_span means overhang

    # Preview spacing
    exploded_gap: float = 30.0  # spacing for exploded-view translation

    # ── derived helpers (not parameters) ────────────────────────────────────

    @property
    def pin_half(self) -> float:
        return self.main_pin_length / 2.0

    @property
    def hole_z(self) -> float:
        """World Z of the main pin axis (center of the through-hole)."""
        open_end_z = -(self.wall_thickness / 2.0 + self.leg_length)
        return open_end_z + self.main_hole_offset_from_open_end

    @property
    def cross_y(self) -> float:
        """World Y of the splint cross-hole center (negative = -Y side)."""
        return -(self.pin_half - self.cross_hole_distance_from_free_end)

    @property
    def head_len(self) -> float:
        return 0.05 * self.main_pin_length

    @property
    def head_r(self) -> float:
        return 2.0 * self.main_pin_radius


# ═══════════════════════════════════════════════════════════════════════════
# Validation & debug
# ═══════════════════════════════════════════════════════════════════════════

def _validate_params(p: DummyParams) -> list[str]:
    w: list[str] = []
    wall_outer_neg = -(p.outer_span / 2.0)
    wall_inner_neg = -(p.outer_span / 2.0 - p.wall_thickness)

    if p.outer_span <= 2.0 * p.wall_thickness:
        w.append("outer_span <= 2*wall_thickness: no inner gap between legs.")
    if p.main_hole_radius * 2.0 >= p.depth:
        w.append("main_hole_diameter >= depth: hole breaks out of frame front/back.")
    if p.main_pin_length <= p.outer_span:
        w.append("main_pin_length <= outer_span: pin does not protrude past walls.")
    if p.main_pin_radius > p.main_hole_radius:
        w.append("main_pin_radius > main_hole_radius: pin cannot fit through holes.")
    if p.splint_radius >= p.cross_hole_radius:
        w.append("splint_radius >= cross_hole_radius: splint won't fit.")
    if wall_outer_neg <= p.cross_y <= wall_inner_neg:
        w.append(f"cross_y={p.cross_y:.1f} inside -Y wall: splint buried in material.")
    if p.cross_y <= -p.pin_half + p.splint_radius:
        w.append("Cross-hole too close to -Y tip: splint overhangs the free end.")
    if p.cross_y >= p.pin_half:
        w.append("Cross-hole beyond +Y head end of pin.")
    if p.overhang_span_y < 0.0:
        w.append("overhang_span_y < 0.")
    return w


def _print_layout(p: DummyParams) -> None:
    print("  Y-axis layout (pin axis):")
    print(f"    free tip (-Y):   y = {-p.pin_half:.1f}")
    print(f"    SPLINT center:   y = {p.cross_y:.1f}")
    print(f"    -Y wall outer:   y = {-p.outer_span / 2:.1f}")
    print(f"    center:          y = 0")
    print(f"    +Y wall outer:   y = {p.outer_span / 2:.1f}")
    print(f"    nail-head (+Y):  y = {p.pin_half:.1f} .. {p.pin_half + p.head_len:.1f}")
    print(f"  Z-axis: hole_z = {p.hole_z:.1f}")
    print(f"    splint Z range:  [{p.hole_z - p.splint_length/2:.1f}, "
          f"{p.hole_z + p.splint_length/2:.1f}]")


# ═══════════════════════════════════════════════════════════════════════════
# Geometry builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_u_frame(p: DummyParams) -> cq.Workplane:
    """Upside-down U: bridge at z ~ 0, legs toward -Z.  Through-hole cut."""
    t = p.wall_thickness
    leg_cz = -(t / 2.0 + p.leg_length / 2.0)
    wall_cy = p.outer_span / 2.0 - t / 2.0

    left  = cq.Workplane("XY").box(p.depth, t, p.leg_length).translate((0, +wall_cy, leg_cz))
    right = cq.Workplane("XY").box(p.depth, t, p.leg_length).translate((0, -wall_cy, leg_cz))
    bridge = cq.Workplane("XY").box(p.depth, p.outer_span, t)
    frame = left.union(right).union(bridge)

    hole_cyl = (
        cq.Workplane("XZ")
        .workplane(offset=-p.outer_span / 2.0 - 1.0)
        .center(0.0, p.hole_z)
        .circle(p.main_hole_radius)
        .extrude(p.outer_span + 2.0)
    )
    return frame.cut(hole_cyl)


def _build_roof(p: DummyParams) -> cq.Workplane:
    """Slab on top of the bridge.  Bottom at z = wall_thickness/2."""
    span = max(float(p.overhang_span_y), 0.0)
    t = p.wall_thickness
    if span < 1e-9:
        return cq.Workplane("XY").box(1e-3, 1e-3, 1e-3).translate((0, 0, t))
    return cq.Workplane("XY").box(p.depth, span, t).translate((0, 0, t))


def build_bracket(p: DummyParams) -> cq.Workplane:
    """Single rigid part: U-frame fused with roof slab."""
    return _build_u_frame(p).union(_build_roof(p))


def build_main_pin(p: DummyParams) -> cq.Workplane:
    """Pin along Y with nail-head at +Y and cross-hole near -Y free end.

    Workplane("XZ") normal = -Y, so:
      offset = -(half)           -->  world y = +half   (shaft start)
      extrude(length) goes -Y    -->  shaft ends at y = +half - length
      offset = -(half + head_r)  -->  world y = +(half + head_r)  (head start)
    """
    h = p.pin_half

    shaft = (
        cq.Workplane("XZ").workplane(offset=-h)
        .center(0, p.hole_z).circle(p.main_pin_radius)
        .extrude(p.main_pin_length)
    )
    head = (
        cq.Workplane("XZ").workplane(offset=-(h + p.head_len))
        .center(0, p.hole_z).circle(p.head_r)
        .extrude(p.head_len)
    )
    bore = (
        cq.Workplane("XY").workplane(offset=p.hole_z)
        .center(0, p.cross_y).circle(p.cross_hole_radius)
        .extrude(p.main_pin_radius * 3.0, both=True)
    )
    return shaft.union(head).cut(bore)


def build_splint(p: DummyParams) -> cq.Workplane:
    """Splint along +Z through cross-hole; small head on +Z side."""
    hl = 0.05 * p.splint_length
    hr = SPLINT_HEAD_RATIO * p.splint_radius
    half_l = p.splint_length / 2.0

    shaft = (
        cq.Workplane("XY").workplane(offset=p.hole_z - half_l)
        .center(0, p.cross_y).circle(p.splint_radius)
        .extrude(p.splint_length)
    )
    head = (
        cq.Workplane("XY").workplane(offset=p.hole_z + half_l)
        .center(0, p.cross_y).circle(hr)
        .extrude(hl)
    )
    return shaft.union(head)


def build_parts(p: DummyParams) -> dict[str, cq.Workplane]:
    return {
        "bracket":  build_bracket(p),
        "main_pin": build_main_pin(p),
        "splint":   build_splint(p),
    }
