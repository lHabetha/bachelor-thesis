"""Fast analytic voxel overlap labels for the Chapter 6 dummy clevis.

The clevis parts are simple unions/differences of boxes and cylinders. For
label acquisition we can evaluate occupancy directly from `DummyParams` without
exporting meshes. This is much faster than mesh-backed signed distance and is
still calibrated against exact mesh booleans before use as the primary label.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .sampler import DummyParams, SPLINT_HEAD_RATIO

ANALYTIC_LABEL_VERSION = "analytic_voxel_clevis_v1"
PART_PAIRS = (
    ("bracket", "main_pin"),
    ("bracket", "splint"),
    ("main_pin", "splint"),
)


@dataclass(frozen=True)
class AnalyticPairOverlap:
    pair: str
    volume: float
    n_grid_points: int
    pitch_mm: float
    wall_time_s: float


@dataclass(frozen=True)
class AnalyticOverlapLabel:
    label_backend_version: str
    pitch_mm: float
    overlap_threshold_norm: float
    total_part_volume_analytic: float
    total_overlap_volume: float
    total_overlap_norm: float
    overlap_binary: bool
    pairs: list[AnalyticPairOverlap]
    wall_time_s: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairs"] = [asdict(p) for p in self.pairs]
        return d


def _box_occ(points: np.ndarray, center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    c = np.array(center, dtype=float)
    half = np.array(size, dtype=float) / 2.0
    return np.all(np.abs(points - c) <= half, axis=1)


def _cyl_y_occ(points: np.ndarray, center_xz: tuple[float, float], y_min: float, y_max: float, radius: float) -> np.ndarray:
    x0, z0 = center_xz
    radial = (points[:, 0] - x0) ** 2 + (points[:, 2] - z0) ** 2 <= radius**2
    along = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    return radial & along


def _cyl_z_occ(points: np.ndarray, center_xy: tuple[float, float], z_min: float, z_max: float, radius: float) -> np.ndarray:
    x0, y0 = center_xy
    radial = (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 <= radius**2
    along = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return radial & along


def part_bounds(p: DummyParams, part: str) -> tuple[np.ndarray, np.ndarray]:
    t = p.wall_thickness
    if part == "bracket":
        frame_lo = np.array([-p.depth / 2, -p.outer_span / 2, -(t / 2 + p.leg_length)], dtype=float)
        frame_hi = np.array([p.depth / 2, p.outer_span / 2, t / 2], dtype=float)
        roof_lo = np.array([-p.depth / 2, -p.overhang_span_y / 2, t / 2], dtype=float)
        roof_hi = np.array([p.depth / 2, p.overhang_span_y / 2, 1.5 * t], dtype=float)
        return np.minimum(frame_lo, roof_lo), np.maximum(frame_hi, roof_hi)
    if part == "main_pin":
        head_r = 2.0 * p.main_pin_radius
        lo = np.array([-head_r, -p.pin_half, p.hole_z - head_r], dtype=float)
        hi = np.array([head_r, p.pin_half + p.head_len, p.hole_z + head_r], dtype=float)
        return lo, hi
    if part == "splint":
        hr = SPLINT_HEAD_RATIO * p.splint_radius
        hl = 0.05 * p.splint_length
        half = p.splint_length / 2
        lo = np.array([-hr, p.cross_y - hr, p.hole_z - half], dtype=float)
        hi = np.array([hr, p.cross_y + hr, p.hole_z + half + hl], dtype=float)
        return lo, hi
    raise KeyError(part)


def _bracket_occ(p: DummyParams, points: np.ndarray) -> np.ndarray:
    t = p.wall_thickness
    leg_cz = -(t / 2 + p.leg_length / 2)
    wall_cy = p.outer_span / 2 - t / 2
    left = _box_occ(points, (0.0, wall_cy, leg_cz), (p.depth, t, p.leg_length))
    right = _box_occ(points, (0.0, -wall_cy, leg_cz), (p.depth, t, p.leg_length))
    bridge = _box_occ(points, (0.0, 0.0, 0.0), (p.depth, p.outer_span, t))
    frame = left | right | bridge
    main_hole = _cyl_y_occ(
        points,
        (0.0, p.hole_z),
        -p.outer_span / 2 - 1.0,
        p.outer_span / 2 + 1.0,
        p.main_hole_radius,
    )
    roof = _box_occ(points, (0.0, 0.0, p.wall_thickness), (p.depth, max(0.0, p.overhang_span_y), t))
    return (frame & ~main_hole) | roof


def _main_pin_occ(p: DummyParams, points: np.ndarray) -> np.ndarray:
    shaft = _cyl_y_occ(points, (0.0, p.hole_z), -p.pin_half, p.pin_half, p.main_pin_radius)
    head = _cyl_y_occ(
        points,
        (0.0, p.hole_z),
        p.pin_half,
        p.pin_half + p.head_len,
        2.0 * p.main_pin_radius,
    )
    bore = _cyl_z_occ(
        points,
        (0.0, p.cross_y),
        p.hole_z - 1.5 * p.main_pin_radius,
        p.hole_z + 1.5 * p.main_pin_radius,
        p.cross_hole_radius,
    )
    return (shaft | head) & ~bore


def _splint_occ(p: DummyParams, points: np.ndarray) -> np.ndarray:
    half = p.splint_length / 2
    hl = 0.05 * p.splint_length
    hr = SPLINT_HEAD_RATIO * p.splint_radius
    shaft = _cyl_z_occ(
        points,
        (0.0, p.cross_y),
        p.hole_z - half,
        p.hole_z + half,
        p.splint_radius,
    )
    head = _cyl_z_occ(
        points,
        (0.0, p.cross_y),
        p.hole_z + half,
        p.hole_z + half + hl,
        hr,
    )
    return shaft | head


def part_occupancy(p: DummyParams, part: str, points: np.ndarray) -> np.ndarray:
    if part == "bracket":
        return _bracket_occ(p, points)
    if part == "main_pin":
        return _main_pin_occ(p, points)
    if part == "splint":
        return _splint_occ(p, points)
    raise KeyError(part)


def _grid(lo: np.ndarray, hi: np.ndarray, pitch: float, max_points: int) -> tuple[np.ndarray, float]:
    ext = hi - lo
    n = np.maximum(1, np.ceil(ext / pitch).astype(int))
    if int(np.prod(n)) > max_points:
        scale = (int(np.prod(n)) / max_points) ** (1 / 3)
        n = np.maximum(1, np.ceil(ext / (pitch * scale)).astype(int))
    axes = [
        np.linspace(lo[i] + 0.5 * ext[i] / n[i], hi[i] - 0.5 * ext[i] / n[i], n[i])
        for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return points, float(np.prod(ext / n))


def _pair_overlap(p: DummyParams, part_a: str, part_b: str, pitch_mm: float, max_points: int) -> AnalyticPairOverlap:
    start = time.perf_counter()
    lo_a, hi_a = part_bounds(p, part_a)
    lo_b, hi_b = part_bounds(p, part_b)
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    if np.any(hi <= lo):
        return AnalyticPairOverlap(f"{part_a}__{part_b}", 0.0, 0, pitch_mm, time.perf_counter() - start)
    points, cell_volume = _grid(lo, hi, pitch_mm, max_points)
    occ = part_occupancy(p, part_a, points) & part_occupancy(p, part_b, points)
    return AnalyticPairOverlap(
        pair=f"{part_a}__{part_b}",
        volume=float(occ.sum() * cell_volume),
        n_grid_points=int(len(points)),
        pitch_mm=float(pitch_mm),
        wall_time_s=time.perf_counter() - start,
    )


def analytic_part_volume(p: DummyParams, part: str, pitch_mm: float = 0.75, max_points: int = 1_000_000) -> float:
    lo, hi = part_bounds(p, part)
    points, cell_volume = _grid(lo, hi, pitch_mm, max_points)
    return float(part_occupancy(p, part, points).sum() * cell_volume)


def compute_analytic_overlap_label(
    p: DummyParams,
    *,
    pitch_mm: float = 0.75,
    overlap_threshold_norm: float = 1e-8,
    max_points_per_pair: int = 500_000,
) -> AnalyticOverlapLabel:
    start = time.perf_counter()
    pairs = [
        _pair_overlap(p, a, b, pitch_mm, max_points_per_pair)
        for a, b in PART_PAIRS
    ]
    total_overlap = float(sum(pair.volume for pair in pairs))
    # Analytic primitive volumes are exact enough for normalization and avoid a
    # second expensive mesh export during params-only acquisition.
    bracket_vol = (
        2 * p.depth * p.wall_thickness * p.leg_length
        + p.depth * p.outer_span * p.wall_thickness
        + p.depth * max(0.0, p.overhang_span_y) * p.wall_thickness
    )
    pin_vol = np.pi * p.main_pin_radius**2 * p.main_pin_length + np.pi * (2 * p.main_pin_radius) ** 2 * p.head_len
    splint_vol = np.pi * p.splint_radius**2 * p.splint_length + np.pi * (SPLINT_HEAD_RATIO * p.splint_radius) ** 2 * (0.05 * p.splint_length)
    total_part_volume = float(max(1e-9, bracket_vol + pin_vol + splint_vol))
    total_norm = total_overlap / total_part_volume
    return AnalyticOverlapLabel(
        label_backend_version=ANALYTIC_LABEL_VERSION,
        pitch_mm=float(pitch_mm),
        overlap_threshold_norm=float(overlap_threshold_norm),
        total_part_volume_analytic=total_part_volume,
        total_overlap_volume=total_overlap,
        total_overlap_norm=float(total_norm),
        overlap_binary=bool(total_norm > overlap_threshold_norm),
        pairs=pairs,
        wall_time_s=time.perf_counter() - start,
    )
