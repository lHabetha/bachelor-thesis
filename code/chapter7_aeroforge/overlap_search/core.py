"""Core ADV build + CadQuery overlap labeling helpers."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import copy
import os
import sys
import time
from pathlib import Path

from chapter7_aeroforge.release_paths import AEROFORGE_ROOT
_OVERLAP_THRESHOLD_MM3 = 1.0

ADV_KEYS = [
    "length",
    "max_width",
    "max_height",
    "wingspan",
    "wing_position",
    "wing_height_ratio",
    "tail_position",
    "design_name",
    "wall_thickness",
    "end_cap_percent",
    "aspect_ratio",
    "taper",
    "sweep",
    "dihedral",
    "twist",
    "root_incidence",
    "airfoil_source",
    "root_naca_code",
    "tip_naca_code",
    "root_csv_filepath",
    "tip_csv_filepath",
    "tail_type",
    "v_tail_angle",
    "hstab_semispan",
    "hstab_aspect_ratio",
    "hstab_taper",
    "hstab_sweep",
    "hstab_dihedral",
    "hstab_root_incidence",
    "hstab_airfoil_source",
    "hstab_root_naca_code",
    "hstab_tip_naca_code",
    "hstab_root_csv_filepath",
    "hstab_tip_csv_filepath",
    "vstab_height",
    "vstab_aspect_ratio",
    "vstab_taper",
    "vstab_sweep",
    "vstab_airfoil_source",
    "vstab_root_naca_code",
    "vstab_tip_naca_code",
    "vstab_root_csv_filepath",
    "vstab_tip_csv_filepath",
]

ACTIVE_DRIVERS = [
    "wing_position",
    "tail_position",
    "length",
    "aspect_ratio",
    "taper",
    "sweep",
    "wingspan",
    "hstab_semispan",
    "hstab_aspect_ratio",
    "hstab_taper",
    "hstab_sweep",
    "wing_height_ratio",
]

DESIGN_NAMES = ["standard", "simple", "transport", "racing", "blended"]
TAIL_TYPES = ["conv_tail", "v_tail"]
ROOT_NACA = ["2412", "2410", "0012"]
TIP_NACA = ["2412", "2408", "0009"]

UAVAssembly = None
DEFAULT_PARAMS: dict | None = None


def ensure_aeroforge_imports() -> None:
    """Import AeroForge modules after chdir (safe for spawn workers)."""
    global UAVAssembly, DEFAULT_PARAMS
    if UAVAssembly is not None:
        return
    sys.path.insert(0, str(AEROFORGE_ROOT))
    os.chdir(AEROFORGE_ROOT)
    from uav_assembly import UAVAssembly as _UAVAssembly  # noqa: PLC0415
    from uav_params import uav_params as _default  # noqa: PLC0415

    UAVAssembly = _UAVAssembly
    DEFAULT_PARAMS = _default


def limit_occt_threads(n: int = 1) -> bool:
    """Pin OCCT's internal thread pool to `n` threads for this process.

    CRITICAL for multiprocessing: by default OCCT's OSD_ThreadPool sizes to the
    full logical CPU count (e.g. 18), so every worker process tries to parallelize
    its boolean/mesh ops across ALL cores. With W workers this oversubscribes to
    W*cores threads and inflates per-task wall time ~8x. Pinning each worker to a
    single OCCT thread lets W workers cleanly occupy W cores with no contention.

    Must be called inside each worker (after OCP is importable). Returns True on
    success. Standard OMP/BLAS env vars do NOT affect OCCT — this is the real knob.
    """
    try:
        from OCP.OSD import OSD_ThreadPool  # noqa: PLC0415

        pool = OSD_ThreadPool.DefaultPool_s()
        pool.Init(max(1, int(n)))
        pool.SetNbDefaultThreadsToLaunch(max(1, int(n)))
        return True
    except Exception:  # noqa: BLE001
        return False


def overlap_threshold_mm3() -> float:
    return _OVERLAP_THRESHOLD_MM3


def is_overlap(overlap_mm3: float | None) -> bool:
    return overlap_mm3 is not None and overlap_mm3 > _OVERLAP_THRESHOLD_MM3


def normalize_driver(key: str, value: float) -> float:
    """Map driver value to [0, 1] using constructible range from sampler specs."""
    from .sampler import DRIVER_SPECS  # noqa: PLC0415

    spec = DRIVER_SPECS[key]
    lo, hi = spec["lo"], spec["hi"]
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def adv_subset(params: dict) -> dict:
    return {k: params[k] for k in ADV_KEYS if k in params}


def build_airframe(params: dict):
    ensure_aeroforge_imports()
    uav = UAVAssembly(params, {})
    t0 = time.perf_counter()
    uav.create_fuselage()
    t_fus = time.perf_counter()
    uav.create_wings()
    t_wing = time.perf_counter()
    uav.create_tail()
    t_tail = time.perf_counter()
    timings = {
        "fuselage_s": t_fus - t0,
        "wings_s": t_wing - t_fus,
        "tail_s": t_tail - t_wing,
        "build_total_s": t_tail - t0,
    }
    return uav, timings


def _solid_volume(solid) -> float:
    if solid is None:
        return 0.0
    val = solid.val() if hasattr(solid, "val") else solid
    if val is None:
        return 0.0
    return float(val.Volume())


def cadquery_overlap_volume(wings, tail) -> tuple[float, float, float]:
    wing_vol = float(wings.val().Volume())
    tail_vol = float(tail.val().Volume())
    inter = wings.intersect(tail)
    inter_vol = _solid_volume(inter)
    return inter_vol, wing_vol, tail_vol


def cadquery_extrinsic_overlap_volume(wings, tail, fuselage_outer) -> dict[str, float]:
    """Wing-tail overlap outside fuselage_outer solid (raw minus in-body intersection)."""
    wing_vol = float(wings.val().Volume())
    tail_vol = float(tail.val().Volume())
    raw_inter = wings.intersect(tail)
    raw_vol = _solid_volume(raw_inter)
    extrinsic_inter = raw_inter.cut(fuselage_outer)
    extrinsic_vol = _solid_volume(extrinsic_inter)
    masked = max(0.0, raw_vol - extrinsic_vol)
    return {
        "overlap_mm3_raw": raw_vol,
        "overlap_mm3_extrinsic": extrinsic_vol,
        "overlap_mm3_fuselage_masked": masked,
        "wing_volume_mm3": wing_vol,
        "tail_volume_mm3": tail_vol,
    }


def cut_fuselage_from_parts(wings, tail, fuselage_outer):
    """Cut the fuselage solid out of wings and tail (cut-from-part rule).

    Implements "overlap between the fuselage and a lifting surface is removed
    from the lifting surface, not the fuselage". Returns (wings_ext, tail_ext),
    each the part trimmed to the fuselage outer surface.
    """
    wings_ext = wings.cut(fuselage_outer)
    tail_ext = tail.cut(fuselage_outer)
    return wings_ext, tail_ext


def cadquery_cut_each_overlap_volume(wings, tail, fuselage_outer) -> dict[str, float]:
    """Extrinsic wing-tail overlap via (wings - fuselage) INTERSECT (tail - fuselage).

    Set-theoretically identical to cadquery_extrinsic_overlap_volume's extrinsic
    term, but computed by cutting the fuselage from each part first, then
    intersecting. This is numerically more robust than intersecting first and
    cutting the (often degenerate) intersection solid, and it can never exceed
    the raw wing-tail overlap (overlap <= raw always).
    """
    raw_inter = wings.intersect(tail)
    raw_vol = _solid_volume(raw_inter)
    wings_ext, tail_ext = cut_fuselage_from_parts(wings, tail, fuselage_outer)
    inter = wings_ext.intersect(tail_ext)
    overlap_vol = _solid_volume(inter)
    return {
        "overlap_mm3_raw": raw_vol,
        "overlap_mm3_cut_each": overlap_vol,
        "wings_ext_volume_mm3": _solid_volume(wings_ext),
        "tail_ext_volume_mm3": _solid_volume(tail_ext),
    }


def cadquery_cut_each_overlap_fast(
    wings, tail, fuselage_outer, threshold: float = _OVERLAP_THRESHOLD_MM3
) -> dict[str, float]:
    """Fast, robust extrinsic wing-tail overlap = (wings ∩ tail) − fuselage.

    Two optimizations over `cadquery_cut_each_overlap_volume`, both exact:

    1. Early-exit: (W−F)∩(T−F) = (W∩T)−F ⊆ (W∩T), so any design with raw
       wing-tail intersect <= threshold has zero extrinsic overlap. Compute the
       one cheap raw intersect first and skip everything else (~no-overlap cases).
    2. Cut the *small* raw-intersection solid by the fuselage instead of cutting
       the two *big* wing/tail solids. `raw_inter.cut(F)` is set-theoretically
       identical to the cut-each result but far cheaper (small operand).

    The v1.1 "OCCT boolean garbage" (cut of a near-empty intersect returning a
    huge volume) is avoided because the early-exit removes degenerate inputs;
    a guard additionally falls back to the cut-each form if OCCT ever returns
    overlap > raw (impossible analytically). `raw_exit`/`fallback` flag the path.
    """
    raw_inter = wings.intersect(tail)
    raw_vol = _solid_volume(raw_inter)
    if raw_vol <= threshold:
        return {"overlap_mm3_raw": raw_vol, "overlap_mm3_cut_each": 0.0, "raw_exit": True}

    overlap_vol = _solid_volume(raw_inter.cut(fuselage_outer))
    if overlap_vol > raw_vol * 1.001:  # OCCT garbage guard; analytically impossible
        wings_ext, tail_ext = cut_fuselage_from_parts(wings, tail, fuselage_outer)
        overlap_vol = min(raw_vol, _solid_volume(wings_ext.intersect(tail_ext)))
        return {
            "overlap_mm3_raw": raw_vol,
            "overlap_mm3_cut_each": overlap_vol,
            "raw_exit": False,
            "fallback": True,
        }
    return {
        "overlap_mm3_raw": raw_vol,
        "overlap_mm3_cut_each": overlap_vol,
        "raw_exit": False,
        "fallback": False,
    }


def label_sample(params: dict, *, extrinsic_primary: bool = True) -> dict:
    """Build stepwise + CadQuery wings-tail overlap label (no STL export)."""
    t0 = time.perf_counter()
    row: dict = {
        "build_ok": False,
        "overlap_mm3": None,
        "overlap_mm3_raw": None,
        "overlap_mm3_extrinsic": None,
        "overlap_mm3_fuselage_masked": None,
        "overlap_norm": None,
        "overlap_norm_extrinsic": None,
        "wing_volume_mm3": None,
        "tail_volume_mm3": None,
        "is_overlap": False,
        "is_overlap_extrinsic": False,
        "error": None,
        "adv": adv_subset(params),
        "timings": {},
    }
    try:
        uav, timings = build_airframe(params)
        row["timings"] = timings
        wings = uav.components["wings"]
        tail = uav.components["tail"]
        fuselage = uav.components["fuselage_outer"]
        t_label = time.perf_counter()
        metrics = cadquery_extrinsic_overlap_volume(wings, tail, fuselage)
        row["timings"]["cq_overlap_s"] = time.perf_counter() - t_label
        row["timings"]["total_s"] = time.perf_counter() - t0
        row["build_ok"] = True
        row["overlap_mm3_raw"] = metrics["overlap_mm3_raw"]
        row["overlap_mm3_extrinsic"] = metrics["overlap_mm3_extrinsic"]
        row["overlap_mm3_fuselage_masked"] = metrics["overlap_mm3_fuselage_masked"]
        row["wing_volume_mm3"] = metrics["wing_volume_mm3"]
        row["tail_volume_mm3"] = metrics["tail_volume_mm3"]
        denom = metrics["wing_volume_mm3"] + metrics["tail_volume_mm3"]
        row["overlap_norm"] = (
            metrics["overlap_mm3_extrinsic"] / denom if denom > 0 else 0.0
        )
        row["overlap_norm_extrinsic"] = row["overlap_norm"]
        row["is_overlap"] = is_overlap(metrics["overlap_mm3_raw"])
        row["is_overlap_extrinsic"] = is_overlap(metrics["overlap_mm3_extrinsic"])
        if extrinsic_primary:
            row["overlap_mm3"] = metrics["overlap_mm3_extrinsic"]
        else:
            row["overlap_mm3"] = metrics["overlap_mm3_raw"]
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["timings"]["total_s"] = time.perf_counter() - t0
    return row


def default_params_copy() -> dict:
    ensure_aeroforge_imports()
    return copy.deepcopy(DEFAULT_PARAMS)
