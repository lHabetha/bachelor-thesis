"""ADV distribution sampling and CEM-style adaptation."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import copy
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .core import (
    ACTIVE_DRIVERS,
    DESIGN_NAMES,
    ROOT_NACA,
    TAIL_TYPES,
    TIP_NACA,
    default_params_copy,
)

# Constructible ranges + broad overlap-leaning centers (truncated normal drivers).
DRIVER_SPECS: dict[str, dict[str, float]] = {
    "wing_position": {"lo": 0.40, "hi": 0.70, "broad_mean": 0.58, "broad_std": 0.08},
    "tail_position": {"lo": 0.82, "hi": 0.98, "broad_mean": 0.90, "broad_std": 0.04},
    "length": {"lo": 800.0, "hi": 1400.0, "broad_mean": 1050.0, "broad_std": 120.0},
    "aspect_ratio": {"lo": 5.5, "hi": 11.0, "broad_mean": 7.0, "broad_std": 1.2},
    "taper": {"lo": 0.35, "hi": 0.55, "broad_mean": 0.48, "broad_std": 0.05},
    "sweep": {"lo": 12.0, "hi": 28.0, "broad_mean": 20.0, "broad_std": 4.0},
    "wingspan": {"lo": 1200.0, "hi": 2000.0, "broad_mean": 1700.0, "broad_std": 180.0},
    "hstab_semispan": {"lo": 200.0, "hi": 350.0, "broad_mean": 290.0, "broad_std": 35.0},
    "hstab_aspect_ratio": {"lo": 2.5, "hi": 5.0, "broad_mean": 3.2, "broad_std": 0.6},
    "hstab_taper": {"lo": 0.35, "hi": 0.55, "broad_mean": 0.48, "broad_std": 0.05},
    "hstab_sweep": {"lo": 15.0, "hi": 30.0, "broad_mean": 24.0, "broad_std": 4.0},
    "wing_height_ratio": {"lo": 0.35, "hi": 0.65, "broad_mean": 0.50, "broad_std": 0.08},
}

BACKGROUND_UNIFORM: dict[str, tuple[float, float]] = {
    "max_width": (100.0, 200.0),
    "max_height": (75.0, 140.0),
    "dihedral": (0.0, 10.0),
    "twist": (0.0, 4.0),
    "root_incidence": (-3.0, 5.0),
    "hstab_dihedral": (0.0, 8.0),
    "hstab_root_incidence": (-6.0, 2.0),
    "vstab_height": (110.0, 230.0),
    "vstab_aspect_ratio": (1.1, 2.2),
    "vstab_taper": (0.35, 0.65),
    "vstab_sweep": (18.0, 42.0),
    "v_tail_angle": (100.0, 130.0),
    "wall_thickness": (3.0, 8.0),
    "end_cap_percent": (0.03, 0.08),
}

# Per-iteration schedule: (phase, alpha, std_mult)
ITERATION_SCHEDULE = [
    ("BROAD", 0.0, 1.0),
    ("BROAD", 0.0, 1.0),
    ("BROAD", 0.0, 1.0),
    ("BROAD", 0.0, 1.0),
    ("TRANSITION", 0.30, 0.80),
    ("TRANSITION", 0.45, 0.70),
    ("TRANSITION", 0.60, 0.60),
    ("NARROW", 0.80, 0.45),
    ("NARROW", 0.90, 0.38),
    ("NARROW", 1.00, 0.30),
]

MIN_ARCHIVE_FOR_ADAPT = 5


@dataclass
class DriverState:
    mean: float
    std: float
    lo: float
    hi: float
    broad_mean: float
    broad_std: float


@dataclass
class DistributionState:
    drivers: dict[str, DriverState] = field(default_factory=dict)
    iteration: int = 0

    @classmethod
    def broad_initial(cls) -> DistributionState:
        drivers = {}
        for key, spec in DRIVER_SPECS.items():
            drivers[key] = DriverState(
                mean=spec["broad_mean"],
                std=spec["broad_std"],
                lo=spec["lo"],
                hi=spec["hi"],
                broad_mean=spec["broad_mean"],
                broad_std=spec["broad_std"],
            )
        return cls(drivers=drivers, iteration=0)

    def to_json(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "drivers": {k: asdict(v) for k, v in self.drivers.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DistributionState:
        drivers = {
            k: DriverState(**v) for k, v in data.get("drivers", {}).items()
        }
        return cls(drivers=drivers, iteration=data.get("iteration", 0))


def _sample_truncated_normal(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    std = max(std, 1e-6)
    for _ in range(200):
        x = rng.gauss(mean, std)
        if lo <= x <= hi:
            return x
    return max(lo, min(hi, mean))


def _round_driver(key: str, value: float) -> float:
    if key in {"length", "wingspan", "hstab_semispan", "vstab_height"}:
        return float(int(round(value)))
    if key in {"wing_position", "tail_position", "wing_height_ratio", "taper", "hstab_taper", "vstab_taper", "end_cap_percent"}:
        return round(value, 3)
    if key in {"wall_thickness"}:
        return round(value, 2)
    return round(value, 2)


def sample_adv_from_distribution(rng: random.Random, dist: DistributionState) -> dict:
    p = default_params_copy()
    for key, state in dist.drivers.items():
        p[key] = _round_driver(key, _sample_truncated_normal(rng, state.mean, state.std, state.lo, state.hi))

    for key, (lo, hi) in BACKGROUND_UNIFORM.items():
        p[key] = round(rng.uniform(lo, hi), 2 if key != "end_cap_percent" else 3)

    p["design_name"] = rng.choice(DESIGN_NAMES)
    p["tail_type"] = rng.choice(TAIL_TYPES)
    p["v_tail_angle"] = round(rng.uniform(100.0, 130.0), 1) if p["tail_type"] == "v_tail" else 0.0

    p["airfoil_source"] = "naca_4"
    p["root_naca_code"] = rng.choice(ROOT_NACA)
    p["tip_naca_code"] = rng.choice(TIP_NACA)
    p["root_csv_filepath"] = None
    p["tip_csv_filepath"] = None
    p["hstab_airfoil_source"] = "naca_4"
    p["hstab_root_naca_code"] = "0012"
    p["hstab_tip_naca_code"] = "0008"
    p["hstab_root_csv_filepath"] = None
    p["hstab_tip_csv_filepath"] = None
    p["vstab_airfoil_source"] = "naca_4"
    p["vstab_root_naca_code"] = "0012"
    p["vstab_tip_naca_code"] = "0008"
    p["vstab_root_csv_filepath"] = None
    p["vstab_tip_csv_filepath"] = None
    return p


def sample_adv_wide(rng: random.Random) -> dict:
    """Maximum-diversity ADV: every active driver sampled UNIFORMLY across its full
    constructible [lo, hi] range (vs the narrow truncated-normals used by the CEM
    search). Background numerics and categoricals stay uniform/random. Use this for
    coverage/stress runs where the distribution should be as wide as possible.
    """
    p = default_params_copy()
    for key, spec in DRIVER_SPECS.items():
        p[key] = _round_driver(key, rng.uniform(spec["lo"], spec["hi"]))

    for key, (lo, hi) in BACKGROUND_UNIFORM.items():
        p[key] = round(rng.uniform(lo, hi), 2 if key != "end_cap_percent" else 3)

    p["design_name"] = rng.choice(DESIGN_NAMES)
    p["tail_type"] = rng.choice(TAIL_TYPES)
    p["v_tail_angle"] = round(rng.uniform(100.0, 130.0), 1) if p["tail_type"] == "v_tail" else 0.0

    p["airfoil_source"] = "naca_4"
    p["root_naca_code"] = rng.choice(ROOT_NACA)
    p["tip_naca_code"] = rng.choice(TIP_NACA)
    p["root_csv_filepath"] = None
    p["tip_csv_filepath"] = None
    p["hstab_airfoil_source"] = "naca_4"
    p["hstab_root_naca_code"] = "0012"
    p["hstab_tip_naca_code"] = "0008"
    p["hstab_root_csv_filepath"] = None
    p["hstab_tip_csv_filepath"] = None
    p["vstab_airfoil_source"] = "naca_4"
    p["vstab_root_naca_code"] = "0012"
    p["vstab_tip_naca_code"] = "0008"
    p["vstab_root_csv_filepath"] = None
    p["vstab_tip_csv_filepath"] = None
    return p


def adapt_distribution(
    dist: DistributionState,
    overlap_archive: list[dict],
    iteration: int,
) -> DistributionState:
    """Return updated distribution after iteration `iteration` completes."""
    phase, alpha, std_mult = ITERATION_SCHEDULE[iteration]
    new_dist = copy.deepcopy(dist)
    new_dist.iteration = iteration + 1

    if phase == "BROAD" or alpha <= 0.0:
        return new_dist

    if len(overlap_archive) < MIN_ARCHIVE_FOR_ADAPT:
        return new_dist

    for key in ACTIVE_DRIVERS:
        values = [
            row["adv"][key]
            for row in overlap_archive
            if row.get("adv") and key in row["adv"]
        ]
        if len(values) < MIN_ARCHIVE_FOR_ADAPT:
            continue
        archive_mean = sum(values) / len(values)
        state = new_dist.drivers[key]
        state.mean = (1.0 - alpha) * state.mean + alpha * archive_mean
        state.mean = max(state.lo, min(state.hi, state.mean))
        state.std = max(state.broad_std * std_mult, (state.hi - state.lo) * 0.02)
    return new_dist


def schedule_for_iteration(iteration: int) -> tuple[str, float, float]:
    return ITERATION_SCHEDULE[iteration]
