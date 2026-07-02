"""Flexible per-parameter ADV sampling from a JSON distribution spec."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import random
from typing import Any

from .core import (
    ADV_KEYS,
    DESIGN_NAMES,
    ROOT_NACA,
    TAIL_TYPES,
    TIP_NACA,
    default_params_copy,
)
from .sampler import (
    BACKGROUND_UNIFORM,
    DRIVER_SPECS,
    DistributionState,
    _round_driver,
    _sample_truncated_normal,
    sample_adv_from_distribution,
    sample_adv_wide,
)

# Keys that are always None / fixed conventions
_CSV_PATH_KEYS = {
    "root_csv_filepath",
    "tip_csv_filepath",
    "hstab_root_csv_filepath",
    "hstab_tip_csv_filepath",
    "vstab_root_csv_filepath",
    "vstab_tip_csv_filepath",
}
_FIXED_STRING_DEFAULTS = {
    "airfoil_source": "naca_4",
    "hstab_airfoil_source": "naca_4",
    "vstab_airfoil_source": "naca_4",
    "hstab_root_naca_code": "0012",
    "hstab_tip_naca_code": "0008",
    "vstab_root_naca_code": "0012",
    "vstab_tip_naca_code": "0008",
}

_NUMERIC_KEYS = set(DRIVER_SPECS) | set(BACKGROUND_UNIFORM)
_CATEGORICAL_KEYS = {"design_name", "tail_type", "root_naca_code", "tip_naca_code"}


def _key_range(key: str) -> tuple[float, float]:
    if key in DRIVER_SPECS:
        s = DRIVER_SPECS[key]
        return s["lo"], s["hi"]
    if key in BACKGROUND_UNIFORM:
        return BACKGROUND_UNIFORM[key]
    return 0.0, 1.0


def _sample_driver_spec(rng: random.Random, key: str, spec: dict[str, Any]) -> Any:
    dist = spec.get("dist", "uniform")
    if dist == "fixed":
        return spec["value"]
    if dist == "choice":
        values = spec["values"]
        weights = spec.get("weights")
        if weights:
            return rng.choices(values, weights=weights, k=1)[0]
        return rng.choice(values)
    lo, hi = spec.get("lo", _key_range(key)[0]), spec.get("hi", _key_range(key)[1])
    if dist == "uniform":
        val = rng.uniform(lo, hi)
    elif dist == "normal":
        mean = spec.get("mean", (lo + hi) / 2)
        std = spec.get("std", (hi - lo) / 6)
        val = _sample_truncated_normal(rng, mean, std, lo, hi)
    else:
        raise ValueError(f"Unknown dist '{dist}' for key '{key}'")
    if key in _NUMERIC_KEYS:
        return _round_driver(key, val)
    return val


def _apply_defaults(rng: random.Random, p: dict, mode: str) -> dict:
    if mode == "wide":
        base = sample_adv_wide(rng)
        p.update({k: base[k] for k in ADV_KEYS if k not in p or p.get(k) is None})
    elif mode == "broad":
        dist = DistributionState.broad_initial()
        base = sample_adv_from_distribution(rng, dist)
        p.update({k: base[k] for k in ADV_KEYS if k not in p or p.get(k) is None})
    elif mode == "aeroforge_default":
        pass  # already from default_params_copy
    else:
        raise ValueError(f"Unknown defaults mode: {mode}")
    return p


def _finalize_conventions(p: dict) -> dict:
    for k in _CSV_PATH_KEYS:
        p[k] = None
    for k, v in _FIXED_STRING_DEFAULTS.items():
        if k not in p or p[k] is None:
            p[k] = v
    if p.get("tail_type") != "v_tail":
        p["v_tail_angle"] = 0.0
    elif not p.get("v_tail_angle"):
        p["v_tail_angle"] = 115.0
    return p


def sample_from_spec(rng: random.Random, spec: dict[str, Any]) -> dict:
    """Build one ADV from a flexible distribution spec.

    spec keys: iteration, name, strategy, hypothesis, defaults, drivers{key: dist_spec}
    """
    defaults = spec.get("defaults", "wide")
    drivers = spec.get("drivers") or {}

    if defaults == "aeroforge_default":
        p = default_params_copy()
        for key, ds in drivers.items():
            p[key] = _sample_driver_spec(rng, key, ds)
        return _finalize_conventions(p)

    p: dict[str, Any] = {}
    for key, ds in drivers.items():
        p[key] = _sample_driver_spec(rng, key, ds)

    p = _apply_defaults(rng, p, defaults)
    for key, ds in drivers.items():
        p[key] = _sample_driver_spec(rng, key, ds)

    return _finalize_conventions(p)
