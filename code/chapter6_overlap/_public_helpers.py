"""Small shared helpers used by the Chapter 6 repair and zigzag scripts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .analytic_overlap import compute_analytic_overlap_label
from .label_cache import DEFAULT_PITCH_MM, STRICT_THRESHOLD_NORM, params_from_row
from .models import OverlapMLP, Standardizer
from .regression_metrics import continuous_label_from_payload
from .sampler import DummyParams, compute_margins
from .release_paths import ensure_chapter3_importable

FEATURES = [
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
]


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _features(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float32)


def _to_params(x: np.ndarray, template: DummyParams) -> DummyParams:
    data = asdict(template)
    for i, name in enumerate(FEATURES):
        data[name] = round(float(x[i]), 3)
    return DummyParams(**data)


def _distance(x: np.ndarray, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    scale = np.maximum(hi - lo, 1e-6)
    return float(np.linalg.norm((x - x0) / scale))


def _bounds_from_pool(pool_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pool_x.min(axis=0), pool_x.max(axis=0)


def _strict_label(p: DummyParams) -> dict:
    label = compute_analytic_overlap_label(
        p,
        pitch_mm=DEFAULT_PITCH_MM,
        overlap_threshold_norm=STRICT_THRESHOLD_NORM,
    )
    payload = {"param_hash": "", "label": label.to_dict()}
    return continuous_label_from_payload(payload)


def _strict_overlap(p: DummyParams) -> float:
    return float(_strict_label(p)["total_overlap_norm"])


def _active_coordinates(x: np.ndarray, x0: np.ndarray, tol: float = 1e-3) -> int:
    return int(np.sum(np.abs(x - x0) > tol))


_BORING_COORDS = {
    "main_hole_radius": "increase",
    "cross_hole_radius": "increase",
    "main_pin_radius": "decrease",
    "splint_radius": "decrease",
}


def _boring_score(x: np.ndarray, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    score = 0.0
    for name, direction in _BORING_COORDS.items():
        i = FEATURES.index(name)
        span = max(float(hi[i] - lo[i]), 1e-9)
        delta = float(x[i] - x0[i]) / span
        score += max(delta, 0.0) if direction == "increase" else max(-delta, 0.0)
    return float(score)


def _load_standardizer(path: Path) -> Standardizer:
    data = np.load(path, allow_pickle=True)
    std = Standardizer()
    std.mean_ = data["mean"].astype(np.float32)
    std.scale_ = data["scale"].astype(np.float32)
    return std


def _load_model(model_dir: Path) -> tuple[OverlapMLP, Standardizer, dict]:
    arch = json.loads((model_dir / "architecture.json").read_text(encoding="utf-8"))
    hidden = tuple(int(v) for v in arch["hidden"])
    model = OverlapMLP(len(FEATURES), hidden=hidden)
    model.load_state_dict(torch.load(model_dir / "model_state.pt", map_location="cpu"))
    model.eval()
    return model, _load_standardizer(model_dir / "standardizer.npz"), arch


ensure_chapter3_importable()
from chapter3_clevis_setup.exact_assemblability import (  # noqa: E402
    compute_exact_terms,
    evaluate_exact_assemblability,
)


def _assemblability_margin(p: DummyParams) -> float:
    terms = compute_exact_terms(p)
    roof_margin = terms.y_overhang_outer_neg - terms.y_splint_head_pos_edge
    return float(max(roof_margin, terms.splint_clearance_margin, terms.inward_movement_margin))


def _params_from_dict(data: dict) -> DummyParams:
    return DummyParams(**{k: float(data[k]) for k in DummyParams.__dataclass_fields__})


def _head_category(row: dict, label: dict) -> str:
    p = params_from_row(row)
    margins = compute_margins(p)
    dominant = str(label["dominant_pair"])
    rules = {r for r in str(row.get("relaxed_violations", "")).split("|") if r}
    intended = {r for r in str(row.get("intended_relaxed_rules", "")).split(",") if r}
    family = rules | intended
    if dominant == "bracket__splint" and (family & {"D4", "D6"} or margins["D4"] < 1.0 or margins["D6"] < 1.0):
        return "splint_head_wall_candidate"
    if dominant == "bracket__main_pin" and (family & {"A1", "D7"} or margins["A1"] < 0.5 or margins["D7"] < 1.0):
        return "pin_head_or_shaft_wall_candidate"
    if dominant == "main_pin__splint":
        return "pin_splint_cross_overlap"
    if dominant == "bracket__splint":
        return "bracket_splint_other"
    if dominant == "bracket__main_pin":
        return "bracket_pin_other"
    return "other_overlap"
