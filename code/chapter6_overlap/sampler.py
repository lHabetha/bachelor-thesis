"""Controlled-overlap parameter sampler for Chapter 6.

The sampler admits bounded violations of the frozen Chapter 6 relaxed constraints
while keeping hole-breakout, retention, and non-degeneracy constraints hard.
It returns parameter-only samples plus diagnostic margins; geometry generation
and overlap labels happen later through the cache/labeler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .release_paths import ensure_chapter3_importable

ensure_chapter3_importable()

from chapter3_clevis_setup.design_space import (  # noqa: E402
    DESIGN_SPACE,
    EPS_ACCESS,
    EPS_DEPTH_WALL,
    EPS_FIT,
    EPS_HEAD_CLEAR,
    EPS_HEAD_SLIP,
    EPS_LEG_WALL,
    EPS_PIN_WALL,
    EPS_RETENTION,
    SPLINT_HEAD_RATIO,
    DummyParams,
    sample_params as sample_clean_params,
)

PARAM_NAMES = [
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
    "exploded_gap",
]

STREAM_WEIGHTS = {
    "clean_valid": 0.25,
    "near_boundary": 0.20,
    "single_relaxation": 0.35,
    "multi_relaxation": 0.15,
    "extreme_meaningful": 0.05,
}

RELAXED_RULES = ("A1", "A2", "D4", "D5", "D6", "D7")


@dataclass(frozen=True)
class RelaxedValidation:
    ok: bool
    hard_failures: list[str]
    relaxed_violations: list[str]
    margins: dict[str, float]


@dataclass(frozen=True)
class RelaxedSample:
    sample_id: str
    params: DummyParams
    stream: str
    intended_relaxed_rules: list[str]
    validation: RelaxedValidation
    seed: int

    def to_row(self) -> dict:
        row = {
            "sample_id": self.sample_id,
            "stream": self.stream,
            "intended_relaxed_rules": ",".join(self.intended_relaxed_rules),
            "seed": self.seed,
            "relaxed_ok": self.validation.ok,
            "hard_failures": "|".join(self.validation.hard_failures),
            "relaxed_violations": "|".join(self.validation.relaxed_violations),
        }
        row.update({k: float(v) for k, v in asdict(self.params).items()})
        row.update({f"margin_{k}": float(v) for k, v in self.validation.margins.items()})
        return row


def params_to_dict(p: DummyParams) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(p).items()}


def _head_len(p: DummyParams) -> float:
    return 0.05 * p.main_pin_length


def _splint_head_len(p: DummyParams) -> float:
    return 0.05 * p.splint_length


def compute_margins(p: DummyParams) -> dict[str, float]:
    """Return positive-is-valid margins for all Chapter 6-relevant constraints."""
    hr = SPLINT_HEAD_RATIO * p.splint_radius
    main_hole_radial_slack = max(0.0, p.main_hole_radius - p.main_pin_radius)
    return {
        "A1": p.main_hole_radius - (p.main_pin_radius + EPS_FIT),
        "A2": p.cross_hole_radius - (p.splint_radius + EPS_FIT),
        "A3": hr - (p.cross_hole_radius + EPS_HEAD_SLIP),
        "B1": p.depth - (2 * p.main_hole_radius + 2 * EPS_DEPTH_WALL),
        "B2_task25": 0.99 * p.leg_length - (p.main_hole_offset_from_open_end + p.main_hole_radius),
        "B3": (p.main_hole_offset_from_open_end - p.main_hole_radius) - EPS_LEG_WALL,
        "B4": p.cross_hole_distance_from_free_end - (p.cross_hole_radius + EPS_PIN_WALL),
        "B5": (p.main_pin_length - _head_len(p)) - (
            p.cross_hole_distance_from_free_end + p.cross_hole_radius + EPS_PIN_WALL
        ),
        "C1_task25": p.outer_span - (p.wall_thickness + EPS_ACCESS),
        "C1_inner_gap": p.outer_span - 2 * p.wall_thickness,
        "D1": p.main_pin_length - (
            p.outer_span + 2 * p.cross_hole_distance_from_free_end + 2 * EPS_ACCESS
        ),
        "D2": p.main_pin_length - (
            p.outer_span
            - 2 * p.wall_thickness
            + 2 * p.cross_hole_distance_from_free_end
            + 2 * p.cross_hole_radius
            + 2 * EPS_ACCESS
        ),
        "D3": p.splint_length - (2 * p.main_pin_radius + 2 * EPS_RETENTION),
        "D4": (
            p.wall_thickness
            + p.leg_length
            - p.main_hole_offset_from_open_end
            - (0.5 * p.splint_length + _splint_head_len(p) + main_hole_radial_slack + EPS_HEAD_CLEAR)
        ),
        "D5": p.main_pin_length - (
            p.outer_span + 2 * p.cross_hole_distance_from_free_end + 2 * hr + 2 * EPS_ACCESS
        ),
        "D6": p.main_hole_radius - EPS_FIT - hr,
        "D7": (p.wall_thickness + p.leg_length) - (
            p.main_hole_offset_from_open_end
            + 2 * p.main_pin_radius
            + main_hole_radial_slack
            + EPS_HEAD_CLEAR
        ),
        "E2_overhang": p.overhang_span_y,
        "E3": p.exploded_gap,
    }


def validate_relaxed_params(p: DummyParams) -> RelaxedValidation:
    margins = compute_margins(p)
    hard_failures: list[str] = []

    for name in PARAM_NAMES:
        if name == "overhang_span_y":
            continue
        value = float(getattr(p, name))
        if name != "exploded_gap" and value < 0.5:
            hard_failures.append(f"E1:{name}")

    hard_rules = ("A3", "B1", "B2_task25", "B3", "B4", "B5", "C1_task25", "D1", "D2", "D3", "E2_overhang", "E3")
    for rule in hard_rules:
        if margins[rule] < 0.0:
            hard_failures.append(rule)

    # C1 is relaxed relative to the clean design space but the physical cavity
    # must not invert; a zero/negative inner gap is not a meaningful bracket.
    if margins["C1_inner_gap"] <= 0.0:
        hard_failures.append("C1_inner_gap")

    relaxed_violations = [rule for rule in RELAXED_RULES if margins[rule] < 0.0]
    return RelaxedValidation(
        ok=len(hard_failures) == 0,
        hard_failures=hard_failures,
        relaxed_violations=relaxed_violations,
        margins=margins,
    )


def _with(p: DummyParams, **updates: float) -> DummyParams:
    data = params_to_dict(p)
    data.update({k: round(float(v), 3) for k, v in updates.items()})
    return DummyParams(**data)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _choose_stream(rng: np.random.Generator) -> str:
    streams = list(STREAM_WEIGHTS)
    weights = np.array([STREAM_WEIGHTS[s] for s in streams], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(streams, p=weights))


def _push_rule(p: DummyParams, rule: str, rng: np.random.Generator, *, near: bool = False) -> DummyParams:
    """Move a clean sample toward or across one relaxed constraint boundary."""
    m = compute_margins(p)
    sign = 0.3 if near else -1.0

    if rule == "A1":
        target_margin = _u(rng, 0.02, 0.20) if near else -_u(rng, 0.05, 1.50)
        return _with(p, main_pin_radius=p.main_hole_radius - EPS_FIT - target_margin)

    if rule == "A2":
        target_margin = _u(rng, 0.02, 0.15) if near else -_u(rng, 0.03, 0.80)
        new_splint = p.cross_hole_radius - EPS_FIT - target_margin
        # Preserve A3 by not shrinking the head relative to cross-hole.
        a3_floor = (p.cross_hole_radius + EPS_HEAD_SLIP) / SPLINT_HEAD_RATIO
        return _with(p, splint_radius=max(new_splint, a3_floor))

    if rule == "D4":
        target_margin = _u(rng, 0.05, 0.50) if near else -_u(rng, 0.25, 8.0)
        current = m["D4"]
        delta_margin = current - target_margin
        return _with(p, main_hole_offset_from_open_end=p.main_hole_offset_from_open_end + delta_margin)

    if rule == "D5":
        target_margin = _u(rng, 0.05, 0.50) if near else -_u(rng, 0.25, 8.0)
        current = m["D5"]
        delta_margin = current - target_margin
        return _with(p, main_pin_length=max(0.5, p.main_pin_length - delta_margin))

    if rule == "D6":
        target_margin = _u(rng, 0.02, 0.30) if near else -_u(rng, 0.05, 3.0)
        head_r = p.main_hole_radius - EPS_FIT - target_margin
        new_splint = max(0.5, head_r / SPLINT_HEAD_RATIO)
        a3_floor = (p.cross_hole_radius + EPS_HEAD_SLIP) / SPLINT_HEAD_RATIO
        return _with(p, splint_radius=max(new_splint, a3_floor))

    if rule == "D7":
        target_margin = _u(rng, 0.05, 0.50) if near else -_u(rng, 0.25, 8.0)
        current = m["D7"]
        delta_margin = current - target_margin
        return _with(p, main_hole_offset_from_open_end=p.main_hole_offset_from_open_end + sign * 0.0 + delta_margin)

    return p


def _make_base(rng: np.random.Generator, stream: str) -> DummyParams:
    if stream != "extreme_meaningful":
        return sample_clean_params(rng)

    # Try a more extreme clean draw by pushing independent dimensions toward
    # corners, then fall back to the clean sampler if a derived constraint binds.
    for _ in range(80):
        p = sample_clean_params(rng)
        updates = {
            "wall_thickness": _u(rng, DESIGN_SPACE["wall_thickness"]["min"], DESIGN_SPACE["wall_thickness"]["max"]),
            "overhang_span_y": _u(rng, 1.6, 2.4) * p.outer_span,
        }
        q = _with(p, **updates)
        if validate_relaxed_params(q).ok:
            return q
    return sample_clean_params(rng)


def sample_relaxed_params(
    rng: np.random.Generator,
    *,
    sample_id: str,
    seed: int,
    stream: str | None = None,
    max_tries: int = 200,
) -> RelaxedSample:
    """Draw one Chapter 6 relaxed-validity sample."""
    selected_stream = stream or _choose_stream(rng)
    for _ in range(max_tries):
        p = _make_base(rng, selected_stream)
        intended: list[str] = []

        if selected_stream == "clean_valid":
            pass
        elif selected_stream == "near_boundary":
            intended = [str(rng.choice(RELAXED_RULES))]
            p = _push_rule(p, intended[0], rng, near=True)
        elif selected_stream == "single_relaxation":
            intended = [str(rng.choice(RELAXED_RULES))]
            p = _push_rule(p, intended[0], rng, near=False)
        elif selected_stream == "multi_relaxation":
            intended = [str(x) for x in rng.choice(RELAXED_RULES, size=2, replace=False)]
            for rule in intended:
                p = _push_rule(p, rule, rng, near=False)
        elif selected_stream == "extreme_meaningful":
            intended = [str(rng.choice(RELAXED_RULES))]
            p = _push_rule(p, intended[0], rng, near=bool(rng.random() < 0.35))

        validation = validate_relaxed_params(p)
        if validation.ok:
            return RelaxedSample(
                sample_id=sample_id,
                params=p,
                stream=selected_stream,
                intended_relaxed_rules=intended,
                validation=validation,
                seed=seed,
            )

    raise RuntimeError(f"Could not draw relaxed sample {sample_id} from stream {selected_stream}")


def sample_relaxed_batch(n: int, *, seed: int, prefix: str = "t25") -> list[RelaxedSample]:
    rng = np.random.default_rng(seed)
    return [
        sample_relaxed_params(rng, sample_id=f"{prefix}_{i:06d}", seed=seed)
        for i in range(n)
    ]


def rows_from_samples(samples: Iterable[RelaxedSample]) -> list[dict]:
    return [s.to_row() for s in samples]
