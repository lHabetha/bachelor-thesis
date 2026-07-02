"""Three-stream "smart" sampler for Chapter 3 dataset v2.

Rationale
---------
Pure uniform draws from :func:`design_space.sample_params` work but waste
coverage: the independent dims are sampled independently, so neighbours in
parameter space can cluster and large regions remain empty. A surrogate
trained on 1000 such draws benefits enormously from better space-filling
and from a deliberate push into the "topologically caging" boundary where
most of the informative label-0 signal lives.

This module implements a hybrid sampling strategy:

 - **Anchored-Gaussian** (5 % of random draws): jitter the 10 canonical
   demos by N(0, 10 % of value) on the independent dims, re-derive
   dependents hierarchically. Gives stability near known-good designs.
 - **Latin-Hypercube** (75 %): proper space-filling via
   :class:`scipy.stats.qmc.LatinHypercube` over the 10 degrees of freedom
   (4 independent frame dims + 6 derivation ratios). Much better surrogate
   coverage than i.i.d. uniform.
 - **Boundary-pushed** (20 %): biased draws on the overhang ratio, hole
   offset, pin/hole fit ratio, and splint/hole fit ratio to raise the
   label-0 rate above v1's ~12 % baseline, which heavily biased the
   binary classifier. Still runs through the validator so every point
   is geometrically valid.

Every emitted ``DummyParams`` passes :func:`design_space.validate_params`.

Public API
----------
* :func:`build_random_plan`   — build the full interleaved plan
* :func:`stream_uniform`      — single uniform draw (for tests)
* :func:`stream_lhs_batch`    — LHS batch
* :func:`stream_anchored_gaussian` — demo-jittered draw
* :func:`stream_boundary_pushed`   — biased draw
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from .design_space import (
    DESIGN_SPACE,
    DummyParams,
    EPS_ACCESS,
    EPS_DEPTH_WALL,
    EPS_FIT,
    EPS_HEAD_CLEAR,
    EPS_HEAD_SLIP,
    EPS_LEG_WALL,
    EPS_PIN_WALL,
    EPS_RETENTION,
    SPLINT_HEAD_RATIO,
    sample_params,
    validate_params,
)


# ──────────────────────────────────────────────────────────────────────
#  Core: 10-D quantile -> valid DummyParams
#
#  The 10 "degrees of freedom" we expose to the sampler streams are:
#     0: wall_thickness_q   -> wall_thickness in [wt_min, wt_max]
#     1: outer_span_q       -> outer_span    in [os_min, os_max]
#     2: leg_length_q       -> leg_length    in [ll_min, ll_max]
#     3: depth_q            -> depth         in [d_min,  d_max]
#     4: pin_ratio_q        -> main_pin_radius / main_hole_radius in [0.75, 0.97]
#     5: off_frac_q         -> main_hole_offset position in [off_lo, off_hi]
#     6: ch_ratio_q         -> cross_hole_radius / main_pin_radius in [0.30, 0.65]
#     7: sp_ratio_q         -> splint_radius / cross_hole_radius in [0.75, 0.90]
#                              (lower-bounded by A3 floor)
#     8: extra_pin_q        -> extra pin length margin in [4, 20]
#     9: overhang_frac_q    -> overhang_span_y / outer_span in [0.8, 2.0]
#
#  Streams differ only in how they draw the 10 quantiles; the derivation
#  is shared, so every stream produces the exact same DummyParams given
#  the same 10 numbers in [0,1]^10.
# ──────────────────────────────────────────────────────────────────────

_IND_RANGES = (
    (DESIGN_SPACE['wall_thickness']['min'], DESIGN_SPACE['wall_thickness']['max']),
    (DESIGN_SPACE['outer_span']['min'],    DESIGN_SPACE['outer_span']['max']),
    (DESIGN_SPACE['leg_length']['min'],    DESIGN_SPACE['leg_length']['max']),
    (DESIGN_SPACE['depth']['min'],         DESIGN_SPACE['depth']['max']),
)

_RATIO_RANGES = {
    'pin_ratio':     (0.75, 0.97),
    'off_frac':      (0.0,  1.0),
    'ch_ratio':      (0.30, 0.65),
    'sp_ratio':      (0.75, 0.90),
    'extra_pin':     (4.0,  20.0),
    'overhang_frac': (0.8,  2.0),
}


def _lerp(q: float, lo: float, hi: float) -> float:
    q = max(0.0, min(1.0, float(q)))
    return lo + q * (hi - lo)


def _derive_from_quantiles(
    q: Sequence[float],
    rng: np.random.Generator,
    *,
    max_tries: int = 40,
) -> Optional[DummyParams]:
    """Hierarchical derivation from 10 [0,1] quantiles.

    Returns ``None`` if the derivation cannot land on a valid tuple within
    ``max_tries``. Early dependents (e.g. splint_length window) can still
    become empty for pathological frame dims — we re-jitter the
    *dependent-only* quantiles with a small amount of randomness in that
    case so the independent "anchor" stays where the caller put it.
    """
    if len(q) != 10:
        raise ValueError(f'expected 10 quantiles, got {len(q)}')
    q = list(q)

    for _ in range(max_tries):
        wall_thickness = _lerp(q[0], *_IND_RANGES[0])
        outer_span     = _lerp(q[1], *_IND_RANGES[1])
        leg_length     = _lerp(q[2], *_IND_RANGES[2])
        depth          = _lerp(q[3], *_IND_RANGES[3])

        if outer_span <= 2 * wall_thickness + EPS_ACCESS + 1.0:
            # Impossible frame; resample only the frame quantiles and retry.
            q[0:4] = rng.random(4).tolist()
            continue

        hole_r_hi = min(DESIGN_SPACE['main_hole_radius']['max'],
                        0.5 * depth - EPS_DEPTH_WALL - 0.25)
        hole_r_lo = DESIGN_SPACE['main_hole_radius']['min']
        if hole_r_hi <= hole_r_lo:
            q[3] = float(rng.random())  # redraw depth only
            continue
        main_hole_radius = rng.uniform(hole_r_lo, hole_r_hi)

        pin_ratio = _lerp(q[4], *_RATIO_RANGES['pin_ratio'])
        main_pin_radius = max(1.0, pin_ratio * main_hole_radius - EPS_FIT)
        if main_pin_radius + EPS_FIT >= main_hole_radius:
            q[4] = float(rng.random())
            continue

        off_lo = main_hole_radius + EPS_LEG_WALL
        off_hi = 0.80 * leg_length - main_hole_radius
        if off_hi <= off_lo:
            q[2] = float(rng.random())  # redraw leg_length
            continue
        main_hole_offset_from_open_end = _lerp(q[5], off_lo, off_hi)

        ch_ratio = _lerp(q[6], *_RATIO_RANGES['ch_ratio'])
        cross_hole_radius = max(1.0, ch_ratio * main_pin_radius)
        if cross_hole_radius + EPS_FIT > main_pin_radius - 0.5:
            q[6] = float(rng.random())
            continue

        a3_splint_lo = (cross_hole_radius + EPS_HEAD_SLIP) / SPLINT_HEAD_RATIO
        sp_ratio = _lerp(q[7], *_RATIO_RANGES['sp_ratio'])
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
            # Cross-hole too small to fit a head-larger-than-hole splint;
            # redraw cross-hole ratio.
            q[6] = float(rng.random())
            continue
        hr = SPLINT_HEAD_RATIO * splint_radius

        cd_lo = cross_hole_radius + EPS_PIN_WALL
        prov_pin_length = outer_span + 2 * cd_lo + 2 * EPS_ACCESS + 12.0
        cd_hi = max(cd_lo + 0.5, 0.25 * prov_pin_length)
        if cd_hi <= cd_lo:
            q[6] = float(rng.random())
            continue
        cross_hole_distance_from_free_end = rng.uniform(cd_lo, cd_hi)

        d1_min = outer_span + 2 * cross_hole_distance_from_free_end + 2 * EPS_ACCESS + 0.5
        d2_min = (outer_span - 2 * wall_thickness
                  + 2 * cross_hole_distance_from_free_end
                  + 2 * cross_hole_radius
                  + 2 * EPS_ACCESS + 0.5)
        d5_min = (outer_span - 2 * wall_thickness
                  + 2 * cross_hole_distance_from_free_end
                  + 2 * hr
                  + 2 * EPS_ACCESS + 0.5)
        pin_len_min = max(d1_min, d2_min, d5_min)
        extra = _lerp(q[8], *_RATIO_RANGES['extra_pin'])
        main_pin_length = pin_len_min + extra

        head_len = max(0.04 * main_pin_length, 3.0)
        if (cross_hole_distance_from_free_end + cross_hole_radius + EPS_PIN_WALL
                > main_pin_length - head_len):
            q[8] = float(rng.random())
            continue

        d3_min = 2 * main_pin_radius + 2 * EPS_RETENTION
        d4_rhs = wall_thickness + leg_length - main_hole_offset_from_open_end
        d4_splint_max_50 = (d4_rhs - EPS_HEAD_CLEAR) / 0.55
        d4_splint_max_floor = 2.0 * (d4_rhs - 1.5 - EPS_HEAD_CLEAR)
        splint_len_max = min(d4_splint_max_50, d4_splint_max_floor)
        if splint_len_max < d3_min + 1.0:
            # Leg + wall_thickness too short to host a splint clearing the roof.
            q[2] = float(rng.random())
            q[5] = float(rng.random())
            continue
        splint_extra = rng.uniform(1.0, max(1.5, min(20.0, splint_len_max - d3_min - 0.5)))
        splint_length = d3_min + splint_extra

        overhang_frac = _lerp(q[9], *_RATIO_RANGES['overhang_frac'])
        overhang_span_y = overhang_frac * outer_span

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
        ok, _reasons = validate_params(p)
        if ok:
            return p
    return None


# ──────────────────────────────────────────────────────────────────────
#  Stream 1: anchored Gaussian around the 10 canonical demos
# ──────────────────────────────────────────────────────────────────────

def _params_to_independent_quantiles(p: DummyParams) -> list[float]:
    """Invert the 4 independent-frame-dim quantile + 6 derivation ratios.

    Returns the best-effort 10-D quantile location of an existing valid
    DummyParams. Used by the anchored-Gaussian stream to figure out where
    the demo sits before jittering.
    """
    wt_q = (p.wall_thickness - _IND_RANGES[0][0]) / (_IND_RANGES[0][1] - _IND_RANGES[0][0])
    os_q = (p.outer_span     - _IND_RANGES[1][0]) / (_IND_RANGES[1][1] - _IND_RANGES[1][0])
    ll_q = (p.leg_length     - _IND_RANGES[2][0]) / (_IND_RANGES[2][1] - _IND_RANGES[2][0])
    d_q  = (p.depth          - _IND_RANGES[3][0]) / (_IND_RANGES[3][1] - _IND_RANGES[3][0])

    pin_ratio = (p.main_pin_radius + EPS_FIT) / max(p.main_hole_radius, 1e-6)
    off_lo = p.main_hole_radius + EPS_LEG_WALL
    off_hi = 0.80 * p.leg_length - p.main_hole_radius
    off_frac = (p.main_hole_offset_from_open_end - off_lo) / max(off_hi - off_lo, 1e-6)
    ch_ratio = p.cross_hole_radius / max(p.main_pin_radius, 1e-6)
    sp_ratio = p.splint_radius / max(p.cross_hole_radius, 1e-6)

    hr = SPLINT_HEAD_RATIO * p.splint_radius
    d1_min = p.outer_span + 2 * p.cross_hole_distance_from_free_end + 2 * EPS_ACCESS + 0.5
    d2_min = (p.outer_span - 2 * p.wall_thickness
              + 2 * p.cross_hole_distance_from_free_end
              + 2 * p.cross_hole_radius + 2 * EPS_ACCESS + 0.5)
    d5_min = (p.outer_span - 2 * p.wall_thickness
              + 2 * p.cross_hole_distance_from_free_end
              + 2 * hr + 2 * EPS_ACCESS + 0.5)
    pin_len_min = max(d1_min, d2_min, d5_min)
    extra = max(4.0, p.main_pin_length - pin_len_min)
    extra_pin = extra
    overhang_frac = p.overhang_span_y / max(p.outer_span, 1e-6)

    def _q(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-6)))

    return [
        max(0.0, min(1.0, wt_q)),
        max(0.0, min(1.0, os_q)),
        max(0.0, min(1.0, ll_q)),
        max(0.0, min(1.0, d_q)),
        _q(pin_ratio,     *_RATIO_RANGES['pin_ratio']),
        max(0.0, min(1.0, off_frac)),
        _q(ch_ratio,      *_RATIO_RANGES['ch_ratio']),
        _q(sp_ratio,      *_RATIO_RANGES['sp_ratio']),
        _q(extra_pin,     *_RATIO_RANGES['extra_pin']),
        _q(overhang_frac, *_RATIO_RANGES['overhang_frac']),
    ]


def stream_anchored_gaussian(
    rng: np.random.Generator,
    anchor: DummyParams,
    *,
    sigma: float = 0.10,
    max_tries: int = 30,
) -> DummyParams:
    """Draw a DummyParams by jittering an anchor's 10-D quantile location
    with N(0, sigma) noise (clipped to [0,1]).

    ``sigma`` is in quantile units (i.e. a fraction of the full design-
    space range per dim, not of the anchor's own value). sigma=0.10 means
    ~10 % of full-range std, which empirically keeps the sample recognis-
    ably close to the anchor while still exploring.
    """
    base_q = _params_to_independent_quantiles(anchor)
    for _ in range(max_tries):
        noise = rng.normal(0.0, sigma, size=10)
        q = [max(0.0, min(1.0, b + n)) for b, n in zip(base_q, noise)]
        p = _derive_from_quantiles(q, rng)
        if p is not None:
            return p
    # Fallback: pure uniform if we cannot land near the anchor.
    return sample_params(rng)


# ──────────────────────────────────────────────────────────────────────
#  Stream 2: Latin Hypercube
# ──────────────────────────────────────────────────────────────────────

def stream_lhs_batch(
    n: int,
    rng: np.random.Generator,
) -> list[DummyParams]:
    """Draw ``n`` valid DummyParams via Latin Hypercube over the 10 DOF.

    Uses :class:`scipy.stats.qmc.LatinHypercube` with scramble=True and
    centred strength so the marginal distributions are exactly uniform
    but correlations between dims are minimised. If the hierarchical
    derivation rejects a point, we substitute a uniform-fallback draw;
    empirically this happens on <2 % of points and does not noticeably
    degrade space-filling.
    """
    if n <= 0:
        return []
    from scipy.stats import qmc  # local import: optional dep surfaces only when used
    seed = int(rng.integers(0, 2**31 - 1))
    engine = qmc.LatinHypercube(d=10, seed=seed)
    u = engine.random(n)
    out: list[DummyParams] = []
    for row in u:
        p = _derive_from_quantiles(row.tolist(), rng)
        if p is None:
            p = sample_params(rng)
        out.append(p)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Stream 3: Boundary-pushed
#
#  We bias four quantiles toward known failure modes:
#   - overhang_frac -> U(0.45, 1.0) (maps to overhang in [1.34, 2.00]*os)
#   - off_frac      -> Beta(5, 1.2) (biases hole toward top of leg, i.e.
#                       B2 near-boundary)
#   - pin_ratio     -> Beta(5, 1.2) (tight A1 fit)
#   - sp_ratio      -> Beta(1.2, 5) (small splint -> small head near A3)
#  The other six quantiles stay uniform.
# ──────────────────────────────────────────────────────────────────────

def stream_boundary_pushed(
    rng: np.random.Generator,
    *,
    max_tries: int = 40,
) -> DummyParams:
    """Draw one boundary-pushed valid DummyParams."""
    for _ in range(max_tries):
        q = rng.random(10).tolist()
        q[9] = float(rng.uniform(0.45, 1.0))          # large overhang
        q[5] = float(rng.beta(5.0, 1.2))              # hole offset toward top
        q[4] = float(rng.beta(5.0, 1.2))              # tight pin/hole fit
        q[7] = float(rng.beta(1.2, 5.0))              # small splint ratio
        p = _derive_from_quantiles(q, rng)
        if p is not None:
            return p
    return sample_params(rng)


def stream_uniform(rng: np.random.Generator) -> DummyParams:
    """Pure-uniform draw via the shared derivation (equivalent to
    design_space.sample_params but going through the 10-D quantile layer)."""
    for _ in range(20):
        q = rng.random(10).tolist()
        p = _derive_from_quantiles(q, rng)
        if p is not None:
            return p
    return sample_params(rng)


# ──────────────────────────────────────────────────────────────────────
#  Stream 4: Extreme / "wild" — Beta(α, α) with α < 1 on all 10 dims
#
#  Where LHS gives uniform marginals and Gauss hugs demos, this stream
#  pushes **every** dim toward its extreme simultaneously. With α < 1
#  the Beta distribution is U-shaped: most mass sits at 0 and 1. So a
#  single draw is likely to be "all low" or "all high" or some random
#  per-dim corner combination — the kind of absurd multi-extreme design
#  (huge overhang + tight fit + tiny splint + long beam) that the
#  single-dim-pushed boundary stream can't produce.
#
#  alpha default 0.25: roughly 50 % of mass in [0, 0.25] ∪ [0.75, 1]
#  per dim, so a 10-D draw sits in a "corner of the hypercube" with
#  very high probability. The shared derivation still guarantees every
#  output is a valid DummyParams.
# ──────────────────────────────────────────────────────────────────────

def stream_extreme(
    rng: np.random.Generator,
    *,
    alpha: float = 0.25,
    max_tries: int = 40,
) -> DummyParams:
    """Draw a single DummyParams with Beta(α, α) quantiles (U-shaped, α<1).

    Tiny ``alpha`` values push harder toward the hypercube corners. 0.25
    is a sensible default — aggressive enough to produce recognisably
    wild designs, loose enough to still find valid tuples without fighting
    the hierarchical derivation. ``max_tries`` caps the inner rejection
    loop in :func:`_derive_from_quantiles`; if it can't land, we fall back
    to a uniform draw rather than block the dataset build.
    """
    for _ in range(max_tries):
        q = rng.beta(alpha, alpha, size=10).tolist()
        p = _derive_from_quantiles(q, rng)
        if p is not None:
            return p
    return sample_params(rng)


# ──────────────────────────────────────────────────────────────────────
#  Plan builder: produce an interleaved list of (kind, DummyParams)
# ──────────────────────────────────────────────────────────────────────

def build_random_plan(
    n_random: int,
    rng: np.random.Generator,
    *,
    demo_anchors: Optional[Sequence[DummyParams]] = None,
    gauss_frac: float = 0.05,
    boundary_frac: float = 0.20,
    gauss_sigma: float = 0.10,
    extreme_frac: float = 0.0,
    extreme_alpha: float = 0.25,
) -> list[tuple[str, DummyParams]]:
    """Build a complete random-sample plan of size ``n_random``.

    Default mix (rounded to integers): ~5 % anchored-Gaussian + ~20 %
    boundary-pushed + remaining ~75 % Latin Hypercube — the v2 phase-1
    baseline. When ``extreme_frac > 0`` that fraction is carved out for
    the U-shaped Beta(α, α) "extreme" stream (see :func:`stream_extreme`)
    which pushes every dim toward its hypercube corners simultaneously;
    LHS absorbs whatever is left after gauss + boundary + extreme. The
    returned ``kind`` tags are one of
    ``'rand_gauss' | 'rand_lhs' | 'rand_boundary' | 'rand_extreme'``.

    The list is interleaved so the first ~20 entries contain at least
    one sample from every active stream — the order matters for the
    first-20 validator loop downstream.
    """
    if n_random <= 0:
        return []

    n_gauss = int(round(n_random * gauss_frac))
    n_bound = int(round(n_random * boundary_frac))
    n_extreme = int(round(n_random * extreme_frac))
    n_lhs = n_random - n_gauss - n_bound - n_extreme
    if n_lhs < 0:
        # If the caller over-specified the non-LHS fractions, trim from
        # the biggest bucket first (usually extreme when wild mixes).
        n_lhs = 0
        overshoot = n_gauss + n_bound + n_extreme - n_random
        trim_order = sorted([('extreme', n_extreme), ('bound', n_bound),
                             ('gauss', n_gauss)], key=lambda kv: -kv[1])
        for name, _val in trim_order:
            if overshoot <= 0:
                break
            if name == 'extreme':
                cut = min(overshoot, n_extreme); n_extreme -= cut; overshoot -= cut
            elif name == 'bound':
                cut = min(overshoot, n_bound);   n_bound   -= cut; overshoot -= cut
            elif name == 'gauss':
                cut = min(overshoot, n_gauss);   n_gauss   -= cut; overshoot -= cut

    gauss_samples: list[DummyParams] = []
    if demo_anchors and n_gauss > 0:
        anchors = list(demo_anchors)
        for i in range(n_gauss):
            anchor = anchors[i % len(anchors)]
            gauss_samples.append(
                stream_anchored_gaussian(rng, anchor, sigma=gauss_sigma))
    else:
        n_lhs += n_gauss
        n_gauss = 0

    bound_samples = [stream_boundary_pushed(rng) for _ in range(n_bound)]
    extreme_samples = [stream_extreme(rng, alpha=extreme_alpha) for _ in range(n_extreme)]
    lhs_samples = stream_lhs_batch(n_lhs, rng)

    # Interleave: we want the first few slots to include at least one of
    # each active stream for the first-20 validator loop. Cheap pattern:
    # (gauss?, boundary?, extreme?, up to 5 LHS) repeating.
    plan: list[tuple[str, DummyParams]] = []
    i_g = i_b = i_e = i_l = 0
    while (i_g < len(gauss_samples) or i_b < len(bound_samples)
           or i_e < len(extreme_samples) or i_l < len(lhs_samples)):
        if i_g < len(gauss_samples):
            plan.append(('rand_gauss', gauss_samples[i_g])); i_g += 1
        if i_b < len(bound_samples):
            plan.append(('rand_boundary', bound_samples[i_b])); i_b += 1
        if i_e < len(extreme_samples):
            plan.append(('rand_extreme', extreme_samples[i_e])); i_e += 1
        for _ in range(5):
            if i_l < len(lhs_samples):
                plan.append(('rand_lhs', lhs_samples[i_l])); i_l += 1
            else:
                break
    return plan


# ──────────────────────────────────────────────────────────────────────
#  Thesis Appendix A.1 five-mode cycle
# ──────────────────────────────────────────────────────────────────────

THESIS_STREAM_NAMES: tuple[str, ...] = (
    "uniform",
    "boundary",
    "extreme",
    "latin_hypercube",
    "uniform",
)


def sample_by_thesis_cycle(seed: int) -> tuple[str, DummyParams]:
    """Draw one valid design using the thesis Appendix A.1 stream cycle.

    Modes repeat every five seeds: uniform, boundary-biased, extreme,
    Latin-hypercube, uniform.
    """
    rng = np.random.default_rng(seed)
    mode = seed % 5
    if mode == 0:
        return THESIS_STREAM_NAMES[0], stream_uniform(rng)
    if mode == 1:
        return THESIS_STREAM_NAMES[1], stream_boundary_pushed(rng)
    if mode == 2:
        return THESIS_STREAM_NAMES[2], stream_extreme(rng)
    if mode == 3:
        batch = stream_lhs_batch(1, rng)
        params = batch[0] if batch else stream_uniform(rng)
        return THESIS_STREAM_NAMES[3], params
    return THESIS_STREAM_NAMES[4], stream_uniform(rng)


__all__ = [
    'build_random_plan',
    'stream_uniform',
    'stream_lhs_batch',
    'stream_anchored_gaussian',
    'stream_boundary_pushed',
    'stream_extreme',
    'sample_by_thesis_cycle',
    'THESIS_STREAM_NAMES',
]
