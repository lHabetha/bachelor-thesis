"""Geometric overlap check for a generated assembly.

Purpose: refuse to plan disassembly on a physically unrealistic assembly. An
"unrealistic" assembly is one where two *different* part meshes
**interpenetrate** (their solid volumes overlap by more than numerical noise).

This is distinct from two parts being *nested* (e.g. a pin sitting inside a
hole with clearance). Nested solids have zero volumetric intersection because
the hole is carved out of the host part's volume. The clevis relies on that
distinction: pin-in-hole is fine, pin-clipping-the-wall is not.

The check is two-stage for speed:

1. **Cheap pre-filter** — AABB overlap per pair. If axis-aligned boxes are
   disjoint, skip the pair.
2. **Exact test** — pairwise `trimesh.boolean.intersection` volume. Positive
   volume above a single global `tolerance` counts as an unintended overlap.

The tolerance is calibrated on canonical demo assemblies to absorb numerical noise
from mesh export / watertight repair only. A negative control (pin radius >
hole radius) is verified to trip the check.

API contract (stable; Chapter 3 introduced the signature, Chapter 3 fills it in)::

    overlap_check(assembly_dir, tolerance=TOL_DEFAULT) -> (ok, report)

where ``report`` is a JSON-safe dict with per-pair fields. See
thesis Chapter 3 validity section for rationale.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh


# Calibrated 2026-04-10 from all 10 demos + negative control.
# Units are cubic millimetres in the generator's native frame (the clevis
# generator exports at mm scale; the check is run *before* normalization so
# the tolerance number is physically meaningful: "less than one cubic mm of
# interpenetration is noise").
TOL_DEFAULT: float = 1e-3


def _list_part_objs(assembly_dir: Path) -> list[Path]:
    """Collect per-part `.obj` files, skipping merged previews."""
    skip = {'assembled.obj', 'exploded.obj'}
    return sorted(p for p in assembly_dir.glob('*.obj') if p.name not in skip)


def _aabb_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh,
                  slack: float = 0.0) -> bool:
    """Axis-aligned bounding box intersection test (fast pre-filter)."""
    a_lo, a_hi = a.bounds
    b_lo, b_hi = b.bounds
    return bool(np.all(a_lo - slack <= b_hi) and np.all(b_lo - slack <= a_hi))


def _boolean_intersection_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Exact volumetric intersection of two watertight meshes.

    Uses trimesh's default boolean engine (manifold3d in modern trimesh).
    Returns 0.0 if the intersection is empty or fails gracefully.
    """
    try:
        inter = trimesh.boolean.intersection([a, b])
    except Exception:
        return float('nan')
    if inter is None or inter.is_empty:
        return 0.0
    vol = float(abs(inter.volume))
    return vol


def overlap_check(
    assembly_dir: str | Path,
    tolerance: float = TOL_DEFAULT,
    *,
    aabb_slack: float = 0.0,
    verbose: bool = False,
) -> tuple[bool, dict]:
    """Return ``(ok, report)`` for the assembly at ``assembly_dir``.

    Parameters
    ----------
    assembly_dir:
        Directory containing per-part ``*.obj`` files. Merged previews
        (``assembled.obj``, ``exploded.obj``) are ignored.
    tolerance:
        Maximum allowed per-pair intersection volume. Default is
        ``TOL_DEFAULT``, calibrated on the 10 demos (see
        ``calibrate_overlap.py``). Units are whatever the loaded meshes use,
        so the check is run **pre-normalization** on the generator's native
        scale (mm for the clevis).
    aabb_slack:
        Extra padding on the AABB pre-filter in the mesh's native units. Set
        non-zero only to diagnose edge cases.
    verbose:
        Print per-pair results to stdout.

    Returns
    -------
    (ok, report):
        ``ok`` is True iff no pair exceeds ``tolerance``. ``report`` is a dict
        with keys:
            - ``ok``: bool
            - ``status``: 'pass' | 'fail' | 'error'
            - ``tolerance``: float (the threshold used)
            - ``part_ids``: list[str]
            - ``pairs``: list of per-pair results, each with:
                - ``a``, ``b``: part ids
                - ``aabb_overlap``: bool
                - ``intersection_volume``: float (NaN if boolean failed)
                - ``over_tolerance``: bool
            - ``max_overlap_volume``: float (max across all pairs)
            - ``max_overlap_pair``: tuple[str, str] | None
            - ``assembly_dir``: str
            - ``warnings``: list[str]
    """
    assembly_dir = Path(assembly_dir).resolve()
    report: dict = {
        'ok': None, 'status': None,
        'tolerance': float(tolerance),
        'assembly_dir': str(assembly_dir),
        'part_ids': [], 'pairs': [],
        'max_overlap_volume': 0.0, 'max_overlap_pair': None,
        'warnings': [],
    }

    obj_paths = _list_part_objs(assembly_dir)
    if len(obj_paths) < 2:
        report.update(ok=False, status='error',
                      warnings=[f'Need at least 2 parts; found {len(obj_paths)}.'])
        return False, report

    meshes: dict[str, trimesh.Trimesh] = {}
    for p in obj_paths:
        m = trimesh.load_mesh(str(p), process=False, maintain_order=True)
        if not m.is_watertight:
            report['warnings'].append(
                f'Mesh {p.name} is not watertight; boolean volume may be '
                f'unreliable. Repair upstream.')
        meshes[p.stem] = m
    report['part_ids'] = list(meshes.keys())

    any_fail = False
    any_error = False

    for pid_a, pid_b in combinations(meshes.keys(), 2):
        a, b = meshes[pid_a], meshes[pid_b]
        aabb = _aabb_overlap(a, b, slack=aabb_slack)
        if not aabb:
            entry = {
                'a': pid_a, 'b': pid_b,
                'aabb_overlap': False,
                'intersection_volume': 0.0,
                'over_tolerance': False,
            }
        else:
            vol = _boolean_intersection_volume(a, b)
            if np.isnan(vol):
                entry = {
                    'a': pid_a, 'b': pid_b,
                    'aabb_overlap': True,
                    'intersection_volume': float('nan'),
                    'over_tolerance': True,
                }
                any_error = True
                report['warnings'].append(
                    f'Boolean intersection failed on ({pid_a}, {pid_b}); '
                    f'treating as failure.')
            else:
                entry = {
                    'a': pid_a, 'b': pid_b,
                    'aabb_overlap': True,
                    'intersection_volume': vol,
                    'over_tolerance': bool(vol > tolerance),
                }
                if vol > report['max_overlap_volume']:
                    report['max_overlap_volume'] = vol
                    report['max_overlap_pair'] = (pid_a, pid_b)
                if entry['over_tolerance']:
                    any_fail = True

        report['pairs'].append(entry)
        if verbose:
            mark = 'OK '
            if entry.get('over_tolerance'):
                mark = 'FAIL'
            v = entry['intersection_volume']
            print(f'  [{mark}] {pid_a:<12s} & {pid_b:<12s}  '
                  f'vol={v:.3e}  aabb={entry["aabb_overlap"]}')

    if any_error:
        report['ok'] = False
        report['status'] = 'error'
    elif any_fail:
        report['ok'] = False
        report['status'] = 'fail'
    else:
        report['ok'] = True
        report['status'] = 'pass'

    if verbose:
        print(f'  result: {report["status"]}  '
              f'max_overlap={report["max_overlap_volume"]:.3e} '
              f'pair={report["max_overlap_pair"]}')

    return bool(report['ok']), report
