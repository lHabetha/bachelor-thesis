"""Reusable clevis assembly generator.

Single entry point: ``generate(params, out_dir)`` produces four .obj files
(bracket, main_pin, splint, assembled) plus a params.json metadata file.

This module is the stable API for dataset generation and future ML pipelines.
Import ``DummyParams`` to configure, then call ``generate()``.

Example::

    from clevis_generator import generate, DummyParams
    generate(DummyParams(leg_length=40, overhang_span_y=90), Path("my_output"))
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import cadquery as cq

from .generate_modified_clevis_dummy import (
    DummyParams,
    SPLINT_HEAD_RATIO,
    _validate_params,
    build_parts,
)

__all__ = [
    "DummyParams",
    "SPLINT_HEAD_RATIO",
    "generate",
]


def _export_obj(shape: cq.Workplane, path: Path) -> None:
    """Export a CadQuery shape to .obj, falling back through STL+trimesh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cq.exporters.export(shape, str(path))
    except Exception:
        import tempfile
        import trimesh
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            cq.exporters.export(shape, tmp.name)
            trimesh.load_mesh(tmp.name, force="mesh").export(str(path))


def _write_params(p: DummyParams, path: Path) -> None:
    """Serialize all params (including derived properties) to JSON."""
    data = asdict(p)
    data["_derived"] = {
        "pin_half": p.pin_half,
        "hole_z": p.hole_z,
        "cross_y": p.cross_y,
        "head_len": p.head_len,
        "head_r": p.head_r,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _verify_sides(p: DummyParams, out_dir: Path) -> bool:
    """Quick check: nail-head and splint must be on opposite Y-sides."""
    import numpy as np
    import trimesh

    pin = trimesh.load_mesh(str(out_dir / "main_pin.obj"), force="mesh")
    spl = trimesh.load_mesh(str(out_dir / "splint.obj"), force="mesh")

    dist = np.sqrt(pin.vertices[:, 0]**2 + (pin.vertices[:, 2] - p.hole_z)**2)
    head_verts = pin.vertices[dist > p.main_pin_radius * 1.15]
    if len(head_verts) == 0:
        return False

    head_y = float(head_verts[:, 1].mean())
    splint_y = float(spl.vertices[:, 1].mean())
    return head_y * splint_y < 0  # opposite signs relative to y=0


def generate(
    params: DummyParams,
    out_dir: Path,
    *,
    verbose: bool = True,
) -> bool:
    """Build and export one clevis assembly.

    Outputs written to *out_dir* (created if needed):
        bracket.obj, main_pin.obj, splint.obj  -- individual parts
        assembled.obj                          -- all three merged
        params.json                            -- full parameter dump

    Returns True if geometry verification passes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = _validate_params(params)
    if warnings and verbose:
        for w in warnings:
            print(f"  WARNING: {w}")

    parts = build_parts(params)

    for name, shape in parts.items():
        _export_obj(shape, out_dir / f"{name}.obj")

    assembled = parts["bracket"].union(parts["main_pin"]).union(parts["splint"])
    _export_obj(assembled, out_dir / "assembled.obj")

    _write_params(params, out_dir / "params.json")

    ok = _verify_sides(params, out_dir)
    if verbose:
        status = "OK" if ok else "FAIL (head & splint on same side)"
        print(f"  [{out_dir.name}] verify: {status}")
    return ok
