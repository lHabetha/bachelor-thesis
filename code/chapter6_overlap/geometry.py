"""Geometry export and hashing helpers for Chapter 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cadquery as cq
import trimesh

from .release_paths import ensure_chapter3_importable
from .sampler import DummyParams

ensure_chapter3_importable()

from chapter3_clevis_setup.generate_modified_clevis_dummy import build_parts  # noqa: E402

GEOMETRY_VERSION = "task25_clevis_geometry_v1"


def canonical_params(p: DummyParams) -> dict[str, float]:
    """Round parameters before hashing to avoid insignificant float drift."""
    return {k: round(float(v), 6) for k, v in asdict(p).items()}


def param_hash(p: DummyParams, *, sampler_version: str = "relaxed_sampler_v1") -> str:
    payload = {
        "params": canonical_params(p),
        "sampler_version": sampler_version,
        "geometry_version": GEOMETRY_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def export_part_geometry(p: DummyParams, out_dir: Path, *, include_stl: bool = True) -> dict[str, str]:
    """Export `bracket`, `main_pin`, and `splint` OBJs for one parameter set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = build_parts(p)
    paths: dict[str, str] = {}

    for name, wp in parts.items():
        obj_path = out_dir / f"{name}.obj"
        stl_path = out_dir / f"{name}.stl"
        if include_stl:
            cq.exporters.export(wp, str(stl_path))
        try:
            cq.exporters.export(wp, str(obj_path))
        except Exception:
            if not stl_path.exists():
                cq.exporters.export(wp, str(stl_path))
            trimesh.load_mesh(str(stl_path), force="mesh").export(str(obj_path))
        paths[f"{name}_obj"] = str(obj_path)
        if include_stl:
            paths[f"{name}_stl"] = str(stl_path)

    (out_dir / "params.json").write_text(
        json.dumps(canonical_params(p), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def load_part_meshes(geometry_dir: Path) -> dict[str, trimesh.Trimesh]:
    meshes: dict[str, trimesh.Trimesh] = {}
    for name in ("bracket", "main_pin", "splint"):
        path = geometry_dir / f"{name}.obj"
        meshes[name] = trimesh.load_mesh(str(path), process=False, maintain_order=True)
    return meshes
