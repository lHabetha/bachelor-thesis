"""On-demand overlap label cache for Chapter 6."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path

from .analytic_overlap import ANALYTIC_LABEL_VERSION, compute_analytic_overlap_label
from .geometry import export_part_geometry, param_hash
from .paths import GEOMETRY_CACHE_DIR
from .release_paths import RESULTS_ROOT
from .sampler import DummyParams

DEFAULT_PITCH_MM = 0.5
DEFAULT_THRESHOLD_NORM = 5e-5
STRICT_THRESHOLD_NORM = 1e-6


def params_from_row(row: dict) -> DummyParams:
    fields = DummyParams.__dataclass_fields__.keys()
    return DummyParams(**{k: float(row[k]) for k in fields})


class LabelCache:
    def __init__(
        self,
        cache_dir: Path | None = None,
        geometry_dir: Path | None = None,
        *,
        pitch_mm: float = DEFAULT_PITCH_MM,
        threshold_norm: float = DEFAULT_THRESHOLD_NORM,
    ) -> None:
        self.cache_dir = cache_dir or RESULTS_ROOT / "labels_cache" / "analytic_voxel_v1"
        self.geometry_dir = geometry_dir or GEOMETRY_CACHE_DIR
        self.pitch_mm = pitch_mm
        self.threshold_norm = threshold_norm
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.geometry_dir.mkdir(parents=True, exist_ok=True)

    def label(self, p: DummyParams, *, retain_geometry: bool = False) -> dict:
        h = param_hash(p)
        cache_key = self._cache_key(h)
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = payload.get("label", {})
            if (
                label.get("label_backend_version") == ANALYTIC_LABEL_VERSION
                and float(label.get("pitch_mm", -1.0)) == float(self.pitch_mm)
                and float(label.get("overlap_threshold_norm", -1.0)) == float(self.threshold_norm)
            ):
                return payload

        label = compute_analytic_overlap_label(
            p,
            pitch_mm=self.pitch_mm,
            overlap_threshold_norm=self.threshold_norm,
        )
        geometry_paths = {}
        if retain_geometry:
            geometry_paths = export_part_geometry(p, self.geometry_dir / h)
        payload = {
            "label_cache_key": cache_key,
            "param_hash": h,
            "label_config": {
                "label_backend_version": ANALYTIC_LABEL_VERSION,
                "pitch_mm": float(self.pitch_mm),
                "overlap_threshold_norm": float(self.threshold_norm),
            },
            "params": {k: float(v) for k, v in asdict(p).items()},
            "label": label.to_dict(),
            "geometry_paths": geometry_paths,
            "retained_geometry": bool(retain_geometry),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _cache_key(self, param_hash_value: str) -> str:
        payload = {
            "param_hash": param_hash_value,
            "label_backend_version": ANALYTIC_LABEL_VERSION,
            "pitch_mm": float(self.pitch_mm),
            "overlap_threshold_norm": float(self.threshold_norm),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:20]
