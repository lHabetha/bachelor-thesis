#!/usr/bin/env python3
"""Performance-MLP wrapper for aero-preserving overlap repair.

The overlap repair optimizers move numeric ADV drivers in normalized ``u`` space.
This module mirrors the differentiable ``u -> features -> MLP`` path from
``model_io.Surrogate``, but for the quick-sim performance MLP trained in §19.14.
It is CAD-free and intended for optimizer-time guidance only; the full VLM
``quicksim_eval`` pass remains the post-hoc truth source.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


from chapter7_aeroforge.ml.models import PerformanceMLP  # noqa: E402
from chapter7_aeroforge.optimizer.model_io import (  # noqa: E402
    ADV_KEYS,
    NUMERIC_BOUNDS,
    _one_hot_col_value,
    adv_from_u,
    opt_keys_for_adv,
    u_from_adv,
)

from chapter7_aeroforge.release_paths import PERFORMANCE_MLP_CKPT

DEFAULT_PERF_CHECKPOINT = PERFORMANCE_MLP_CKPT

CORE4_NAMES = ("L_D_vlm", "CD0", "CL_vlm", "Cm_vlm")
REPORTABLE_REL_NAMES = ("L_D_vlm", "CD0")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class PerfSurrogate:
    """Loaded performance MLP with u-space value, drift, and Jacobian helpers."""

    model: PerformanceMLP
    ckpt: dict

    @classmethod
    def load(cls, checkpoint: Path | str = DEFAULT_PERF_CHECKPOINT) -> "PerfSurrogate":
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if ckpt.get("model_type") != "performance_mlp":
            raise ValueError(f"not a performance MLP checkpoint: {checkpoint}")
        model = PerformanceMLP(
            int(ckpt["in_dim"]),
            int(ckpt["out_dim"]),
            hidden=tuple(ckpt.get("hidden", (256, 128, 64, 32))),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model=model, ckpt=ckpt)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.ckpt["target_names"])

    @property
    def core_indices(self) -> list[int]:
        by_name = {name: i for i, name in enumerate(self.target_names)}
        missing = [name for name in CORE4_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"performance checkpoint missing core targets: {missing}")
        return [by_name[name] for name in CORE4_NAMES]

    def _feature_from_u_torch(self, adv_fixed: dict, opt_keys: list[str], u: torch.Tensor) -> torch.Tensor:
        values: dict[str, torch.Tensor] = {}
        u_by_key = {k: u[i] for i, k in enumerate(opt_keys)}
        for key in ADV_KEYS:
            if key not in adv_fixed:
                continue
            if key in u_by_key:
                lo, hi = NUMERIC_BOUNDS[key]
                values[key] = torch.tensor(lo, dtype=torch.float32) + u_by_key[key] * float(hi - lo)
            elif isinstance(adv_fixed.get(key), (int, float)):
                values[key] = torch.tensor(float(adv_fixed[key]), dtype=torch.float32)

        features: list[torch.Tensor] = []
        for col in self.ckpt["feature_names"]:
            if col in values:
                features.append(values[col])
            else:
                features.append(torch.tensor(_one_hot_col_value(adv_fixed, col), dtype=torch.float32))
        return torch.stack(features)

    def _predict_y_torch(self, feature_raw: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.ckpt["x_mean"], dtype=torch.float32)
        scale = torch.tensor(self.ckpt["x_scale"], dtype=torch.float32)
        y_mean = torch.tensor(self.ckpt["y_mean"], dtype=torch.float32)
        y_scale = torch.tensor(self.ckpt["y_scale"], dtype=torch.float32)
        x = ((feature_raw - mean) / scale).unsqueeze(0)
        return (self.model(x).squeeze(0) * y_scale) + y_mean

    def predict_from_u(self, adv0: dict, opt_keys: list[str], u: np.ndarray) -> dict[str, float]:
        ut = torch.tensor(np.asarray(u, dtype=np.float32), dtype=torch.float32)
        feature = self._feature_from_u_torch(adv0, opt_keys, ut)
        with torch.no_grad():
            y = self._predict_y_torch(feature).detach().cpu().numpy().astype(float)
        return {name: float(y[i]) for i, name in enumerate(self.target_names)}

    def predict(self, adv: dict) -> dict[str, float]:
        opt_keys = opt_keys_for_adv(adv)
        return self.predict_from_u(adv, opt_keys, u_from_adv(adv, opt_keys))

    def core_vector_from_u(self, adv0: dict, opt_keys: list[str], u: np.ndarray) -> np.ndarray:
        pred = self.predict_from_u(adv0, opt_keys, u)
        return np.array([pred[name] for name in CORE4_NAMES], dtype=np.float64)

    def core_scale(self) -> np.ndarray:
        y_scale = np.asarray(self.ckpt["y_scale"], dtype=np.float64)
        return y_scale[self.core_indices]

    def target_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {}
        for spec in self.ckpt.get("target_specs", []):
            if isinstance(spec, dict) and {"name", "lo", "hi"} <= set(spec):
                bounds[str(spec["name"])] = (float(spec["lo"]), float(spec["hi"]))
        return bounds

    def standardized_delta(
        self,
        adv0: dict,
        opt_keys: list[str],
        u: np.ndarray,
        *,
        u_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        if u_ref is None:
            u_ref = u_from_adv(adv0, opt_keys)
        return (self.core_vector_from_u(adv0, opt_keys, u) - self.core_vector_from_u(adv0, opt_keys, u_ref)) / self.core_scale()

    def drift_from_u(self, adv0: dict, opt_keys: list[str], u: np.ndarray, *, u_ref: np.ndarray | None = None) -> float:
        d = self.standardized_delta(adv0, opt_keys, u, u_ref=u_ref)
        return float(np.dot(d, d))

    def drift_score(self, adv0: dict, adv: dict, opt_keys: list[str]) -> float:
        return self.drift_from_u(adv0, opt_keys, u_from_adv(adv, opt_keys))

    def drift_radius_from_u(
        self, adv0: dict, opt_keys: list[str], u: np.ndarray, *, u_ref: np.ndarray | None = None
    ) -> float:
        return float(np.sqrt(max(self.drift_from_u(adv0, opt_keys, u, u_ref=u_ref), 0.0)))

    def jacobian_core4(self, adv0: dict, opt_keys: list[str], u: np.ndarray | None = None) -> np.ndarray:
        """Return standardized core-4 Jacobian, rows = δ metrics, cols = u drivers."""
        if u is None:
            u = u_from_adv(adv0, opt_keys)
        ut = torch.tensor(np.asarray(u, dtype=np.float32), requires_grad=True)
        feature = self._feature_from_u_torch(adv0, opt_keys, ut)
        y = self._predict_y_torch(feature)
        rows = []
        scale = torch.tensor(self.core_scale(), dtype=torch.float32)
        for row_i, idx in enumerate(self.core_indices):
            if ut.grad is not None:
                ut.grad.zero_()
            (y[idx] / scale[row_i]).backward(retain_graph=True)
            rows.append(ut.grad.detach().cpu().numpy().astype(np.float64).copy())
        return np.vstack(rows) if rows else np.zeros((0, len(opt_keys)), dtype=np.float64)

    def drift_gradient(
        self,
        adv0: dict,
        opt_keys: list[str],
        u: np.ndarray | None = None,
        *,
        u_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        if u is None:
            u = u_from_adv(adv0, opt_keys)
        if u_ref is None:
            u_ref = u_from_adv(adv0, opt_keys)
        delta = self.standardized_delta(adv0, opt_keys, u, u_ref=u_ref)
        j = self.jacobian_core4(adv0, opt_keys, u=u)
        return (2.0 * delta) @ j

    def summary_for_u(self, adv0: dict, opt_keys: list[str], u: np.ndarray, *, u_ref: np.ndarray | None = None) -> dict:
        if u_ref is None:
            u_ref = u_from_adv(adv0, opt_keys)
        start = self.core_vector_from_u(adv0, opt_keys, u_ref)
        final = self.core_vector_from_u(adv0, opt_keys, u)
        std_delta = (final - start) / self.core_scale()
        rel = {}
        for name, s, f in zip(CORE4_NAMES, start, final):
            if name in REPORTABLE_REL_NAMES and abs(float(s)) > 1e-12:
                rel[name] = float((f - s) / abs(s))
        bounds = self.target_bounds()
        in_bounds = True
        for name, value in zip(CORE4_NAMES, start):
            if name in bounds:
                lo, hi = bounds[name]
                in_bounds = in_bounds and (lo <= float(value) <= hi)
        return {
            "core4": list(CORE4_NAMES),
            "start": {name: float(v) for name, v in zip(CORE4_NAMES, start)},
            "final": {name: float(v) for name, v in zip(CORE4_NAMES, final)},
            "delta": {name: float(f - s) for name, s, f in zip(CORE4_NAMES, start, final)},
            "standardized_delta": {name: float(v) for name, v in zip(CORE4_NAMES, std_delta)},
            "reportable_relative_delta": rel,
            "D_aero": float(np.dot(std_delta, std_delta)),
            "R_aero": float(np.linalg.norm(std_delta)),
            "start_within_training_bounds": bool(in_bounds),
        }


_CACHE: dict[str, PerfSurrogate] = {}


def get_perf_surrogate(checkpoint: Path | str | None = DEFAULT_PERF_CHECKPOINT) -> PerfSurrogate:
    path = str(Path(checkpoint or DEFAULT_PERF_CHECKPOINT).resolve())
    sur = _CACHE.get(path)
    if sur is None:
        sur = PerfSurrogate.load(path)
        _CACHE[path] = sur
    return sur


def adv_from_u_for_perf(adv0: dict, opt_keys: list[str], u: np.ndarray) -> dict:
    """Tiny public alias used by tests/importers to avoid reaching into model_io."""
    return adv_from_u(adv0, opt_keys, u)
