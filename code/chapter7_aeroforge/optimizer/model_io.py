#!/usr/bin/env python3
"""Surrogate I/O + ADV<->normalized transforms for the Chapter 7 optimizer workbench.

This module is the single source of truth for turning a 43-key AeroForge design
vector (ADV) into the feature vector the overlap MLP consumes, for predicting
overlap with the MLP, and for computing the MLP gradient w.r.t. the *normalized*
original numeric drivers. It is deliberately CAD-free: no airframe is ever built
here. The only thing it does is fast MLP inference and autograd. CadQuery
verification lives in `verify.py`; geometry export lives in `render_worker.py`.

Normalized space: every optimizable numeric driver is mapped to u in [0, 1] via
its sampler bounds (`DRIVER_SPECS` / `BACKGROUND_UNIFORM`). The optimizer always
walks in this u-space so step sizes and distances are comparable across drivers.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


from chapter7_aeroforge.ml.models import (  # noqa: E402
    DEFAULT_TAU_MM3,
    OverlapMLP,
    OverlapMLPMultitask,
    inverse_transformed_target,
)
from chapter7_aeroforge.overlap_search.core import ADV_KEYS  # noqa: E402
from chapter7_aeroforge.overlap_search.sampler import (  # noqa: E402
    BACKGROUND_UNIFORM,
    DRIVER_SPECS,
)

CATEGORICAL_COLS = (
    "design_name",
    "tail_type",
    "root_naca_code",
    "tip_naca_code",
    "airfoil_source",
    "hstab_airfoil_source",
    "vstab_airfoil_source",
    "hstab_root_naca_code",
    "hstab_tip_naca_code",
    "vstab_root_naca_code",
    "vstab_tip_naca_code",
)

# Smallest meaningful change in normalized space; below this a driver is "unmoved".
CHANGE_TOL = 1e-9

# Legacy sampler ranges — used only as a fallback. The 100k dataset is generated
# with a wider, multi-family distribution, so the real normalized space is the
# data-driven domain in `normalization_bounds.json` (see build_normalization.py).
_SAMPLER_BOUNDS: dict[str, tuple[float, float]] = {
    **{k: (float(v["lo"]), float(v["hi"])) for k, v in DRIVER_SPECS.items()},
    **{k: (float(v[0]), float(v[1])) for k, v in BACKGROUND_UNIFORM.items()},
}

from chapter7_aeroforge.release_paths import NORMALIZATION_JSON

NORMALIZATION_PATH = NORMALIZATION_JSON


def _load_numeric_bounds() -> dict[str, tuple[float, float]]:
    bounds = {k: (lo, hi) for k, (lo, hi) in _SAMPLER_BOUNDS.items()}
    if NORMALIZATION_PATH.exists():
        payload = json.loads(NORMALIZATION_PATH.read_text(encoding="utf-8"))
        for k, (lo, hi) in payload.get("bounds", {}).items():
            if hi > lo:
                bounds[k] = (float(lo), float(hi))
    return bounds


# Normalized-space bounds for every numeric driver the optimizer may move.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = _load_numeric_bounds()

# Drivers rounded to integers when written back to an ADV (manufacturing/grid).
_INT_KEYS = {"length", "wingspan", "hstab_semispan", "vstab_height"}
_R3_KEYS = {
    "wing_position",
    "tail_position",
    "wing_height_ratio",
    "taper",
    "hstab_taper",
    "vstab_taper",
    "end_cap_percent",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _one_hot_col_value(adv: dict, one_hot_col: str) -> float:
    for col in sorted(CATEGORICAL_COLS, key=len, reverse=True):
        prefix = f"{col}_"
        if one_hot_col.startswith(prefix):
            return 1.0 if str(adv.get(col)) == one_hot_col[len(prefix) :] else 0.0
    return 0.0


def _engineered_values_np(values: dict[str, float]) -> dict[str, float]:
    tail_wing_dx = (values["tail_position"] - values["wing_position"]) * values["length"]
    wing_mean_chord = values["wingspan"] / max(values["aspect_ratio"], 1e-9)
    hstab_mean_chord = (2.0 * values["hstab_semispan"]) / max(values["hstab_aspect_ratio"], 1e-9)
    return {
        "tail_wing_dx": tail_wing_dx,
        "wing_x": values["wing_position"] * values["length"],
        "tail_x": values["tail_position"] * values["length"],
        "wing_mean_chord": wing_mean_chord,
        "hstab_mean_chord": hstab_mean_chord,
        "effective_dx": tail_wing_dx - 0.5 * wing_mean_chord - 0.5 * hstab_mean_chord,
    }


def _engineered_values_torch(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tail_wing_dx = (values["tail_position"] - values["wing_position"]) * values["length"]
    wing_mean_chord = values["wingspan"] / torch.clamp(values["aspect_ratio"], min=1e-9)
    hstab_mean_chord = (2.0 * values["hstab_semispan"]) / torch.clamp(
        values["hstab_aspect_ratio"], min=1e-9
    )
    return {
        "tail_wing_dx": tail_wing_dx,
        "wing_x": values["wing_position"] * values["length"],
        "tail_x": values["tail_position"] * values["length"],
        "wing_mean_chord": wing_mean_chord,
        "hstab_mean_chord": hstab_mean_chord,
        "effective_dx": tail_wing_dx - 0.5 * wing_mean_chord - 0.5 * hstab_mean_chord,
    }


def opt_keys_for_adv(adv: dict) -> list[str]:
    """The numeric drivers the optimizer is allowed to move for this design.

    Categoricals and null CSV-path keys are frozen; `v_tail_angle` is frozen for
    conventional tails (it has no geometric effect there).
    """
    keys: list[str] = []
    for key in ADV_KEYS:
        if key not in NUMERIC_BOUNDS:
            continue
        if not isinstance(adv.get(key), (int, float)):
            continue
        if key == "v_tail_angle" and adv.get("tail_type") == "conv_tail":
            continue
        keys.append(key)
    return keys


def u_from_adv(adv: dict, opt_keys: list[str]) -> np.ndarray:
    vals = []
    for key in opt_keys:
        lo, hi = NUMERIC_BOUNDS[key]
        vals.append(np.clip((_safe_float(adv.get(key)) - lo) / (hi - lo), 0.0, 1.0))
    return np.array(vals, dtype=np.float64)


def adv_from_u(adv: dict, opt_keys: list[str], u: np.ndarray) -> dict:
    """Write a normalized vector back into an ADV, touching only moved drivers.

    Every key the optimizer did NOT move (|u - u_start| <= CHANGE_TOL) is copied
    verbatim from `adv`, so the round-trip is exactly lossless for unchanged
    coordinates. This avoids silently snapping in-distribution-but-near-edge
    values when a design barely moves.
    """
    out = copy.deepcopy(adv)
    u0 = u_from_adv(adv, opt_keys)
    for i, key in enumerate(opt_keys):
        if abs(float(u[i]) - float(u0[i])) <= CHANGE_TOL:
            continue  # unmoved: keep the exact original value
        lo, hi = NUMERIC_BOUNDS[key]
        raw = lo + float(np.clip(u[i], 0.0, 1.0)) * (hi - lo)
        if key in _INT_KEYS:
            out[key] = float(int(round(raw)))
        elif key in _R3_KEYS:
            out[key] = round(raw, 3)
        else:
            out[key] = round(raw, 2)
    if out.get("tail_type") == "conv_tail":
        out["v_tail_angle"] = 0.0
    return out


def _adv_to_feature_np(adv: dict, ckpt: dict) -> np.ndarray:
    raw_values = {k: _safe_float(adv.get(k)) for k in ADV_KEYS if k in adv}
    if ckpt.get("engineered", False):
        raw_values.update(_engineered_values_np(raw_values))
    vals: list[float] = [float(raw_values[col]) for col in ckpt["numeric_cols"]]
    vals.extend(_one_hot_col_value(adv, col) for col in ckpt["one_hot_cols"])
    return np.array(vals, dtype=np.float64)


def _feature_from_u_torch(
    u: torch.Tensor,
    *,
    adv_fixed: dict,
    opt_keys: list[str],
    ckpt: dict,
    feature_grad_mode: str = "full",
) -> torch.Tensor:
    """Build the model's feature vector as a differentiable function of `u`.

    `feature_grad_mode` controls how gradients flow through the six engineered
    features (only relevant for engineered checkpoints):
      * ``full``     — autograd chains through the engineered features back to the
                       raw drivers (total derivative).
      * ``detached`` — the engineered columns keep their correct numeric value in
                       the forward pass (prediction unchanged) but are detached in
                       the backward pass, so only the direct ∂f/∂x path survives.
      * ``raw``      — for non-engineered checkpoints; identical to ``full`` since
                       there are no engineered features to chain through.
    """
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
    if ckpt.get("engineered", False):
        engineered = _engineered_values_torch(values)
        if feature_grad_mode == "detached":
            # Forward value unchanged; zero gradient contribution from the
            # engineered pathway (treat engineered columns as constants).
            engineered = {k: v.detach() for k, v in engineered.items()}
        values.update(engineered)
    features: list[torch.Tensor] = [values[col] for col in ckpt["numeric_cols"]]
    one_hot_vals = [_one_hot_col_value(adv_fixed, col) for col in ckpt["one_hot_cols"]]
    features.extend(torch.tensor(v, dtype=torch.float32) for v in one_hot_vals)
    return torch.stack(features)


@dataclass
class Surrogate:
    """A loaded overlap MLP + its standardizers. Pure-MLP, no CAD."""

    model: OverlapMLP | OverlapMLPMultitask
    ckpt: dict
    tau_mm3: float = DEFAULT_TAU_MM3

    @classmethod
    def load(cls, checkpoint: Path | str, tau_mm3: float = DEFAULT_TAU_MM3) -> "Surrogate":
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        hidden = tuple(ckpt["hidden"])
        in_dim = int(ckpt["in_dim"])
        multitask = bool(ckpt.get("multitask")) or ckpt.get("loss_variant") == "multitask_gate"
        if multitask:
            model: OverlapMLP | OverlapMLPMultitask = OverlapMLPMultitask(in_dim, hidden=hidden)
        else:
            model = OverlapMLP(in_dim, hidden=hidden)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model=model, ckpt=ckpt, tau_mm3=tau_mm3)

    @property
    def model_id(self) -> str:
        return str(self.ckpt.get("model_id", "unknown"))

    @property
    def is_multitask(self) -> bool:
        return isinstance(self.model, OverlapMLPMultitask)

    @property
    def engineered(self) -> bool:
        return bool(self.ckpt.get("engineered", False))

    def _standardize(self, feature_raw: torch.Tensor) -> torch.Tensor:
        ckpt = self.ckpt
        outer_mean = torch.tensor(ckpt["outer_mean"], dtype=torch.float32)
        outer_scale = torch.tensor(ckpt["outer_scale"], dtype=torch.float32)
        inner_mean = torch.tensor(ckpt["inner_mean"], dtype=torch.float32)
        inner_scale = torch.tensor(ckpt["inner_scale"], dtype=torch.float32)
        x_outer = (feature_raw - outer_mean) / outer_scale
        return (x_outer - inner_mean) / inner_scale

    def _z_from_feature(self, feature_raw: torch.Tensor) -> torch.Tensor:
        x_batch = self._standardize(feature_raw).unsqueeze(0)
        if isinstance(self.model, OverlapMLPMultitask):
            return self.model.forward_z(x_batch).squeeze(0)
        return self.model(x_batch).squeeze(0)

    def predict(self, adv: dict) -> tuple[float, float]:
        """Return (z, overlap_mm3) for one design via a single forward pass."""
        feature = torch.tensor(_adv_to_feature_np(adv, self.ckpt), dtype=torch.float32)
        with torch.no_grad():
            z = torch.clamp(self._z_from_feature(feature), min=0.0).item()
        pred = float(inverse_transformed_target(np.array([z]), tau_mm3=self.tau_mm3)[0])
        return z, pred

    def predict_with_prob(self, adv: dict) -> tuple[float, float, float | None]:
        """Return (z, overlap_mm3, p_overlap). p_overlap is None for non-multitask.

        For the multitask model both the regression head (z) and the binary
        overlap head (P(overlap)) come from a single forward pass.
        """
        feature = torch.tensor(_adv_to_feature_np(adv, self.ckpt), dtype=torch.float32)
        with torch.no_grad():
            x_batch = self._standardize(feature).unsqueeze(0)
            if isinstance(self.model, OverlapMLPMultitask):
                z_raw, binary_logit = self.model(x_batch)
                z = float(torch.clamp(z_raw.squeeze(0), min=0.0).item())
                p_overlap: float | None = float(torch.sigmoid(binary_logit).reshape(-1)[0].item())
            else:
                z = float(torch.clamp(self.model(x_batch).squeeze(0), min=0.0).item())
                p_overlap = None
        pred = float(inverse_transformed_target(np.array([z]), tau_mm3=self.tau_mm3)[0])
        return z, pred, p_overlap

    def predict_binary(self, adv: dict) -> float | None:
        """P(overlap) from the explicit binary head (multitask only, else None)."""
        return self.predict_with_prob(adv)[2]

    def is_clean(
        self,
        adv: dict,
        *,
        gate: str = "reg",
        tau_decide: float | None = None,
        p_star: float = 0.5,
    ) -> bool:
        """Surrogate 'repaired' verdict at an operating point.

        * ``reg``        — pred <= tau_decide.
        * ``binary_and`` — pred <= tau_decide AND P(overlap) < p_star (multitask).
        ``tau_decide`` defaults to the ground-truth tau (1 mm^3) when None.
        """
        td = self.tau_mm3 if tau_decide is None else float(tau_decide)
        _z, pred, p_overlap = self.predict_with_prob(adv)
        if pred > td:
            return False
        if gate == "binary_and" and p_overlap is not None:
            return p_overlap < float(p_star)
        return True

    def gradient(self, adv: dict, opt_keys: list[str], *, feature_grad_mode: str = "full") -> np.ndarray:
        """d z / d u for every optimizable driver (normalized space).

        `feature_grad_mode` (``full`` | ``detached`` | ``raw``) controls whether the
        engineered-feature pathway contributes to the gradient (see
        `_feature_from_u_torch`). For non-engineered checkpoints it is a no-op.
        """
        u0 = u_from_adv(adv, opt_keys)
        u = torch.tensor(u0.astype(np.float32), requires_grad=True)
        feature = _feature_from_u_torch(
            u, adv_fixed=adv, opt_keys=opt_keys, ckpt=self.ckpt, feature_grad_mode=feature_grad_mode
        )
        z = self._z_from_feature(feature)
        z.backward()
        return u.grad.detach().cpu().numpy().astype(np.float64)

    def gradient_top_k(
        self, adv: dict, k: int, *, feature_grad_mode: str = "full"
    ) -> tuple[list[dict], np.ndarray, list[str]]:
        """Top-k most active normalized gradients + a unit descent direction.

        ``k`` may be an int or the string ``"all"`` (use every driver).
        """
        opt_keys = opt_keys_for_adv(adv)
        u0 = u_from_adv(adv, opt_keys)
        grad = self.gradient(adv, opt_keys, feature_grad_mode=feature_grad_mode)
        ranked = sorted(
            (
                {
                    "key": key,
                    "grad_normalized": float(grad[i]),
                    "abs_grad_normalized": float(abs(grad[i])),
                    "u_start": float(u0[i]),
                }
                for i, key in enumerate(opt_keys)
            ),
            key=lambda r: r["abs_grad_normalized"],
            reverse=True,
        )
        kk = len(opt_keys) if (isinstance(k, str) and k == "all") else int(k)
        top = ranked[:kk]
        direction = np.zeros_like(u0)
        for row in top:
            idx = opt_keys.index(row["key"])
            direction[idx] = -grad[idx]  # descend overlap
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction /= norm
        return top, direction, opt_keys
