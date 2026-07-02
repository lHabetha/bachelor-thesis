"""Model loading and gradient utilities for Chapter 5 optimization."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

_SPEC = importlib.util.spec_from_file_location("_chapter5_release_paths", _PKG / "release_paths.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load Chapter 5 release_paths from {_PKG / 'release_paths.py'}")
_RELEASE_PATHS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RELEASE_PATHS)

ensure_mlp_lib_importable = _RELEASE_PATHS.ensure_mlp_lib_importable

ensure_mlp_lib_importable()

from lib.model_mlp64 import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    FittedModel,
    MLP64,
    Standardizer,
    eval_holdout,
    fit_mlp,
)
from lib.formula_oracle import (  # noqa: E402
    label_params,
    params_from_dict,
    params_to_array,
)

__all__ = [
    "FittedModel",
    "MLP64",
    "Standardizer",
    "fit_mlp",
    "eval_holdout",
    "FEATURE_NAMES",
    "N_FEATURES",
    "label_params",
    "params_from_dict",
    "params_to_array",
    "compute_gradient",
    "predict_proba_with_grad",
    "load_model_artifact",
    "save_model_artifact",
]


def compute_gradient(model: FittedModel, x: np.ndarray) -> tuple[np.ndarray, float]:
    if hasattr(model, "gradient"):
        return model.gradient(x)

    model.model.eval()
    xz = model.scaler.transform(x.reshape(1, -1))
    xt = torch.as_tensor(xz, dtype=torch.float32)
    xt.requires_grad_(True)

    logit = model.model(xt)
    prob = torch.sigmoid(logit)
    prob.backward()

    grad_z = xt.grad.detach().numpy().flatten()
    grad_raw = grad_z / model.scaler.std
    return grad_raw.astype(np.float64), float(prob.item())


def predict_proba_with_grad(model: FittedModel, x: np.ndarray) -> tuple[float, np.ndarray]:
    grad, prob = compute_gradient(model, x)
    return prob, grad


def save_model_artifact(model: FittedModel, out_dir: Path, metadata: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), out_dir / "model.pt")
    np.savez(
        out_dir / "standardizer.npz",
        mean=model.scaler.mean,
        std=model.scaler.std,
    )
    import json

    card = {
        "architecture": "MLP64 (13->64->32->1)",
        "best_epoch": model.best_epoch,
        "best_val_loss": model.best_val_loss,
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
    }
    if metadata:
        card.update(metadata)
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))


def load_model_artifact(model_dir: Path) -> FittedModel:
    import json

    card = json.loads((model_dir / "model_card.json").read_text())
    if card.get("task30_tree_surrogate"):
        import joblib

        class Task30TreeArtifact:
            def __init__(self, model, model_card):
                self.model = model
                self.model_card = model_card

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(X)
                    if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
                        return proba[:, 1].astype(np.float64)
                    return np.asarray(proba).reshape(-1).astype(np.float64)
                if hasattr(self.model, "predict_proba_task30"):
                    return self.model.predict_proba_task30(X).astype(np.float64)
                raise TypeError("Chapter 4 tree artifact does not expose predict_proba")

        model = joblib.load(model_dir / "model.joblib")
        return Task30TreeArtifact(model, card)

    state_dict = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    npz = np.load(model_dir / "standardizer.npz")
    scaler = Standardizer(mean=npz["mean"], std=npz["std"])

    model = MLP64()
    model.load_state_dict(state_dict)
    model.eval()

    return FittedModel(
        scaler=scaler,
        model=model,
        best_epoch=card.get("best_epoch", 0),
        best_val_loss=card.get("best_val_loss", 0.0),
    )
