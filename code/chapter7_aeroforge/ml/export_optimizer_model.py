#!/usr/bin/env python3
"""Export full-data overlap MLP checkpoints for the optimizer workbench."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


from chapter7_aeroforge.ml.features import (  # noqa: E402
    ENGINEERED_FEATURE_NAMES,
    load_labeled_dataset,
)
from chapter7_aeroforge.ml.models import (  # noqa: E402
    DEFAULT_GATE_LOGIT_SCALE,
    DEFAULT_TAU_MM3,
    GRID_VARIANTS,
    MLP_ARCHITECTURES,
    OverlapMLP,
    OverlapMLPMultitask,
    train_mlp,
    train_mlp_gate_augmented,
    train_mlp_multitask_gate,
)

from chapter7_aeroforge.release_paths import CHECKPOINTS_DIR, LABELS_CLOUD, REGISTRY_JSON

DEFAULT_LABELS_PATH = LABELS_CLOUD
DEFAULT_MODELS_DIR = CHECKPOINTS_DIR
DEFAULT_ARCH_ID = "mlp_256_128_64_32"
DEFAULT_REGISTRY = REGISTRY_JSON

# Optimizer-facing model ids for full-100k exports.
EXPORT_MODEL_IDS = {
    "raw_log_huber": "raw_log_huber_100k",
    "eng_log_huber": "eng_log_huber_100k",
    "eng_gate_aug_strong": "eng_gate_aug_strong_100k",
    "eng_multitask_gate_strong": "eng_multitask_gate_strong_100k",
}


def _outer_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (x - mean) / scale, mean, scale


def export_checkpoint(
    *,
    variant_id: str,
    labels_path: Path,
    out_path: Path,
    arch_id: str = DEFAULT_ARCH_ID,
    seed: int = 42,
    epochs: int = 180,
    max_rows: int | None = None,
) -> dict:
    if variant_id not in GRID_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_id}")

    spec = GRID_VARIANTS[variant_id]
    hidden = MLP_ARCHITECTURES[arch_id]
    engineered = bool(spec["engineered"])

    t0 = time.perf_counter()
    dataset = load_labeled_dataset(labels_path, engineered=engineered)
    n = len(dataset.x_raw) if max_rows is None else min(max_rows, len(dataset.x_raw))
    x_raw = dataset.x_raw[:n]
    y = dataset.y_mm3[:n]

    x_outer, outer_mean, outer_scale = _outer_standardize(x_raw)

    loss_variant = str(spec["loss_variant"])
    multitask = bool(spec.get("multitask", False))
    gate_lambda = float(spec.get("gate_lambda", 0.0))
    gate_logit_scale = float(spec.get("gate_logit_scale", DEFAULT_GATE_LOGIT_SCALE))

    if multitask:
        model, inner_std = train_mlp_multitask_gate(
            x_outer,
            y,
            hidden=hidden,
            seed=seed,
            epochs=epochs,
            tau_mm3=DEFAULT_TAU_MM3,
            gate_lambda=gate_lambda,
        )
    elif loss_variant == "gate_augmented":
        model, inner_std = train_mlp_gate_augmented(
            x_outer,
            y,
            hidden=hidden,
            seed=seed,
            epochs=epochs,
            tau_mm3=DEFAULT_TAU_MM3,
            gate_lambda=gate_lambda,
            gate_logit_scale=gate_logit_scale,
        )
    else:
        model, inner_std = train_mlp(
            x_outer,
            y,
            hidden=hidden,
            seed=seed,
            epochs=epochs,
            tau_mm3=DEFAULT_TAU_MM3,
        )

    model_id = EXPORT_MODEL_IDS.get(variant_id, f"{variant_id}_100k")
    ckpt = {
        "model_id": model_id,
        "variant_id": variant_id,
        "arch_id": arch_id,
        "hidden": list(hidden),
        "loss_variant": loss_variant,
        "engineered": engineered,
        "engineered_features": list(ENGINEERED_FEATURE_NAMES) if engineered else [],
        "multitask": multitask,
        "gate_lambda": gate_lambda,
        "gate_logit_scale": gate_logit_scale,
        "train_rows": int(n),
        "labels_path": str(labels_path.resolve()),
        "feature_names": list(dataset.spec.feature_names),
        "numeric_cols": list(dataset.spec.numeric_cols),
        "one_hot_cols": list(dataset.spec.one_hot_cols),
        "in_dim": int(x_raw.shape[1]),
        "outer_mean": outer_mean.astype(np.float64).tolist(),
        "outer_scale": outer_scale.astype(np.float64).tolist(),
        "inner_mean": inner_std.mean_.astype(np.float64).tolist(),
        "inner_scale": inner_std.scale_.astype(np.float64).tolist(),
        "state_dict": model.state_dict(),
        "trained_wall_s": time.perf_counter() - t0,
        "export_seed": seed,
        "export_epochs": epochs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return ckpt


def _update_registry(models_dir: Path, entries: list[dict]) -> None:
    registry_path = models_dir / "registry.json"
    existing: dict = {}
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))

    by_id = {e["model_id"]: e for e in existing.get("models", [])}
    for entry in entries:
        by_id[entry["model_id"]] = entry

    payload = {
        "note": (
            "Optimizer-ready overlap surrogate checkpoints. "
            "Gradients use the regression overlap head (z). "
            "Multitask models ignore the binary head at inference/optimization time."
        ),
        "default_for_optimizer": {
            "best_volume_mae": "eng_log_huber_100k",
            "best_gate_f1": "eng_gate_aug_strong_100k",
            "best_spearman": "eng_gate_aug_strong_100k",
            "best_multitask_binary": "eng_multitask_gate_strong_100k",
        },
        "legacy_20k_models": ["eng_log_huber_20k", "eng_gate_aug_20k"],
        "models": sorted(by_id.values(), key=lambda e: e["model_id"]),
    }
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export optimizer overlap MLP checkpoints")
    ap.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--variant", type=str, default=None, choices=list(GRID_VARIANTS.keys()))
    ap.add_argument("--all", action="store_true", help="Export all four grid variants")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Output .pt path (single variant)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(GRID_VARIANTS.keys()) if args.all else [args.variant]
    if not variants or variants == [None]:
        raise SystemExit("Specify --variant <id> or --all")

    entries: list[dict] = []
    for variant_id in variants:
        model_id = EXPORT_MODEL_IDS[variant_id]
        out_path = args.out if (args.out and not args.all) else args.models_dir / f"{model_id}.pt"
        print(f"Exporting {variant_id} -> {out_path} ...")
        ckpt = export_checkpoint(
            variant_id=variant_id,
            labels_path=args.labels_path,
            out_path=out_path,
            seed=args.seed,
            epochs=args.epochs,
            max_rows=args.max_rows,
        )
        print(
            f"  saved {out_path.name} rows={ckpt['train_rows']} "
            f"in_dim={ckpt['in_dim']} wall={ckpt['trained_wall_s']:.1f}s"
        )
        entries.append(
            {
                "model_id": ckpt["model_id"],
                "variant_id": variant_id,
                "path": str(out_path.resolve()),
                "loss_variant": ckpt["loss_variant"],
                "engineered": ckpt["engineered"],
                "multitask": ckpt["multitask"],
                "train_rows": ckpt["train_rows"],
                "labels_path": ckpt["labels_path"],
                "optimizer_use": "regression head z for predict + gradient",
                "recommended_for": {
                    "raw_log_huber": "volume MAE baseline without engineered features",
                    "eng_log_huber": "best headline overlap volume accuracy",
                    "eng_gate_aug_strong": "best gate F1 / Spearman for boundary-aware repair",
                    "eng_multitask_gate_strong": "explicit binary overlap head (reg head still used by optimizer)",
                }.get(variant_id, "general"),
            }
        )

    _update_registry(args.models_dir, entries)
    print(f"Updated registry: {args.models_dir / 'registry.json'}")


if __name__ == "__main__":
    main()
