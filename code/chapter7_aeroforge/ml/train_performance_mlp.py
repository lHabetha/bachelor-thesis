#!/usr/bin/env python3
"""Train a single performance MLP: raw ADV → quick-sim vector (§19.14)."""

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


from chapter7_aeroforge.ml.models import (  # noqa: E402
    MLP_ARCHITECTURES,
    PerformanceMLP,
    performance_metrics,
    predict_performance_mlp,
    train_performance_mlp,
)
from chapter7_aeroforge.ml.sim_features import (  # noqa: E402
    DEFAULT_SIM_LABELS_PATH,
    SIM_TARGET_NAMES,
    SIM_TARGET_SPECS,
    load_sim_dataset,
)

from chapter7_aeroforge.release_paths import PERFORMANCE_MLP_CKPT, PERFORMANCE_MLP_DIR

DEFAULT_OUT_DIR = PERFORMANCE_MLP_DIR
DEFAULT_CHECKPOINT = PERFORMANCE_MLP_CKPT
DEFAULT_ARCH_ID = "mlp_256_128_64_32"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 180
DEFAULT_VAL_FRAC = 0.1


def _split_train_val(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _gradient_ranking_demo(
    model: PerformanceMLP,
    x_std,
    y_std,
    x_row: np.ndarray,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    *,
    target_index: int,
) -> list[tuple[str, float]]:
    """Return |∂y_j/∂x_i| ranked descending for one ADV row."""
    model.eval()
    xt = torch.tensor(x_std.transform(x_row), dtype=torch.float32, requires_grad=True)
    pred_z = model(xt)
    pred_z[0, target_index].backward()
    grad = xt.grad.detach().cpu().numpy()[0]
    ranked = sorted(
        ((feature_names[i], float(abs(grad[i]))) for i in range(len(feature_names))),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(description="Train performance MLP on quick-sim labels.")
    ap.add_argument("--labels", type=Path, default=DEFAULT_SIM_LABELS_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--arch-id", default=DEFAULT_ARCH_ID)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--skip-full-retrain", action="store_true", help="Only train on train split.")
    args = ap.parse_args()

    hidden = MLP_ARCHITECTURES[args.arch_id]
    t0 = time.perf_counter()

    dataset = load_sim_dataset(args.labels)
    n = len(dataset.x_raw)
    train_idx, val_idx = _split_train_val(n, args.val_frac, args.seed)

    x_train = dataset.x_raw[train_idx]
    y_train = dataset.y[train_idx]
    x_val = dataset.x_raw[val_idx]
    y_val = dataset.y[val_idx]

    print(f"Loaded {n} filtered rows from {args.labels}")
    print(f"  features={dataset.x_raw.shape[1]} targets={dataset.y.shape[1]}")
    print(f"  train={len(train_idx)} val={len(val_idx)} seed={args.seed}")

    model, x_std, y_std = train_performance_mlp(
        x_train,
        y_train,
        hidden=hidden,
        seed=args.seed,
        epochs=args.epochs,
    )
    y_val_pred = predict_performance_mlp(model, x_std, y_std, x_val)
    val_metrics = performance_metrics(y_val, y_val_pred, SIM_TARGET_NAMES)

    if not args.skip_full_retrain:
        print("Retraining on full filtered pool for export checkpoint...")
        model, x_std, y_std = train_performance_mlp(
            dataset.x_raw,
            dataset.y,
            hidden=hidden,
            seed=args.seed,
            epochs=args.epochs,
        )

    y_full_pred = predict_performance_mlp(model, x_std, y_std, dataset.x_raw)
    full_metrics = performance_metrics(dataset.y, y_full_pred, SIM_TARGET_NAMES)

    mean_adv = dataset.x_raw.mean(axis=0, keepdims=True)
    ld_analytic_idx = SIM_TARGET_NAMES.index("L_D_analytic")
    grad_rank_ld = _gradient_ranking_demo(
        model,
        x_std,
        y_std,
        mean_adv,
        dataset.spec.feature_names,
        SIM_TARGET_NAMES,
        target_index=ld_analytic_idx,
    )
    cd0_idx = SIM_TARGET_NAMES.index("CD0")
    grad_rank_cd0 = _gradient_ranking_demo(
        model,
        x_std,
        y_std,
        mean_adv,
        dataset.spec.feature_names,
        SIM_TARGET_NAMES,
        target_index=cd0_idx,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_type": "performance_mlp",
        "arch_id": args.arch_id,
        "hidden": hidden,
        "in_dim": int(dataset.x_raw.shape[1]),
        "out_dim": int(dataset.y.shape[1]),
        "engineered": False,
        "feature_names": list(dataset.spec.feature_names),
        "target_names": list(SIM_TARGET_NAMES),
        "target_specs": [
            {"name": s.name, "path": s.path, "lo": s.lo, "hi": s.hi} for s in SIM_TARGET_SPECS
        ],
        "x_mean": x_std.mean_.tolist(),
        "x_scale": x_std.scale_.tolist(),
        "y_mean": y_std.mean_.tolist(),
        "y_scale": y_std.scale_.tolist(),
        "seed": args.seed,
        "epochs": args.epochs,
        "n_train_rows": int(n),
        "n_filtered_from": 100_000,
        "labels_path": str(args.labels),
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, args.checkpoint)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch_id": args.arch_id,
        "hidden": list(hidden),
        "seed": args.seed,
        "epochs": args.epochs,
        "n_filtered": n,
        "n_train_split": int(len(train_idx)),
        "n_val_split": int(len(val_idx)),
        "val_frac": args.val_frac,
        "in_dim": int(dataset.x_raw.shape[1]),
        "out_dim": int(dataset.y.shape[1]),
        "target_names": list(SIM_TARGET_NAMES),
        "feature_names": list(dataset.spec.feature_names),
        "val_metrics": val_metrics,
        "full_fit_metrics": full_metrics,
        "checkpoint": str(args.checkpoint),
        "filter": {
            "vlm_ok": True,
            "physical_bounds": [
                {"name": s.name, "lo": s.lo, "hi": s.hi} for s in SIM_TARGET_SPECS
            ],
        },
        "gradient_demo_mean_adv": {
            "L_D_analytic_top5_sensitive": [
                {"feature": f, "abs_grad": g} for f, g in grad_rank_ld[:5]
            ],
            "L_D_analytic_bottom5_insensitive": [
                {"feature": f, "abs_grad": g} for f, g in grad_rank_ld[-5:]
            ],
            "CD0_bottom5_insensitive": [
                {"feature": f, "abs_grad": g} for f, g in grad_rank_cd0[-5:]
            ],
        },
        "wall_s": time.perf_counter() - t0,
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nValidation MAE (physical units, {len(val_idx)} rows):")
    for name in SIM_TARGET_NAMES:
        print(f"  {name:14s} {val_metrics[f'mae_{name}']:.6f}")
    print(f"  {'mean':14s} {val_metrics['mae_mean']:.6f}")
    print(f"\nFull-fit MAE ({n} rows): mean={full_metrics['mae_mean']:.6f}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Metrics:    {metrics_path}")
    print(f"Wall:       {report['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
