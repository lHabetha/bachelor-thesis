"""Verification checks for Chapter 4 graph-surrogate artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .lib.data_utils import FEATURE_LIST, load_bundle
from .lib.paths import ARCH_SCREEN_DIR, FIXED_GRID_DIR, HOLDOUT, MODELS_DIR, REPORTS_DIR, TASK22_DENSE
from .lib.train_eval import load_artifact


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _check_paths() -> list[str]:
    checked = []
    for path in [TASK22_DENSE, HOLDOUT]:
        if not path.exists():
            _fail(f"Missing input path: {path}")
        checked.append(str(path))
    return checked


def _check_no_holdout_leakage() -> None:
    bundle = load_bundle()
    holdout_ids = set(bundle.holdout["param_id"].astype(str))
    train_ids = set(bundle.pool["param_id"].astype(str))
    overlap = train_ids & holdout_ids
    if overlap:
        _fail(f"Dense50k pool overlaps holdout: {len(overlap)} IDs")


def _check_results_schema(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    required = {"architecture", "status", "balanced_accuracy", "recall_blocked", "total_labels"}
    missing = required - set(df.columns)
    if missing:
        _fail(f"{path} missing columns: {sorted(missing)}")
    return int(len(df))


def _check_model_artifacts() -> list[str]:
    checked = []
    for card_path in MODELS_DIR.glob("*/model_card.json"):
        model_dir = card_path.parent
        for name in ["model.pt", "standardizer.npz"]:
            if not (model_dir / name).exists():
                _fail(f"Missing {name} in {model_dir}")
        fitted = load_artifact(model_dir)
        x = load_bundle().holdout_X[0]
        prob = float(fitted.predict_proba(x.reshape(1, -1))[0])
        grad, prob2 = fitted.gradient(x)
        if not (0.0 <= prob <= 1.0) or abs(prob - prob2) > 1e-6:
            _fail(f"Bad probability/gradient agreement in {model_dir}")
        if grad.shape != (len(FEATURE_LIST),) or not np.all(np.isfinite(grad)):
            _fail(f"Bad gradient shape/values in {model_dir}")
        checked.append(str(model_dir))
    return checked


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "input_paths": _check_paths(),
        "no_holdout_leakage": True,
        "arch_rows": 0,
        "fixed_rows": 0,
        "model_artifacts": [],
    }
    _check_no_holdout_leakage()
    report["arch_rows"] = _check_results_schema(ARCH_SCREEN_DIR / "architecture_metrics.csv")
    report["fixed_rows"] = _check_results_schema(FIXED_GRID_DIR / "fixed_grid_metrics.csv")
    report["model_artifacts"] = _check_model_artifacts()
    (REPORTS_DIR / "verification_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[verify] ERROR: {exc}", file=sys.stderr)
        raise
