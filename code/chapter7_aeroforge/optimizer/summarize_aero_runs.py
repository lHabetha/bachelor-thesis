#!/usr/bin/env python3
"""Robust summary tables for the aero-preserving optimizer study."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


from chapter7_aeroforge.optimizer.perf_surrogate import (  # noqa: E402
    CORE4_NAMES,
    DEFAULT_PERF_CHECKPOINT,
    get_perf_surrogate,
)
from chapter7_aeroforge.optimizer.run_aero_sweep import GROUP_ID, MODEL_ID  # noqa: E402

from chapter7_aeroforge.release_paths import RUNS_DIR, TABLES_DIR

DEFAULT_RUNS_DIR = RUNS_DIR
DEFAULT_ANALYSIS_DIR = TABLES_DIR

SIM_KEY = {
    "L_D_vlm": "L_over_D_from_vlm",
    "CD0": "CD0",
    "CL_vlm": "CL_from_vlm",
    "Cm_vlm": "Cm_from_vlm",
}

TWIN_MAP = {
    "aero_penalized_receding": "receding_multistep_penalty",
    "aero_tangent_receding": "receding_multistep_gradient",
    "aero_budget_trust_region": "trust_region_hybrid_shrinkage",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def _iqr(xs: list[float]) -> float | None:
    return float(np.percentile(xs, 75) - np.percentile(xs, 25)) if xs else None


def _p90(xs: list[float]) -> float | None:
    return float(np.percentile(xs, 90)) if xs else None


def _fmt(v):
    return "" if v is None else f"{float(v):.8g}"


def _rankdata(xs: list[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    sx, sy = float(np.std(rx)), float(np.std(ry))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _actual_r_by_rank(run_dir: Path, scales: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[int, float]:
    path = run_dir / "sim_eval.json"
    if not path.exists():
        return {}
    sim = _read_json(path)
    out: dict[int, float] = {}
    for row in sim.get("starts", []):
        if not row.get("success") or not row.get("start_vlm_ok") or not row.get("final_vlm_ok"):
            continue
        vals = []
        in_bounds = True
        for name in CORE4_NAMES:
            key = SIM_KEY[name]
            start_value = (row.get("start") or {}).get(key)
            if start_value is None:
                vals = []
                break
            if name in bounds:
                lo, hi = bounds[name]
                in_bounds = in_bounds and (lo <= float(start_value) <= hi)
            d = (row.get("delta") or {}).get(key)
            if d is None:
                vals = []
                break
            vals.append(float(d) / scales[name])
        if vals and in_bounds:
            out[int(row["rank"])] = float(np.linalg.norm(vals))
    return out


def _discover_aero_runs(runs_dir: Path) -> list[dict]:
    rows = []
    for vd in sorted(runs_dir.glob(f"{GROUP_ID}__*/viewer_data.json")):
        run_dir = vd.parent
        data = _read_json(vd)
        if data.get("optimizer_id") == "all_aero":
            continue
        stats = data.get("statistics", {})
        ap = stats.get("aero_preserve", {})
        rows.append(
            {
                "run_dir": run_dir,
                "run_id": data.get("run_id", run_dir.name),
                "optimizer": data.get("optimizer_id"),
                "optimizer_label": data.get("optimizer_label"),
                "strength_param": ap.get("operating_param"),
                "strength": ap.get("operating_strength"),
                "stats": stats,
                "aero": ap,
            }
        )
    return rows


def _twin_run_dir(runs_dir: Path, aero_optimizer: str) -> Path | None:
    twin = TWIN_MAP.get(aero_optimizer)
    if not twin:
        return None
    matches = list(runs_dir.glob(f"model_a_eng_multitask_gate_strong_featgrad_detached__{twin}__{MODEL_ID}"))
    return matches[0] if matches else None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# SOURCE: generated={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize aero-preserving optimizer runs")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument("--grid-id", type=str, default="aero_preserve_v1")
    ap.add_argument("--perf-checkpoint", type=Path, default=DEFAULT_PERF_CHECKPOINT)
    args = ap.parse_args()

    perf = get_perf_surrogate(args.perf_checkpoint)
    scales = {name: float(scale) for name, scale in zip(CORE4_NAMES, perf.core_scale())}
    bounds = perf.target_bounds()
    runs = _discover_aero_runs(args.runs_dir)
    out_dir = args.analysis_dir / args.grid_id
    tables = out_dir / "tables"

    repair_rows = []
    pred_rows = []
    drift_rows = []
    paired_rows = []
    pareto_rows = []
    fidelity_rows = []
    for run in runs:
        stats = run["stats"]
        aero = run["aero"]
        vd = _read_json(run["run_dir"] / "viewer_data.json")
        pred_by_rank = {
            int(s["rank"]): float((s.get("aero_preserve") or {}).get("R_aero"))
            for s in vd.get("starts", [])
            if isinstance(s.get("aero_preserve"), dict)
            and s.get("verified_clean")
            and (s.get("aero_preserve") or {}).get("start_within_training_bounds", True)
        }
        pred_vals = list(pred_by_rank.values())
        actual_by_rank = _actual_r_by_rank(run["run_dir"], scales, bounds)
        actual_vals = list(actual_by_rank.values())
        repair_rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "strength_param": run["strength_param"],
                "strength": run["strength"],
                "verified_clean_count": stats.get("verified_clean_count"),
                "false_success_count": stats.get("false_success_count"),
                "median_R_pred": _fmt(aero.get("median_R_aero")),
                "median_R_actual": _fmt(_median(actual_vals)),
            }
        )
        pred_rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "strength_param": run["strength_param"],
                "strength": run["strength"],
                "n_pred": len(pred_vals),
                "median_R_pred": _fmt(_median(pred_vals)),
                "iqr_R_pred": _fmt(_iqr(pred_vals)),
                "p90_R_pred": _fmt(_p90(pred_vals)),
                "within_R_0p5": _fmt(len([v for v in pred_vals if v <= 0.5]) / len(pred_vals) if pred_vals else None),
                "within_R_1p0": _fmt(len([v for v in pred_vals if v <= 1.0]) / len(pred_vals) if pred_vals else None),
                "n_excluded_degenerate_baseline": aero.get("n_excluded_degenerate_baseline"),
            }
        )
        drift_rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "strength_param": run["strength_param"],
                "strength": run["strength"],
                "n_actual": len(actual_vals),
                "median_R_actual": _fmt(_median(actual_vals)),
                "iqr_R_actual": _fmt(_iqr(actual_vals)),
                "p90_R_actual": _fmt(_p90(actual_vals)),
                "within_R_0p5": _fmt(len([v for v in actual_vals if v <= 0.5]) / len(actual_vals) if actual_vals else None),
                "within_R_1p0": _fmt(len([v for v in actual_vals if v <= 1.0]) / len(actual_vals) if actual_vals else None),
            }
        )
        paired_pred_actual = [(pred_by_rank[r], actual_by_rank[r]) for r in sorted(pred_by_rank) if r in actual_by_rank]
        fidelity_rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "strength_param": run["strength_param"],
                "strength": run["strength"],
                "n_paired_pred_actual": len(paired_pred_actual),
                "spearman_pred_vs_actual_R": _fmt(
                    _spearman([p for p, _a in paired_pred_actual], [a for _p, a in paired_pred_actual])
                ),
            }
        )
        pareto_rows.append(
            {
                "run_id": run["run_id"],
                "optimizer": run["optimizer"],
                "strength_param": run["strength_param"],
                "strength": run["strength"],
                "verified_clean_rate": _fmt((stats.get("verified_clean_count") or 0) / max((stats.get("n_starts") or 1), 1)),
                "verified_clean_count": stats.get("verified_clean_count"),
                "median_R_pred": _fmt(_median(pred_vals)),
                "median_R_actual": _fmt(_median(actual_vals)),
            }
        )
        twin_dir = _twin_run_dir(args.runs_dir, str(run["optimizer"]))
        if twin_dir is not None and twin_dir.exists():
            twin = _actual_r_by_rank(twin_dir, scales, bounds)
            diffs = [actual_by_rank[r] - twin[r] for r in sorted(actual_by_rank) if r in twin]
            paired_rows.append(
                {
                    "run_id": run["run_id"],
                    "optimizer": run["optimizer"],
                    "strength_param": run["strength_param"],
                    "strength": run["strength"],
                    "twin_run_id": twin_dir.name,
                    "n_paired": len(diffs),
                    "median_delta_R_aero_minus_twin": _fmt(_median(diffs)),
                    "win_rate_aero_closer": _fmt(len([d for d in diffs if d < 0]) / len(diffs) if diffs else None),
                    "verified_clean_count": stats.get("verified_clean_count"),
                }
            )

    write_csv(
        tables / "T1_repair.csv",
        repair_rows,
        ["run_id", "optimizer", "strength_param", "strength", "verified_clean_count", "false_success_count", "median_R_pred", "median_R_actual"],
    )
    write_csv(
        tables / "T2_drift_predicted.csv",
        pred_rows,
        ["run_id", "optimizer", "strength_param", "strength", "n_pred", "median_R_pred", "iqr_R_pred", "p90_R_pred", "within_R_0p5", "within_R_1p0", "n_excluded_degenerate_baseline"],
    )
    write_csv(
        tables / "T3_drift_actual.csv",
        drift_rows,
        ["run_id", "optimizer", "strength_param", "strength", "n_actual", "median_R_actual", "iqr_R_actual", "p90_R_actual", "within_R_0p5", "within_R_1p0"],
    )
    write_csv(
        tables / "T4_paired.csv",
        paired_rows,
        ["run_id", "optimizer", "strength_param", "strength", "twin_run_id", "n_paired", "median_delta_R_aero_minus_twin", "win_rate_aero_closer", "verified_clean_count"],
    )
    write_csv(
        tables / "T5_pareto.csv",
        pareto_rows,
        ["run_id", "optimizer", "strength_param", "strength", "verified_clean_rate", "verified_clean_count", "median_R_pred", "median_R_actual"],
    )
    write_csv(
        tables / "T6_fidelity.csv",
        fidelity_rows,
        ["run_id", "optimizer", "strength_param", "strength", "n_paired_pred_actual", "spearman_pred_vs_actual_R"],
    )
    (out_dir / "README.md").write_text(
        "# Aero-preserving optimizer analysis\n\n"
        "Tables summarize standardized core-4 drift radius `R_aero` and paired comparisons against grid_100k_v2 model-a twins.\n",
        encoding="utf-8",
    )
    print(f"[summarize_aero_runs] wrote {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
