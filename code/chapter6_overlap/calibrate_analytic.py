"""Calibrate fast analytic voxel overlap labels against mesh booleans."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .analytic_overlap import compute_analytic_overlap_label
from .geometry import export_part_geometry, load_part_meshes, param_hash
from .paths import RUNS_DIR
from .sampler import RELAXED_RULES, sample_relaxed_params


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _sample_plan(n: int, seed: int):
    rng = np.random.default_rng(seed)
    streams = ["clean_valid", "near_boundary"]
    streams.extend(["single_relaxation"] * len(RELAXED_RULES))
    streams.extend(["multi_relaxation", "extreme_meaningful"])
    samples = []
    for i in range(n):
        stream = streams[i % len(streams)]
        samples.append(
            sample_relaxed_params(
                rng,
                sample_id=f"analytic_calib_{i:04d}",
                seed=seed,
                stream=stream,
            )
        )
    return samples


def run(args: argparse.Namespace) -> Path:
    try:
        from .sdf_overlap import compute_overlap_label
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "calibrate_analytic requires the mesh-backed sdf_overlap module "
            "(not shipped in the public release; use the frozen calibration_rows.csv "
            "under datasets/chapter6_overlap_clevis/calibration/ instead)."
        ) from exc

    run_dir = RUNS_DIR / "calibration" / args.run_id
    geom_root = run_dir / "geometry"
    geom_root.mkdir(parents=True, exist_ok=True)
    pitches = _parse_float_list(args.pitch)
    samples = _sample_plan(args.n, args.seed)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for idx, sample in enumerate(samples):
        h = param_hash(sample.params)
        geom_dir = geom_root / h
        if not (geom_dir / "bracket.obj").exists():
            export_part_geometry(sample.params, geom_dir)
        meshes = load_part_meshes(geom_dir)
        boolean_label = compute_overlap_label(
            meshes,
            pitch_mm=1.0,
            epsilon_mm=0.05,
            mode="signed_hard",
            include_boolean=True,
            max_points_per_pair=args.boolean_max_points,
        )
        bool_total = float(boolean_label.total_boolean_volume or 0.0)
        bool_norm = float(boolean_label.total_boolean_norm or 0.0)

        for pitch in pitches:
            analytic = compute_analytic_overlap_label(
                sample.params,
                pitch_mm=pitch,
                overlap_threshold_norm=args.threshold_norm,
                max_points_per_pair=args.max_points_per_pair,
            )
            abs_err = abs(analytic.total_overlap_volume - bool_total)
            rel_err = None if bool_total <= 1e-9 else abs_err / bool_total
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "param_hash": h,
                    "stream": sample.stream,
                    "intended_relaxed_rules": ",".join(sample.intended_relaxed_rules),
                    "relaxed_violations": "|".join(sample.validation.relaxed_violations),
                    "pitch_mm": pitch,
                    "analytic_raw": analytic.total_overlap_volume,
                    "analytic_norm": analytic.total_overlap_norm,
                    "boolean_raw": bool_total,
                    "boolean_norm": bool_norm,
                    "abs_error_raw": abs_err,
                    "rel_error_nonzero": rel_err,
                    "analytic_wall_time_s": analytic.wall_time_s,
                    "boolean_wall_time_s": boolean_label.wall_time_s,
                    "analytic_points": sum(p.n_grid_points for p in analytic.pairs),
                    "boolean_status": "|".join(boolean_label.warnings),
                    "boolean_nonzero": bool_total > args.boolean_tol,
                    "analytic_nonzero": analytic.total_overlap_volume > args.boolean_tol,
                }
            )
        print(f"[{idx + 1}/{len(samples)}] {sample.sample_id} boolean={bool_total:.4f}")

    csv_path = run_dir / "analytic_calibration_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    summary.update(
        {
            "run_id": args.run_id,
            "seed": args.seed,
            "n_samples": args.n,
            "elapsed_s": time.perf_counter() - t0,
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir, summary)
    return run_dir


def _summarize(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for pitch in sorted({float(r["pitch_mm"]) for r in rows}):
        vals = [r for r in rows if float(r["pitch_mm"]) == pitch]
        abs_err = np.array([float(v["abs_error_raw"]) for v in vals], dtype=float)
        rel_err = np.array(
            [float(v["rel_error_nonzero"]) for v in vals if v["rel_error_nonzero"] not in (None, "")],
            dtype=float,
        )
        agree = np.mean([bool(v["boolean_nonzero"]) == bool(v["analytic_nonzero"]) for v in vals])
        out[f"pitch_{pitch:g}"] = {
            "mean_abs_error_raw": float(abs_err.mean()),
            "median_abs_error_raw": float(np.median(abs_err)),
            "max_abs_error_raw": float(abs_err.max()),
            "mean_rel_error_nonzero": None if len(rel_err) == 0 else float(rel_err.mean()),
            "median_rel_error_nonzero": None if len(rel_err) == 0 else float(np.median(rel_err)),
            "max_rel_error_nonzero": None if len(rel_err) == 0 else float(rel_err.max()),
            "zero_nonzero_agreement": float(agree),
            "mean_analytic_wall_time_s": float(np.mean([float(v["analytic_wall_time_s"]) for v in vals])),
            "mean_boolean_wall_time_s": float(np.mean([float(v["boolean_wall_time_s"]) for v in vals])),
        }
    return {"settings": out, "row_count": len(rows)}


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _write_report(run_dir: Path, summary: dict) -> None:
    lines = [
        "# Analytic Voxel Overlap Calibration",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Samples: `{summary['n_samples']}`",
        f"Elapsed wall time: `{summary['elapsed_s']:.1f} s`",
        "",
        "## Summary",
        "",
        "| Pitch (mm) | Mean abs err (mm^3) | Median abs err | Max abs err | Median rel err nonzero | Zero/nonzero agreement | Analytic wall (s) | Boolean wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, val in summary["settings"].items():
        pitch = key.replace("pitch_", "")
        lines.append(
            f"| {pitch} | {_fmt(val['mean_abs_error_raw'])} | {_fmt(val['median_abs_error_raw'])} | "
            f"{_fmt(val['max_abs_error_raw'])} | {_fmt(val['median_rel_error_nonzero'])} | "
            f"{_fmt(val['zero_nonzero_agreement'])} | {_fmt(val['mean_analytic_wall_time_s'])} | "
            f"{_fmt(val['mean_boolean_wall_time_s'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The analytic voxel label is calibrated against exact mesh booleans exported from the same `DummyParams` values. If a pitch has acceptable zero/nonzero agreement and near-boundary error, it should become the active-learning acquisition label because it avoids CadQuery and mesh signed-distance calls.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="analytic_calibration_v1")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=25026)
    parser.add_argument("--pitch", default="1.0,0.75,0.5")
    parser.add_argument("--threshold-norm", type=float, default=1e-8)
    parser.add_argument("--boolean-tol", type=float, default=1e-6)
    parser.add_argument("--max-points-per-pair", type=int, default=750000)
    parser.add_argument("--boolean-max-points", type=int, default=100000)
    args = parser.parse_args()
    run_dir = run(args)
    print(run_dir)


if __name__ == "__main__":
    main()
