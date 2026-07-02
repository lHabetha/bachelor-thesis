"""Run Chapter 6 overlap-label calibration on a small relaxed sample set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .geometry import export_part_geometry, load_part_meshes, param_hash
from .paths import RUNS_DIR, CHAPTER6_ROOT
from .sampler import RELAXED_RULES, sample_relaxed_params


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _sample_plan(n: int, seed: int):
    rng = np.random.default_rng(seed)
    streams = ["clean_valid", "near_boundary"]
    streams.extend(["single_relaxation"] * len(RELAXED_RULES))
    streams.append("multi_relaxation")
    rows = []
    for i in range(n):
        stream = streams[i % len(streams)]
        rows.append(
            sample_relaxed_params(
                rng,
                sample_id=f"calib_{i:04d}",
                seed=seed,
                stream=stream,
            )
        )
    return rows


def run_calibration(args: argparse.Namespace) -> Path:
    try:
        from .sdf_overlap import compute_overlap_label
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "calibrate_labels requires the mesh-backed sdf_overlap module "
            "(not shipped in the public release; use the frozen labeled datasets "
            "under datasets/chapter6_overlap_clevis/labeled/ instead)."
        ) from exc

    run_dir = RUNS_DIR / "calibration" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    geom_root = run_dir / "geometry"
    labels_root = run_dir / "labels"
    geom_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)

    samples = _sample_plan(args.n, args.seed)
    pitch_values = _parse_float_list(args.pitch)
    eps_values = _parse_float_list(args.epsilon)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    rows: list[dict] = []
    t0 = time.perf_counter()
    for s in samples:
        h = param_hash(s.params)
        geom_dir = geom_root / h
        if not (geom_dir / "bracket.obj").exists():
            export_part_geometry(s.params, geom_dir)
        meshes = load_part_meshes(geom_dir)

        for pitch in pitch_values:
            for eps in eps_values:
                for mode in modes:
                    label = compute_overlap_label(
                        meshes,
                        pitch_mm=pitch,
                        epsilon_mm=eps,
                        mode=mode,
                        include_boolean=True,
                        max_points_per_pair=args.max_points_per_pair,
                    )
                    label_path = labels_root / f"{s.sample_id}_{mode}_p{pitch:g}_e{eps:g}.json"
                    label_path.write_text(json.dumps(label.to_dict(), indent=2), encoding="utf-8")
                    bool_raw = label.total_boolean_volume
                    voxel_raw = label.total_voxel_volume
                    abs_err = None if bool_raw is None else abs(voxel_raw - bool_raw)
                    rel_err = None
                    if bool_raw is not None and bool_raw > 1e-9:
                        rel_err = abs_err / bool_raw
                    rows.append(
                        {
                            "sample_id": s.sample_id,
                            "param_hash": h,
                            "stream": s.stream,
                            "intended_relaxed_rules": ",".join(s.intended_relaxed_rules),
                            "relaxed_violations": "|".join(s.validation.relaxed_violations),
                            "pitch_mm": pitch,
                            "epsilon_mm": eps,
                            "mode": mode,
                            "voxel_raw": voxel_raw,
                            "voxel_norm": label.total_voxel_norm,
                            "boolean_raw": bool_raw,
                            "boolean_norm": label.total_boolean_norm,
                            "abs_error_raw": abs_err,
                            "rel_error_nonzero": rel_err,
                            "wall_time_s": label.wall_time_s,
                            "n_grid_points": sum(p.n_grid_points for p in label.pairs),
                            "warnings": "|".join(label.warnings),
                        }
                    )

    csv_path = run_dir / "calibration_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(rows)
    summary["run_id"] = args.run_id
    summary["seed"] = args.seed
    summary["n_samples"] = args.n
    summary["elapsed_s"] = time.perf_counter() - t0
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir, summary)
    return run_dir


def _summarize(rows: list[dict]) -> dict:
    by_setting: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['mode']} pitch={row['pitch_mm']} eps={row['epsilon_mm']}"
        by_setting.setdefault(key, []).append(row)

    settings = {}
    for key, vals in by_setting.items():
        abs_errors = [float(v["abs_error_raw"]) for v in vals if v["abs_error_raw"] not in (None, "")]
        rel_errors = [float(v["rel_error_nonzero"]) for v in vals if v["rel_error_nonzero"] not in (None, "")]
        times = [float(v["wall_time_s"]) for v in vals]
        bool_nonzero = [float(v["boolean_raw"]) > 1e-6 for v in vals if v["boolean_raw"] is not None]
        voxel_nonzero = [float(v["voxel_raw"]) > 1e-6 for v in vals]
        agreement = sum(a == b for a, b in zip(bool_nonzero, voxel_nonzero)) / max(1, len(voxel_nonzero))
        settings[key] = {
            "mean_abs_error_raw": float(np.mean(abs_errors)) if abs_errors else None,
            "max_abs_error_raw": float(np.max(abs_errors)) if abs_errors else None,
            "mean_rel_error_nonzero": float(np.mean(rel_errors)) if rel_errors else None,
            "max_rel_error_nonzero": float(np.max(rel_errors)) if rel_errors else None,
            "mean_wall_time_s": float(np.mean(times)),
            "max_wall_time_s": float(np.max(times)),
            "zero_nonzero_agreement": agreement,
        }
    return {
        "settings": settings,
        "row_count": len(rows),
        "streams": dict(Counter(row["stream"] for row in rows)),
    }


def _write_report(run_dir: Path, summary: dict) -> None:
    lines = [
        "# SDF/Voxel Label Calibration",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Samples: `{summary['n_samples']}`",
        f"Elapsed wall time: `{summary['elapsed_s']:.1f} s`",
        "",
        "## Setting Summary",
        "",
        "| Setting | Mean abs err (mm^3) | Max abs err | Mean rel err nonzero | Zero/nonzero agreement | Mean wall (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, val in summary["settings"].items():
        lines.append(
            "| "
            + key
            + f" | {val['mean_abs_error_raw'] if val['mean_abs_error_raw'] is not None else 'n/a'}"
            + f" | {val['max_abs_error_raw'] if val['max_abs_error_raw'] is not None else 'n/a'}"
            + f" | {val['mean_rel_error_nonzero'] if val['mean_rel_error_nonzero'] is not None else 'n/a'}"
            + f" | {val['zero_nonzero_agreement']:.3f}"
            + f" | {val['mean_wall_time_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Note",
            "",
            "This report is an initial calibration artifact. Do not scale active learning until the selected label settings are documented in `configs/sdf_label_v1.json` or a superseding config.",
            "",
            f"Repository root: `{CHAPTER6_ROOT}`",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="calibration_v1_smoke")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=25025)
    parser.add_argument("--pitch", default="1.0,0.75")
    parser.add_argument("--epsilon", default="0.1,0.05")
    parser.add_argument("--modes", default="signed_hard,soft")
    parser.add_argument("--max-points-per-pair", type=int, default=250000)
    args = parser.parse_args()
    run_dir = run_calibration(args)
    print(run_dir)


if __name__ == "__main__":
    main()
