"""Run and aggregate Chapter 5 parameter-lock robustness studies."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from shared.model_utils import FEATURE_NAMES
from shared.parameter_locks import (
    all_scenario_ids,
    scenario_registry,
)
from shared.paths import CONSTRAINT_STUDIES_DIR, RUNS_DIR


DEFAULT_STUDY_ID = "parameter_locks_v1"

SMOKE_BASELINES = [
    "penalized_proximity_tau0.75",
    "trust_region_tau0.90",
    "coordinate_axis_tau0.90",
]
SMOKE_SCENARIOS = [
    "grad_top_1",
    "grad_bottom_1",
    "delta_top_1",
    "lock_roof_path",
    "random_1_seed0",
]
LEADER_FAMILY_OPTIMIZERS = {
    "penalized_proximity_descent_v1",
    "trust_region_hybrid_v1",
    "coordinate_axis_bracket_v1",
    "random_sphere_coordinate_shrink_v1",
    "momentum_receding_gradient_v1",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def discover_baselines() -> dict[str, dict[str, Any]]:
    baselines = {}
    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json")):
        manifest = load_json(manifest_path)
        run_id = manifest_path.parent.name
        if manifest.get("constraint_id"):
            continue
        model_id = str(manifest.get("model_id", "")).lower()
        if run_id.startswith("task27_") or "graph" in model_id or "soft" in model_id:
            continue
        baselines[run_id] = manifest
    return baselines


def phase_baselines(phase: str, baselines: dict[str, dict[str, Any]]) -> list[str]:
    if phase == "smoke":
        return [run_id for run_id in SMOKE_BASELINES if run_id in baselines]
    if phase == "leader":
        return [
            run_id for run_id, manifest in baselines.items()
            if manifest.get("optimizer_id") in LEADER_FAMILY_OPTIMIZERS
        ]
    if phase == "full":
        return list(baselines)
    raise ValueError(f"Unknown phase: {phase}")


def phase_scenarios(phase: str) -> list[str]:
    if phase == "smoke":
        return SMOKE_SCENARIOS
    return all_scenario_ids()


def constrained_run_id(baseline_run_id: str, scenario_id: str) -> str:
    return f"{baseline_run_id}_lock_{scenario_id}"


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run_constrained_job(
    *,
    root: Path,
    baseline_run_id: str,
    manifest: dict[str, Any],
    scenario_id: str,
    skip_existing: bool,
) -> str:
    run_id = constrained_run_id(baseline_run_id, scenario_id)
    run_dir = RUNS_DIR / run_id
    if skip_existing and (run_dir / "manifest.json").exists():
        return run_id

    cmd = [
        sys.executable,
        "-m",
        "run_workbench",
        "--optimizer",
        manifest["optimizer_id"],
        "--tau",
        str(manifest["tau"]),
        "--run-id",
        run_id,
        "--constraint-id",
        scenario_id,
        "--baseline-run-id",
        baseline_run_id,
    ]
    if manifest.get("model_dir"):
        cmd.extend(["--model-dir", manifest["model_dir"]])
    print("[parameter_lock_study]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=root,
        env=_single_thread_env(),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return run_id


def run_campaign(
    *,
    phase: str,
    baseline_ids: list[str] | None,
    scenarios: list[str] | None,
    skip_existing: bool,
    skip_verify: bool,
    workers: int,
) -> list[str]:
    root = Path(__file__).resolve().parent
    baselines = discover_baselines()
    selected_baselines = baseline_ids or phase_baselines(phase, baselines)
    missing = [run_id for run_id in selected_baselines if run_id not in baselines]
    if missing:
        raise ValueError(f"Unknown baseline run(s): {', '.join(missing)}")
    selected_scenarios = scenarios or phase_scenarios(phase)
    generated: list[str] = []

    jobs = [
        {
            "root": root,
            "baseline_run_id": baseline_run_id,
            "manifest": baselines[baseline_run_id],
            "scenario_id": scenario_id,
            "skip_existing": skip_existing,
        }
        for baseline_run_id in selected_baselines
        for scenario_id in selected_scenarios
    ]

    if workers <= 1:
        for job in jobs:
            generated.append(_run_constrained_job(**job))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_constrained_job, **job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                run_id = future.result()
                generated.append(run_id)
                print(f"[parameter_lock_study] completed {run_id} ({job['scenario_id']})")

    if generated and not skip_verify:
        verify_module = root / "verify_runs.py"
        if verify_module.exists():
            cmd = [
                sys.executable,
                "-m",
                "verify_runs",
                "--skip-browser",
                *generated,
            ]
            subprocess.run(cmd, cwd=root, check=True)
        else:
            print("[parameter_lock_study] verify_runs.py is not shipped; skipping post-run verification.")
    return generated


def aggregate_study(
    run_ids: list[str] | None = None,
    *,
    study_id: str = DEFAULT_STUDY_ID,
    baseline_run_ids: list[str] | None = None,
) -> Path:
    study_dir = CONSTRAINT_STUDIES_DIR / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "scenario_registry.json").write_text(json.dumps(scenario_registry(), indent=2))

    selected = set(run_ids or [])
    constrained_dirs: list[Path] = []
    if RUNS_DIR.exists():
        constrained_dirs = [
            p for p in sorted(RUNS_DIR.iterdir())
            if p.is_dir()
            and (p / "manifest.json").exists()
            and load_json(p / "manifest.json").get("constraint_id")
            and (not selected or p.name in selected)
        ]

    per_run_rows = []
    per_start_rows = []
    criticality: dict[str, dict[str, float]] = {
        name: {
            "locked_start_count": 0,
            "oracle_success_drop_sum": 0,
            "distance_delta_sum": 0.0,
            "false_success_sum": 0,
        }
        for name in FEATURE_NAMES
    }

    for run_dir in constrained_dirs:
        manifest = load_json(run_dir / "manifest.json")
        stats = load_json(run_dir / "statistics.json")
        trajectories = load_json(run_dir / "trajectories.json")
        viewer_subgroups = {
            rec["start_id"]: rec.get("subgroup", "")
            for rec in load_json(run_dir / "viewer_data.json")
        }
        baseline_run_id = manifest.get("baseline_run_id")
        baseline_by_start = {}
        if baseline_run_id:
            baseline_by_start = {
                rec["start_id"]: rec
                for rec in load_json(RUNS_DIR / baseline_run_id / "trajectories.json")
            }
        lock_stats = stats.get("parameter_locks", {})
        paired = lock_stats.get("paired_baseline", {})
        sparse = stats.get("sparse_metrics", {})
        per_run_rows.append({
            "run_id": run_dir.name,
            "baseline_run_id": baseline_run_id or "",
            "optimizer_id": manifest.get("optimizer_id", ""),
            "tau": manifest.get("tau", ""),
            "constraint_id": manifest.get("constraint_id", ""),
            "scenario_kind": lock_stats.get("scenario_kind", ""),
            "oracle_confirmed": stats.get("oracle_confirmed_count", 0),
            "surrogate_success": stats.get("surrogate_success_count", 0),
            "false_success": stats.get("false_success_count", 0),
            "mean_distance": stats.get("distances", {}).get("all_mean", 0.0),
            "median_distance": stats.get("distances", {}).get("all_median", 0.0),
            "mean_l1_distance": sparse.get("l1_distance_mean", ""),
            "mean_active_coordinates": sparse.get("active_coordinate_count_mean", ""),
            "baseline_oracle_confirmed": paired.get("baseline_oracle_confirmed", ""),
            "oracle_success_drop": paired.get("oracle_success_drop", ""),
            "recoverability": paired.get("recoverability_among_baseline_successes", ""),
            "mean_distance_delta_vs_baseline": paired.get("mean_distance_delta_vs_baseline", ""),
            "mean_locked_gradient_mass": lock_stats.get("locked_gradient_mass_mean", ""),
            "mean_locked_delta_mass": lock_stats.get("locked_delta_mass_mean", ""),
        })
        for traj in trajectories:
            baseline = baseline_by_start.get(traj["start_id"], {})
            baseline_ok = int(baseline.get("oracle_label", 0)) == 1
            constrained_ok = int(traj.get("oracle_label", 0)) == 1
            baseline_dist = float(baseline.get("normalized_distance", 0.0))
            dist_delta = float(traj.get("normalized_distance", 0.0)) - baseline_dist
            constraint = traj.get("constraint", {})
            row = {
                "run_id": run_dir.name,
                "baseline_run_id": baseline_run_id or "",
                "start_id": traj["start_id"],
                "subgroup": viewer_subgroups.get(traj["start_id"], ""),
                "constraint_id": constraint.get("constraint_id", ""),
                "locked_names": ",".join(constraint.get("locked_names", [])),
                "locked_gradient_mass": constraint.get("locked_gradient_mass", ""),
                "locked_delta_mass": constraint.get("locked_delta_mass", ""),
                "baseline_oracle": int(baseline_ok),
                "constrained_oracle": int(constrained_ok),
                "oracle_success_drop": int(baseline_ok) - int(constrained_ok),
                "baseline_distance": baseline_dist,
                "constrained_distance": float(traj.get("normalized_distance", 0.0)),
                "distance_delta": dist_delta,
                "false_success": int(traj.get("status") == "surrogate_success" and not constrained_ok),
            }
            per_start_rows.append(row)
            for name in constraint.get("locked_names", []):
                entry = criticality[name]
                entry["locked_start_count"] += 1
                entry["oracle_success_drop_sum"] += row["oracle_success_drop"]
                entry["distance_delta_sum"] += dist_delta
                entry["false_success_sum"] += row["false_success"]

    write_csv(per_run_rows, study_dir / "per_run_summary.csv")
    write_csv(per_start_rows, study_dir / "per_start_results.csv")

    baseline_rows = [
        _baseline_summary_row(run_id)
        for run_id in (baseline_run_ids or [])
        if (RUNS_DIR / run_id / "manifest.json").exists()
    ]
    write_csv(baseline_rows, study_dir / "baseline_summary.csv")
    write_csv(
        [
            *baseline_rows,
            *[
                {
                    "run_id": row["run_id"],
                    "baseline_run_id": row["baseline_run_id"],
                    "optimizer_id": row["optimizer_id"],
                    "tau": row["tau"],
                    "constraint_id": row["constraint_id"],
                    "oracle_confirmed": row["oracle_confirmed"],
                    "surrogate_success": row["surrogate_success"],
                    "false_success": row["false_success"],
                    "mean_distance": row["mean_distance"],
                    "median_distance": row["median_distance"],
                    "mean_l1_distance": row["mean_l1_distance"],
                    "mean_active_coordinates": row["mean_active_coordinates"],
                    "oracle_success_drop": row["oracle_success_drop"],
                    "recoverability": row["recoverability"],
                }
                for row in per_run_rows
            ],
        ],
        study_dir / "thesis_ready_summary.csv",
    )

    criticality_rows = []
    for name, values in criticality.items():
        count = int(values["locked_start_count"])
        if count == 0:
            continue
        criticality_rows.append({
            "parameter": name,
            "locked_start_count": count,
            "mean_oracle_success_drop_when_locked": values["oracle_success_drop_sum"] / count,
            "mean_distance_delta_when_locked": values["distance_delta_sum"] / count,
            "false_success_count_when_locked": int(values["false_success_sum"]),
        })
    criticality_rows.sort(
        key=lambda row: (
            -float(row["mean_oracle_success_drop_when_locked"]),
            -float(row["mean_distance_delta_when_locked"]),
        )
    )
    write_csv(criticality_rows, study_dir / "parameter_criticality.csv")
    (study_dir / "robustness_summary.md").write_text(
        render_summary(
            study_id=study_id,
            baseline_rows=baseline_rows,
            per_run_rows=per_run_rows,
            criticality_rows=criticality_rows,
        )
    )
    return study_dir


def _baseline_summary_row(run_id: str) -> dict[str, Any]:
    manifest = load_json(RUNS_DIR / run_id / "manifest.json")
    stats = load_json(RUNS_DIR / run_id / "statistics.json")
    sparse = stats.get("sparse_metrics", {})
    return {
        "run_id": run_id,
        "baseline_run_id": "",
        "optimizer_id": manifest.get("optimizer_id", ""),
        "tau": manifest.get("tau", ""),
        "constraint_id": "none",
        "oracle_confirmed": stats.get("oracle_confirmed_count", 0),
        "surrogate_success": stats.get("surrogate_success_count", 0),
        "false_success": stats.get("false_success_count", 0),
        "mean_distance": stats.get("distances", {}).get("all_mean", 0.0),
        "median_distance": stats.get("distances", {}).get("all_median", 0.0),
        "mean_l1_distance": sparse.get("l1_distance_mean", ""),
        "mean_active_coordinates": sparse.get("active_coordinate_count_mean", ""),
        "oracle_success_drop": "",
        "recoverability": "",
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(
    *,
    study_id: str,
    baseline_rows: list[dict[str, Any]],
    per_run_rows: list[dict[str, Any]],
    criticality_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Parameter Lock Robustness Study",
        "",
        f"Study ID: `{study_id}`",
        "",
        "This suite reruns Chapter 5 optimizers while selected parameters are forced to remain",
        "equal to their blocked-start values. Each constrained run is paired with its",
        "unconstrained baseline.",
        "",
        "## Run Summary",
        "",
        "| Run | Constraint | Oracle OK | False OK | Mean Dist | Drop vs Baseline | Recoverability |",
        "|-----|------------|-----------|----------|-----------|------------------|----------------|",
    ]
    for row in sorted(baseline_rows, key=lambda r: (r["optimizer_id"], str(r["tau"]))):
        lines.append(
            f"| `{row['run_id']}` | `none` | {row['oracle_confirmed']} | "
            f"{row['false_success']} | {float(row['mean_distance']):.4f} | n/a | n/a |"
        )
    for row in sorted(per_run_rows, key=lambda r: (r["optimizer_id"], str(r["tau"]), r["constraint_id"])):
        recoverability = row["recoverability"]
        recoverability_text = f"{float(recoverability):.1%}" if recoverability != "" else "n/a"
        drop = row["oracle_success_drop"] if row["oracle_success_drop"] != "" else "n/a"
        lines.append(
            f"| `{row['run_id']}` | `{row['constraint_id']}` | {row['oracle_confirmed']} | "
            f"{row['false_success']} | {float(row['mean_distance']):.4f} | {drop} | {recoverability_text} |"
        )
    lines.extend([
        "",
        "## Parameter Criticality",
        "",
        "| Parameter | Locked Starts | Mean Success Drop | Mean Distance Delta | False OK |",
        "|-----------|---------------|-------------------|---------------------|----------|",
    ])
    for row in criticality_rows:
        lines.append(
            f"| `{row['parameter']}` | {row['locked_start_count']} | "
            f"{float(row['mean_oracle_success_drop_when_locked']):.4f} | "
            f"{float(row['mean_distance_delta_when_locked']):.4f} | "
            f"{row['false_success_count_when_locked']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "leader", "full"], default="smoke")
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID)
    parser.add_argument("--baselines", default=None, help="Comma-separated baseline run IDs; overrides --phase baselines")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario IDs; defaults by phase")
    parser.add_argument("--workers", type=int, default=1, help="Parallel constrained run workers")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    baseline_ids = _parse_csv_arg(args.baselines)
    scenarios = _parse_csv_arg(args.scenarios)
    run_ids: list[str] | None = None
    if not args.aggregate_only:
        run_ids = run_campaign(
            phase=args.phase,
            baseline_ids=baseline_ids,
            scenarios=scenarios,
            skip_existing=args.skip_existing,
            skip_verify=args.skip_verify,
            workers=max(1, args.workers),
        )
    study_dir = aggregate_study(
        run_ids,
        study_id=args.study_id,
        baseline_run_ids=baseline_ids,
    )
    print(f"Wrote parameter-lock study artifacts to {study_dir}")


if __name__ == "__main__":
    main()
