#!/usr/bin/env python3
"""Small executor for the aero-preserving overlap-repair study.

Runs 14 strength-specific subgroups (O1/O2/O3) plus an aero-aware All meta-run.
It mirrors run_sweep.py's subprocess/resume contract but keeps this second study
separate from the original 42-subgroup grid.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


from chapter7_aeroforge.release_paths import REPO_ROOT, subprocess_env
from chapter7_aeroforge.optimizer.run_sweep import _run_sim_pass  # noqa: E402
from chapter7_aeroforge.optimizer.workbench import (  # noqa: E402
    DEFAULT_BENCHMARK,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_VERIFY_TIMEOUT_S,
    DEFAULT_WORKERS,
    MODEL_TITLES,
    PART_COLORS,
    _filter_starts,
    _is_complete,
    _safe_float,
    _statistics,
    _write_json,
)

WORKBENCH_MOD = "chapter7_aeroforge.optimizer.workbench"
MODEL_ID = "eng_multitask_gate_strong_100k"
GROUP_ID = "aero_preserve"
GROUP_LABEL = "Aero-preserving repair (perf-MLP gradients)"
DEFAULT_SUBGROUP_TIMEOUT_S = 5400.0
DEFAULT_SIM_TIMEOUT_S = 10800.0
NO_AERO_BUDGET = 1.0e99

AERO_PLAN = [
    {
        "optimizer": "aero_penalized_receding",
        "param": "lambda_aero",
        "cli": "--aero-lambda",
        "slug_prefix": "lambda",
        "values": [0.0, 0.1, 1.0, 10.0, 100.0],
    },
    {
        "optimizer": "aero_tangent_receding",
        "param": "alpha",
        "cli": "--aero-alpha",
        "slug_prefix": "alpha",
        "values": [0.0, 0.5, 0.9, 0.99],
    },
    {
        "optimizer": "aero_budget_trust_region",
        "param": "beta",
        "cli": "--aero-beta",
        "slug_prefix": "beta",
        # Use a finite no-budget sentinel so viewer_data.json stays valid JSON.
        "values": [NO_AERO_BUDGET, 4.0, 1.0, 0.25, 0.04],
        "selection": "budget_lowest_drift",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strength_slug(prefix: str, value: float) -> str:
    if np.isinf(value) or value >= 1.0e90:
        return f"{prefix}_inf"
    s = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{s}"


def _display_value(value: float) -> str:
    return "inf" if value >= 1.0e90 else f"{value:g}"


def _member_run_id(row: dict) -> str:
    slug = _strength_slug(row["slug_prefix"], float(row["value"]))
    return f"{GROUP_ID}__{row['optimizer']}__{slug}__{MODEL_ID}"


def _all_run_id() -> str:
    return f"{GROUP_ID}__all_aero__{MODEL_ID}"


def _is_complete_run(run_dir: Path, expected_n: int) -> bool:
    return _is_complete(run_dir, expected_n)


class Progress:
    def __init__(self, path: Path, config: dict):
        self.path = path
        self.data = {"started_at": _now(), "config": config, "subgroups": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
                self.data["config"] = config
            except (json.JSONDecodeError, OSError):
                pass
        self.data.setdefault("subgroups", {})

    def update(self, run_id: str, **fields) -> None:
        row = self.data["subgroups"].setdefault(run_id, {})
        row.update(fields)
        row["updated_at"] = _now()
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def _run_subprocess(cmd: list[str], log_path: Path, timeout_s: float) -> tuple[str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {_now()}  CMD: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
            return ("ok" if rc == 0 else "failed", rc)
        except subprocess.TimeoutExpired:
            log.write(f"\n# {_now()}  TIMEOUT after {timeout_s:.0f}s -- killing process group\n")
            log.flush()
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return ("timeout", -1)


def _flat_plan() -> list[dict]:
    out = []
    for family in AERO_PLAN:
        for value in family["values"]:
            row = {**family, "value": float(value)}
            row["slug"] = _strength_slug(family["slug_prefix"], float(value))
            row["run_id"] = _member_run_id(row)
            out.append(row)
    return out


def _run_all_aero(*, runs_dir: Path, benchmark: dict, resume: bool, seed: int, plan: list[dict]) -> dict:
    run_id = _all_run_id()
    run_dir = runs_dir / run_id
    expected_n = len(benchmark["starts"])
    if resume and _is_complete_run(run_dir, expected_n):
        stats = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
        print(f"[{run_id}] complete -> skip (resume)")
        return stats

    sibling_starts: dict[str, dict[int, dict]] = {}
    for row in plan:
        vd = runs_dir / row["run_id"] / "viewer_data.json"
        if not vd.exists():
            raise SystemExit(f"[all_aero] missing sibling {row['run_id']} ({vd})")
        data = json.loads(vd.read_text(encoding="utf-8"))
        sibling_starts[row["run_id"]] = {int(s["rank"]): s for s in data["starts"]}

    ranks = sorted({int(s["rank"]) for s in benchmark["starts"]})
    chosen: list[dict] = []
    winner_hist: dict[str, int] = {row["run_id"]: 0 for row in plan}
    for rank in ranks:
        cand = [(rid, starts[rank]) for rid, starts in sibling_starts.items() if rank in starts]
        if not cand:
            continue
        clean = [(rid, s) for rid, s in cand if s.get("verified_clean")]
        if clean:
            rid, row = min(
                clean,
                key=lambda t: (
                    _safe_float((t[1].get("aero_preserve") or {}).get("R_aero"), float("inf")),
                    t[1]["normalized_distance"],
                    t[1]["final_pred_overlap_mm3"],
                ),
            )
            reason = "all_aero_lowest_drift_clean"
        else:
            verified = [(rid, s) for rid, s in cand if s.get("final_verified_overlap_mm3") is not None]
            if verified:
                rid, row = min(verified, key=lambda t: t[1]["final_verified_overlap_mm3"])
            else:
                rid, row = min(cand, key=lambda t: t[1]["final_pred_overlap_mm3"])
            reason = "all_aero_lowest_overlap"
        winner_hist[rid] = winner_hist.get(rid, 0) + 1
        merged = dict(row)
        merged["selection_reason"] = reason
        merged["source_optimizer"] = rid
        merged.pop("sweep", None)
        chosen.append(merged)

    chosen.sort(key=lambda s: int(s["rank"]))
    stats = _statistics(chosen, verify=True, tau=1.0, tau_decide=1.0, p_star=None, seed=seed, operating_points=[])
    stats["all_aero_verified_clean_count"] = len([s for s in chosen if s.get("verified_clean")])
    stats["winner_histogram"] = winner_hist
    stats["wall_s_total"] = 0.0
    stats["wall_s_per_start"] = 0.0

    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_title = MODEL_TITLES.get(MODEL_ID, MODEL_ID)
    viewer_data = {
        "run_id": run_id,
        "title": f"All (aero-preserving best drift) — {model_title}",
        "group_id": GROUP_ID,
        "group_label": GROUP_LABEL,
        "subgroup_id": f"all_aero__{MODEL_ID}",
        "subgroup_label": f"All (aero-preserving best drift) — {model_title}",
        "model_id": MODEL_ID,
        "optimizer_id": "all_aero",
        "optimizer_label": "All (aero-preserving best drift)",
        "benchmark_id": benchmark.get("benchmark_id"),
        "tau_mm3": 1.0,
        "tau_decide_mm3": 1.0,
        "binary_p_star": None,
        "seed": seed,
        "part_colors": PART_COLORS,
        "optimizer_config": {
            "meta": "all_aero",
            "members": [row["run_id"] for row in plan],
            "selection_rule": "per start: lowest predicted R_aero among CAD-verified-clean finals; else lowest verified overlap",
            "seed": seed,
        },
        "statistics": stats,
        "starts": chosen,
    }
    _write_json(run_dir / "viewer_data.json", viewer_data)
    _write_json(run_dir / "statistics.json", stats)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "title": viewer_data["title"],
            "group_id": GROUP_ID,
            "group_label": GROUP_LABEL,
            "subgroup_id": f"all_aero__{MODEL_ID}",
            "subgroup_label": viewer_data["subgroup_label"],
            "model_id": MODEL_ID,
            "optimizer_id": "all_aero",
            "benchmark_id": benchmark.get("benchmark_id"),
            "n_starts": len(chosen),
            "statistics": stats,
        },
    )
    print(f"[{run_id}] saved -> {run_dir} | verified {stats['all_aero_verified_clean_count']}/{len(chosen)}")
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run aero-preserving Chapter 7 optimizer sweep")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--verify-timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT_S)
    ap.add_argument("--subgroup-timeout", type=float, default=DEFAULT_SUBGROUP_TIMEOUT_S)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--sim-workers", type=int, default=None)
    ap.add_argument("--sim-timeout", type=float, default=DEFAULT_SIM_TIMEOUT_S)
    resume = ap.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", help="Skip complete subgroups (default)")
    resume.add_argument("--no-resume", dest="resume", action="store_false", help="Re-run even complete subgroups")
    ap.set_defaults(resume=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-starts", type=int, default=None)
    ap.add_argument("--rank-stride", type=int, default=None)
    ap.add_argument("--ranks", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    benchmark = _filter_starts(benchmark, rank_stride=args.rank_stride, ranks=args.ranks, limit_starts=args.limit_starts)
    expected_n = len(benchmark["starts"])
    plan = _flat_plan()
    config = {
        "group_id": GROUP_ID,
        "model_id": MODEL_ID,
        "expected_n": expected_n,
        "workers": args.workers,
        "seed": args.seed,
        "verify": not args.no_verify,
        "sweep": not args.no_sweep,
        "plan": [{k: v for k, v in row.items() if k != "values"} for row in plan],
    }
    progress = Progress(args.runs_dir / "aero_sweep_progress.json", config)

    if args.sim_only:
        return _run_sim_pass(args, progress)

    print("[run_aero_sweep] plan:")
    for i, row in enumerate(plan, 1):
        print(f"  {i:02d}. {row['run_id']}  ({row['param']}={_display_value(float(row['value']))})")
    print(f"  15. {_all_run_id()}  (meta)")
    if args.dry_run:
        return 0

    logs_dir = args.runs_dir / "_aero_sweep_logs"
    for i, row in enumerate(plan, 1):
        run_dir = args.runs_dir / row["run_id"]
        if args.resume and _is_complete_run(run_dir, expected_n):
            print(f"[{i:02d}/14] {row['run_id']} complete -> skip")
            progress.update(row["run_id"], status="done", skipped=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            WORKBENCH_MOD,
            "--variant",
            "a",
            "--group-id",
            GROUP_ID,
            "--group",
            GROUP_LABEL,
            "--optimizer",
            row["optimizer"],
            "--run-suffix",
            row["slug"],
            "--aero-param",
            row["param"],
            "--aero-strength",
            str(row["value"]),
            row["cli"],
            str(row["value"]),
            "--benchmark",
            str(args.benchmark),
            "--runs-dir",
            str(args.runs_dir),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--verify-timeout",
            str(args.verify_timeout),
        ]
        if row.get("selection"):
            if not (row["optimizer"] == "aero_budget_trust_region" and float(row["value"]) >= 1.0e90):
                cmd += ["--aero-selection", row["selection"]]
        if args.no_verify:
            cmd.append("--no-verify")
        if args.no_sweep:
            cmd.append("--no-sweep")
        if args.resume:
            cmd.append("--resume")
        if args.limit_starts is not None:
            cmd += ["--limit-starts", str(args.limit_starts)]
        if args.rank_stride is not None:
            cmd += ["--rank-stride", str(args.rank_stride)]
        if args.ranks is not None:
            cmd += ["--ranks", args.ranks]

        print("=" * 78)
        print(f"[run_aero_sweep] {i:02d}/14 {row['run_id']} -> {logs_dir / (row['run_id'] + '.log')}")
        progress.update(row["run_id"], status="running", attempt=1)
        t0 = time.perf_counter()
        outcome, rc = _run_subprocess(cmd, logs_dir / f"{row['run_id']}.log", args.subgroup_timeout)
        wall = time.perf_counter() - t0
        status = "done" if outcome == "ok" else outcome
        progress.update(row["run_id"], status=status, wall_s=round(wall, 1), returncode=rc)
        if outcome != "ok":
            raise SystemExit(f"[run_aero_sweep] subgroup failed: {row['run_id']} ({outcome}, rc={rc})")

    progress.update(_all_run_id(), status="running")
    stats = _run_all_aero(runs_dir=args.runs_dir, benchmark=benchmark, resume=args.resume, seed=args.seed, plan=plan)
    progress.update(_all_run_id(), status="done", verified_clean_count=stats.get("all_aero_verified_clean_count"))

    if not args.no_sim:
        return _run_sim_pass(args, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
