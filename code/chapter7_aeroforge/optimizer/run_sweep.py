#!/usr/bin/env python3
"""Resumable central executor for the Chapter 7 optimization grid.

Runs the full master list: six model variants a-f, each with the six
canonical optimizers and then the per-variant `All` meta-run = 6 x (6 + 1) = 42
subgroups. One subgroup = (one model variant) x (one optimizer) over all 100
benchmark starts.

Design choices:
  * **Sequential subgroups, each fanned across all cores.** One subgroup runs at a
    time and internally parallelizes its 100 starts over `--workers` processes
    (default 15, saturating a 16-core laptop). This avoids nested oversubscription
    and means each finished subgroup is fully persisted before the next begins.
  * **Subprocess isolation.** Each subgroup is its own `workbench` process started
    in a fresh process group; a hard wall-clock timeout kills the whole group
    (workers included) if a CadQuery build truly wedges past the per-verify
    watchdog. A crash/timeout is caught, recorded, and the grid continues.
  * **Resumable.** Progress is tracked in `runs/sweep_progress.json` and every
    subgroup's artifacts are written atomically by `workbench`. On restart,
    already-complete subgroups are skipped, so a dead laptop only costs the
    in-flight subgroup.
  * **`All` last per group.** The meta-run reuses its six siblings' verified
    finals, so it only runs once all six of its variant's optimizers are complete.

Run it under the aeroforge env (so the subgroups can build geometry):
    conda run --no-capture-output -n bachelor-thesis python3 -m \
        chapter7_aeroforge.optimizer.run_sweep --variants a..f --resume
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


from chapter7_aeroforge.optimizer.optimizers import CANONICAL_SUITE  # noqa: E402
from chapter7_aeroforge.optimizer.workbench import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VERIFY_TIMEOUT_S,
    VARIANTS,
    _filter_starts,
)

from chapter7_aeroforge.release_paths import BENCHMARK_JSON, REPO_ROOT, RUNS_DIR, subprocess_env

DEFAULT_BENCHMARK = BENCHMARK_JSON
DEFAULT_RUNS_DIR = RUNS_DIR
DEFAULT_WORKERS = 14  # 14 of 16 cores (§L.2): leave 2 for OS/writer/CAD I/O, avoid oversubscription stalls
DEFAULT_SUBGROUP_TIMEOUT_S = 5400.0  # 90 min hard backstop per subgroup (per-verify watchdog is finer)
DEFAULT_SIM_TIMEOUT_S = 10800.0  # 3 h hard backstop for the whole VLM quick-sim pass (typically ~15-25 min)
PREFLIGHT_MOD = "chapter7_aeroforge.optimizer.preflight"
WORKBENCH_MOD = "chapter7_aeroforge.optimizer.workbench"
QUICKSIM_MOD = "chapter7_aeroforge.optimizer.quicksim_eval"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_variants(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec in ("all", "a..f", ""):
        return list("abcdef")
    if ".." in spec:
        a, b = spec.split("..", 1)
        order = "abcdef"
        return list(order[order.index(a) : order.index(b) + 1])
    return [v.strip() for v in spec.split(",") if v.strip()]


def _subgroup_run_id(variant: str, optimizer: str) -> str:
    v = VARIANTS[variant]
    return f"{v['group_id']}__{optimizer}__{v['model_id']}"


def _is_complete(run_dir: Path, expected_n: int) -> bool:
    manifest = run_dir / "manifest.json"
    if not manifest.exists() or not (run_dir / "viewer_data.json").exists():
        return False
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return int(m.get("n_starts", -1)) == expected_n


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

    def get(self, run_id: str) -> dict:
        return self.data["subgroups"].get(run_id, {})

    def update(self, run_id: str, **fields) -> None:
        row = self.data["subgroups"].setdefault(run_id, {})
        row.update(fields)
        row["updated_at"] = _now()
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def _ensure_preflight(mode: str, benchmark: Path) -> None:
    if mode == "skip":
        print("[run_sweep] skipping pre-flight (--preflight skip)")
        return
    need_build = mode == "run" or not benchmark.exists()
    cmd = [sys.executable, "-m", PREFLIGHT_MOD]
    if not need_build:
        cmd += ["--skip-normalization", "--skip-benchmark"]  # read-only clip-check gate
        print("[run_sweep] pre-flight: clip-check gate on existing benchmark")
    else:
        print("[run_sweep] pre-flight: rebuilding normalization + benchmark v2")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=subprocess_env())
    if proc.returncode != 0:
        raise SystemExit("[run_sweep] pre-flight gate FAILED — aborting (fix domain/benchmark first)")


def _run_subprocess(cmd: list[str], log_path: Path, timeout_s: float) -> tuple[str, int]:
    """Run one subgroup in its own process group; SIGKILL the group on timeout."""
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
            start_new_session=True,  # own process group so we can kill workers too
        )
        try:
            rc = proc.wait(timeout=timeout_s)
            return ("ok" if rc == 0 else "failed", rc)
        except subprocess.TimeoutExpired:
            log.write(f"\n# {_now()}  TIMEOUT after {timeout_s:.0f}s — killing process group\n")
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


def _run_sim_pass(args: argparse.Namespace, progress: "Progress | None") -> int:
    """Run the deterministic full-VLM quick-sim pass over the runs dir (§J).

    A standalone subprocess in its own process group (same isolation contract as a
    subgroup): the grid's CadQuery workers are gone by now, so the pass owns the
    cores with aerosandbox-only workers. It is itself resumable (ADV-hash cache +
    per-run sim_eval.json), so a kill/restart only redoes the in-flight run.
    """
    sim_workers = args.sim_workers or args.workers
    cmd = [
        sys.executable,
        "-m",
        QUICKSIM_MOD,
        "--runs-dir", str(args.runs_dir),
        "--benchmark", str(args.benchmark),
        "--workers", str(sim_workers),
    ]
    logs_dir = args.runs_dir / "_sweep_logs"
    log_path = logs_dir / "_sim_pass.log"
    if progress is not None:
        progress.update("_sim_pass", status="running")
    print("=" * 78)
    print(f"[run_sweep] quick-sim performance pass (workers={sim_workers}) -> {log_path}")
    t0 = time.perf_counter()
    outcome, rc = _run_subprocess(cmd, log_path, args.sim_timeout)
    wall = time.perf_counter() - t0
    status = "done" if outcome == "ok" else outcome
    if progress is not None:
        progress.update("_sim_pass", status=status, wall_s=round(wall, 1))
    print(f"[run_sweep] sim pass {status.upper()} (rc={rc}, {wall/60:.1f}m); see {log_path}")
    return 0 if outcome == "ok" else 1


def _build_plan(variants: list[str], optimizers: list[str], include_all: bool) -> list[tuple[str, str]]:
    """Ordered (variant, optimizer) list; `all` placed last within each variant."""
    plan: list[tuple[str, str]] = []
    for v in variants:
        for opt in optimizers:
            plan.append((v, opt))
        if include_all:
            plan.append((v, "all"))
    return plan


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resumable central executor for the Chapter 7 optimization grid")
    ap.add_argument("--variants", type=str, default="a..f", help="e.g. 'a..f', 'a,c,f', or 'all'")
    ap.add_argument(
        "--optimizers",
        type=str,
        default="suite",
        help="'suite' (the six canonical) or a comma list of optimizer ids",
    )
    ap.add_argument("--no-all", action="store_true", help="Skip the per-variant 'All' meta-run")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--verify-timeout", type=float, default=DEFAULT_VERIFY_TIMEOUT_S)
    ap.add_argument(
        "--subgroup-timeout", type=float, default=DEFAULT_SUBGROUP_TIMEOUT_S, help="Hard wall-clock cap per subgroup"
    )
    ap.add_argument("--no-verify", action="store_true", help="MLP-only (no CadQuery) — fast dry pass")
    ap.add_argument("--no-sweep", action="store_true", help="Single headline operating point (no tau/p* sweep)")
    ap.add_argument("--no-sim", action="store_true", help="Skip the post-grid full-VLM quick-sim performance pass")
    ap.add_argument("--sim-only", action="store_true", help="Skip the grid; only run the quick-sim performance pass over existing runs")
    ap.add_argument("--sim-workers", type=int, default=None, help="Workers for the sim pass (default: --workers)")
    ap.add_argument("--sim-timeout", type=float, default=DEFAULT_SIM_TIMEOUT_S, help="Hard wall-clock cap for the whole sim pass")
    ap.add_argument("--resume", action="store_true", help="Skip already-complete subgroups (default behavior anyway)")
    ap.add_argument("--retries", type=int, default=1, help="Retry a failed subgroup this many times before skipping")
    ap.add_argument("--preflight", choices=["auto", "run", "skip"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    # smoke-test passthroughs (kept off for the real overnight grid)
    ap.add_argument("--limit-starts", type=int, default=None)
    ap.add_argument("--rank-stride", type=int, default=None)
    ap.add_argument("--ranks", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    variants = _parse_variants(args.variants)
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"[run_sweep] unknown variant {v!r}; valid: {sorted(VARIANTS)}")
    optimizers = CANONICAL_SUITE if args.optimizers == "suite" else [
        o.strip() for o in args.optimizers.split(",") if o.strip()
    ]
    include_all = not args.no_all and args.optimizers == "suite"
    plan = _build_plan(variants, optimizers, include_all)

    print("=" * 78)
    print(f"[run_sweep] {len(plan)} subgroups over variants {variants}")
    print(f"[run_sweep] optimizers: {optimizers}{'  + All' if include_all else ''}")
    print(f"[run_sweep] workers/subgroup={args.workers}  verify={'off' if args.no_verify else 'cad'}  "
          f"sweep={'off' if args.no_sweep else 'on'}  seed={args.seed}")
    print("=" * 78)
    for i, (v, opt) in enumerate(plan, 1):
        print(f"  {i:02d}. variant {v} :: {opt}  ->  {_subgroup_run_id(v, opt)}")
    if args.dry_run:
        return 0

    if args.sim_only:
        print("[run_sweep] --sim-only: skipping grid; running quick-sim pass over existing runs")
        sim_progress = Progress(args.runs_dir / "sweep_progress.json", config={"sim_only": True})
        return _run_sim_pass(args, sim_progress)

    _ensure_preflight(args.preflight, args.benchmark)

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    expected_n = len(
        _filter_starts(
            benchmark, rank_stride=args.rank_stride, ranks=args.ranks, limit_starts=args.limit_starts
        )["starts"]
    )
    print(f"[run_sweep] benchmark={benchmark.get('benchmark_id')} expected_n={expected_n}")

    progress = Progress(
        args.runs_dir / "sweep_progress.json",
        config={
            "variants": variants,
            "optimizers": optimizers,
            "include_all": include_all,
            "workers": args.workers,
            "seed": args.seed,
            "verify": not args.no_verify,
            "sweep": not args.no_sweep,
            "benchmark": str(args.benchmark),
            "expected_n": expected_n,
        },
    )
    logs_dir = args.runs_dir / "_sweep_logs"

    grid_t0 = time.perf_counter()
    done = skipped = failed = 0
    for i, (variant, optimizer) in enumerate(plan, 1):
        run_id = _subgroup_run_id(variant, optimizer)
        run_dir = args.runs_dir / run_id
        prefix = f"[{i:02d}/{len(plan)}] {run_id}"

        if _is_complete(run_dir, expected_n):
            progress.update(run_id, variant=variant, optimizer=optimizer, status="done", note="already complete")
            print(f"{prefix}  COMPLETE -> skip")
            skipped += 1
            continue

        if optimizer == "all":  # All needs all six siblings complete
            missing = [
                o for o in CANONICAL_SUITE if not _is_complete(args.runs_dir / _subgroup_run_id(variant, o), expected_n)
            ]
            if missing:
                progress.update(run_id, variant=variant, optimizer=optimizer, status="blocked", missing=missing)
                print(f"{prefix}  BLOCKED (siblings incomplete: {missing}) -> skip")
                failed += 1
                continue

        cmd = [
            sys.executable,
            "-m",
            WORKBENCH_MOD,
            "--variant", variant,
            "--optimizer", optimizer,
            "--benchmark", str(args.benchmark),
            "--runs-dir", str(args.runs_dir),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--verify-timeout", str(args.verify_timeout),
            "--resume",
        ]
        if args.no_verify:
            cmd.append("--no-verify")
        cmd.append("--no-sweep" if args.no_sweep else "--sweep")
        if args.limit_starts is not None:
            cmd += ["--limit-starts", str(args.limit_starts)]
        if args.rank_stride is not None:
            cmd += ["--rank-stride", str(args.rank_stride)]
        if args.ranks is not None:
            cmd += ["--ranks", args.ranks]

        attempts = 0
        status = "failed"
        sub_t0 = time.perf_counter()
        while attempts <= max(0, args.retries):
            attempts += 1
            progress.update(run_id, variant=variant, optimizer=optimizer, status="running", attempts=attempts)
            print(f"{prefix}  RUN (attempt {attempts}) ...")
            log_path = logs_dir / f"{run_id}.log"
            outcome, rc = _run_subprocess(cmd, log_path, args.subgroup_timeout)
            if outcome == "ok" and _is_complete(run_dir, expected_n):
                status = "done"
                break
            print(f"{prefix}  attempt {attempts} {outcome} (rc={rc}); see {log_path}")
        wall = time.perf_counter() - sub_t0

        verified = None
        if status == "done":
            try:
                stats = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
                verified = stats.get("verified_clean_count")
            except (json.JSONDecodeError, OSError):
                pass
            done += 1
        else:
            failed += 1
        progress.update(
            run_id,
            variant=variant,
            optimizer=optimizer,
            status=status,
            attempts=attempts,
            wall_s=round(wall, 1),
            verified_clean_count=verified,
        )

        elapsed = time.perf_counter() - grid_t0
        completed = done + skipped + failed
        eta = (elapsed / completed) * (len(plan) - completed) if completed else 0.0
        vtxt = f" verified={verified}" if verified is not None else ""
        print(
            f"{prefix}  {status.upper()}{vtxt}  ({wall:.0f}s)  | "
            f"progress {completed}/{len(plan)} done={done} skip={skipped} fail={failed} | "
            f"elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m"
        )

    print("=" * 78)
    print(
        f"[run_sweep] FINISHED: done={done} skipped={skipped} failed/blocked={failed} "
        f"of {len(plan)} in {(time.perf_counter()-grid_t0)/60:.1f} min"
    )
    print(f"[run_sweep] progress -> {progress.path}")
    print(f"[run_sweep] per-subgroup logs -> {logs_dir}")

    sim_rc = 0
    if not args.no_sim:
        # Performance pass over whatever finished (resumable; safe even if some
        # subgroups failed — it just evaluates the complete ones).
        sim_rc = _run_sim_pass(args, progress)
    else:
        print("[run_sweep] --no-sim: skipping the quick-sim performance pass")

    return 0 if (failed == 0 and sim_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
